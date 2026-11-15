"""Hackathon backlog: one-click sample (VD-01), output contract (VD-02), drift decision consistency (VD-03),
review queue from the same decisions (VD-04), reading-passage handling (VD-05), calibration artifacts (VD-07)."""
import json

import numpy as np
import pytest

from server import store


# ------------------------------------------------------------------ VD-01 + VD-02
def test_one_click_sample_runs_the_real_pipeline_and_fills_the_contract(client):
    r = client.post("/api/demo/sample")
    assert r.status_code == 200, r.text
    j = r.json()
    ck = j["checkin"]
    assert j["baseline_built_now"] is True and "synthetic" in j["label"]
    assert ck["source"] == "demo" and ck["synthetic_voice"] is True
    assert ck["quality_status"] in ("pass", "warning") and ck["speech_duration_seconds"] > 15
    assert ck["drift_status"] in ("within_range", "elevated") and ck["drift_threshold"] == 1.0
    assert ck["drift_score"] == pytest.approx(ck["drift"]["d"] / ck["drift"]["tau"], abs=1e-3)
    assert (ck["drift_status"] == "elevated") == (ck["drift"]["d"] > ck["drift"]["tau"])
    for k in ("checkin_id", "patient_id", "created_at", "speaking_rate_change_pct", "pause_duration_change_pct",
              "review_required", "model_version"):
        assert k in ck
    assert ck["acoustic_features"]["pause_fraction"]["status"] == "VALID"
    assert ck["evidence"] == [] and ck["protocol"]["task_type"] == "read_speech"   # VD-05: no symptom mining
    # second click is served from the measured snapshot: same result, no new analysis
    j2 = client.post("/api/demo/sample").json()
    assert j2["cached"] is True and j2["checkin"]["checkin_id"] == ck["checkin_id"] and j["cached"] is False
    assert j2["ran_at"] and j2["model"] and j2["compute_seconds"] >= 0
    # fresh=1 re-measures live and reuses the baseline; tmp stays empty; bundled files are never deleted
    j3 = client.post("/api/demo/sample", params={"fresh": 1}).json()
    assert j3["cached"] is False and j3["baseline_built_now"] is False and j3["checkin"]["checkin_id"] != ck["checkin_id"]
    assert list(client.tmp_path.iterdir()) == []
    tl = client.get("/api/patients/p_sample/timeline").json()
    assert tl["patient"]["data_mode"] == "demo_synthetic_voice" and len(tl["checkins"]) == 7
    assert all(c["data_mode"] == "demo_synthetic_voice" for c in tl["checkins"])
    res = client.get("/api/results").json()
    assert res["sample"]["present"] and res["demo"]["profile_baseline_ready"]


def test_noise_replay_demo_works_with_bundled_audio(client, tmp_path, monkeypatch):
    from server.config import get_config
    monkeypatch.setitem(get_config()["results"], "demo_snapshot", str(tmp_path / "snap.json"))
    r = client.post("/api/demo/run")
    assert r.status_code == 200, r.text
    assert r.json()["noisy"]["data_mode"] == "controlled_experiment"


def test_demo_cache_is_served_instantly_never_stale_and_warm_fills_both(client, monkeypatch):
    import time
    from server.config import get_config
    w = client.post("/api/demo/warm").json()
    assert w["state"] == "ready" and w["sample_cached"] and w["replay_cached"]
    res = client.get("/api/results").json()
    assert res["sample_snapshot"] and res["demo_snapshot"] and res["demo_warm"]["state"] == "ready"
    for path in ("/api/demo/sample", "/api/demo/run"):
        t = time.perf_counter()
        j = client.post(path).json()
        assert j["cached"] is True and time.perf_counter() - t < 1.0      # instant: no analysis on click
    # anything that could change the result invalidates the snapshot: it is never served stale
    monkeypatch.setitem(get_config()["demo"], "snr_db", 5)
    assert client.get("/api/results").json()["demo_snapshot"] is None
    assert client.post("/api/demo/run").json()["cached"] is False             # re-measured with the new setting


def test_sample_cache_is_stale_once_its_checkin_is_gone(client):
    first = client.post("/api/demo/sample").json()
    assert client.post("/api/demo/sample").json()["cached"] is True
    assert client.delete("/api/patients/p_sample").status_code == 200       # e.g. profile deleted / fresh database
    assert client.get("/api/results").json()["sample_snapshot"] is None
    again = client.post("/api/demo/sample").json()
    assert again["cached"] is False and again["checkin"]["checkin_id"] != first["checkin"]["checkin_id"]
    assert client.get(f"/api/checkins/{again['checkin']['checkin_id']}").status_code == 200


def test_real_recordings_are_refused_on_synthetic_profiles(client, wav_bytes):
    client.post("/api/demo/reset", json={"scenario": "drift"})
    r = client.post("/api/patients/p_drift/checkins", files={"audio": ("a.wav", wav_bytes(seconds=40), "audio/wav")},
                    data={"prompt_id": "standard_reading_v1"})
    assert r.status_code == 409 and "synthetic demo profile" in r.json()["detail"]
    # the frontend must never run a stale script after an update
    for path in ("/", "/static/app.js", "/static/styles.css"):
        assert client.get(path).headers.get("cache-control") == "no-cache"
    assert list(client.tmp_path.iterdir()) == []                              # nothing written for the refused upload


def test_missing_sample_audio_is_a_clear_409_not_a_crash(client, monkeypatch):
    from server.config import get_config
    monkeypatch.setitem(get_config()["demo"], "sample", {**get_config()["demo"]["sample"], "recording": "nope.wav"})
    assert client.post("/api/demo/sample").status_code == 409
    assert client.get("/api/results").json()["sample"]["present"] is False
    assert client.get("/").status_code == 200


# ------------------------------------------------------------------ VD-03
def _scored(tmp_env, d, tau=1.47):
    from server import scoring
    from server.pipeline import drift as D
    dm = D.fallback_model()
    with store.session() as s:
        s.add(store.Patient(id="p_b", display_name="b")); s.commit()
        s.merge(store.Baseline(patient_id="p_b", n=5, mu=store.vec_to_blob(np.zeros(dm.k)),
                               sigma=store.vec_to_blob(np.ones(dm.k)), timing_mu={}, timing_sigma={},
                               frozen_at=store.utcnow()))
        s.commit()
        ck = store.Checkin(id="c_b", patient_id="p_b", quality={"verdict": "PASS"}, versions={}, extra={})
        orig = D.drift_score
        D.drift_score = lambda *a, **k: d                     # isolate the comparison itself
        try:
            scoring.apply_scoring(s, ck, vec=np.ones(768), vec_raw=np.ones(768), timing=None,
                                  safety={"triggered": False, "source": None}, dm=dm)
        finally:
            D.drift_score = orig
        return ck, scoring.contract(ck)


def test_boundary_equality_is_within_range_and_status_matches(tmp_env):
    from server.pipeline import drift as D
    tau = D.fallback_model().tau
    ck, c = _scored(tmp_env, tau)
    assert ck.drift["exceeds"] is False and c["drift_status"] == "within_range" and ck.status == "GREEN"


def test_just_above_boundary_is_elevated(tmp_env):
    from server.pipeline import drift as D
    tau = D.fallback_model().tau
    ck, c = _scored(tmp_env, tau + 1e-3)
    assert ck.drift["exceeds"] is True and c["drift_status"] == "elevated" and ck.status in ("AMBER", "RED")
    assert c["drift_score"] > 1.0


def test_missing_baseline_is_unavailable(client, wav_bytes):
    r = client.post("/api/patients/p_nb/checkins", files={"audio": ("a.wav", wav_bytes(seconds=40), "audio/wav")},
                    data={"prompt_id": "standard_reading_v1"}).json()
    from tests.conftest import wait_result
    res = wait_result(client, r["checkin_id"])
    full = client.get(f"/api/checkins/{res['checkin_id']}").json()
    assert full["drift_status"] == "unavailable" and full["drift_score"] is None and full["review_required"] is False


# ------------------------------------------------------------------ VD-04
def test_queue_items_carry_trigger_score_and_reviewed_leaves_pending(client):
    client.post("/api/demo/reset", json={"scenario": "all"})
    pend = client.get("/api/v1/review-flags", params={"status": "OPEN,ACKNOWLEDGED"}).json()
    assert [f["patient_id"] for f in pend] == ["p_drift"]                   # stable profile: no false item
    f = pend[0]
    assert f["trigger"]["normalized_score"] > 1.0 and f["trigger"]["threshold"] == 1.0 and f["trigger"]["checkin_id"]
    assert f["review_state"] == "pending"
    assert f["qualifying_checkins"] == 2 and f["considered_checkins"] == 3      # "2 of 3", stated explicitly
    tl = client.get("/api/patients/p_drift/timeline").json()
    flagged = [c for c in tl["checkins"] if c.get("review_flag")]
    assert flagged and all(c["drift_status"] == "elevated" for c in flagged)  # same decisions as the queue
    assert f["trigger"]["checkin_id"] in {c["checkin_id"] for c in flagged}
    fid = f["flag_id"]
    act = lambda a: client.post(f"/api/v1/review-flags/{fid}/actions", json={"action": a, "actor": "r", "note": "ok"})
    act("acknowledge"); assert act("review").json()["review_state"] == "reviewed"
    assert client.get("/api/v1/review-flags", params={"status": "OPEN,ACKNOWLEDGED"}).json() == []
    assert client.get("/api/v1/notifications").json()["unresolved"] == 0


# ------------------------------------------------------------------ synthetic drift showcase spec (another.md)
def test_seeded_sam_raises_exactly_one_open_review_by_the_rule_and_reset_is_idempotent(client):
    for _ in range(2):                                                       # loading twice: still exactly one
        assert client.post("/api/demo/reset", json={"scenario": "all"}).status_code == 200
        open_ = client.get("/api/v1/review-flags", params={"status": "OPEN"}).json()
        assert [f["patient_id"] for f in open_] == ["p_drift"]
    fid = open_[0]["flag_id"]
    d = client.get(f"/api/v1/review-flags/{fid}").json()
    qual = sorted(e["recorded_at"][:10] for e in d["evidence"] if e["role"] == "qualifying")
    tl = client.get("/api/patients/p_drift/timeline", params={"days": 365}).json()
    by_day = {c["created_at"][:10]: c for c in tl["checkins"]}
    assert qual == sorted(day for day, c in by_day.items() if c.get("acted"))  # the two acted days (Nov 13, 14)
    assert d["audit"][0]["action"] == "created"                              # made by the rule, not inserted
    # one boundary everywhere: baseline, timeline and every scored check-in
    taus = {c["drift"]["tau"] for c in tl["checkins"] if c.get("drift")}
    assert taus == {tl["tau"]} == {tl["baseline"]["tau"]}
    last = tl["checkins"][-1]
    assert last["drift_status"] == "elevated" and last["review_flag"]["flag_id"] == fid
    assert sum(1 for c in tl["checkins"] if c.get("drift") and c["drift"]["exceeds"]) == 2
    pts = {p["id"]: p for p in client.get("/api/patients").json()}
    assert pts["p_drift"]["status"] == last["status"] and pts["p_drift"]["open_review_flags"] == 1
    assert pts["p_stable"]["open_review_flags"] == 0
    assert client.get("/api/v1/review-flags", params={"status": "OPEN"}).json()[0]["patient_id"] == "p_drift"
    # acknowledge -> leaves OPEN; a fresh demo load brings it back as OPEN
    client.post(f"/api/v1/review-flags/{fid}/actions", json={"action": "acknowledge", "actor": "r", "note": ""})
    assert client.get("/api/v1/review-flags", params={"status": "OPEN"}).json() == []
    client.post("/api/demo/reset", json={"scenario": "all"})
    assert [f["status"] for f in client.get("/api/v1/review-flags", params={"status": "OPEN"}).json()] == ["OPEN"]


# ------------------------------------------------------------------ VD-07
def test_aggregate_writes_traceable_calibration_artifacts(tmp_path):
    pytest.importorskip("torch")
    from research.lite import aggregate
    from tests.test_results import _lite_fixture
    feats, ck = _lite_fixture(tmp_path)
    out_dir = tmp_path / "results"
    aggregate.run(feats, ck, layer=6, out_path=out_dir / "aggregate.json", B=50, target_rate=0.10)
    cal = json.loads((out_dir / "calibration.json").read_text())
    meta = json.loads((out_dir / "evaluation_metadata.json").read_text())
    met = json.loads((out_dir / "evaluation_metrics.json").read_text())
    assert cal["target_clean_false_alert_rate"] == 0.10 and set(cal["methods"]) == {"B0", "B1", "P1"}
    for m in cal["methods"].values():
        assert m["thresholds_by_speaker"] and "observed_clean_false_alert_rate" in m
    assert meta["generated_at"] == met["generated_at"] == cal["generated_at"]
    assert meta["seeds"]["bootstrap"] == 0 and meta["splits"]["eval_speakers"] and meta["splits"]["train_speakers"]
    assert "source_commit" in meta and meta["calibration_method"]
    assert "sample_size_note" in cal

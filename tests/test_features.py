"""Final feature spec v1.0: standardized protocol (F1), quality (F2), verification (F3), acoustic evidence (F4),
patient context (F5) and review resolution (F6)."""
from datetime import timedelta

import numpy as np
import pytest

from server import monitoring, protocol, store, verification
from server.pipeline import evidence
from server.pipeline.decode import write_wav
from tests.conftest import wait_result
from tests.test_monitoring import T0, _ck, _patient

STD = protocol.STANDARD_ID


def post(client, pid, wav, prompt_id="default", **extra):
    r = client.post(f"/api/patients/{pid}/checkins", files={"audio": ("a.wav", wav, "audio/wav")},
                    data={"prompt_id": prompt_id, **extra})
    assert r.status_code == 202, r.text
    return wait_result(client, r.json()["checkin_id"])


# ------------------------------------------------------------------ F1 protocol
def test_protocol_endpoint_and_prompt_metadata_saved(client, wav_bytes):
    p = client.get("/api/checkin/protocol").json()
    assert p["prompt_id"] == STD and p["prompt_version"] == 1 and p["task_type"] == "read_speech"
    assert len(p["text"].split()) >= 100 and client.get("/api/prompt").json() == p
    r = post(client, "p_f1", wav_bytes(seconds=40), prompt_id=STD)
    assert r["protocol"] == {"prompt_id": STD, "prompt_version": 1, "protocol_version": "1.0", "task_type": "read_speech"}
    bad = client.post("/api/patients/p_f1/checkins", files={"audio": ("a.wav", wav_bytes(seconds=40), "audio/wav")},
                      data={"prompt_id": "made_up"})
    assert bad.status_code == 422


def test_incompatible_prompt_withholds_comparison(client, wav_bytes):
    for i in range(5):
        post(client, "p_inc", wav_bytes(seconds=40, seed=i))                 # legacy free-speech baseline
    r = post(client, "p_inc", wav_bytes(seconds=40, seed=9), prompt_id=STD)
    assert r["status"] == "INCOMPATIBLE" and r["drift"] is None
    assert r["analysis"]["status"] == "INCOMPATIBLE_PROTOCOL" and r["comparison"]["baseline_prompt"]["prompt_id"] == "default"
    pv = client.get(f"/api/checkins/{r['checkin_id']}", params={"role": "patient"}).json()
    assert "different speaking task" in pv["message"]


# ------------------------------------------------------------------ F2 quality
def test_no_speech_rejected_with_reason_codes_and_baseline_unchanged(client, wav_bytes, tmp_path):
    first = post(client, "p_q", wav_bytes(seconds=40, seed=1))
    assert first["baseline_progress"] == [1, 5] and first["quality"]["status"] == "PASS"
    q = first["quality"]
    for k in ("duration_seconds", "speech_seconds", "speech_fraction", "clipping_fraction", "rms_amplitude"):
        assert k in q and q[k] is not None
    p = tmp_path / "silence.wav"
    write_wav(p, (1e-4 * np.random.default_rng(0).standard_normal(16000 * 30)).astype(np.float32))
    rej = post(client, "p_q", p.read_bytes())
    assert rej["quality"]["status"] == "REJECT" and rej["drift"] is None and rej["analysis"]["status"] == "NOT_SCORED"
    assert rej["quality"]["reason_codes"] and "acoustic_features" in rej and rej["acoustic_features"] is None
    nxt = post(client, "p_q", wav_bytes(seconds=40, seed=2))
    assert nxt["baseline_progress"] == [2, 5]                                  # the rejected recording didn't count
    pv = client.get(f"/api/checkins/{nxt['checkin_id']}", params={"role": "patient"}).json()
    assert pv["quality"]["message"] == "Recording quality passed."


def test_quality_excessive_silence_rule():
    from server.config import get_config
    from server.pipeline import quality
    x = np.zeros(16000 * 60, np.float32)
    t = np.arange(16000 * 20) / 16000
    x[:len(t)] = 0.3 * np.sin(2 * np.pi * 150 * t) * (np.sin(2 * np.pi * 2 * t) > -0.6)   # 20 s of 60 s
    q = quality.assess(x, get_config())
    assert q.verdict == "REJECT" and q.to_json()["reason_codes"][0] in ("EXCESSIVE_SILENCE", "INSUFFICIENT_SPEECH")


def test_zero_norm_embedding_is_refused(tmp_env):
    from server import scoring
    from server.pipeline import drift
    with store.session() as s:
        s.add(store.Patient(id="p_z", display_name="z")); s.commit()
        ck = store.Checkin(id="c_z", patient_id="p_z", quality={"verdict": "PASS"}, versions={})
        with pytest.raises(ValueError):
            scoring.apply_scoring(s, ck, vec=np.zeros(768), vec_raw=np.ones(768), timing=None,
                                  safety={"triggered": False, "source": None}, dm=drift.fallback_model())


# ------------------------------------------------------------------ F3 verification
def test_classification_is_deterministic():
    base = dict(repeat_quality_ok=True, comparable=True, repeat_scored=True, calibrated=True, original_exceeded=True)
    assert verification.classify(**base, repeat_exceeded=True) == ("REPRODUCED", None)
    assert verification.classify(**base, repeat_exceeded=False) == ("NOT_REPRODUCED", None)
    assert verification.classify(**{**base, "repeat_quality_ok": False}, repeat_exceeded=True)[0] == "INCONCLUSIVE"
    assert verification.classify(**{**base, "comparable": False}, repeat_exceeded=True)[0] == "INCONCLUSIVE"
    assert verification.classify(**{**base, "calibrated": False}, repeat_exceeded=True)[0] == "UNCALIBRATED"


def _real(s, pid, i, when, r, artifacts="v3", **extra):
    cid = _ck(s, pid, i, when, r=r, simulated=False, artifacts=artifacts, encoder="wavlm-base+head")
    ck = s.get(store.Checkin, cid)
    ck.extra = {**ck.extra, "protocol": protocol.meta(STD), **extra}
    s.add(ck); s.commit()
    return ck


def test_offer_and_evaluate_reproduced_inconclusive_uncalibrated(tmp_env):
    with store.session() as s:
        pid = _patient(s, "p_v", simulated=False)
        orig = _real(s, pid, 5, T0 + timedelta(days=5), 1.6)
        off = verification.offer(s, orig)
        assert off["available"] and off["kind"] == "VERIFY"
        rep = _real(s, pid, 6, T0 + timedelta(days=5, hours=1), 1.4, verification_of=orig.id)
        assert verification.evaluate(s, rep)["result"] == "REPRODUCED"
        assert not verification.offer(s, orig)["available"]                    # superseded by its repeat
        rep2 = _real(s, pid, 7, T0 + timedelta(days=5, hours=2), 0.6, verification_of=orig.id)
        assert verification.evaluate(s, rep2)["result"] == "NOT_REPRODUCED"
        bad = _real(s, pid, 8, T0 + timedelta(days=5, hours=3), None, verification_of=orig.id)
        bad.quality = {"verdict": "REJECT"}; s.add(bad); s.commit()
        assert verification.evaluate(s, bad)["result"] == "INCONCLUSIVE"
        # fallback boundary: a manual repeat is offered, but nothing is classified
        o2 = _real(s, pid, 9, T0 + timedelta(days=6), 1.8, artifacts="fallback")
        assert verification.offer(s, o2)["kind"] == "MANUAL_REPEAT"
        r2 = _real(s, pid, 10, T0 + timedelta(days=6, hours=1), 1.8, artifacts="fallback", verification_of=o2.id)
        assert verification.evaluate(s, r2)["result"] == "UNCALIBRATED"
        within = _real(s, pid, 11, T0 + timedelta(days=7), 0.5)
        assert not verification.offer(s, within)["available"]


def test_same_day_repeat_never_adds_to_persistence(tmp_env):
    with store.session() as s:
        pid = _patient(s, "p_rep")
        for i in range(5):
            _ck(s, pid, i, T0 + timedelta(days=i))
        a = _ck(s, pid, 5, T0 + timedelta(days=5), r=1.6)
        rep = _ck(s, pid, 6, T0 + timedelta(days=5, hours=1), r=1.7)
        ck = s.get(store.Checkin, rep); ck.extra = {**ck.extra, "verification_of": a}; s.add(ck); s.commit()
        b = _ck(s, pid, 7, T0 + timedelta(days=6), r=0.5)
        for c in (a, rep):
            monitoring.evaluate(s, pid, c); s.commit()
        assert monitoring.evaluate(s, pid, rep)["monitoring_state"] == "VERIFICATION_REPEAT"
        out = monitoring.evaluate(s, pid, b); s.commit()
        assert rep not in out["persistence"]["window_checkins"] and out["persistence"]["elevated_sessions"] == 1
        assert s.exec(store.select(store.ReviewFlag)).all() == []


def test_api_repeat_flow_links_and_validates(client, wav_bytes):
    for i in range(5):
        post(client, "p_api", wav_bytes(seconds=40, seed=i), prompt_id=STD)
    r = post(client, "p_api", wav_bytes(seconds=40, seed=7), prompt_id=STD)
    v = client.get(f"/api/checkins/{r['checkin_id']}/verification").json()
    assert v["original"]["checkin_id"] == r["checkin_id"] and v["repeats"] == []
    resp = client.post("/api/patients/p_api/checkins", files={"audio": ("a.wav", wav_bytes(seconds=40), "audio/wav")},
                       data={"prompt_id": "default", "verification_of": r["checkin_id"]})
    if v["offer"]["available"]:
        assert resp.status_code == 409                                         # repeat must use the same passage
        rep = post(client, "p_api", wav_bytes(seconds=40, seed=8), prompt_id=STD, verification_of=r["checkin_id"])
        assert rep["checkin_type"] == "verification" and rep["verification"]["result"] in (
            "UNCALIBRATED", "INCONCLUSIVE")                                    # no trained boundary in tests
        assert len(client.get(f"/api/checkins/{r['checkin_id']}/verification").json()["repeats"]) == 1
    else:
        assert resp.status_code == 409                                         # nothing elevated: no repeat
    assert client.post("/api/patients/p_other/checkins", files={"audio": ("a.wav", wav_bytes(seconds=40), "audio/wav")},
                       data={"prompt_id": STD, "verification_of": r["checkin_id"]}).status_code == 404


# ------------------------------------------------------------------ F4 acoustic evidence
def test_pause_fraction_rules():
    m = np.zeros(400, bool)
    m[20:120] = True; m[125:200] = True; m[220:320] = True                  # 5-frame (150 ms) gap, 20-frame gap
    pf = evidence.pause_fraction(m)
    assert pf == pytest.approx(20 / 300, abs=1e-3)                         # short gap and edges excluded
    assert evidence.pause_fraction(np.zeros(100, bool)) is None
    assert evidence.pause_fraction(m[:110]) is None or evidence.pause_fraction(m[:110]) >= 0


def test_compare_marks_missing_as_unavailable_not_zero():
    cur = {"pause_fraction": {"value": 0.21}, "speaking_rate_wpm": {"value": None, "reason": "no transcript"},
           "pitch_variability": {"value": None}}
    out = evidence.compare(cur, [{"pause_fraction": {"value": 0.10}}, {"pause_fraction": {"value": 0.14}}])
    assert out["pause_fraction"]["status"] == "VALID" and out["pause_fraction"]["delta"] == pytest.approx(0.09)
    assert out["pause_fraction"]["baseline"] == pytest.approx(0.12) and out["pause_fraction"]["relative"] == pytest.approx(0.75)
    assert out["speaking_rate_wpm"]["status"] == "UNAVAILABLE" and out["speaking_rate_wpm"]["current"] is None
    assert evidence.describe(out)[0].startswith("Your latest recording contained more pause time")
    nob = evidence.compare(cur, [{}])
    assert nob["pause_fraction"]["status"] == "NO_BASELINE" and nob["pause_fraction"]["delta"] is None


def test_live_checkin_reports_acoustic_comparison(client, wav_bytes):
    for i in range(5):
        post(client, "p_ac", wav_bytes(seconds=40, seed=i), prompt_id=STD)
    r = post(client, "p_ac", wav_bytes(seconds=40, seed=11), prompt_id=STD)
    pf = r["acoustic_features"]["pause_fraction"]
    assert pf["status"] == "VALID" and pf["baseline_n"] == 5 and 0 < pf["current"] < 1
    assert r["acoustic_features"]["pitch_variability"]["status"] == "UNAVAILABLE"
    assert r["acoustic_summary"]
    pv = client.get(f"/api/checkins/{r['checkin_id']}", params={"role": "patient"}).json()
    assert pv["acoustic_features"]["pause_fraction"]["current"] == pf["current"]


# ------------------------------------------------------------------ F5 context
def test_context_saved_without_changing_scores(client, wav_bytes):
    for i in range(6):
        r = post(client, "p_ctx", wav_bytes(seconds=40, seed=i))
    before = r["drift"]
    out = client.patch(f"/api/checkins/{r['checkin_id']}/context", json={"tags": ["TIRED", "DIFFERENT_ROOM_OR_MIC"],
                                                                         "note": "new headset"}).json()
    assert out["context"]["tags"] == ["DIFFERENT_ROOM_OR_MIC", "TIRED"]
    after = client.get(f"/api/checkins/{r['checkin_id']}").json()
    assert after["drift"] == before and after["context"]["note"] == "new headset"
    assert client.patch(f"/api/checkins/{r['checkin_id']}/context", json={"tags": ["SICK"]}).status_code == 422
    assert client.patch("/api/checkins/nope/context", json={"tags": []}).status_code == 404


def test_context_note_is_safety_screened(client, wav_bytes):
    """A reading passage can't surface crisis language, but the typed note can: it gets the same screen."""
    r = post(client, "p_note", wav_bytes(seconds=40))
    ok = client.patch(f"/api/checkins/{r['checkin_id']}/context", json={"tags": ["TIRED"], "note": "slept badly"}).json()
    assert "safety" not in ok
    hit = client.patch(f"/api/checkins/{r['checkin_id']}/context",
                       json={"tags": [], "note": "honestly some days I just want to die"}).json()
    assert hit["safety"]["triggered"] and "988" in hit["safety"]["message"]
    tl = client.get("/api/patients/p_note/timeline").json()
    assert [a["kind"] for a in tl["alerts"]] == ["SAFETY"]                 # the care team sees it
    client.patch(f"/api/checkins/{r['checkin_id']}/context", json={"tags": [], "note": "I want to die"})
    assert len(client.get("/api/patients/p_note/timeline").json()["alerts"]) == 1   # no duplicate alert
    assert client.get(f"/api/checkins/{r['checkin_id']}").json()["drift"] == r["drift"]   # scores unchanged


# ------------------------------------------------------------------ F6 review resolution
def test_request_repeat_close_with_reason_and_no_external_delivery(client):
    client.post("/api/demo/reset", json={"scenario": "drift"})
    fid = client.get("/api/v1/review-flags").json()[0]["flag_id"]
    act = lambda a, note="": client.post(f"/api/v1/review-flags/{fid}/actions", json={"action": a, "actor": "r", "note": note})
    assert act("request_repeat").status_code == 200
    assert client.get("/api/patients/p_drift/timeline").json()["repeat_requested"] is True
    assert act("acknowledge").json()["status"] == "ACKNOWLEDGED"
    assert act("close").status_code == 409                                       # reason required
    d = act("close", "Reviewed repeated speech change.").json()
    assert d["status"] == "CLOSED" and [a["action"] for a in d["audit"]][-1] == "close"
    assert d["audit"][-1]["detail"]["note"] == "Reviewed repeated speech change."
    assert all(n["external_delivery"] is False for n in d["notifications"])
    ev = d["evidence"][0]
    assert {"acoustic_features", "context", "quality", "normalized_score"} <= set(ev)

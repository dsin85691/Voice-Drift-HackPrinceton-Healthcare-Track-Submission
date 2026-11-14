import json

import pytest

from server.pipeline import asr as asr_mod
from tests.conftest import wait_result


def post(client, pid, wav):
    r = client.post(f"/api/patients/{pid}/checkins", files={"audio": ("a.wav", wav, "audio/wav")},
                    data={"prompt_id": "default"})
    assert r.status_code == 202
    return r.json()["checkin_id"]


def test_health(client):
    h = client.get("/api/health").json()
    assert h["models_loaded"] and "artifacts_version" in h and h["cohort_risk"] is False


def test_calibration_endpoint_without_artifacts(client):
    assert client.get("/api/calibration").json()["available"] is False


def test_live_checkin_without_classifier_reports_cohort_unavailable(client, wav_bytes):
    for i in range(6):
        cid = post(client, "p_cr", wav_bytes(seconds=40, seed=i))
        r = wait_result(client, cid)
    assert r["cohort_risk"]["available"] is False and "prob" not in r["cohort_risk"]


def test_live_checkin_baseline_then_scored_and_tmp_empty(client, wav_bytes):
    for i in range(6):
        cid = post(client, "p_live", wav_bytes(seconds=40, seed=i))
        res = wait_result(client, cid)
    assert res["quality"]["verdict"] == "PASS"
    assert res["drift"] is not None and res["drift"]["baseline_n"] == 5       # 6th is scored
    assert res["status"] in ("GREEN", "AMBER", "RED")
    assert "latency_ms" in res and res["disclaimer"].startswith("Screening")
    assert list(client.tmp_path.iterdir()) == []                              # audio deleted


def test_first_five_are_baseline_and_patient_sees_no_numbers(client, wav_bytes):
    cid = post(client, "p_live", wav_bytes(seconds=40))
    wait_result(client, cid)
    pv = client.get(f"/api/checkins/{cid}", params={"role": "patient"}).json()
    assert pv["baseline_progress"] == [1, 5] and "Building your baseline (1/5)" in pv["message"]
    for forbidden in ("drift", "reference_risk", "cohort_risk", "transcript", "evidence", "timing"):
        assert forbidden not in pv


def test_reject_noisy_and_too_short(client, wav_bytes):
    cid = post(client, "p_live", wav_bytes(seconds=40, snr_db=2))
    r = wait_result(client, cid)
    assert r["quality"]["verdict"] == "REJECT" and r["drift"] is None
    cid = post(client, "p_live", wav_bytes(seconds=8))
    assert wait_result(client, cid)["quality"]["verdict"] == "REJECT"
    pv = client.get(f"/api/checkins/{cid}", params={"role": "patient"}).json()
    assert pv["message"]


def test_audio_deleted_on_error_path_100_checkins(client):
    ids = [post(client, "p_err", b"not audio at all %d" % i) for i in range(100)]
    for cid in ids:
        r = wait_result(client, cid, timeout=60)
        assert r["quality"]["verdict"] == "REJECT"
    assert list(client.tmp_path.iterdir()) == []


def _fake_transcript(client, text):
    rt = client.app.state.holder["rt"]
    rt.asr.transcribe = lambda x: asr_mod.Transcript(text, [], True, "fake")
    return rt


def test_safety_lexicon_works_with_llm_down_and_hides_scores(client, wav_bytes):
    _fake_transcript(client, "Honestly some days I just want to die.")
    cid = post(client, "p_safe", wav_bytes(seconds=40))
    r = wait_result(client, cid)
    assert r["status"] == "SAFETY" and r["safety"] == {"triggered": True, "source": "lexicon"}
    assert "drift" not in r and "reference_risk" not in r and "cohort_risk" not in r
    pv = client.get(f"/api/checkins/{cid}", params={"role": "patient"}).json()
    assert pv["status"] == "SAFETY" and "988" in pv["safety"]["message"]
    tl = client.get("/api/patients/p_safe/timeline").json()
    assert [a["kind"] for a in tl["alerts"]] == ["SAFETY"]
    assert client.get("/api/patients").json()[0]["status"] == "SAFETY"


def test_llm_safety_flag_triggers(client, wav_bytes):
    rt = _fake_transcript(client, "Things feel heavy lately.")
    rt.language.analyze = lambda t: {"available": True, "safety_flag": True, "evidence": [],
                                     "evidence_dropped": 0, "summary": ""}
    cid = post(client, "p_safe2", wav_bytes(seconds=40))
    r = wait_result(client, cid)
    assert r["status"] == "SAFETY" and r["safety"]["source"] == "llm" and "drift" not in r


def test_sse_stages_then_result(client, wav_bytes):
    cid = post(client, "p_sse", wav_bytes(seconds=40))
    wait_result(client, cid)
    body = client.get(f"/api/checkins/{cid}/events", params={"role": "patient"}).text
    names = [l.split(": ", 1)[1] for l in body.splitlines() if l.startswith("event: ")]
    assert names[-1] == "result" and names.count("stage") >= 4
    assert '"drift"' not in body


def test_scenarios_alerts_ack_compare_and_delete(client):
    assert client.post("/api/demo/reset", json={"scenario": "all"}).status_code == 200
    pts = {p["id"]: p for p in client.get("/api/patients").json()}
    assert pts["p_drift"]["status"] == "RED" and pts["p_stable"]["status"] == "GREEN"
    assert client.get("/api/patients").json()[0]["id"] == "p_drift"          # severity sort
    tl = client.get("/api/patients/p_drift/timeline").json()
    assert tl["baseline"]["n"] == 5 and any(a["kind"] == "DRIFT" for a in tl["alerts"])
    assert tl["checkins"][-1]["simulated"] and tl["checkins"][-1]["acted"]
    aid = next(a["id"] for a in tl["alerts"] if a["kind"] == "DRIFT")
    acked = client.post(f"/api/alerts/{aid}/ack", json={"clinician": "dr_x", "note": "called patient"}).json()
    assert acked["acked_by"] == "dr_x"
    # robustness: noisy stable day crosses tau with raw features but not with the invariance head
    stable = client.get("/api/patients/p_stable/timeline").json()["checkins"]
    noisy = next(c for c in stable if c["quality"]["verdict"] == "WARN")
    cmp = client.post("/api/demo/compare", json={"checkin_id": noisy["checkin_id"]}).json()
    assert cmp["raw"]["exceeds"] and not cmp["invariant"]["exceeds"]
    assert client.delete("/api/patients/p_drift").status_code == 200
    assert client.get("/api/patients/p_drift/timeline").status_code == 404
    assert client.get("/api/demo/nope").status_code == 404


def test_frontend_served(client):
    assert client.get("/").status_code == 200


"""Speech-change review flags (alerting spec v1.0): gates, persistence rule, lifecycle, dedup, cooldown, idempotency."""
from datetime import datetime, timedelta

import numpy as np
import pytest

from server import monitoring, store
from server.store import vec_to_blob

T0 = datetime(2026, 11, 1, 9, 0)


def _patient(s, pid="p_t", simulated=True, consent="demo_consent", enrolled="enrolled"):
    s.add(store.Patient(id=pid, display_name=pid, simulated=simulated, monitoring_enrollment_status=enrolled,
                        review_consent_status=consent, assigned_reviewer_id="rev_1"))
    s.commit()
    return pid


def _ck(s, pid, i, when, r=None, verdict="PASS", status=None, simulated=True, artifacts="seed", encoder="simulated"):
    """r=None -> baseline check-in (no drift). r=float -> scored check-in with normalized score r (tau = 1)."""
    ck = store.Checkin(id=f"{pid}_{i:02d}", patient_id=pid, created_at=when,
                       quality={"verdict": verdict}, session_vec=vec_to_blob(np.ones(4)) if verdict != "REJECT" else None,
                       drift=None if r is None else {"d": r, "tau": 1.0, "exceeds": r > 1, "weight": 1.0},
                       status=status or ("BASELINE" if r is None else ("AMBER" if r > 1 else "GREEN")),
                       versions={"encoder": encoder, "artifacts": artifacts},
                       extra={"simulated": simulated})
    s.add(ck); s.commit()
    return ck.id


def _history(s, pid, scores, start=T0, **kw):
    """5 baseline days, then one scored check-in per entry: (day_offset, r)."""
    for i in range(5):
        _ck(s, pid, i, start + timedelta(days=i), **kw)
    ids = []
    for j, (day, r) in enumerate(scores):
        ids.append(_ck(s, pid, 5 + j, start + timedelta(days=5 + day, hours=j % 3), r=r, **kw))
    return ids


@pytest.fixture()
def s(tmp_env):
    with store.session() as sess:
        yield sess


def ev(s, pid, cid):
    out = monitoring.evaluate(s, pid, cid)
    s.commit()
    return out


def test_baseline_and_insufficient_history(s):
    pid = _patient(s)
    ids = _history(s, pid, [(0, 2.0), (1, 2.0)])
    assert ev(s, pid, f"{pid}_04")["monitoring_state"] == "BASELINE_PENDING"
    out = ev(s, pid, ids[-1])
    assert out["monitoring_state"] == "INSUFFICIENT_DATA" and "INSUFFICIENT_HISTORY" in out["reason_codes"]
    assert out["flag"]["created"] is False


def test_two_of_three_on_distinct_days_creates_one_flag_with_evidence_and_minimal_notification(s):
    pid = _patient(s)
    ids = _history(s, pid, [(0, 0.5), (1, 1.6), (2, 1.9)])
    for c in ids[:-1]:
        ev(s, pid, c)
    out = ev(s, pid, ids[-1])
    assert out["monitoring_state"] == "REVIEW_ELIGIBLE" and out["flag"]["created"] and out["flag"]["status"] == "OPEN"
    assert out["persistence"] == {"eligible_sessions": 3, "elevated_sessions": 2, "distinct_days": 2,
                                  "window_checkins": ids, "rule_satisfied": True}
    assert out["notification"] == {"mode": "demo", "external_delivery": False}
    f = monitoring.flag_json(s.get(store.ReviewFlag, out["flag"]["flag_id"]), s, detail=True)
    assert f["summary"]["explanation"].startswith("2 of the last 3 quality-passing recordings exceeded")
    assert sorted(e["role"] for e in f["evidence"]) == ["qualifying", "qualifying", "supporting"]
    n = f["notifications"][0]
    assert n["external_delivery"] is False and n["delivery_status"] == "delivered_in_app"
    blob = str(n["payload"]).lower()
    assert pid not in blob and "1.9" not in blob and "depress" not in blob     # no id, score or interpretation
    assert [a["action"] for a in f["audit"]] == ["created", "queued"]


def test_same_day_crossings_and_old_crossings_do_not_count(s):
    pid = _patient(s)
    ids = _history(s, pid, [(0, 0.5), (1, 1.6), (1, 1.9)])                  # two crossings, same calendar day
    out = ev(s, pid, ids[-1])
    assert out["monitoring_state"] == "ELEVATED_OBSERVATION" and not out["persistence"]["rule_satisfied"]
    pid2 = _patient(s, "p_old")
    ids2 = _history(s, pid2, [(0, 1.8), (1, 1.8), (20, 1.8)])               # earlier crossings are > 7 days back
    out2 = ev(s, pid2, ids2[-1])
    assert out2["monitoring_state"] == "INSUFFICIENT_DATA"


def test_idempotent_and_no_duplicate_flag_for_the_same_episode(s):
    pid = _patient(s)
    ids = _history(s, pid, [(0, 1.5), (1, 1.6), (2, 1.7), (3, 1.8)])
    first = ev(s, pid, ids[2])
    again = ev(s, pid, ids[2])
    assert again["replayed"] and again["flag"] == first["flag"]
    later = ev(s, pid, ids[3])                                                # same episode continues
    assert later["flag"]["created"] is False and later["flag"]["flag_id"] == first["flag"]["flag_id"]
    assert "ACTIVE_FLAG_EXISTS" in later["reason_codes"]
    assert len(s.exec(store.select(store.ReviewFlag)).all()) == 1


def test_lifecycle_audit_and_invalid_transitions(s):
    pid = _patient(s)
    ids = _history(s, pid, [(0, 1.5), (1, 1.6), (2, 1.7)])
    fid = ev(s, pid, ids[-1])["flag"]["flag_id"]
    with pytest.raises(ValueError):
        monitoring.act(s, fid, "review", "rev_1")                             # must acknowledge first
    monitoring.act(s, fid, "acknowledge", "rev_1"); monitoring.act(s, fid, "note", "rev_1", "spoke to them")
    monitoring.act(s, fid, "review", "rev_1")
    with pytest.raises(ValueError):
        monitoring.act(s, fid, "close", "rev_1")                              # a documented reason is required
    monitoring.act(s, fid, "close", "rev_1", "reviewed, no action")
    s.commit()
    f = monitoring.flag_json(s.get(store.ReviewFlag, fid), s, detail=True)
    assert f["status"] == "CLOSED" and f["acknowledged_at"] and f["reviewed_at"] and f["closed_at"]
    assert [a["action"] for a in f["audit"]][-4:] == ["acknowledge", "note", "review", "close"]
    with pytest.raises(ValueError):
        monitoring.act(s, fid, "dismiss", "rev_1", "x")                       # closed is final


def test_cooldown_after_dismissal_then_new_flag(s):
    pid = _patient(s)
    now = datetime.utcnow().replace(microsecond=0)
    ids = _history(s, pid, [(0, 1.5), (1, 1.6), (2, 1.7)], start=now - timedelta(days=7, hours=12))
    fid = ev(s, pid, ids[-1])["flag"]["flag_id"]
    monitoring.act(s, fid, "dismiss", "rev_1", "context explains it"); s.commit()
    soon = _ck(s, pid, 20, datetime.utcnow() + timedelta(hours=1), r=1.9)
    out = ev(s, pid, soon)
    assert out["monitoring_state"] == "SUPPRESSED" and "COOLDOWN" in out["reason_codes"]
    later = _ck(s, pid, 21, datetime.utcnow() + timedelta(hours=80), r=1.9)
    out2 = ev(s, pid, later)
    assert out2["flag"]["created"] is True and out2["flag"]["flag_id"] != fid


def test_consent_gate_suppresses_flag(s):
    pid = _patient(s, consent="none")
    ids = _history(s, pid, [(0, 1.5), (1, 1.6), (2, 1.7)])
    out = ev(s, pid, ids[-1])
    assert out["monitoring_state"] == "SUPPRESSED" and "NOT_ENROLLED_OR_NO_REVIEW_CONSENT" in out["reason_codes"]
    assert s.exec(store.select(store.ReviewFlag)).all() == []


def test_uncalibrated_real_recordings_never_flag_and_rejects_are_ineligible(s):
    pid = _patient(s, simulated=False)
    ids = _history(s, pid, [(0, 3.0), (1, 3.0), (2, 3.0)], simulated=False, artifacts="fallback",
                   encoder="spectral-fallback")
    out = ev(s, pid, ids[-1])
    assert out["monitoring_state"] == "CALIBRATION_PENDING" and "CALIBRATION_PENDING" in out["reason_codes"]
    pid2 = _patient(s, "p_rej")
    _history(s, pid2, [(0, 1.5), (1, 1.6)])
    rej = _ck(s, pid2, 30, T0 + timedelta(days=8), r=None, verdict="REJECT", status="GREEN")
    assert "QUALITY_NOT_PASSED" in ev(s, pid2, rej)["reason_codes"]
    saf = _ck(s, pid2, 31, T0 + timedelta(days=8, hours=2), r=None, status="SAFETY")
    assert "SAFETY_PATH" in ev(s, pid2, saf)["reason_codes"]


def test_model_context_change_starts_a_new_window(s):
    pid = _patient(s)
    ids = _history(s, pid, [(0, 1.5), (1, 1.6)])
    new = _ck(s, pid, 40, T0 + timedelta(days=8), r=1.7, artifacts="v2")     # different projection version
    out = ev(s, pid, new)
    assert out["persistence"]["eligible_sessions"] == 1 and out["flag"]["created"] is False


# ---------------------------------------------------------------- API
def test_api_evaluate_forbids_client_thresholds_and_flag_endpoints(client):
    client.post("/api/demo/reset", json={"scenario": "drift"})
    cid = client.get("/api/patients/p_drift/timeline").json()["checkins"][-1]["checkin_id"]
    bad = client.post("/api/v1/monitoring/evaluate", json={"patient_id": "p_drift", "checkin_id": cid,
                                                           "session_threshold": 99})
    assert bad.status_code == 422
    assert client.post("/api/v1/monitoring/evaluate", json={"patient_id": "p_drift", "checkin_id": cid,
                                                            "data_mode": "real_recording"}).status_code == 409
    r = client.post("/api/v1/monitoring/evaluate", json={"patient_id": "p_drift", "checkin_id": cid}).json()
    assert r["replayed"] and r["monitoring_state"] == "REVIEW_ELIGIBLE"       # seeding already evaluated it
    flags = client.get("/api/v1/review-flags", params={"status": "open"}).json()
    assert len(flags) == 1 and flags[0]["flag"] == "Speech Change Review" and flags[0]["supporting_checkins"] == 3
    fid = flags[0]["flag_id"]
    d = client.get(f"/api/v1/review-flags/{fid}").json()
    assert d["summary"]["calibration"].startswith("illustrative") and d["limitations"]
    assert client.post(f"/api/v1/review-flags/{fid}/actions", json={"action": "review", "actor": "r"}).status_code == 409
    ok = client.post(f"/api/v1/review-flags/{fid}/actions", json={"action": "acknowledge", "actor": "r"}).json()
    assert ok["status"] == "ACKNOWLEDGED"
    n = client.get("/api/v1/notifications").json()
    assert n["external_delivery"] is False and n["unresolved"] == 1
    for word in ("risk", "depress", "diagnos"):
        assert word not in str(d).lower().replace("not a health condition", "")

"""Server side of the bug-fix specification (2026-11-15): progress stream ends, stable review ids, live recordings
survive "Load demo", status filter validation, local days for the 2-of-3 rule, daytime seed times."""
from datetime import datetime

from server import monitoring, store
from tests.conftest import wait_result


def _post(client, pid, wav, **data):
    r = client.post(f"/api/patients/{pid}/checkins", files={"audio": ("a.wav", wav, "audio/wav")},
                    data={"prompt_id": "default", **data})
    assert r.status_code == 202
    return r.json()["checkin_id"]


def _event_names(body: str) -> list[str]:
    return [l.split(": ", 1)[1] for l in body.splitlines() if l.startswith("event: ")]


# ------------------------------------------------------------------ P1-5: the progress stream always ends
def test_events_for_unknown_checkin_is_404(client):
    assert client.get("/api/checkins/c_nope/events", params={"role": "patient"}).status_code == 404


def test_events_for_finished_checkin_after_restart_sends_stored_result_and_ends(client, wav_bytes):
    cid = _post(client, "p_ev", wav_bytes(seconds=40))
    wait_result(client, cid)
    client.app.state.holder["rt"].bus._events.clear()          # what a server restart leaves behind
    body = client.get(f"/api/checkins/{cid}/events", params={"role": "patient"}).text
    assert _event_names(body) == ["result"]
    assert '"outcome"' in body and '"drift"' not in body      # still the patient view


# ------------------------------------------------------------------ P1-4: review ids survive "Load demo"
def _open_flags(client, pid="p_drift"):
    return [f for f in client.get("/api/v1/review-flags", params={"status": "OPEN,ACKNOWLEDGED"}).json()
            if f["patient_id"] == pid]


def test_review_flag_id_is_the_same_after_loading_demo_again(client):
    client.post("/api/demo/reset", json={"scenario": "all"})
    first = _open_flags(client)
    client.post("/api/demo/reset", json={"scenario": "all"})
    second = _open_flags(client)
    assert len(first) == len(second) == 1                       # M2: still exactly one open flag
    assert first[0]["flag_id"] == second[0]["flag_id"]           # I12: same id, so links keep working
    assert client.get(f"/api/v1/review-flags/{first[0]['flag_id']}").status_code == 200


def test_flag_id_depends_on_patient_rule_and_trigger():
    a = monitoring.flag_id_for("p", "r", "c1")
    assert a == monitoring.flag_id_for("p", "r", "c1") and a != monitoring.flag_id_for("p", "r", "c2")


# ------------------------------------------------------------------ P1-7: "Load demo" keeps live recordings
def test_load_demo_all_keeps_live_recordings_and_blank_clears_them(client, wav_bytes):
    cid = _post(client, "p_demo", wav_bytes(seconds=40))
    wait_result(client, cid)
    assert client.get("/api/demo/live-count").json()["checkins"] == 1
    for scenario in ("all", "drift", "stable"):
        assert client.post("/api/demo/reset", json={"scenario": scenario}).status_code == 200
        ids = [c["checkin_id"] for c in client.get("/api/patients/p_demo/timeline").json()["checkins"]]
        assert ids == [cid], scenario
    client.post("/api/demo/reset", json={"scenario": "blank"})
    assert client.get("/api/patients/p_demo/timeline").json()["checkins"] == []
    assert client.get("/api/demo/live-count").json()["checkins"] == 0


def test_load_demo_all_creates_the_live_profile_when_missing(client):
    client.post("/api/demo/reset", json={"scenario": "all"})
    assert any(p["id"] == "p_demo" for p in client.get("/api/patients").json())


# ------------------------------------------------------------------ P3-7: unknown status filter
def test_unknown_review_status_filter_is_422(client):
    assert client.get("/api/v1/review-flags", params={"status": "BOGUS"}).status_code == 422
    assert client.get("/api/v1/review-flags", params={"status": "open,Reviewed"}).status_code == 200
    assert client.get("/api/v1/review-flags").status_code == 200


# ------------------------------------------------------------------ P5-1: separate days are the person's days
def _ck(created, off=None):
    return store.Checkin(id="c", patient_id="p", created_at=created,
                         extra={} if off is None else {"tz_offset_min": off})


def test_one_evening_either_side_of_midnight_utc_is_one_local_day():
    # 7:30 PM and 8:30 PM US Eastern (UTC-5, getTimezoneOffset() = 300) = 00:30 and 01:30 UTC the NEXT day,
    # and 6:30 PM = 23:30 UTC the same day: one evening, two UTC dates
    a, b = datetime(2026, 11, 10, 23, 30), datetime(2026, 11, 11, 1, 30)
    assert a.date() != b.date()
    assert monitoring.local_day(_ck(a, 300)) == monitoring.local_day(_ck(b, 300))
    assert monitoring.local_day(_ck(a)) != monitoring.local_day(_ck(b))        # no offset: UTC dates (seed data)
    assert monitoring.local_day(_ck(a, 99999)) == a.date()                      # nonsense offsets are ignored


def test_upload_stores_the_browser_time_zone_offset(client, wav_bytes):
    cid = _post(client, "p_tz", wav_bytes(seconds=40), tz_offset_min="300")
    wait_result(client, cid)
    with store.session() as s:
        assert s.get(store.Checkin, cid).extra.get("tz_offset_min") == 300
    cid2 = _post(client, "p_tz", wav_bytes(seconds=40), tz_offset_min="not a number")   # ignored, never fatal
    wait_result(client, cid2)
    with store.session() as s:
        assert "tz_offset_min" not in s.get(store.Checkin, cid2).extra


# ------------------------------------------------------------------ P4-5: seeded check-ins read as daytime
def test_seeded_checkins_are_at_15_utc(client):
    client.post("/api/demo/reset", json={"scenario": "drift"})
    times = [c["created_at"] for c in client.get("/api/patients/p_drift/timeline", params={"days": 365}).json()["checkins"]]
    assert times and all(t[11:16] == "15:00" for t in times)

"""Window-level raw-vs-projected visualization (spec v0.1): math, states, calibration, API, replay, migration."""
import json

import numpy as np
import pytest

from server.config import get_config
from server.pipeline import noise, quality, windows
from server.pipeline.acoustic import Embedding
from server.store import mat_to_blob
from tests.conftest import SR, speechlike, wait_result

WCFG = {"reject_speech_s": 3.0, "reject_snr_db": 5, "warn_snr_db": 12}


class FakeCk:
    def __init__(self, i, W, Wr, enc="enc-A", wq=None):
        self.id = f"c{i}"
        self.window_vecs, self.window_vecs_raw = mat_to_blob(W), mat_to_blob(Wr)
        self.versions = {"encoder": enc}
        self.extra = {"window_analysis": {"windows": [{"quality": q} for q in wq]}} if wq else {}


def _baseline(rng, n=5, per=5, d=16, enc="enc-A"):
    base = rng.normal(0, 1, d) + 3
    return [FakeCk(i, base + 0.1 * rng.normal(0, 1, (per, d)), base + 0.1 * rng.normal(0, 1, (per, d)), enc)
            for i in range(n)], base


def _emb(W, Wr, enc="enc-A"):
    return Embedding(W.mean(0), Wr.mean(0), len(W), enc, None, [(8 * i, 8 * i + 8) for i in range(len(W))], W, Wr)


def _q(x):
    return quality.assess(x, get_config())


def test_cosine_distance_basics():
    mu = np.array([1.0, 0, 0])
    d = windows.cosine_distance(np.array([[1.0, 0, 0], [0, 1.0, 0], [-1.0, 0, 0]]), mu)
    assert np.allclose(d, [0, 1, 2])


def test_patient_calibration_loso_and_insufficient_data():
    rng = np.random.default_rng(0)
    base, _ = _baseline(rng)
    from server.store import blob_to_mat
    P = [blob_to_mat(c.window_vecs) for c in base]
    cal = windows.calibrate_patient(P, P)
    assert cal and cal["tau_proj"] > 0 and cal["metric"] == "cosine" and cal["n_windows"] == 25
    assert windows.calibrate_patient(P[:2], P[:2]) is None                      # too few sessions
    assert windows.calibrate_patient([p[:1] for p in P], [p[:1] for p in P]) is None   # too few windows


def test_state_C_ratios_use_each_models_own_threshold_and_fixed_line():
    rng = np.random.default_rng(1)
    base, b = _baseline(rng)
    x = speechlike(40, snr_db=30)
    W = b + 0.1 * rng.normal(0, 1, (5, 16))
    Wr = W.copy(); Wr[3:] += 2.0                                             # raw drifts late, projected doesn't
    a = windows.analyze(x=x, quality=_q(x), emb=_emb(W, Wr), baseline=base, encoder="enc-A", wcfg=WCFG,
                        interventions=[{"start_seconds": 24, "end_seconds": 40}])
    assert a["state"] == "C" and a["window_calibration_ready"] and a["raw_threshold"] > 0
    for w in a["windows"]:
        assert w["raw_ratio"] == pytest.approx(w["raw_distance"] / a["raw_threshold"], rel=1e-3)
        assert w["projected_ratio"] == pytest.approx(w["projected_distance"] / a["projected_threshold"], rel=1e-3)
    assert a["windows"][4]["raw_ratio"] > 1 >= a["windows"][4]["projected_ratio"]
    assert a["summary"]["raw_only_crossings"] >= 1 and a["interventions"][0]["start_seconds"] == 24
    assert a["calibration"]["source"].startswith("patient-baseline") and a["calibration"]["version"]


def test_state_A_no_baseline_and_encoder_mismatch():
    rng = np.random.default_rng(2)
    x = speechlike(40)
    W = rng.normal(0, 1, (5, 16))
    a = windows.analyze(x=x, quality=_q(x), emb=_emb(W, W), baseline=[], encoder="enc-A", wcfg=WCFG)
    assert a["state"] == "A" and a["baseline_progress"] == [0, 5]
    assert all(w["raw_ratio"] is None and w["raw_distance"] is None for w in a["windows"])
    base, _ = _baseline(rng, enc="OLD")
    a = windows.analyze(x=x, quality=_q(x), emb=_emb(W, W), baseline=base, encoder="enc-A", wcfg=WCFG)
    assert a["state"] == "A" and "re-record" in a["baseline_note"]


def test_state_B_uncalibrated_shows_distances_without_ratios():
    rng = np.random.default_rng(3)
    base, b = _baseline(rng, per=1)                                         # 5 windows total: too few to calibrate
    x = speechlike(40)
    W = b + 0.1 * rng.normal(0, 1, (5, 16))
    a = windows.analyze(x=x, quality=_q(x), emb=_emb(W, W), baseline=base, encoder="enc-A", wcfg=WCFG)
    assert a["state"] == "B" and not a["window_calibration_ready"] and a["raw_threshold"] is None
    assert all(w["raw_distance"] is not None and w["raw_ratio"] is None for w in a["windows"])
    # population thresholds (exported) are used when the patient's own calibration isn't possible
    pop = {"window": {"metric": "cosine", "proj": {"tau": 0.01}, "raw": {"tau": 0.02}}}
    a = windows.analyze(x=x, quality=_q(x), emb=_emb(W, W), baseline=base, encoder="enc-A", wcfg=WCFG,
                        population_meta=pop)
    assert a["state"] == "C" and a["calibration"]["source"].startswith("population")
    bad = {"window": {"proj": {"tau": 0.0}, "raw": {"tau": 0.02}}}             # tau must be positive: never divide by 0
    assert windows.population_calibration(bad) is None


def test_state_D_rejected_recording_and_state_E_gap_never_zero():
    rng = np.random.default_rng(4)
    base, b = _baseline(rng)
    x = np.zeros(SR * 30, np.float32) + 1e-5 * rng.normal(size=SR * 30).astype(np.float32)
    a = windows.analyze(x=x, quality=_q(x), emb=None, baseline=base, encoder="enc-A", wcfg=WCFG)
    assert a["state"] == "D" and a["recording_quality"] == "REJECT"
    # 40 s recording whose 3rd window is silence -> that window is a gap with nulls, the rest are scored
    x = speechlike(40, snr_db=30)
    x[16 * SR:24 * SR] = 1e-5 * rng.normal(size=8 * SR)
    W = b + 0.1 * rng.normal(0, 1, (5, 16))
    a = windows.analyze(x=x, quality=_q(x), emb=_emb(W, W), baseline=base, encoder="enc-A", wcfg=WCFG)
    assert a["state"] == "E"
    gap = a["windows"][2]
    assert gap["quality"] == "REJECT" and gap["raw_ratio"] is None and gap["raw_distance"] is None
    assert a["summary"]["valid_windows"] == 4


def test_noise_mixing_hits_target_snr_inside_interval_only():
    x = speechlike(20, snr_db=60).astype(np.float64)
    nz = noise.make_noise("pink", len(x), seed=0)
    y = noise.mix_interval(x, nz, 5, 15, snr_db=6, ramp_s=0)
    a, b = 5 * SR, 15 * SR
    added = y[a:b] - x[a:b]
    assert abs(10 * np.log10(np.mean(x[a:b] ** 2) / np.mean(added ** 2)) - 6) < 0.3
    assert np.allclose(y[:a], x[:a], atol=1e-6) and np.allclose(y[b:], x[b:], atol=1e-6)
    assert np.array_equal(noise.make_noise("babble_self", 1000, 3, ref=x), noise.make_noise("babble_self", 1000, 3, ref=x))
    with pytest.raises(ValueError):
        noise.make_noise("nope", 10, 0)


def test_migration_adds_new_columns_to_an_old_database(tmp_path):
    import sqlite3
    from server import store
    db = tmp_path / "old.db"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE checkin (id VARCHAR PRIMARY KEY, patient_id VARCHAR, created_at DATETIME, status VARCHAR,"
                " quality JSON, drift JSON, timing JSON, reference_risk JSON, evidence JSON, transcript VARCHAR,"
                " session_vec BLOB, session_vec_raw BLOB, versions JSON, extra JSON)")
    con.commit(); con.close()
    store.reset_engine(f"sqlite:///{db.as_posix()}")
    cols = {r[1] for r in sqlite3.connect(db).execute("PRAGMA table_info(checkin)")}
    assert {"window_vecs", "window_vecs_raw"} <= cols


# ---------------------------------------------------------------- API
def _post(client, pid, wav, interventions=None):
    data = {"prompt_id": "default"}
    if interventions is not None:
        data["interventions"] = json.dumps(interventions)
    r = client.post(f"/api/patients/{pid}/checkins", files={"audio": ("a.wav", wav, "audio/wav")}, data=data)
    assert r.status_code == 202
    return r.json()["checkin_id"]


def test_live_window_analysis_states_over_a_baseline(client, wav_bytes):
    ids = [_post(client, "p_w", wav_bytes(seconds=40, seed=i)) for i in range(1)]
    wait_result(client, ids[0])
    first = client.get(f"/api/checkins/{ids[0]}/windows").json()
    assert first["state"] == "A" and first["baseline_progress"][0] == 0
    for i in range(1, 5):
        wait_result(client, _post(client, "p_w", wav_bytes(seconds=40, seed=i)))
    cid = _post(client, "p_w", wav_bytes(seconds=40, seed=9),
                interventions=[{"type": "controlled_background_noise", "start_seconds": 16, "end_seconds": 99}])
    r = wait_result(client, cid)
    assert r["has_window_analysis"]
    a = client.get(f"/api/checkins/{cid}/windows").json()
    assert a["state"] in ("C", "E") and a["raw_threshold"] > 0 and a["projected_threshold"] > 0
    assert a["interventions"] == [{"type": "controlled_background_noise", "start_seconds": 16.0,
                                   "end_seconds": 40.0, "source": "marker"}]          # clipped to the recording
    assert len(a["windows"]) == 5 and a["envelope"]["hz"] == 10 and len(a["envelope"]["db"]) == 400
    # the longitudinal drift is still the session-level score, untouched by the window analysis
    assert r["drift"]["tau"] == pytest.approx(get_config().drift["tau"], rel=1e-3)
    # patient never sees any of it
    pv = client.get(f"/api/checkins/{cid}", params={"role": "patient"}).json()
    assert "has_window_analysis" not in pv and "windows" not in json.dumps(pv)
    # malformed interventions are ignored, never fatal
    r = client.post("/api/patients/p_w/checkins", files={"audio": ("a.wav", wav_bytes(seconds=40, seed=11), "audio/wav")},
                    data={"interventions": "{not json"})
    assert r.status_code == 202
    bad = wait_result(client, r.json()["checkin_id"])
    assert client.get(f"/api/checkins/{bad['checkin_id']}/windows").json()["interventions"] == []


def test_replay_is_deterministic_and_stores_nothing(client, wav_bytes):
    for i in range(5):
        wait_result(client, _post(client, "p_r", wav_bytes(seconds=40, seed=i)))
    n_before = len(client.get("/api/patients/p_r/timeline").json()["checkins"])
    form = {"patient_id": "p_r", "kind": "babble_self", "start_seconds": "16", "end_seconds": "40",
            "snr_db": "0", "seed": "7"}
    clean = wav_bytes(seconds=40, seed=21)
    r1 = client.post("/api/demo/replay", data=form, files={"audio": ("c.wav", clean, "audio/wav")})
    r2 = client.post("/api/demo/replay", data=form, files={"audio": ("c.wav", clean, "audio/wav")})
    assert r1.status_code == 200, r1.text
    j1, j2 = r1.json(), r2.json()
    assert j1["noisy"]["windows"] == j2["noisy"]["windows"]                       # reproducible
    assert j1["clean"]["interventions"][0]["type"] == "reference_interval_no_noise"
    # windows that end before the noise starts must be IDENTICAL in both versions (windows are independent)
    for cw_, nw_ in zip(j1["clean"]["windows"], j1["noisy"]["windows"]):
        if nw_["end_seconds"] <= 16:
            assert cw_ == nw_, (cw_, nw_)
    assert j1["noisy"]["interventions"][0]["snr_db"] == 0
    # the noise is real: windows inside the interval have a lower local SNR estimate than the clean version
    cw, nw = j1["clean"]["windows"], j1["noisy"]["windows"]
    inside = [i for i, w in enumerate(nw) if w["start_seconds"] >= 16 and nw[i]["snr_estimate_db"] is not None]
    assert inside and all(nw[i]["snr_estimate_db"] < cw[i]["snr_estimate_db"] for i in inside)
    assert len(client.get("/api/patients/p_r/timeline").json()["checkins"]) == n_before   # analysis only
    assert list(client.tmp_path.iterdir()) == []                                          # audio deleted
    assert client.post("/api/demo/replay", data={**form, "kind": "nope"},
                       files={"audio": ("c.wav", clean, "audio/wav")}).status_code == 422


def test_safety_checkin_has_no_window_analysis(client, wav_bytes):
    from server.pipeline import asr as asr_mod
    rt = client.app.state.holder["rt"]
    rt.asr.transcribe = lambda x: asr_mod.Transcript("I want to die", [], True, "fake")
    cid = _post(client, "p_s", wav_bytes(seconds=40))
    r = wait_result(client, cid)
    assert r["status"] == "SAFETY" and "has_window_analysis" not in r
    assert client.get(f"/api/checkins/{cid}/windows").status_code == 409

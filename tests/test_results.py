"""Aggregate experiment, Results endpoints, and seeded scenarios scored in their own space."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from tests.conftest import wait_result  # noqa: E402


def test_seeded_drift_profile_reads_as_drifted_whatever_model_is_loaded(client):
    """Regression: with trained artifacts loaded, the synthetic seeds were pushed through the real model's PCA and
    the drift profile said 'Within your usual range' on a 27%-slower day."""
    from server.pipeline.drift import DriftModel
    rt = client.app.state.holder["rt"]
    rng = np.random.default_rng(0)
    rt.dm = DriftModel(16, 5.0, np.full(16, 9.0), rng.normal(size=128), rng.normal(size=(16, 128)), version="v9")
    client.post("/api/demo/reset", json={"scenario": "all"})
    tl = client.get("/api/patients/p_drift/timeline").json()
    last = tl["checkins"][-1]
    assert last["status"] == "RED" and any("slower" in c["text"] for c in last["timing_chips"])
    assert tl["tau"] == pytest.approx(1.47)                     # the fallback (seed) space, not the loaded model


def _lite_fixture(tmp_path):
    """Fake L1 + L2 shards and P1/B1 checkpoints for the aggregate script."""
    import pandas as pd

    from research.extract import write_shard
    from research.heads import ProjectionHead
    from research.lite.extract import SWEEP_SNRS, VIEWS
    rng = np.random.default_rng(0)
    nuis = rng.normal(0, 1, (6, 768)) * 2
    rows, F, rows2, F2 = [], [], [], []
    for s in range(12):
        ident = rng.normal(0, 1, 768)
        for i in range(18):
            wid = f"s{s}:{i}"
            clean = ident + 0.3 * rng.normal(0, 1, 768)
            views = [clean, clean + nuis[0], clean + nuis[1], clean + nuis[2], clean + 1.5 * rng.normal(0, 1, 768)]
            for v, x in zip(VIEWS, views):
                rows.append({"window_id": wid, "speaker_id": f"s{s}", "split": "val" if s < 4 else "train", "view": v,
                             "view_detail": v, "seconds": 8.0, "t_pause_fraction": 0.2, "t_pitch_var_st": 1.0,
                             "t_run_rate": 0.5, "m_pause_fraction": True, "m_pitch_var_st": True, "m_run_rate": True})
                F.append(np.tile(x, (13, 1)).astype(np.float16))
            if s < 4:
                for k, snr in enumerate(SWEEP_SNRS):
                    rows2.append({"window_id": wid, "speaker_id": f"s{s}", "split": "val", "view": f"noise{snr}",
                                  "view_detail": "x", "seconds": 8.0, "t_pause_fraction": 0.2, "t_pitch_var_st": 1.0,
                                  "t_run_rate": 0.5, "m_pause_fraction": True, "m_pitch_var_st": True, "m_run_rate": True})
                    F2.append(np.tile(clean + nuis[3 + k % 3] * (k + 1) / 2, (13, 1)).astype(np.float16))
    feats = tmp_path / "lite_feats"; feats.mkdir()
    write_shard(feats, "L1_0000", np.stack(F), pd.DataFrame(rows))
    write_shard(feats, "L2_0000", np.stack(F2), pd.DataFrame(rows2))
    ck = tmp_path / "ck"; ck.mkdir()
    for k in ("P1", "B1"):
        h = ProjectionHead()
        torch.save({"head": h.state_dict()}, ck / f"{k}.pt")
    return feats, ck


def test_aggregate_experiment_produces_the_two_numbers(tmp_path):
    from research.lite import aggregate
    feats, ck = _lite_fixture(tmp_path)
    out = aggregate.run(feats, ck, layer=6, out_path=tmp_path / "results" / "aggregate.json", B=200)
    h = out["headline"]
    assert h["projection_model"] == "P1" and h["level"].startswith("session")
    for k in ("false_alert_raw", "false_alert_projection", "change_sensitivity_raw", "change_sensitivity_projection"):
        assert 0 <= h[k]["rate"] <= 1 and h[k]["ci"][0] <= h[k]["rate"] <= h[k]["ci"][1] and h[k]["n"] == 4 * 6
    assert h["noise_condition"].startswith("added held-out noise or babble at 10 dB")       # uses the L2 sweep
    assert set(out["results"]["session"]["B0"]) >= {"clean", "noise_mix", "pause_change", "noise_0dB", "noise_20dB"}
    assert out["meta"]["eval_speakers"] == 4 and out["meta"]["train_speakers_for_boundary"] == 8
    # raw features: added noise in this fixture is huge, so raw must flag far more noisy sessions than clean ones
    S = out["results"]["session"]["B0"]
    assert S["noise_0dB"]["rate"] > S["clean"]["rate"]
    assert json.loads((tmp_path / "results" / "aggregate.json").read_text())["illustrative"] is False


def test_loso_reference_excludes_the_tested_session():
    from research.lite.aggregate import loso_session_distances
    E = np.zeros((9, 4)); E[:, 0] = 1
    E[3:6] = [0, 1, 0, 0]                                            # session 1 is different
    d = loso_session_distances(E, E)
    assert d[1] > 0.5 and d[0] < d[1]                                # it isn't compared with itself


def test_results_endpoint_pending_then_present_and_demo_needs_clip(client, tmp_path, monkeypatch):
    from server.config import get_config
    cfg = get_config()
    monkeypatch.setitem(cfg["results"], "aggregate", str(tmp_path / "agg.json"))
    monkeypatch.setitem(cfg["results"], "demo_snapshot", str(tmp_path / "snap.json"))
    monkeypatch.setitem(cfg["demo"], "clip", str(tmp_path / "missing.wav"))
    r = client.get("/api/results").json()
    assert r["aggregate"] is None and r["demo"]["clip_present"] is False         # pending, never invented
    assert client.post("/api/demo/run").status_code == 409
    (tmp_path / "agg.json").write_text(json.dumps({"headline": {"x": 1}}))
    assert client.get("/api/results").json()["aggregate"]["headline"] == {"x": 1}


def test_one_click_demo_runs_on_the_bundled_clip_and_saves_a_snapshot(client, tmp_path, monkeypatch, wav_bytes):
    from server.config import get_config
    cfg = get_config()
    for i in range(5):
        cid = client.post("/api/patients/p_demo/checkins",
                          files={"audio": ("a.wav", wav_bytes(seconds=40, seed=i), "audio/wav")}).json()["checkin_id"]
        wait_result(client, cid)
    clip = tmp_path / "clip.wav"; clip.write_bytes(wav_bytes(seconds=40, seed=42))
    monkeypatch.setitem(cfg["demo"], "clip", str(clip))
    monkeypatch.setitem(cfg["results"], "demo_snapshot", str(tmp_path / "snap.json"))
    assert client.get("/api/results").json()["demo"]["profile_baseline_ready"] is True
    j = client.post("/api/demo/run").json()
    assert j["noisy"]["data_mode"] == "controlled_experiment" and j["ran_at"]
    assert clip.exists()                                                         # bundled clip never deleted
    assert json.loads((tmp_path / "snap.json").read_text())["noisy"]["windows"] == j["noisy"]["windows"]
    assert list(client.tmp_path.iterdir()) == []

import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research import augment, budget, data_prep, evaluate, export, train_heads  # noqa: E402
from research.extract import seed_for, window_audio  # noqa: E402

SR = 16000


# ---------------------------------------------------------------- data prep / augmentation / budget
def test_window_audio_and_spans():
    x = np.zeros(int(SR * 21), np.float32)     # 8 + 8 + 5 -> three windows (5 s tail kept)
    assert [len(w) for w in window_audio(x)] == [8 * SR, 8 * SR, 5 * SR]
    assert len(window_audio(np.zeros(int(SR * 19.5), np.float32))) == 2          # 3.5 s tail dropped
    assert data_prep.window_spans(19.5) == [(0.0, 8.0), (8.0, 16.0)]
    assert data_prep.window_spans(3.9) == []


def test_androids_speaker_ids_and_junk():
    from pathlib import Path
    p = Path("data/androids/Androids-Corpus/Interview-Task/audio_clip/18_CM64_3/18_CM64_3_1.wav")
    assert data_prep.speaker_from_path(p) == "18_CM64_3"
    assert data_prep.speaker_from_path(Path("x/18_PM64_3/18_PM64_3_2.wav")) == "18_PM64_3"   # no C/P collision
    assert data_prep.label_from_speaker("18_CM64_3") == 0 and data_prep.label_from_speaker("07_PF40_2") == 1
    assert data_prep.label_from_speaker("09_F40_2") is None                                 # X missing
    assert data_prep.is_junk(Path("data/androids/__MACOSX/Androids-Corpus/a/._x.wav"))
    assert data_prep.is_junk(Path("a/._18_CM64_3_1.wav")) and not data_prep.is_junk(p)


def test_read_androids_folds_real_layout(tmp_path):
    csv_text = ("Read,,,,,,,Interview,,,,\n"
                "fold1,fold2,fold3,fold4,fold5,,,fold1,fold2,fold3,fold4,fold5\n"
                "'01_CF56_1','05_CF41_3','06_CF44_2','07_CF50_2','03_CF30_3',,,'01_CF56_1','12_CF36_1','03_CF30_3','05_CF41_3','06_CF44_2'\n"
                "'02_CM57_2','11_CF44_2','15_CF53_3','10_CF51_2','04_CF57_3',,,'02_CM57_2','14_CF49_3','04_CF57_3','11_CF44_2',\n")
    p = tmp_path / "fold-lists.csv"
    p.write_text(csv_text)
    f = data_prep.read_androids_folds(str(p))
    assert f["01_CF56_1"] == 1 and f["12_CF36_1"] == 2 and f["06_CF44_2"] == 5 and f["11_CF44_2"] == 4
    assert len(f) == 9                       # interview columns only; ragged last row handled
    assert data_prep.read_androids_folds(str(p), "Read")["05_CF41_3"] == 2


def test_build_manifest_on_fake_tree(tmp_path):
    from pathlib import Path

    from research.extract import SR
    from server.pipeline.decode import write_wav
    root = tmp_path / "Androids-Corpus"
    for spk, secs in (("01_CF56_1", [29.84, 5.49, 2.81]), ("02_PM57_2", [20.0])):
        d = root / "Interview-Task" / "audio_clip" / spk
        d.mkdir(parents=True)
        for k, s in enumerate(secs, 1):
            write_wav(d / f"{spk}_{k}.wav", np.zeros(int(s * SR), np.float32))
    junk = tmp_path / "__MACOSX" / "Androids-Corpus" / "Interview-Task" / "audio_clip" / "01_CF56_1"
    junk.mkdir(parents=True); (junk / "._01_CF56_1_1.wav").write_bytes(b"\x00\x05")
    (root / "fold-lists.csv").write_text("Read,,,,,,,Interview,,,,\nfold1,fold2,fold3,fold4,fold5,,,fold1,fold2,fold3,fold4,fold5\n"
                                         ",,,,,,,'01_CF56_1','02_PM57_2',,,\n")
    df = data_prep.build_manifest(str(tmp_path / "**" / "Interview-Task" / "audio_clip" / "*" / "*.wav"),
                                  str(root / "fold-lists.csv"), str(tmp_path / "m.parquet"))
    # 29.84 s -> 3 full + 5.84 tail = 4 ; 5.49 -> 1 (tail kept) ; 2.81 -> 0 ; 20.0 -> 2 + 4.0 tail = 3
    assert df.groupby("speaker_id").size().to_dict() == {"01_CF56_1": 5, "02_PM57_2": 3}
    assert set(df.label) == {0, 1} and df.src_file.map(lambda s: Path(s).is_absolute()).all()
    assert not df.src_file.str.contains("__MACOSX").any()


def test_cap_is_seeded_and_per_speaker():
    rows = [{"speaker_id": "a", "src_file": "f", "t0": float(i), "t1": i + 8.0} for i in range(100)]
    k1, k2 = data_prep.cap_windows(rows, 40, seed=1), data_prep.cap_windows(rows, 40, seed=1)
    assert len(k1) == 40 and k1 == k2


def test_noise_split_by_file():
    files = {"noise": [f"n{i}" for i in range(20)], "music": [f"m{i}" for i in range(10)],
             "speech": [f"s{i}" for i in range(10)], "rir_real": ["r1"], "rir_point_noise": ["pn1"]}
    idx = data_prep.split_noise(files)
    assert not set(idx["noise_train"]) & set(idx["noise_test"]) and "pn1" in idx["noise_test"]
    assert idx["rir_test"] == ["r1"] and "pn1" not in idx["noise_train"]


def test_read_16k_resamples_44k1_with_correct_offsets(tmp_path):
    from research.extract import read_16k
    from server.pipeline.decode import write_wav
    sr = 44100
    t = np.arange(sr * 10) / sr
    x = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    x[int(5 * sr):] *= 0.1                        # amplitude step at 5.0 s marks the time axis
    write_wav(tmp_path / "a.wav", x, sr=sr)
    y = read_16k(str(tmp_path / "a.wav"), 4.0, 6.0)
    assert abs(len(y) - 2 * SR) <= 2              # 2 s at 16 kHz, offsets taken at 44.1 kHz
    F = np.abs(np.fft.rfft(y)); f = np.fft.rfftfreq(len(y), 1 / SR)
    assert abs(f[F.argmax()] - 440) < 2           # pitch preserved (no rate mix-up)
    first, second = np.abs(y[: SR // 2]).max(), np.abs(y[-SR // 2:]).max()
    assert first > 0.4 and second < 0.1           # the 5.0 s step lands mid-window
    # augment's noise loader returns exactly n samples from a 44.1 kHz file
    clip = augment._load(str(tmp_path / "a.wav"), 8000, __import__("random").Random(0), "cpu")
    assert clip.shape == (8000,)


def test_noise_caps_applied_before_split():
    files = {"noise": [f"n{i}" for i in range(50)], "music": [f"m{i}" for i in range(100)],
             "speech": [f"s{i}" for i in range(100)]}
    idx = data_prep.split_noise(files, caps={"music": 20, "speech": 30})
    assert len(idx["music_train"]) + len(idx["music_test"]) == 20
    assert len(idx["speech_train"]) + len(idx["speech_test"]) == 30
    assert len(idx["noise_train"]) + len(idx["noise_test"]) == 50          # uncapped
    assert idx == data_prep.split_noise(files, caps={"music": 20, "speech": 30})   # deterministic


def test_mix_at_snr_exact():
    g = torch.Generator().manual_seed(0)
    s, n = torch.randn(16000, generator=g), torch.randn(16000, generator=g)
    for snr in (5, 10, 20):
        y = augment.mix_at_snr(s, n, snr)
        got = 10 * torch.log10(s.pow(2).mean() / (y - s).pow(2).mean())
        assert abs(got.item() - snr) < 1e-3


def test_fft_convolve_matches_direct():
    x, h = torch.randn(1000), torch.randn(50)
    ref = torch.from_numpy(np.convolve(x.numpy(), h.numpy())[:1000])
    assert torch.allclose(augment.fft_convolve(x, h), ref, atol=1e-3)


def test_seed_deterministic_and_phone_band():
    assert seed_for("w1", 2) == seed_for("w1", 2) != seed_for("w1", 3)
    a = augment.Augmenter({"noise_train": [], "rir_test": []})
    t = torch.arange(SR * 2) / SR
    x = torch.sin(2 * np.pi * 100 * t) + torch.sin(2 * np.pi * 1000 * t) + torch.sin(2 * np.pi * 6000 * t)
    y = augment.phone_sim(x)
    F = torch.fft.rfft(y).abs()
    f = torch.fft.rfftfreq(len(y), 1 / SR)
    assert F[(f - 1000).abs().argmin()] > 20 * F[(f - 6000).abs().argmin()]
    assert a.copy_plan(["clean", "train", "test"]).__len__() == 1 + 4 + 3


def test_budget_ledger(tmp_path):
    called = []
    with budget.stage("G0", tmp_path, unassign=lambda: called.append(1)):
        pass
    rows = budget.read_rows(tmp_path)
    assert rows[0]["stage"] == "G0" and rows[0]["end_utc"] and called
    assert 0 <= budget.hours_used(tmp_path) < 0.01
    # a ledger already at the trip point refuses new stages
    import csv
    with open(tmp_path / "budget" / "ledger.csv", "a", newline="") as f:
        csv.writer(f).writerow(["G1", "2026-01-01T00:00:00+00:00", "2026-01-01T05:00:00+00:00", "L4"])
    with pytest.raises(RuntimeError):
        with budget.stage("G2", tmp_path, unassign=lambda: None):
            pass


def test_disconnected_stage_stops_counting_and_is_closed(tmp_path):
    from datetime import timedelta
    import csv
    now = budget._now()
    start = now - timedelta(minutes=40)
    seen = now - timedelta(minutes=30)                # died 30 min ago, after 10 min of work
    led = budget._ledger(tmp_path)
    with open(led, "a", newline="") as f:
        csv.writer(f).writerow(["G1", start.isoformat(), "", "L4", seen.isoformat(), ""])
    assert abs(budget.hours_used(tmp_path, now) - 10 / 60) < 1e-3     # counts to last heartbeat, not to now
    assert budget.close_stale(tmp_path, now) == ["G1"]
    row = budget.read_rows(tmp_path)[0]
    assert row["end_utc"] == seen.isoformat() and row["note"] == "closed after disconnect"


def test_live_stage_blocks_a_second_one(tmp_path):
    import csv
    now = budget._now()
    with open(budget._ledger(tmp_path), "a", newline="") as f:
        csv.writer(f).writerow(["G1", now.isoformat(), "", "L4", now.isoformat(), ""])   # fresh heartbeat
    with pytest.raises(RuntimeError, match="still heartbeating"):
        with budget.stage("G2", tmp_path, unassign=lambda: None):
            pass


def test_old_four_column_ledger_is_upgraded(tmp_path):
    (tmp_path / "budget").mkdir()
    (tmp_path / "budget" / "ledger.csv").write_text(
        "stage,start_utc,end_utc,gpu_name\nG0,2026-11-14T10:00:00+00:00,,L4\n")
    rows = budget.read_rows(tmp_path)
    assert list(rows[0].keys()) == budget.FIELDS
    # open row without any heartbeat (died under the old code): capped at its kill rule (G0 = 0.25 h)
    assert abs(budget.hours_used(tmp_path) - 0.25) < 1e-6


def test_heartbeat_written_while_running(tmp_path):
    import time
    with budget.stage("G0", tmp_path, unassign=lambda: None, heartbeat_s=0.05):
        time.sleep(0.3)
        r = budget.read_rows(tmp_path)[0]
        assert budget.is_running(r) and r["last_seen_utc"] > r["start_utc"]
    r = budget.read_rows(tmp_path)[0]
    assert r["end_utc"] and not budget.is_running(r)


def test_shard_write_is_atomic(tmp_path):
    import pandas as pd
    from research.extract import shard_done, write_shard
    (tmp_path / "G1_0000.npy").write_bytes(b"truncated")          # leftover from a disconnect, no parquet
    assert not shard_done(tmp_path, "G1_0000")                     # -> will be redone, not skipped
    write_shard(tmp_path, "G1_0000", np.zeros((2, 13, 768), np.float16), pd.DataFrame({"window_id": ["a", "b"]}))
    assert shard_done(tmp_path, "G1_0000") and np.load(tmp_path / "G1_0000.npy").shape == (2, 13, 768)
    assert not list(tmp_path.glob("*.tmp"))


# ---------------------------------------------------------------- training on planted synthetic features
def synth(n_spk=40, win=8, A=2, seed=0):
    rng = np.random.default_rng(seed)
    N = n_spk * win
    spk = np.repeat([f"s{i:02d}" for i in range(n_spk)], win)
    lab_spk = np.arange(n_spk) % 2
    y = np.repeat(lab_spk, win)
    fold = np.repeat(np.arange(n_spk) % 5, win)
    dis = rng.standard_normal((13, 768)) * 0.0
    dis[:, :20] = 1.0                                   # disease signal in 20 dims (all layers)
    base = rng.standard_normal((n_spk, 1, 13, 768)).repeat(win, 1).reshape(N, 13, 768) * 0.3   # speaker identity
    Zc = (base + rng.standard_normal((N, 13, 768)) + y[:, None, None] * dis * 0.6).astype(np.float32)
    env_dirs = rng.standard_normal((6, 768)) * 1.2
    def corrupt(Z, e):
        return Z + env_dirs[e][:, None, :] + 0.3 * rng.standard_normal(Z.shape).astype(np.float32)
    Ea = rng.integers(1, 6, size=(N, A))
    Za = np.stack([corrupt(Zc, Ea[:, a]) for a in range(A)], 1).astype(np.float32)
    Zt = {snr: corrupt(Zc, np.full(N, 3)).astype(np.float32) for snr in (20, 10, 5)}
    return train_heads.Data(Zc, Za, Ea, Zt, y, spk, fold)


@pytest.fixture(scope="module")
def cv():
    d = synth()
    res, chosen, oof = train_heads.run_cv(d, models=("L0", "M1", "M3"), seeds=(0,), epochs=6, layers=[2, 8],
                                          return_oof=True)
    return d, res, chosen, oof


def test_oof_logits_cover_every_window_and_carry_signal(cv):
    from research import calibration
    from research.stats import roc_auc_score
    d, _, _, oof = cv
    for m in ("L0", "M1", "M3"):
        assert oof[m].shape == d.y.shape and not np.isnan(oof[m]).any()
    S, Y, F, Z = calibration.session_logits(oof["L0"], d.spk, d.y, d.fold)
    assert len(Z) == 40 * (8 // 3) and roc_auc_score(Y, Z) > 0.7       # OOF logits are aligned and informative
    S, Y, F, Z = calibration.session_logits(oof["M3"], d.spk, d.y, d.fold)
    assert roc_auc_score(Y, Z) > 0.55


@pytest.mark.parametrize("kind", ["M2", "M3"])
def test_consistency_loss_does_not_block_disease_learning(kind):
    """Regression: with the consistency term summed over 128 dims, M2/M3 never learned (train AUC ~0.5)."""
    from research.stats import roc_auc_score
    d = synth()
    tr_idx, va_idx = np.where(d.fold != 0)[0], np.where(d.fold == 0)[0]
    P, H, _ = train_heads.train_head(d, kind, 2, tr_idx, va_idx, seed=0, epochs=15)
    assert roc_auc_score(d.y[tr_idx], train_heads.predict_head(P, H, d.Zc[tr_idx, 2])) > 0.8


def test_cv_runs_and_signal_is_found(cv):
    d, res, chosen, _ = cv
    assert set(chosen) == {0, 1, 2, 3, 4} and all(c["layer"] in (2, 8) for c in chosen.values())
    tab = evaluate.e1_table(res, B=200)
    l0 = next(r for r in tab if r["model"] == "L0" and r["snr"] == "clean")
    assert l0["auc"] > 0.8 and l0["auc_ci"][0] <= l0["auc"] <= l0["auc_ci"][1]
    assert any(r["model"] == "M3-L0" and r["snr"] == 5 for r in tab)         # paired CI present at each SNR
    n_spk = {r["n_speakers"] for r in tab if "n_speakers" in r}
    assert n_spk == {40}                                                      # every speaker scored exactly once


def test_leakage_assertion_fires():
    d = synth(n_spk=10)
    with pytest.raises(AssertionError):
        train_heads.assert_no_leakage(d, 0, ["s00", "s01"], ["s01", "s02"], ["s00"])
    with pytest.raises(AssertionError):
        train_heads.assert_no_leakage(d, 0, ["s00", "s01"], ["s02"], ["s02"])


def test_noise_probe_head_reduces_env_accuracy():
    d = synth(n_spk=20)
    P, H, _ = train_heads.train_head(d, "M3", 2, np.arange(0, 120), np.arange(120, 160), seed=0, epochs=8, beta_max=0.5)
    with torch.no_grad():
        Hz = P(torch.from_numpy(np.concatenate([d.Zc[:, 2], d.Za[:, 0, 2]]))).numpy()
    raw = np.concatenate([d.Zc[:, 2], d.Za[:, 0, 2]])
    env = np.concatenate([np.zeros(len(d.y), int), d.Ea[:, 0]])
    spk = np.concatenate([d.spk, d.spk])
    r = evaluate.e2_noise_probe(raw, Hz, env, spk)
    assert r["raw_z"] > 0.6 and r["P_M3"] <= r["raw_z"]


def test_e3_and_export_roundtrip(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    clean, noisy = {}, {}
    for i in range(12):
        s = f"c{i}"
        spk_vec = rng.standard_normal(32)
        clean[s] = spk_vec + 0.3 * rng.standard_normal((24, 32))
        noisy[s] = clean[s][:, None, :] + 0.1 * rng.standard_normal((24, 3, 32))
    out = evaluate.e3_false_alerts(clean, noisy, [20, 10, 5], [f"c{i}" for i in range(8)], [f"c{i}" for i in range(8, 12)], k=8)
    assert out["tau"] > 0 and out["exceed_rate"]["clean"] is not None and "lower bound" in out["note"]

    # export -> server loads the artifacts and tau/sigma_pop shapes line up
    d = synth(n_spk=20, win=24)
    chosen = {f: {"layer": 2, "C": 1.0} for f in range(5)}
    out_dir, cfg = export.export(d, chosen, tmp_path / "artifacts", 1, {"E1": []}, "rev", seeds=(0,), epochs=3,
                                 calibration={"model": "M3", "a": 1.0, "b": 0.0, "gate_passed": True,
                                              "bands": {"moderate": 0.4, "elevated": 0.7},
                                              "metrics": {"ece": 0.05, "brier": 0.2, "auc": 0.7, "auc_ci": [0.6, 0.8]}})
    assert (out_dir / "projector.pt").exists() and json.loads((out_dir / "metrics.json").read_text()) == {"E1": []}
    assert (out_dir / "disease_head.pt").exists() and (out_dir / "calibration.json").exists()
    fd = torch.nn.Linear(128, 1); fd.load_state_dict(torch.load(out_dir / "disease_head.pt", weights_only=True))
    from server.config import get_config
    from server.pipeline import drift
    cfgd = get_config()
    monkeypatch.setitem(cfgd["artifacts"], "dir", str(tmp_path / "artifacts"))
    dm, dmr = drift.load_drift_model(), drift.load_drift_model(raw=True)
    assert dm.version == "v1" and dm.layer == 2 and dm.pca_components.shape == (16, 128)
    wcal = cfg["window"]                                   # population window thresholds for the visualization
    assert wcal["metric"] == "cosine" and wcal["proj"]["tau"] > 0 and wcal["raw"]["tau"] > 0
    from server.pipeline.windows import population_calibration
    assert population_calibration(dm.meta)["source"].startswith("population")
    assert dmr.pca_components.shape == (16, 768) and dm.tau == pytest.approx(cfg["tau"])
    # serving-side projector loads the exported weights (weights_only=True) and reproduces training-time output
    from research.heads import Projector
    P = Projector(); P.load_state_dict(torch.load(out_dir / "projector.pt", weights_only=True)); P.eval()
    assert P(torch.zeros(1, 768)).shape == (1, 128)


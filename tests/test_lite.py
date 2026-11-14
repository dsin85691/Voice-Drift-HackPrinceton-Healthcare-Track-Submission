"""Lightweight projection experiment (projection-head spec v0.1, section 11 test list)."""
import json
import random
import sys
import types

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from research.heads import AcousticHead, ProjectionHead  # noqa: E402
from research.lite import acoustic as A  # noqa: E402
from research.lite import losses as Lf  # noqa: E402
from research.lite import train as T  # noqa: E402
from tests.conftest import SR, speechlike  # noqa: E402


# ---------------------------------------------------------------- model / gradients
def test_wavlm_is_frozen_and_in_eval_mode(monkeypatch):
    class Fake(torch.nn.Linear):
        @classmethod
        def from_pretrained(cls, *a, **k):
            assert k.get("use_safetensors") is True
            return cls(4, 4)
    monkeypatch.setitem(sys.modules, "transformers", types.SimpleNamespace(WavLMModel=Fake))
    from research.extract import load_wavlm
    m = load_wavlm()
    assert not m.training and all(not p.requires_grad for p in m.parameters())


def test_only_projection_and_aux_heads_receive_gradients():
    backbone = torch.nn.Linear(16, 768)
    for p in backbone.parameters():
        p.requires_grad_(False)
    head, ah = ProjectionHead(), AcousticHead()
    x = torch.randn(8, 16)
    z, zt = head(backbone(x)), head(backbone(x + 0.1))
    loss, _ = Lf.composite(z, zt, T.WEIGHTS["P1"], ah, torch.zeros(8, 3), torch.ones(8, 3, dtype=torch.bool))
    loss.backward()
    assert all(p.grad is None for p in backbone.parameters())
    assert all(p.grad is not None for p in head.net.parameters()) and all(p.grad is not None for p in ah.parameters())
    assert z.shape == (8, 128) and ah(z).shape == (8, 3)


def test_eval_determinism_and_checkpoint_roundtrip(tmp_path):
    head = ProjectionHead().eval()
    x = torch.randn(5, 768)
    assert torch.equal(head(x), head(x))
    torch.save(head.state_dict(), tmp_path / "h.pt")
    h2 = ProjectionHead(); h2.load_state_dict(torch.load(tmp_path / "h.pt", weights_only=True)); h2.eval()
    assert torch.allclose(head(x), h2(x))
    from server.pipeline.acoustic import load_projection        # what the server actually deploys
    torch.save(head.state_dict(), tmp_path / "projection_head.pt")
    assert torch.allclose(load_projection(tmp_path)(x), head(x))


# ---------------------------------------------------------------- losses
def test_losses_finite_batch_of_one_and_masking():
    z = torch.randn(1, 128)
    assert Lf.variance(z) == 0 and Lf.covariance(z) == 0                      # no NaN for N = 1
    tot, parts = Lf.composite(z, z + 0.1, T.WEIGHTS["P1"], AcousticHead(), torch.zeros(1, 3),
                              torch.ones(1, 3, dtype=torch.bool))
    assert torch.isfinite(tot) and all(np.isfinite(v) for v in parts.values())
    pred, tgt = torch.zeros(4, 3), torch.zeros(4, 3)
    mask = torch.tensor([[1, 0, 1]] * 4, dtype=torch.bool)
    base = Lf.preservation(pred, tgt, mask)
    tgt2 = tgt.clone(); tgt2[:, 1] = 99.0                                     # change only MASKED entries
    assert Lf.preservation(pred, tgt2, mask) == base == 0
    assert Lf.preservation(pred, tgt, torch.zeros(4, 3, dtype=torch.bool)) == 0
    a, b = torch.randn(6, 128), torch.randn(6, 128)
    assert torch.isclose(Lf.invariance(a, b, "sum"), 128 * Lf.invariance(a, b, "mean"))
    # a collapsed (constant) representation is penalized by the variance term
    assert Lf.variance(torch.zeros(10, 128)) > 0.9


# ---------------------------------------------------------------- acoustic targets / views
def test_acoustic_targets_and_validity():
    v, ok = A.acoustic_targets(speechlike(8, seed=0))
    assert ok[0] and abs(v[0] - 0.25) < 0.12                                  # 1.5 s on / 0.5 s off
    sil = np.random.default_rng(0).normal(0, 1e-5, 8 * SR).astype(np.float32)
    _, ok_sil = A.acoustic_targets(sil)
    assert not ok_sil[1] and not ok_sil[2]                                    # no pitch, no speech: masked
    t = np.arange(8 * SR) / SR
    steady = np.sin(2 * np.pi * 140 * t)
    vib = np.sin(2 * np.pi * np.cumsum(140 * 2 ** (3 * np.sin(2 * np.pi * 0.5 * t) / 12)) / SR)
    vs, oks = A.acoustic_targets(steady.astype(np.float32))
    vv, okv = A.acoustic_targets(vib.astype(np.float32))
    assert oks[1] and okv[1] and vv[1] > vs[1] + 1.0                          # vibrato -> more pitch variability


def test_pause_view_adds_pauses_without_digital_silence_and_mic_keeps_length():
    x = speechlike(8, seed=1, snr_db=35)
    y = A.insert_pauses(x)
    assert len(y) > len(x) + 0.4 * SR
    assert A.acoustic_targets(y)[0][0] > A.acoustic_targets(x)[0][0] + 0.05
    assert np.mean(np.abs(y[len(x):]) == 0) < 0.01                           # room tone, not zeros
    m = A.mic_sim(x, np.random.default_rng(0))
    assert len(m) == len(x) and 0.25 < np.sqrt(np.mean(m ** 2) / np.mean(x ** 2)) < 4


def test_selection_is_speaker_disjoint_and_consecutive():
    import pandas as pd
    from research.lite.extract import select_windows
    rows = [{"window_id": f"s{s}:{i}", "speaker_id": f"s{s}", "src_file": f"f{s}", "t0": 8.0 * i, "t1": 8.0 * i + 8}
            for s in range(30) for i in range(25)]
    sel = select_windows(pd.DataFrame(rows), n_speakers=24, per_speaker=20)
    assert sel.groupby("speaker_id").size().eq(20).all() and sel.speaker_id.nunique() == 24
    assert sel.groupby("speaker_id").split.nunique().eq(1).all()             # a speaker lives in ONE split
    assert sel[sel.split == "val"].speaker_id.nunique() == 6
    assert (sel.groupby("speaker_id").t0.min() == 0).all()                   # first, consecutive windows


def test_nuisance_views_are_seeded_length_preserving_and_use_held_out_pools(tmp_path):
    from research.lite.extract import make_views
    from server.pipeline.decode import write_wav
    idx = {}
    for pool in ("noise", "speech"):
        for split in ("train", "test"):
            f = tmp_path / f"{pool}_{split}.wav"
            write_wav(f, (0.1 * np.random.default_rng(len(pool)).standard_normal(SR * 12)).astype(np.float32))
            idx[f"{pool}_{split}"] = [str(f)]
    x = speechlike(8, seed=3)
    v1, l1 = make_views(x, "w1", idx)
    v2, l2 = make_views(x, "w1", idx)
    assert l1 == l2 and all(np.array_equal(a, b) for a, b in zip(v1, v2))   # reproducible
    assert [len(v) for v in v1[:4]] == [len(x)] * 4 and l1[4] == "synthetic_pauses"


# ---------------------------------------------------------------- end to end on synthetic cached embeddings
def _synthetic(n_spk=16, per=20, seed=0):
    """Speaker identity + 'delivery' dims (tied to the pause target) + view-specific nuisance offsets."""
    rng = np.random.default_rng(seed)
    H, spk, split, wid, tg = [], [], [], [], []
    nuis_dirs = rng.normal(0, 1, (8, 768)) * 2.0
    deliver = rng.normal(0, 1, 768)
    for s in range(n_spk):
        ident = rng.normal(0, 1, 768)
        for i in range(per):
            pf = rng.uniform(0.1, 0.4)
            clean = ident + 0.3 * rng.normal(0, 1, 768) + 4 * pf * deliver
            views = [clean, clean + nuis_dirs[rng.integers(0, 4)], clean + nuis_dirs[rng.integers(0, 4)],
                     clean + nuis_dirs[rng.integers(4, 8)],                 # held-out nuisance directions
                     clean + 4 * 0.3 * deliver]                             # synthetic 'more pauses'
            H.append(np.stack(views)); spk.append(f"s{s}"); split.append("val" if s < 4 else "train")
            wid.append(f"s{s}:{i}"); tg.append([pf, 1.0, 0.5])
    tg = np.array(tg)
    return T.LiteData(np.array(H, np.float32), np.array(spk), np.array(split), np.array(wid), tg,
                      np.ones_like(tg, bool), 6)


def test_train_evaluate_report_export_end_to_end(tmp_path):
    d = _synthetic()
    opt = {**T.OPT, "epochs": 6}
    runs = {k: T.train_variant(d, k, opt=opt, log=lambda *_: None) for k in ("B1", "P1")}
    for r in runs.values():
        assert r["epochs_run"] >= 1 and np.isfinite(r["best_val"])
        assert all(np.isfinite(v) for h in r["history"] for v in h["train"].values())
        assert not set(r["inner_val_speakers"]) & set(d.spk[d.split == "val"])   # stopping never saw eval speakers
    assert "pres" in runs["P1"]["history"][0]["train"] and runs["B1"]["weights"]["pres"] == 0
    res = T.evaluate(d, {"B0": None, "B1": runs["B1"]["head"], "P1": runs["P1"]["head"]}, B=100)
    for k in ("B0", "B1", "P1"):
        m = res[k]
        assert 0 <= m["auroc_change_vs_nuisance"] <= 1 and 0 <= m["nuisance_false_alert_rate"] <= 1
        assert m["auroc_change_vs_nuisance_ci"][0] <= m["auroc_change_vs_nuisance_ci"][1]
    assert not T.collapsed(res["B1"]) and not T.collapsed(res["P1"])
    text = T.report(res, runs, d, tmp_path / "r.md")
    assert "SYNTHETIC" in text.upper() and "| P1 |" in text
    for word in ("depress", "diagnos", "risk"):                              # no health inference anywhere
        assert word not in json.dumps({k: v for k, v in res.items() if k != "_meta"}).lower()
    adir = T.export_for_dashboard(d, runs["P1"], tmp_path / "artifacts", 1, res)
    cfg = json.loads((adir / "drift_config.json").read_text())
    assert cfg["head"] == "lite-P1" and cfg["window"]["proj"]["tau"] > 0 and cfg["window"]["raw"]["tau"] > 0
    from server.pipeline.acoustic import load_projection
    from server.pipeline.windows import population_calibration
    x = torch.from_numpy(d.H[:3, 0])
    assert torch.allclose(load_projection(adir)(x), runs["P1"]["head"](x))   # checkpoint reproduces outputs
    assert population_calibration(cfg)["tau_proj"] > 0                        # model-specific window thresholds


def test_disjointness_assertion_fires():
    d = _synthetic(n_spk=6, per=4)
    with pytest.raises(AssertionError):
        T.assert_disjoint(d, {"s4", "s5"}, {"s5"})
    with pytest.raises(AssertionError):
        T.assert_disjoint(d, {"s0"}, {"s5"})                                  # s0 is an evaluation speaker


def test_load_lite_from_shards(tmp_path):
    import pandas as pd
    from research.extract import write_shard
    from research.lite.extract import VIEWS
    rows, F = [], []
    for w in range(3):
        for v in VIEWS:
            rows.append({"window_id": f"w{w}", "speaker_id": "s0", "split": "train", "view": v, "view_detail": v,
                         "seconds": 8.0, "t_pause_fraction": 0.2, "t_pitch_var_st": 1.0, "t_run_rate": 0.5,
                         "m_pause_fraction": True, "m_pitch_var_st": True, "m_run_rate": False})
            F.append(np.full((13, 768), w + VIEWS.index(v) / 10, np.float16))
    write_shard(tmp_path, "L1_0000", np.stack(F), pd.DataFrame(rows))
    d = T.load_lite(tmp_path, layer=6)
    assert d.H.shape == (3, 5, 768) and d.mask[:, 2].sum() == 0
    assert np.allclose(d.H[1, 4], 1.4, atol=1e-2)                            # window 1, view 'pause'

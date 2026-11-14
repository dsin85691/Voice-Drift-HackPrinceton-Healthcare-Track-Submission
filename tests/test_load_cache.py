import numpy as np
import pandas as pd
import pytest

pytest.importorskip("torch")

from research import load_cache  # noqa: E402


def test_load_roundtrip(tmp_path):
    rng = np.random.default_rng(0)
    wids = [f"s{i}:f:{j}" for i in range(2) for j in range(3)]
    man = pd.DataFrame({"window_id": wids, "speaker_id": [w.split(":")[0] for w in wids],
                        "label": [int(w[1]) for w in wids], "fold": [0] * 6})
    man.to_parquet(tmp_path / "m.parquet")
    for stage, copies in (("G1", [(0, "clean"), (1, "train"), (2, "train"), (3, "train"), (4, "train")]),
                          ("G2", [(100, "test"), (101, "test"), (102, "test")])):
        rows = [dict(window_id=w, copy=k, copy_idx=i, snr_db=0.0, env=(i % 5) + 1 if k == "train" else 0)
                for w in wids for i, k in copies]
        pd.DataFrame(rows).to_parquet(tmp_path / f"{stage}_0000.parquet")
        np.save(tmp_path / f"{stage}_0000.npy", rng.standard_normal((len(rows), 13, 768)).astype(np.float16))
    d = load_cache.load_data(tmp_path, tmp_path / "m.parquet")
    assert d.Zc.shape == (6, 13, 768) and d.Za.shape == (6, 4, 13, 768) and set(d.Zt) == {20, 10, 5}
    assert d.Ea.shape == (6, 4) and list(d.spk[:3]) == ["s0"] * 3 and d.y.tolist() == [0, 0, 0, 1, 1, 1]

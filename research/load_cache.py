"""Merge extract.py shards (feats/{stage}_NNNN.npy + .parquet) with the manifest into train_heads.Data.

Row layout written by extract.py: one row per (window, copy) with columns
window_id, copy ("clean" | "train" | "test"), copy_idx (0 | 1..4 | 100..102), snr_db, env.
Only windows that have every copy are kept, so G1 + G2 must both have finished for them.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .train_heads import Data

N_TRAIN, TEST_IDX = 4, {100: 20, 101: 10, 102: 5}


def load_data(feat_dir: str | Path, manifest: str | Path, stages=("G1", "G2")) -> Data:
    feat_dir = Path(feat_dir)
    idx_parts, feat_parts = [], []
    for st in stages:
        for pq in sorted(feat_dir.glob(f"{st}_*.parquet")):
            npy = pq.with_suffix(".npy")
            if not npy.exists():
                continue                                   # half-written shard: ignore
            df = pd.read_parquet(pq)
            f = np.load(npy, mmap_mode="r")
            assert len(df) == len(f), f"{pq.name}: {len(df)} index rows vs {len(f)} feature rows"
            idx_parts.append(df)
            feat_parts.append(f)
    if not idx_parts:
        raise FileNotFoundError(f"no complete shards for {stages} in {feat_dir}")
    idx = pd.concat(idx_parts, ignore_index=True)
    feats = np.concatenate([np.asarray(f) for f in feat_parts])      # fp16 [rows, 13, 768]

    man = pd.read_parquet(manifest).drop_duplicates("window_id")
    need = {0, *range(1, 1 + N_TRAIN), *TEST_IDX}
    have = idx.groupby("window_id")["copy_idx"].agg(lambda s: need <= set(s))
    keep = [w for w in man.window_id if have.get(w, False)]
    man = man.set_index("window_id").loc[keep].reset_index()
    pos = {(w, c): i for i, (w, c) in enumerate(zip(idx.window_id, idx.copy_idx))}
    env = dict(zip(zip(idx.window_id, idx.copy_idx), idx.env))

    get = lambda c: np.stack([feats[pos[(w, c)]] for w in man.window_id])
    Zc = get(0).astype(np.float32)
    Za = np.stack([get(c) for c in range(1, 1 + N_TRAIN)], 1)                    # fp16 [N, 4, 13, 768]
    Ea = np.array([[env[(w, c)] for c in range(1, 1 + N_TRAIN)] for w in man.window_id], dtype=np.int64)
    Zt = {snr: get(c) for c, snr in TEST_IDX.items()}
    return Data(Zc, Za, Ea, Zt, man.label.to_numpy(int), man.speaker_id.to_numpy(str), man.fold.to_numpy(int))

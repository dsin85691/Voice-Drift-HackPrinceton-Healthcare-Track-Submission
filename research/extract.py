"""WavLM 13-layer feature cache (spec 3.3). The core functions here are also imported by the server
(server/pipeline/acoustic.py) so training and serving share one code path (train/serve parity).

GPU use: only in `main()`. Run it from a notebook cell that has written the budget ledger (budget.py).
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np

SR = 16000
WIN_S = 8.0
MIN_TAIL_S = 4.0
N_LAYERS = 13


def read_16k(path: str, t0: float | None = None, t1: float | None = None) -> np.ndarray:
    """Read [t0, t1) seconds of any wav as float32 mono 16 kHz.
    Androids is stored at 44.1 kHz, so offsets are computed at the FILE's rate, then resampled (polyphase).
    MUSAN is already 16 kHz and passes through untouched."""
    from math import gcd

    import soundfile as sf
    from scipy.signal import resample_poly

    with sf.SoundFile(path) as f:                     # ONE file open (each open is slow on a Drive mount)
        sr = f.samplerate
        start = 0 if t0 is None else int(round(t0 * sr))
        stop = f.frames if t1 is None else min(f.frames, int(round(t1 * sr)))
        f.seek(start)
        x = f.read(max(0, stop - start), dtype="float32", always_2d=True)
    x = x.mean(1)                                     # downmix (no-op for mono)
    if sr != SR:
        g = gcd(sr, SR)
        x = resample_poly(x, SR // g, sr // g).astype(np.float32)
    return x


def window_bounds(n_samples: int, sr: int = SR, win_s: float = WIN_S, min_tail_s: float = MIN_TAIL_S,
                  max_windows: int | None = None) -> list[tuple[int, int]]:
    """Sample ranges of the 8.0 s non-overlapping windows; a tail shorter than 4 s is dropped."""
    w = int(win_s * sr)
    out = []
    for i in range(0, n_samples, w):
        e = min(i + w, n_samples)
        if e - i == w or e - i >= min_tail_s * sr:
            out.append((i, e))
    return out[:max_windows] if max_windows else out


def window_audio(x: np.ndarray, sr: int = SR, win_s: float = WIN_S, min_tail_s: float = MIN_TAIL_S,
                 max_windows: int | None = None) -> list[np.ndarray]:
    """8.0 s non-overlapping windows; a tail shorter than 4 s is dropped, a longer tail is kept as-is."""
    return [x[a:b] for a, b in window_bounds(len(x), sr, win_s, min_tail_s, max_windows)]


def load_wavlm(name: str = "microsoft/wavlm-base", revision: str = "main", device: str = "cpu"):
    import torch
    from transformers import WavLMModel

    # safetensors only: avoids also downloading pytorch_model.bin (378 MB twice) and never unpickles weights
    try:
        m = WavLMModel.from_pretrained(name, revision=revision, use_safetensors=True)
    except OSError:
        m = _wavlm_from_cache(name, revision)      # offline (HF_HUB_OFFLINE=1): see below
    m = m.to(device).eval()
    for p in m.parameters():
        p.requires_grad_(False)
    return m


def _wavlm_from_cache(name: str, revision: str):
    """Offline fallback. microsoft/wavlm-base publishes safetensors only on the Hub's auto-conversion PR (refs/pr/1),
    so the cache holds config.json under `main` and model.safetensors under a different snapshot. Online,
    transformers follows that PR; offline it cannot. Rebuild the model from the cached config + cached safetensors,
    loading STRICTLY so any mismatch raises instead of silently producing a wrong model."""
    import glob
    import os

    from safetensors.torch import load_file
    from transformers import WavLMConfig, WavLMModel

    cfg = WavLMConfig.from_pretrained(name, revision=revision, local_files_only=True)
    path = None
    try:
        from huggingface_hub import try_to_load_from_cache
        for rev in (revision, "refs/pr/1"):
            p = try_to_load_from_cache(name, "model.safetensors", revision=rev)
            if isinstance(p, str) and os.path.exists(p):
                path = p
                break
    except Exception:
        path = None
    if path is None:
        hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")), "hub")
        hits = glob.glob(os.path.join(hub, "models--" + name.replace("/", "--"), "snapshots", "*", "model.safetensors"))
        path = hits[0] if hits else None
    if path is None:
        raise OSError(f"{name}: no cached model.safetensors (run once online to download it)")
    m = WavLMModel(cfg)
    m.load_state_dict(load_file(path), strict=True)
    return m


def layer_means(model, wavs: list[np.ndarray], device: str = "cpu", dtype=None) -> np.ndarray:
    """wavs: list of float32 mono 16 kHz arrays (variable length OK, zero-padded + attention-masked).
    Returns [B, 13, 768] float32: mean over valid frames of each hidden state (embedding + 12 layers)."""
    import torch

    B = len(wavs)
    L = max(len(w) for w in wavs)
    x = torch.zeros(B, L)
    mask = torch.zeros(B, L, dtype=torch.long)
    for i, w in enumerate(wavs):
        t = torch.from_numpy(np.asarray(w, dtype=np.float32))
        # WavLM-base was trained on per-utterance normalized waveforms
        t = (t - t.mean()) / (t.std() + 1e-7)
        x[i, : len(t)] = t
        mask[i, : len(t)] = 1
    x, mask = x.to(device), mask.to(device)
    ctx = torch.autocast(device_type=device.split(":")[0], dtype=dtype) if dtype else torch.no_grad()
    with torch.no_grad(), ctx:
        out = model(x, attention_mask=mask, output_hidden_states=True)
    hs = torch.stack(out.hidden_states, dim=1).float()            # [B, 13, T, 768]
    T = hs.shape[2]
    # frame-level validity: conv stack downsamples ~320x
    flen = model._get_feat_extract_output_lengths(mask.sum(1)).clamp(max=T)
    fmask = (torch.arange(T, device=hs.device)[None, :] < flen[:, None]).float()
    num = (hs * fmask[:, None, :, None]).sum(2)
    return (num / fmask.sum(1)[:, None, None]).cpu().numpy()


def seed_for(window_id: str, copy_idx: int) -> int:
    """Deterministic augmentation seed from hash(window_id, copy_idx)."""
    h = hashlib.sha256(f"{window_id}|{copy_idx}".encode()).digest()
    return int.from_bytes(h[:4], "little")


def shard_done(out: Path, shard_id: str) -> bool:
    """A shard counts as finished only if BOTH files exist (each is written atomically by write_shard)."""
    return (out / f"{shard_id}.npy").exists() and (out / f"{shard_id}.parquet").exists()


def write_shard(out: Path, shard_id: str, feats: np.ndarray, index) -> None:
    """Write to temp names, then rename. A disconnect mid-write leaves only *.tmp files, never a truncated
    shard that a resume would mistake for a finished one. The .npy is renamed last."""
    import os

    tmp_pq, tmp_np = out / f"{shard_id}.parquet.tmp", out / f"{shard_id}.npy.tmp"
    index.to_parquet(tmp_pq)
    with open(tmp_np, "wb") as f:
        np.save(f, feats)
    os.replace(tmp_pq, out / f"{shard_id}.parquet")
    os.replace(tmp_np, out / f"{shard_id}.npy")


def main() -> None:  # pragma: no cover - Colab entry point
    import pandas as pd
    import soundfile as sf
    import torch

    from . import augment, budget

    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["G0", "G1", "G2", "G3", "G4"])
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--noise-index", required=True, help="json from data_prep.split_noise")
    ap.add_argument("--batch", type=int, default=64)
    # windows per checkpoint shard. Spec says 1000, but G0 = 200 windows was then ONE shard that the watchdog killed
    # before it was written (nothing saved, every retry restarted). 100 windows = 800 clips: lose at most ~1 min.
    ap.add_argument("--shard", type=int, default=100)
    ap.add_argument("--limit", type=int, default=0, help="G0 smoke: windows to process")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW",
                    help="rewrite audio/noise path prefixes, e.g. Drive -> local disk (repeatable)")
    ap.add_argument("--io-only", action="store_true",
                    help="CPU profiling of the data path (read + resample + augment), no model, no GPU, no ledger")
    ap.add_argument("--keep-runtime", action="store_true",
                    help="don't unassign the runtime when THIS stage ends (chain stages: G1 --keep-runtime && G2). "
                         "Still metered: ledger row, heartbeat and the kill-rule watchdog all stay active")
    a = ap.parse_args()

    import json
    import time
    from contextlib import nullcontext

    remap = [kv.split("=", 1) for kv in a.path_map]
    fix = lambda p: next((p.replace(o, n, 1) for o, n in remap if p.startswith(o)), p)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    man = pd.read_parquet(a.manifest)
    if a.limit:
        man = man.head(a.limit)
    man["src_file"] = man.src_file.map(fix)
    idx = {k: [fix(p) for p in v] for k, v in json.load(open(a.noise_index)).items()}
    missing = [p for p in man.src_file.drop_duplicates().head(20).tolist() + sum(idx.values(), [])[:20]
               if not Path(p).exists()]
    if missing:
        raise SystemExit(f"{len(missing)} input files not found, e.g. {missing[0]} (check --path-map / local copy)")
    copies = {"G0": ["clean", "train", "test"], "G1": ["clean", "train"], "G2": ["test"],
              "G3": [], "G4": ["clean", "train", "test"]}[a.stage]
    dev = "cpu" if a.io_only else "cuda"
    n_done = 0
    clock = {"read": 0.0, "augment": 0.0, "model": 0.0}
    # --io-only never holds a GPU, so it is not metered; real stages always go through the ledger + watchdog
    stage_ctx = (nullcontext() if a.io_only else
                 budget.stage(a.stage, out.parent, **({"unassign_on_exit": False} if a.keep_runtime else {})))
    with stage_ctx:
        if not a.io_only:
            print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE - aborting")
            assert torch.cuda.is_available(), "no GPU: switch the runtime to an L4/T4 first"
            model = load_wavlm(device=dev)
        t_run = time.time()
        aug = augment.Augmenter(idx, device=dev)
        for s0 in range(0, len(man), a.shard):
            shard_id = f"{a.stage}_{s0 // a.shard:04d}"
            if not a.io_only and shard_done(out, shard_id):
                print(f"{shard_id}: already complete, skipping", flush=True)
                continue  # resumable: skip finished shards (both files present, written atomically)
            rows, feats = [], []
            chunk = man.iloc[s0:s0 + a.shard]
            jobs = [(r, c) for _, r in chunk.iterrows() for c in aug.copy_plan(copies)]
            seg_cache: dict[str, np.ndarray] = {}            # each window is read ONCE for all its copies
            for bi, b0 in enumerate(range(0, len(jobs), a.batch)):
                batch = jobs[b0:b0 + a.batch]
                wavs, meta = [], []
                t = time.time()
                for r, (kind, cidx, snr) in batch:
                    if r.window_id not in seg_cache:
                        seg_cache = {r.window_id: read_16k(r.src_file, r.t0, r.t1)} | (
                            seg_cache if len(seg_cache) < 32 else {})
                    wavs.append(torch.from_numpy(seg_cache[r.window_id]).to(dev))
                    meta.append((r.window_id, kind, cidx, snr))
                clock["read"] += time.time() - t; t = time.time()
                wavs, envs = aug.apply(wavs, meta)
                clock["augment"] += time.time() - t; t = time.time()
                if not a.io_only:
                    feats.append(layer_means(model, [w.cpu().numpy() for w in wavs], device=dev, dtype=torch.bfloat16))
                clock["model"] += time.time() - t
                rows += [dict(window_id=m[0], copy=m[1], copy_idx=m[2], snr_db=m[3], env=e)
                         for m, e in zip(meta, envs)]
                n_done += len(batch)
                if bi % 5 == 0:   # progress within a minute, not only at the end of a shard
                    tot = sum(clock.values()) or 1e-9
                    print(f"  {n_done} clips | {n_done / (time.time() - t_run):.1f} clips/s | time in read "
                          f"{clock['read'] / tot:.0%}, augment {clock['augment'] / tot:.0%}, model {clock['model'] / tot:.0%}"
                          f"{'' if a.io_only else ' | G0 rule: need >= 20 clips/s'}", flush=True)
            if not a.io_only:
                write_shard(out, shard_id, np.concatenate(feats).astype(np.float16), pd.DataFrame(rows))
            print(f"{shard_id}: {n_done} clips, {n_done / (time.time() - t_run):.1f} clips/s overall", flush=True)


if __name__ == "__main__":
    main()

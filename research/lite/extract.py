"""GPU stage L1: frozen WavLM features for the lightweight projection experiment.

Per selected 8 s window, 5 views go through the frozen encoder (all 13 layers cached, fp16):
  0 clean          1 aug_a (train noise)   2 aug_b (train noise, independent draw)
  3 heldout        (HELD-OUT noise files: Experiment A, never used for training)
  4 pause          (SYNTHETIC extra pauses: Experiment B, never used for training)
Acoustic targets are computed on the clean view. Speakers are split train/val BEFORE anything is computed.

Colab (after unpacking the tars to /content/data):
  python -m research.lite.extract --manifest manifest.parquet --noise-index noise_index.json --out lite_feats \
      --path-map <drive prefix>=<local prefix> ...
"""
from __future__ import annotations

import argparse
import json
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np

from ..extract import read_16k, seed_for, shard_done, write_shard
from .acoustic import TARGETS, acoustic_targets, insert_pauses, mic_sim, time_stretch

SR = 16000
VIEWS = ("clean", "aug_a", "aug_b", "heldout", "pause")
NUISANCES = ("noise", "babble", "reverb", "mic")


def select_windows(man, n_speakers: int = 24, per_speaker: int = 20, val_frac: float = 0.25, seed: int = 42):
    """Seeded choice of speakers with >= per_speaker windows; their first per_speaker CONSECUTIVE windows (needed
    to form 3-window sessions for drift constants). Speaker-disjoint split decided here, once."""
    counts = man.groupby("speaker_id").size()
    eligible = sorted(counts[counts >= per_speaker].index)
    rng = random.Random(seed)
    spk = sorted(rng.sample(eligible, min(n_speakers, len(eligible))))
    val = set(rng.sample(spk, max(1, round(len(spk) * val_frac))))
    sel = man[man.speaker_id.isin(spk)].sort_values(["speaker_id", "src_file", "t0"]).groupby("speaker_id").head(per_speaker)
    sel = sel.copy()
    sel["split"] = np.where(sel.speaker_id.isin(val), "val", "train")
    return sel.reset_index(drop=True)


def _mix(x: np.ndarray, n: np.ndarray, snr_db: float) -> np.ndarray:
    n = np.resize(n, len(x)).astype(np.float64)
    ps, pn = float(np.mean(x ** 2)), float(np.mean(n ** 2)) + 1e-20
    return (x + n * np.sqrt(ps / (pn * 10 ** (snr_db / 10)))).astype(np.float32)


def nuisance_view(x: np.ndarray, rng: random.Random, idx: dict, held_out: bool) -> tuple[np.ndarray, str]:
    """One or two nuisance transforms (spec 3.3): additive noise or babble at 10-30 dB SNR, simulated reverb,
    mild mic response. No pitch/time changes. held_out=True draws noise from the TEST pools only."""
    import torch

    from ..augment import _load, fft_convolve, simulated_rir

    pool = "test" if held_out else "train"
    kinds = rng.sample(NUISANCES, rng.choice([1, 2]))
    y = x.astype(np.float32)
    for k in kinds:
        if k == "noise":
            y = _mix(y, _load(rng.choice(idx[f"noise_{pool}"]), len(y), rng, "cpu").numpy(), rng.uniform(10, 30))
        elif k == "babble":
            b = sum(_load(rng.choice(idx[f"speech_{pool}"]), len(y), rng, "cpu").numpy() for _ in range(rng.randint(3, 5)))
            y = _mix(y, b, rng.uniform(10, 30))
        elif k == "reverb":
            y = fft_convolve(torch.from_numpy(y), simulated_rir(rng)).numpy().astype(np.float32)
        else:
            y = mic_sim(y, np.random.default_rng(rng.getrandbits(32)))
    return y, "+".join(kinds)


def make_views(seg: np.ndarray, window_id: str, idx: dict) -> tuple[list[np.ndarray], list[str]]:
    views, labels = [seg], ["clean"]
    for ci, held in ((1, False), (2, False), (3, True)):
        y, lab = nuisance_view(seg, random.Random(seed_for(window_id, ci)), idx, held)
        views.append(y); labels.append(lab)
    views.append(insert_pauses(seg)); labels.append("synthetic_pauses")
    return views, labels


SWEEP_SNRS = (20, 10, 5, 0)


def sweep_views(seg: np.ndarray, window_id: str, idx: dict) -> tuple[list[np.ndarray], list[str]]:
    """L2 (aggregate experiment): ADDITIVE noise only, at fixed SNRs, from HELD-OUT noise files (noise or babble,
    seeded per window and level). Same clean speech in every view, so the only change is the added noise."""
    from ..augment import _load

    views, labels = [], []
    for i, snr in enumerate(SWEEP_SNRS):
        n, kind = _heldout_noise(random.Random(seed_for(window_id, 200 + i)), len(seg), idx)
        views.append(_mix(seg, n, snr)); labels.append(f"{kind}@{snr}dB")
    return views, labels


def _heldout_noise(rng: random.Random, n: int, idx: dict) -> tuple[np.ndarray, str]:
    from ..augment import _load

    if rng.random() < 0.5:
        return _load(rng.choice(idx["noise_test"]), n, rng, "cpu").numpy(), "noise"
    return sum(_load(rng.choice(idx["speech_test"]), n, rng, "cpu").numpy() for _ in range(rng.randint(3, 5))), "babble"


CHANGE_VIEWS = ("stretch", "pause_noise10", "stretch_noise10")
STRETCH = 1.2          # 20% slower delivery, pitch unchanged (spec 3.4 fallback: 15-25%)


def change_views(seg: np.ndarray, window_id: str, idx: dict) -> tuple[list[np.ndarray], list[str]]:
    """L3 (evidence spec 3.4/3.5), eval speakers only:
      stretch          SYNTHETIC slower delivery (time-stretch x1.2, no pitch shift)
      pause_noise10    the L1 synthetic-pause change PLUS held-out noise at 10 dB
      stretch_noise10  the slower delivery PLUS held-out noise at 10 dB
    The 10 dB noise uses the same seeded draw as the L2 10 dB view, so change+noise differs from noise-only only by
    the speech change."""
    st, pa = time_stretch(seg, STRETCH), insert_pauses(seg)
    i10 = SWEEP_SNRS.index(10)
    views, labels = [st], [f"time_stretch x{STRETCH}"]
    for name, y in (("pause", pa), ("stretch", st)):
        n, kind = _heldout_noise(random.Random(seed_for(window_id, 200 + i10)), len(y), idx)
        views.append(_mix(y, n, 10)); labels.append(f"{name}+{kind}@10dB")
    return views, labels


def main() -> None:  # pragma: no cover - Colab entry point
    import pandas as pd
    import torch

    from .. import budget
    from ..extract import layer_means, load_wavlm

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--noise-index", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--speakers", type=int, default=24)
    ap.add_argument("--per-speaker", type=int, default=20)
    ap.add_argument("--shard", type=int, default=40, help="windows per checkpoint shard")
    ap.add_argument("--batch-windows", type=int, default=12, help="windows per forward pass (x5 views)")
    ap.add_argument("--path-map", action="append", default=[], metavar="OLD=NEW")
    ap.add_argument("--io-only", action="store_true", help="time the data path on CPU, no model, no ledger")
    ap.add_argument("--noise-sweep", action="store_true",
                    help="stage L2: held-out (eval) speakers only, additive held-out noise at 20/10/5/0 dB")
    ap.add_argument("--change-sweep", action="store_true",
                    help="stage L3: eval speakers only, slower delivery (time-stretch) and change+noise views")
    ap.add_argument("--cpu", action="store_true",
                    help="run WavLM on CPU (no GPU ledger); fine for L2/L3, which are a few hundred clips")
    a = ap.parse_args()

    remap = [kv.split("=", 1) for kv in a.path_map]
    fix = lambda p: next((p.replace(o, n, 1) for o, n in remap if p.startswith(o)), p)
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    sel = select_windows(pd.read_parquet(a.manifest), a.speakers, a.per_speaker)
    sel["src_file"] = sel.src_file.map(fix)
    stage = "L3" if a.change_sweep else "L2" if a.noise_sweep else "L1"
    prefix = stage
    if a.noise_sweep or a.change_sweep:
        sel = sel[sel.split == "val"].reset_index(drop=True)      # same seeded split as L1: eval speakers only
    idx = {k: [fix(p) for p in v] for k, v in json.load(open(a.noise_index)).items()}
    bad = [p for p in sel.src_file.drop_duplicates().tolist()[:20] + sum(idx.values(), [])[:20] if not Path(p).exists()]
    if bad:
        raise SystemExit(f"input files not found, e.g. {bad[0]} (check --path-map / local copy)")
    if stage == "L1":
        sel.to_parquet(out / "selection.parquet")
    nviews = {"L1": len(VIEWS), "L2": len(SWEEP_SNRS), "L3": len(CHANGE_VIEWS)}[stage]
    print(f"[{stage}] {len(sel)} windows | {sel.speaker_id.nunique()} speakers | val speakers "
          f"{sel[sel.split == 'val'].speaker_id.nunique()} | {len(sel) * nviews} clips", flush=True)
    dev = "cpu" if (a.io_only or a.cpu) else "cuda"
    done, t0 = 0, time.time()
    with (nullcontext() if (a.io_only or a.cpu) else budget.stage(stage, out.parent)):
        if not a.io_only:
            if dev == "cuda":
                assert torch.cuda.is_available(), "no GPU: switch the runtime to an L4/T4 first, or pass --cpu"
                print("GPU:", torch.cuda.get_device_name(0), flush=True)
            model = load_wavlm(device=dev)
        for s0 in range(0, len(sel), a.shard):
            sid = f"{prefix}_{s0 // a.shard:04d}"
            if not a.io_only and shard_done(out, sid):
                print(f"{sid}: already complete, skipping", flush=True); continue
            chunk = sel.iloc[s0:s0 + a.shard]
            feats, rows = [], []
            for b0 in range(0, len(chunk), a.batch_windows):
                wavs = []
                for _, r in chunk.iloc[b0:b0 + a.batch_windows].iterrows():
                    seg = read_16k(r.src_file, r.t0, r.t1)
                    tv, tm = acoustic_targets(seg)
                    if a.change_sweep:
                        views, labels = change_views(seg, r.window_id, idx)
                        names = list(CHANGE_VIEWS)
                    elif a.noise_sweep:
                        views, labels = sweep_views(seg, r.window_id, idx)
                        names = [f"noise{snr}" for snr in SWEEP_SNRS]
                    else:
                        views, labels = make_views(seg, r.window_id, idx)
                        names = list(VIEWS)
                    for vi, (v, lab) in enumerate(zip(views, labels)):
                        wavs.append(v)
                        rows.append({"window_id": r.window_id, "speaker_id": r.speaker_id, "split": r.split,
                                     "view": names[vi], "view_detail": lab, "seconds": round(len(v) / SR, 2),
                                     **{f"t_{k}": float(tv[i]) for i, k in enumerate(TARGETS)},
                                     **{f"m_{k}": bool(tm[i]) for i, k in enumerate(TARGETS)}})
                if not a.io_only:
                    feats.append(layer_means(model, wavs, device=dev, dtype=torch.bfloat16 if dev == "cuda" else None))
                done += len(wavs)
                print(f"  {done} clips | {done / (time.time() - t0):.1f} clips/s", flush=True)
            if not a.io_only:
                write_shard(out, sid, np.concatenate(feats).astype(np.float16), pd.DataFrame(rows))
            print(f"{sid}: written ({done} clips so far)", flush=True)


if __name__ == "__main__":
    main()

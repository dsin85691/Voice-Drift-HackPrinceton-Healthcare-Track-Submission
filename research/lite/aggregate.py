"""Aggregate experiment: the two numbers the project's claim rests on, raw WavLM (B0) vs the projection (P1/B1).

  1. FALSE-ALERT RATE UNDER ADDED NOISE: share of unchanged sessions (same speaker, same words) that cross the
     model's own boundary once noise is added. Lower is better.
  2. SENSITIVITY TO A CONTROLLED SPEECH CHANGE: share of sessions with a controlled delivery change (SYNTHETIC extra
     pauses, same words, same conditions) that cross the same boundary. Higher is better.

Design (no tuning on evaluation data):
  - sessions = 3 consecutive 8 s windows (~24 s), as the app scores a check-in; every eval speaker's sessions are
    scored leave-one-session-out: baseline reference = mean of that speaker's OTHER sessions' clean windows
  - distance = cosine (same as the app's window analysis); each model is judged against ITS OWN boundary
  - boundary tau_m = 90th percentile of clean leave-one-session-out distances on TRAINING speakers only, so the
    expected clean false-alert rate is ~10% by construction (reported as a sanity check)
  - noise: L1 'heldout' view (held-out noise/babble 10-30 dB, sometimes + reverb/mic) and, if the L2 noise sweep
    exists, pure additive held-out noise at 20 / 10 / 5 / 0 dB
  - 95% CIs: bootstrap over evaluation SPEAKERS

Run (CPU, seconds):  python -m research.lite.aggregate --feats lite_feats --checkpoints reports/checkpoints
Writes results/aggregate.json, which the app's Results and Home pages read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ..heads import ProjectionHead
from .train import load_lite

SESSION = 3
CONDITIONS_L1 = {"clean": 0, "noise_mix": 3, "pause_change": 4}   # view index in LiteData.H


def _cos(A, mu):
    A = np.atleast_2d(A)
    return 1 - (A @ mu) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(mu), 1e-12)


def load_sweep(feat_dir, layer: int) -> dict:
    """{(window_id, 'noise20'|...): vec}. Empty if the L2 sweep wasn't run."""
    import pandas as pd

    out = {}
    for pq in sorted(Path(feat_dir).glob("L2_*.parquet")):
        npy = pq.with_suffix(".npy")
        if npy.exists():
            df = pd.read_parquet(pq)
            f = np.load(npy, mmap_mode="r")
            for i, (w, v) in enumerate(zip(df.window_id, df.view)):
                out[(w, v)] = np.asarray(f[i, layer], dtype=np.float32)
    return out


def embedders(ckpt_dir) -> dict:
    """B0 = raw WavLM layer; P1 / B1 from their checkpoints (if present)."""
    out = {"B0": None}
    for k in ("P1", "B1"):
        f = Path(ckpt_dir) / f"{k}.pt"
        if f.exists():
            ck = torch.load(f, weights_only=True)
            h = ProjectionHead(); h.load_state_dict(ck["head"]); h.eval()
            out[k] = h
    return out


def _embed(model, X: np.ndarray) -> np.ndarray:
    if model is None:
        return X.astype(np.float64)
    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(X, np.float32))).numpy().astype(np.float64)


def loso_session_distances(E_clean: np.ndarray, E_test: np.ndarray, size: int = SESSION) -> np.ndarray:
    """One speaker. E_*: [n_windows, d] in window order. For each session i: reference = mean of the clean windows of
    all OTHER sessions; distance of the session-mean of E_test over session i's windows."""
    n = len(E_clean) // size
    if n < 2:
        return np.zeros(0)
    d = []
    for i in range(n):
        other = np.r_[0:i * size, (i + 1) * size:n * size]
        mu = E_clean[other].mean(0)
        d.append(_cos(E_test[i * size:(i + 1) * size].mean(0), mu)[0])
    return np.array(d)


def loso_window_distances(E_clean: np.ndarray, E_test: np.ndarray, size: int = SESSION) -> np.ndarray:
    n = len(E_clean) // size
    d = []
    for i in range(n):
        other = np.r_[0:i * size, (i + 1) * size:n * size]
        mu = E_clean[other].mean(0)
        d.extend(_cos(E_test[i * size:(i + 1) * size], mu).tolist())
    return np.array(d)


def _rates(per_spk: dict, tau: float) -> dict:
    return {s: (float((v > tau).sum()), len(v)) for s, v in per_spk.items()}


def _summ(per_spk: dict, tau: float, B: int, rng) -> dict:
    r = _rates(per_spk, tau)
    sp = sorted(r)
    hits, n = sum(r[s][0] for s in sp), sum(r[s][1] for s in sp)
    if n == 0:
        return {"rate": None, "ci": [None, None], "n": 0}
    boot = []
    for _ in range(B):
        ss = rng.choice(sp, len(sp), replace=True)
        h, m = sum(r[s][0] for s in ss), sum(r[s][1] for s in ss)
        boot.append(h / m if m else np.nan)
    return {"rate": round(hits / n, 4), "ci": [round(float(np.nanpercentile(boot, 2.5)), 4),
                                                round(float(np.nanpercentile(boot, 97.5)), 4)], "n": int(n)}


def run(feat_dir="lite_feats", ckpt_dir="reports/checkpoints", layer: int = 6, out_path="results/aggregate.json",
        B: int = 2000, seed: int = 0, q: float = 0.9) -> dict:
    d = load_lite(feat_dir, layer)
    sweep = load_sweep(feat_dir, layer)
    models = embedders(ckpt_dir)
    rng = np.random.default_rng(seed)
    train_spk = sorted(set(d.spk[d.split == "train"]))
    eval_spk = sorted(set(d.spk[d.split == "val"]))
    sweep_levels = sorted({int(v[5:]) for (_, v) in sweep}, reverse=True)
    res = {"session": {}, "window": {}, "tau": {}}
    for name, m in models.items():
        E = {v: _embed(m, d.H[:, i]) for v, i in {"clean": 0, "noise_mix": 3, "pause_change": 4}.items()}
        idx = {s: np.where(d.spk == s)[0] for s in set(d.spk)}
        # boundary from TRAINING speakers' clean audio only
        tr_s = np.concatenate([loso_session_distances(E["clean"][idx[s]], E["clean"][idx[s]]) for s in train_spk])
        tr_w = np.concatenate([loso_window_distances(E["clean"][idx[s]], E["clean"][idx[s]]) for s in train_spk])
        tau_s, tau_w = float(np.quantile(tr_s, q)), float(np.quantile(tr_w, q))
        res["tau"][name] = {"session": tau_s, "window": tau_w}
        conds = {c: E[c] for c in ("clean", "noise_mix", "pause_change")}
        for lvl in sweep_levels:
            arr = np.full_like(E["clean"], np.nan)
            ev = np.where(d.split == "val")[0]
            keys = [(d.window_id[i], f"noise{lvl}") for i in ev]
            if all(k in sweep for k in keys):
                arr[ev] = _embed(m, np.stack([sweep[k] for k in keys]))
                conds[f"noise_{lvl}dB"] = arr
        res["session"][name], res["window"][name] = {}, {}
        for c, Ec in conds.items():
            ps = {s: loso_session_distances(E["clean"][idx[s]], Ec[idx[s]]) for s in eval_spk}
            pw = {s: loso_window_distances(E["clean"][idx[s]], Ec[idx[s]]) for s in eval_spk}
            res["session"][name][c] = _summ(ps, tau_s, B, rng)
            res["window"][name][c] = _summ(pw, tau_w, B, rng)
    proj = "P1" if "P1" in models else ("B1" if "B1" in models else None)
    noise_key = "noise_10dB" if any("noise_10dB" in v for v in res["session"].values()) else "noise_mix"
    noise_label = {"noise_10dB": "added held-out noise or babble at 10 dB SNR",
                   "noise_mix": "added held-out noise or babble at 10-30 dB (some with reverb or mic coloring)"}[noise_key]
    head = None
    if proj:
        S = res["session"]
        head = {"level": "session (3 windows, ~24 s)", "projection_model": proj, "noise_condition": noise_label,
                "false_alert_raw": S["B0"][noise_key], "false_alert_projection": S[proj][noise_key],
                "change_sensitivity_raw": S["B0"]["pause_change"], "change_sensitivity_projection": S[proj]["pause_change"],
                "clean_false_alert_raw": S["B0"]["clean"], "clean_false_alert_projection": S[proj]["clean"]}
    out = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "illustrative": False,
           "headline": head, "results": res,
           "conditions": {"clean": "unchanged clean speech (expected ~10% by construction)",
                          "noise_mix": "held-out noise/babble 10-30 dB, sometimes with reverb or mic coloring",
                          **{f"noise_{l}dB": f"held-out noise or babble at {l} dB SNR, additive only" for l in sweep_levels},
                          "pause_change": "SYNTHETIC controlled change: room-tone pauses inserted after each speech run"},
           "meta": {"wavlm_layer": layer, "eval_speakers": len(eval_spk), "train_speakers_for_boundary": len(train_spk),
                    "session_windows": SESSION, "boundary_quantile": q, "distance": "cosine",
                    "bootstrap": f"{B} resamples over evaluation speakers", "models": list(models)},
           "limitations": [
               f"Small: {len(eval_spk)} held-out speakers (Androids interview speech, Italian).",
               "The speech change is synthetic (inserted pauses), not real slower or flatter delivery.",
               "Leave-one-session-out pseudo-sessions come from one interview: same day, same room. Real day-to-day "
               "variation is larger, so these false-alert rates are a lower bound.",
               "Speech-change sensitivity says nothing about detecting any health condition."]}
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    Path(out_path).write_text(json.dumps(out, indent=1))
    return out


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="lite_feats")
    ap.add_argument("--checkpoints", default="reports/checkpoints")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--out", default="results/aggregate.json")
    a = ap.parse_args()
    o = run(a.feats, a.checkpoints, a.layer, a.out)
    h = o["headline"]
    if h:
        f = lambda r: f"{r['rate']:.0%} [{r['ci'][0]:.0%}-{r['ci'][1]:.0%}] (n={r['n']})"
        print(f"False alerts under {h['noise_condition']}:  raw {f(h['false_alert_raw'])}  vs  "
              f"{h['projection_model']} {f(h['false_alert_projection'])}")
        print(f"Detects the controlled change:  raw {f(h['change_sensitivity_raw'])}  vs  "
              f"{h['projection_model']} {f(h['change_sensitivity_projection'])}")
        print(f"(sanity, clean: raw {f(h['clean_false_alert_raw'])}, {h['projection_model']} {f(h['clean_false_alert_projection'])})")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

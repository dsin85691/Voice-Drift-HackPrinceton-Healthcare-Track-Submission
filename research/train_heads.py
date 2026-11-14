"""L0 / M1 / M3 (+ optional M2) on cached frozen WavLM features, 5-fold CV, CPU only (spec 3.4).

Data layout (all arrays aligned to `win` = manifest rows):
  Zc  [N, 13, 768]      clean windows
  Za  [N, A, 13, 768]   train-aug copies (A=4);  Ea [N, A] env class 1..5
  Zt  {snr: [N, 13, 768]}  test-corrupt copies (held-out noise), snr in (20, 10, 5)
  y [N] window label, spk [N] speaker id (str), fold [N] official fold id.
Leakage assertions live in `assert_no_leakage` and run inside `run_cv`.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from .stats import LogisticRegression, roc_auc_score
from torch import nn
from torch.nn import functional as F

from .heads import Heads, Projector

C_GRID = (0.01, 0.1, 1.0, 10.0)
ALPHA, BETA_MAX, BETA_RAMP = 1.0, 0.5, 0.3
EPOCHS, BATCH, LR, WD, PATIENCE = 40, 256, 1e-3, 1e-2, 6


@dataclass
class Data:
    Zc: np.ndarray
    Za: np.ndarray
    Ea: np.ndarray
    Zt: dict
    y: np.ndarray
    spk: np.ndarray
    fold: np.ndarray


# ------------------------------------------------------------------ helpers
def speaker_scores(p: np.ndarray, spk: np.ndarray, y: np.ndarray):
    """Speaker score = mean of window probabilities. Returns (speakers, scores, labels)."""
    us = np.unique(spk)
    sc = np.array([p[spk == s].mean() for s in us])
    lb = np.array([y[spk == s][0] for s in us])
    return us, sc, lb


def speaker_auc(p, spk, y) -> float:
    _, sc, lb = speaker_scores(p, spk, y)
    return float(roc_auc_score(lb, sc)) if len(set(lb)) > 1 else float("nan")


def inner_split(train_spk: np.ndarray, labels: dict, frac=0.15, seed=0):
    """Stratified 15% of TRAINING speakers held out as inner validation."""
    rng = np.random.default_rng(seed)
    val = []
    for lab in (0, 1):
        s = sorted(s for s in train_spk if labels[s] == lab)
        rng.shuffle(s)
        val += s[: max(1, int(round(len(s) * frac)))]
    val = set(val)
    return [s for s in train_spk if s not in val], sorted(val)


def assert_no_leakage(d: Data, fold: int, train_spk, test_spk, inner_val):
    assert not set(train_spk) & set(test_spk), "speaker in both train and test"
    assert not set(inner_val) & (set(test_spk)), "inner-val touches test speakers"
    assert set(inner_val) <= set(train_spk), "inner-val must come from training speakers"
    # augmented and test copies share their source's fold (they are indexed by the source window)
    for s in test_spk:
        assert np.all(d.fold[d.spk == s] == fold), "test speaker spans folds"


def standardize_fit(X):
    m, s = X.mean(0), X.std(0) + 1e-6
    return m, s


# ------------------------------------------------------------------ L0 and layer choice
def fit_l0(X, y, C, device="cpu"):
    m, s = standardize_fit(X.astype(np.float32))
    clf = LogisticRegression(C=C, max_iter=500, device=device).fit((X - m) / s, y)
    return lambda Z: clf.predict_proba((Z - m) / s)[:, 1]


def choose_layer_and_C(d: Data, inner_train, inner_val, layers=range(13), device="cpu"):
    """l* and C by inner-validation speaker AUC of L0 (uses training-fold speakers only)."""
    tr = np.isin(d.spk, inner_train)
    va = np.isin(d.spk, inner_val)
    best = (-1.0, 0, 1.0)
    table = {}
    for l in layers:
        for C in C_GRID:
            f = fit_l0(d.Zc[tr, l], d.y[tr], C, device)
            a = speaker_auc(f(d.Zc[va, l]), d.spk[va], d.y[va])
            table[(l, C)] = a
            if np.nan_to_num(a, nan=-1) > best[0]:
                best = (a, l, C)
    return best[1], best[2], table


# ------------------------------------------------------------------ M1 / M2 / M3
def train_head(d: Data, kind: str, layer: int, train_idx: np.ndarray, val_idx: np.ndarray, seed: int,
               epochs: int = EPOCHS, beta_max: float = BETA_MAX, alpha: float = ALPHA, device: str = "cpu"):
    """kind in {"M1", "M2", "M3"}. Returns (projector, heads, best_val_auc) on CPU with the best-AUC weights.
    device="cuda" trains on the GPU (opt-in; the spec keeps training on CPU by default)."""
    assert kind in ("M1", "M2", "M3")
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    Zc = torch.from_numpy(d.Zc[:, layer].astype(np.float32)).to(device)
    Za = torch.from_numpy(d.Za[:, :, layer].astype(np.float32)).to(device)
    Ea = torch.from_numpy(d.Ea.astype(np.int64)).to(device)
    y = torch.from_numpy(d.y.astype(np.float32)).to(device)
    P, H = Projector(), Heads()
    m, s = standardize_fit(d.Zc[train_idx, layer])
    P.mean.copy_(torch.from_numpy(m.astype(np.float32)))
    P.std.copy_(torch.from_numpy(s.astype(np.float32)))
    P, H = P.to(device), H.to(device)
    opt = torch.optim.AdamW(list(P.parameters()) + list(H.parameters()), lr=LR, weight_decay=WD)
    n_steps = epochs * int(np.ceil(len(train_idx) / BATCH))
    step, best, best_state, bad = 0, -1.0, None, 0
    A = Za.shape[1]
    for ep in range(epochs):
        P.train(); H.train()
        perm = rng.permutation(train_idx)
        for b in range(0, len(perm), BATCH):
            ix = torch.from_numpy(perm[b:b + BATCH]).to(device)
            a = torch.from_numpy(rng.integers(0, A, len(ix))).to(device)
            zc, zn, e, yy = Zc[ix], Za[ix, a], Ea[ix, a], y[ix]
            hc, hn = P(zc), P(zn)
            loss = 0.5 * (F.binary_cross_entropy_with_logits(H.disease(hc), yy)
                          + F.binary_cross_entropy_with_logits(H.disease(hn), yy))
            if kind in ("M2", "M3"):
                loss = loss + alpha * (hc - hn).pow(2).sum(1).mean()
            if kind == "M3":
                beta = beta_max * min(1.0, step / max(1, BETA_RAMP * n_steps))   # ramp 0 -> beta over first 30%
                loss = loss + beta * 0.5 * (F.cross_entropy(H.env(hc, 1.0), torch.zeros_like(e))
                                            + F.cross_entropy(H.env(hn, 1.0), e))
            opt.zero_grad(); loss.backward(); opt.step(); step += 1
        P.eval(); H.eval()
        with torch.no_grad():
            pv = torch.sigmoid(H.disease(P(Zc[torch.from_numpy(val_idx).to(device)]))).cpu().numpy()
        auc = speaker_auc(pv, d.spk[val_idx], d.y[val_idx])
        if np.nan_to_num(auc, nan=-1) > best:
            best, bad = auc, 0
            best_state = ({k: v.clone() for k, v in P.state_dict().items()}, {k: v.clone() for k, v in H.state_dict().items()})
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    P.load_state_dict(best_state[0]); H.load_state_dict(best_state[1])
    P, H = P.cpu().eval(), H.cpu().eval()      # always hand back CPU modules (serving/export are device-free)
    return P, H, best


def predict_head(P, H, Z: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        return torch.sigmoid(H.disease(P(torch.from_numpy(Z.astype(np.float32))))).numpy()


# ------------------------------------------------------------------ 5-fold CV driver
def run_cv(d: Data, models=("L0", "M1", "M3"), seeds=(0, 1, 2), folds=None, epochs=EPOCHS,
           layers=range(13), device: str = "cpu"):
    """Returns dict: results[model][cond] = DataFrame-like list of (fold, speaker, label, score), plus chosen layers.
    cond in {"clean", 20, 10, 5}."""
    labels = {s: int(d.y[d.spk == s][0]) for s in np.unique(d.spk)}
    folds = folds if folds is not None else sorted(np.unique(d.fold))
    res = {m: {c: [] for c in ["clean", *d.Zt.keys()]} for m in models}
    chosen = {}
    for f in folds:
        test_mask = d.fold == f
        test_spk = sorted(np.unique(d.spk[test_mask]))
        train_spk = sorted(np.unique(d.spk[~test_mask]))
        tr_in, va_in = inner_split(np.array(train_spk), labels, seed=f)
        assert_no_leakage(d, f, train_spk, test_spk, va_in)
        layer, C, _ = choose_layer_and_C(d, tr_in, va_in, layers, device)       # only training-fold speakers
        chosen[f] = {"layer": layer, "C": C}
        tr_idx = np.where(np.isin(d.spk, tr_in))[0]
        va_idx = np.where(np.isin(d.spk, va_in))[0]
        fit_idx = np.where(~test_mask)[0]
        conds = {"clean": d.Zc, **{k: v for k, v in d.Zt.items()}}

        def record(model, preds):
            for cond, p in preds.items():
                us, sc, lb = speaker_scores(p[test_mask], d.spk[test_mask], d.y[test_mask])
                res[model][cond] += list(zip([f] * len(us), us, lb, sc))

        if "L0" in models:
            f0 = fit_l0(d.Zc[fit_idx, layer], d.y[fit_idx], C, device)
            record("L0", {c: f0(Z[:, layer]) for c, Z in conds.items()})
        for kind in [m for m in models if m != "L0"]:
            ps = {c: [] for c in conds}
            for sd in seeds:
                P, H, _ = train_head(d, kind, layer, tr_idx, va_idx, seed=sd, epochs=epochs, device=device)
                for c, Z in conds.items():
                    ps[c].append(predict_head(P, H, Z[:, layer]))
            record(kind, {c: np.mean(v, 0) for c, v in ps.items()})   # 3 seeds averaged
    return res, chosen


"""Basic post-hoc calibration of the disease head (Platt scaling) + calibration metrics.

Design (see README "Cohort reference risk"):
- Fit ONLY on out-of-fold (OOF) predictions from the 5-fold CV, never on training predictions.
- Fit at SESSION level, at live check-in length: mean of the window logits over 3-window pseudo-sessions
  (~24 s), because live check-ins are 2-7 windows, not whole 5-minute interviews.
- Platt (2 parameters) because 116 speakers is too few for isotonic regression.
- This calibrates to the SOURCE cohort (Androids: Italian interview speech, ~55% depressed). It does not
  correct for language/microphone shift or for a different prevalence; those are documented limitations.
"""
from __future__ import annotations

import numpy as np

from .stats import LogisticRegression, roc_auc_score

SESSION_WINDOWS = 3
N_BINS = 10


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float)))


# ------------------------------------------------------------------ Platt
def platt_fit(z: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """p = sigmoid(a * z + b). Essentially unregularized (C large); 2 parameters."""
    clf = LogisticRegression(C=1e6, max_iter=1000).fit(np.asarray(z, float).reshape(-1, 1), np.asarray(y))
    return float(clf.W[0, 0]), float(clf.b[0])


def platt_apply(z, a: float, b: float):
    return sigmoid(a * np.asarray(z, dtype=float) + b)


def prior_shift(b: float, pi_source: float, pi_target: float) -> float:
    """Intercept correction for a different prevalence (label shift). Not applied by default: choosing the
    target prevalence is a clinical decision, not something this data can tell us."""
    return b + float(logit(pi_target) - logit(pi_source))


# ------------------------------------------------------------------ metrics
def brier(p, y) -> float:
    return float(np.mean((np.asarray(p) - np.asarray(y)) ** 2))


def reliability(p, y, n_bins: int = N_BINS) -> list[dict]:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, n_bins + 1)
    out = []
    for i in range(n_bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < n_bins - 1 else (p <= edges[i + 1]))
        out.append({"bin_lo": round(edges[i], 2), "bin_hi": round(edges[i + 1], 2), "n": int(m.sum()),
                    "mean_pred": round(float(p[m].mean()), 4) if m.any() else None,
                    "frac_pos": round(float(y[m].mean()), 4) if m.any() else None})
    return out


def ece(p, y, n_bins: int = N_BINS) -> float:
    """Expected calibration error: bin-weighted |mean predicted - observed frequency|."""
    p, y = np.asarray(p, float), np.asarray(y, float)
    tot = 0.0
    for r in reliability(p, y, n_bins):
        if r["n"]:
            tot += r["n"] * abs(r["mean_pred"] - r["frac_pos"])
    return float(tot / len(p))


# ------------------------------------------------------------------ sessions from OOF window logits
def session_logits(window_logits: np.ndarray, spk: np.ndarray, y: np.ndarray, fold: np.ndarray,
                   size: int = SESSION_WINDOWS):
    """Group each speaker's consecutive windows into pseudo-sessions of `size`, average the logits.
    Returns arrays (speaker, label, fold, session_logit)."""
    S, Y, F, Z = [], [], [], []
    for s in np.unique(spk):
        m = spk == s
        zl = window_logits[m]
        n = len(zl) // size
        for i in range(n):
            S.append(s); Y.append(int(y[m][0])); F.append(int(fold[m][0]))
            Z.append(float(zl[i * size:(i + 1) * size].mean()))
    return np.array(S), np.array(Y, int), np.array(F, int), np.array(Z, float)


# ------------------------------------------------------------------ the full fit
def fit_calibration(spk, y, fold, z, model: str = "M3", spec_elevated: float = 0.8, spec_moderate: float = 0.5,
                    B: int = 1000, seed: int = 0) -> dict:
    """Platt on OOF session logits + metrics with speaker-bootstrap CIs + band cutoffs + the E1 gate.

    Bands (on the calibrated probability): 'elevated' >= the control-session quantile at `spec_elevated`
    (80% of control sessions fall below it), 'moderate' >= the control median, else 'low'.
    Gate: shown only if the session-level AUC's 95% CI lower bound is above 0.5 (spec section 11)."""
    spk, y, fold, z = map(np.asarray, (spk, y, fold, z))
    a, b = platt_fit(z, y)
    p = platt_apply(z, a, b)
    p_raw = sigmoid(z)
    ctrl = p[y == 0]
    t_mod, t_hi = float(np.quantile(ctrl, spec_moderate)), float(np.quantile(ctrl, spec_elevated))

    rng = np.random.default_rng(seed)
    us = np.unique(spk)
    idx_by = {s: np.where(spk == s)[0] for s in us}
    boot = {"auc": [], "brier": [], "ece": []}
    for _ in range(B):                      # resample SPEAKERS (sessions of one speaker are not independent)
        ix = np.concatenate([idx_by[s] for s in rng.choice(us, len(us), replace=True)])
        if len(set(y[ix])) < 2:
            continue
        boot["auc"].append(roc_auc_score(y[ix], z[ix]))
        boot["brier"].append(brier(p[ix], y[ix]))
        boot["ece"].append(ece(p[ix], y[ix]))
    ci = lambda v: [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
    auc = roc_auc_score(y, z)
    auc_ci = ci(boot["auc"])
    return {
        "model": model,
        "method": "platt",
        "a": a, "b": b,
        "session_windows": SESSION_WINDOWS,
        "n_sessions": int(len(z)), "n_speakers": int(len(us)),
        "source_prevalence": round(float(y.mean()), 4),
        "bands": {"moderate": round(t_mod, 4), "elevated": round(t_hi, 4),
                  "rule": f"elevated >= control q{int(spec_elevated * 100)}, moderate >= control median"},
        "metrics": {
            "auc": round(float(auc), 4), "auc_ci": auc_ci,
            "brier": round(brier(p, y), 4), "brier_ci": ci(boot["brier"]),
            "brier_uncalibrated": round(brier(p_raw, y), 4),
            "ece": round(ece(p, y), 4), "ece_ci": ci(boot["ece"]),
            "ece_uncalibrated": round(ece(p_raw, y), 4),
            "reliability": reliability(p, y),
        },
        "gate_passed": bool(auc_ci[0] > 0.5),
        "label": "Androids-cohort reference (Italian interview speech; not validated on English speech)",
        "caveats": [
            "Calibrated on the source cohort only (out-of-fold, 3-window sessions).",
            "Language, microphone and task differ for live users; calibration may not transfer.",
            f"Source prevalence is {y.mean():.0%}; no prevalence correction is applied.",
            "Reference only: never a diagnosis, never combined with drift, never shown to the patient.",
        ],
    }


def band(p: float, cal: dict) -> str:
    t = cal["bands"]
    return "elevated" if p >= t["elevated"] else "moderate" if p >= t["moderate"] else "low"

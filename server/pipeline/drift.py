"""Drift and alert math (spec section 6). Pure numpy, no I/O besides loading artifacts."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from ..config import get_config, resolve


@dataclass
class DriftModel:
    k: int
    tau: float
    sigma_pop: np.ndarray            # (k,)
    pca_mean: np.ndarray | None      # (d,)
    pca_components: np.ndarray | None  # (k, d)
    version: str = "fallback"
    layer: int | None = None
    meta: dict | None = None

    def project(self, v: np.ndarray) -> np.ndarray:
        v = np.asarray(v, dtype=np.float64)
        if self.pca_components is None:
            return v[: self.k]
        return (v - self.pca_mean) @ self.pca_components.T


def load_drift_model(raw: bool = False) -> DriftModel:
    """raw=True loads the raw-z drift constants (for the robustness toggle); falls back to the head's."""
    cfg = get_config()
    d = cfg.drift
    adir = resolve(cfg.artifacts["dir"])
    ver = cfg.artifacts["version"]
    cands = sorted(adir.glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1)
    chosen = (adir / ver) if ver != "auto" else (cands[-1] if cands else None)
    pca_file = "pca_raw.npz" if raw else "pca.npz"
    if chosen and (chosen / "drift_config.json").exists() and (chosen / pca_file).exists():
        dc = json.loads((chosen / "drift_config.json").read_text())
        if raw:
            dc = {**dc, **dc.get("raw", {})}
        pz = np.load(chosen / pca_file)
        k = int(dc.get("k", d["k"]))
        sp = np.asarray(dc["sigma_pop"], dtype=np.float64)
        if sp.ndim == 0:
            sp = np.full(k, float(sp))
        return DriftModel(k, float(dc["tau"]), sp, pz["mean"], pz["components"][:k],
                          version=chosen.name, layer=dc.get("layer"), meta=dc)
    return fallback_model()


def fallback_model() -> DriftModel:
    """Config-default drift space (no PCA). The SIMULATED seed scenarios are generated in this space, so they must
    always be scored with it, whatever trained model is loaded for live recordings."""
    d = get_config().drift
    k = int(d["k"])
    return DriftModel(k, float(d["tau"]), np.full(k, float(d["sigma_pop"])), None, None)


# ----------------------------------------------------------------------------- baseline
def shrunk_sigma(s: np.ndarray, sigma_pop: np.ndarray, n: int, n0: int) -> np.ndarray:
    """sigma_hat^2 = (n s^2 + n0 sigma_pop^2) / (n + n0)."""
    return np.sqrt((n * s ** 2 + n0 * sigma_pop ** 2) / (n + n0))


def fit_baseline(xs: np.ndarray, sigma_pop: np.ndarray, n0: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """xs: (n, k) projected PASS sessions. Returns (mu, sigma_hat). Uses the population (ddof=0) SD s_j."""
    xs = np.atleast_2d(np.asarray(xs, dtype=np.float64))
    n = xs.shape[0]
    mu = xs.mean(axis=0)
    s = xs.std(axis=0, ddof=0)
    return mu, shrunk_sigma(s, sigma_pop, n, n0)


def drift_score(x: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> float:
    """D_t = sqrt(mean_j ((x_j - mu_j)/sigma_j)^2)."""
    z = (np.asarray(x) - mu) / sigma
    return float(np.sqrt(np.mean(z ** 2)))


# ----------------------------------------------------------------------------- timing
def timing_z(cur: dict, mu: dict, sigma: dict) -> dict:
    return {k: (None if cur.get(k) is None or k not in mu else round((cur[k] - mu[k]) / sigma[k], 2))
            for k in ("speech_rate", "pause_rate", "mean_pause")}


def fit_timing_baseline(rows: list[dict], sigma_pop: dict, n0: int = 5) -> tuple[dict, dict]:
    mu, sg = {}, {}
    for k in ("speech_rate", "pause_rate", "mean_pause"):
        vals = np.array([r[k] for r in rows if r.get(k) is not None], dtype=float)
        if len(vals) == 0:
            continue
        mu[k] = float(vals.mean())
        sg[k] = float(shrunk_sigma(np.array([vals.std()]), np.array([sigma_pop[k]]), len(vals), n0)[0])
    return mu, sg


def timing_chips(cur: dict, mu: dict, z: dict) -> list[dict]:
    """Human-readable evidence chips; never part of D_t."""
    labels = {"speech_rate": ("speech", "faster", "slower"),
              "pause_rate": ("pauses", "more frequent", "less frequent"),
              "mean_pause": ("pause length", "longer", "shorter")}
    chips = []
    for k, (name, up, down) in labels.items():
        zz = z.get(k)
        if zz is None or abs(zz) < 1.0 or not mu.get(k):
            continue
        pct = round(abs(cur[k] - mu[k]) / mu[k] * 100)
        chips.append({"key": k, "z": zz, "text": f"{name} {pct}% {up if zz > 0 else down} than baseline"})
    return chips


# ----------------------------------------------------------------------------- alert rule
def confirmed_drift(history: list[tuple[bool, float]], confirm_sum: float = 1.5, last: int = 3) -> bool:
    """history: chronological (exceeds, weight) for SCORED check-ins (PASS/WARN only).
    Fires when the weighted exceedances among the last `last` scored check-ins reach `confirm_sum`."""
    recent = history[-last:]
    return sum(w for ex, w in recent if ex) >= confirm_sum - 1e-9


def reject_streak(verdicts: list[str]) -> int:
    n = 0
    for v in reversed(verdicts):
        if v != "REJECT":
            break
        n += 1
    return n


def decide_status(*, safety: bool, baseline_ready: bool, confirmed: bool, exceeds: bool,
                  dam_high: bool, reject_run: int) -> str:
    if safety:
        return "SAFETY"
    if not baseline_ready:
        return "BASELINE"
    if confirmed:
        return "RED"
    if exceeds or dam_high or reject_run >= 2:
        return "AMBER"
    return "GREEN"

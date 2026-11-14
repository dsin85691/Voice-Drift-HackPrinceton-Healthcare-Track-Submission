"""Export artifacts/v{n}/ (spec 3.6): projector.pt, pca.npz (+ pca_raw.npz), drift_config.json, metrics.json.

Retrain P_M3 on ALL folds with the median-chosen l*, then fit PCA and drift constants on all control speakers.
Total size must stay under 5 MB. Artifacts go to the private store, NOT the public repo.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import torch

from server.pipeline import drift as D

from .evaluate import pseudo_sessions
from .heads import Projector
from .train_heads import Data, inner_split, train_head

K = 16


def fit_pca(V: np.ndarray, k: int = K):
    mu = V.mean(0)
    _, S, Vt = np.linalg.svd(V - mu, full_matrices=False)
    var = (S ** 2) / (S ** 2).sum()
    return mu, Vt[:k], float(var[:k].sum())


def drift_constants(sessions: dict[str, np.ndarray], mu, comp, base_n=5, q=0.9):
    """sigma_pop (pooled within-speaker SD in PCA space) and tau (q-quantile of D_t on clean new sessions)."""
    P = {s: (v - mu) @ comp.T for s, v in sessions.items() if len(v) > base_n}
    if not P:
        raise ValueError(f"no control speaker has more than {base_n} pseudo-sessions (need >= {3 * (base_n + 1)} windows)")
    sigma_pop = np.concatenate([p - p.mean(0) for p in P.values()]).std(0)
    Ds = []
    for p in P.values():
        m, sg = D.fit_baseline(p[:base_n], sigma_pop)
        Ds += [D.drift_score(x, m, sg) for x in p[base_n:]]
    return sigma_pop, float(np.quantile(Ds, q))


def export(d: Data, chosen_layers: dict, out_dir: str | Path, version: int, metrics: dict, wavlm_revision: str,
           seeds=(0, 1, 2), epochs=40, device: str = "cpu"):
    out = Path(out_dir) / f"v{version}"
    out.mkdir(parents=True, exist_ok=True)
    layer = int(np.median([c["layer"] for c in chosen_layers.values()]))
    labels = {s: int(d.y[d.spk == s][0]) for s in np.unique(d.spk)}
    tr_in, va_in = inner_split(np.array(sorted(labels)), labels, seed=123)
    tr_idx, va_idx = np.where(np.isin(d.spk, tr_in))[0], np.where(np.isin(d.spk, va_in))[0]
    # best seed by inner-val AUC (one projector is shipped)
    runs = [train_head(d, "M3", layer, tr_idx, va_idx, seed=s, epochs=epochs, device=device) for s in seeds]
    P, _, _ = max(runs, key=lambda r: np.nan_to_num(r[2], nan=-1))
    torch.save(P.state_dict(), out / "projector.pt")

    with torch.no_grad():
        H = P(torch.from_numpy(d.Zc[:, layer].astype(np.float32))).numpy()
    raw = d.Zc[:, layer].astype(np.float64)
    ctrl = [s for s in labels if labels[s] == 0]
    sess_h = {s: pseudo_sessions(H[d.spk == s]) for s in ctrl}
    sess_r = {s: pseudo_sessions(raw[d.spk == s]) for s in ctrl}
    mu, comp, var = fit_pca(np.concatenate(list(sess_h.values())))
    mu_r, comp_r, var_r = fit_pca(np.concatenate(list(sess_r.values())))
    sp, tau = drift_constants(sess_h, mu, comp)
    sp_r, tau_r = drift_constants(sess_r, mu_r, comp_r)
    np.savez(out / "pca.npz", mean=mu, components=comp)
    np.savez(out / "pca_raw.npz", mean=mu_r, components=comp_r)
    try:
        sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    except Exception:
        sha = "nogit"
    cfg = {"layer": layer, "k": K, "tau": tau, "sigma_pop": sp.tolist(), "pca_variance_retained": var,
           "wavlm": "microsoft/wavlm-base", "wavlm_revision": wavlm_revision, "git_sha": sha,
           "raw": {"tau": tau_r, "sigma_pop": sp_r.tolist(), "pca_variance_retained": var_r}}
    (out / "drift_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "metrics.json").write_text(json.dumps(metrics, indent=1, default=float))
    size = sum(f.stat().st_size for f in out.iterdir())
    assert size < 5 * 1024 * 1024, f"artifacts are {size/1e6:.1f} MB (limit 5 MB)"
    return out, cfg

"""E1 (AUC vs SNR, paired bootstrap), E2 (noise probe), E3 (false drift alerts) - spec 3.5. CPU only."""
from __future__ import annotations

import numpy as np
from .stats import LogisticRegression, roc_auc_score

from server.pipeline import drift as D


# ------------------------------------------------------------------ E1
def _metrics(sc: np.ndarray, lb: np.ndarray) -> dict:
    auc = roc_auc_score(lb, sc) if len(set(lb)) > 1 else np.nan
    pred = sc >= 0.5
    sens, spec = (pred[lb == 1].mean() if (lb == 1).any() else np.nan), (~pred[lb == 0]).mean() if (lb == 0).any() else np.nan
    thr = np.percentile(sc[lb == 0], 80) if (lb == 0).any() else np.nan      # 80% specificity
    return {"auc": auc, "bal_acc": (sens + spec) / 2, "sens_at_80spec": (sc[lb == 1] > thr).mean() if (lb == 1).any() else np.nan}


def _table(rows):
    """rows: list of (fold, speaker, label, score) -> aligned arrays sorted by speaker."""
    rows = sorted(rows, key=lambda r: str(r[1]))
    return np.array([r[1] for r in rows]), np.array([r[2] for r in rows], int), np.array([r[3] for r in rows], float)


def e1_table(res: dict, conds=("clean", 20, 10, 5), B: int = 2000, seed: int = 0, base="L0", cand="M3") -> list[dict]:
    """Speaker-bootstrap 95% CIs of AUC / balanced accuracy / sens@80%spec per model x SNR, plus the PAIRED
    bootstrap CI of (cand - base) using identical resamples."""
    rng = np.random.default_rng(seed)
    out = []
    for cond in conds:
        tabs = {m: _table(res[m][cond]) for m in res if res[m].get(cond)}
        if not tabs:
            continue
        spk0, lb0, _ = next(iter(tabs.values()))
        n = len(spk0)
        draws = rng.integers(0, n, size=(B, n))
        boot = {m: {k: np.full(B, np.nan) for k in ("auc", "bal_acc", "sens_at_80spec")} for m in tabs}
        for b, ix in enumerate(draws):
            if len(set(lb0[ix])) < 2:
                continue
            for m, (_, lb, sc) in tabs.items():
                for k, v in _metrics(sc[ix], lb[ix]).items():
                    boot[m][k][b] = v
        for m, (_, lb, sc) in tabs.items():
            pt = _metrics(sc, lb)
            row = {"model": m, "snr": cond, "n_speakers": n}
            for k in pt:
                lo, hi = np.nanpercentile(boot[m][k], [2.5, 97.5])
                row[k] = round(float(pt[k]), 4); row[k + "_ci"] = [round(float(lo), 4), round(float(hi), 4)]
            out.append(row)
        if base in tabs and cand in tabs:
            for k in ("auc", "bal_acc", "sens_at_80spec"):
                diff = boot[cand][k] - boot[base][k]
                lo, hi = np.nanpercentile(diff, [2.5, 97.5])
                pt = _metrics(tabs[cand][2], tabs[cand][1])[k] - _metrics(tabs[base][2], tabs[base][1])[k]
                out.append({"model": f"{cand}-{base}", "snr": cond, "metric": k, "diff": round(float(pt), 4),
                            "diff_ci": [round(float(lo), 4), round(float(hi), 4)]})
    return out


# ------------------------------------------------------------------ E2
def e2_noise_probe(Z_raw: np.ndarray, Z_proj: np.ndarray, env: np.ndarray, spk: np.ndarray, seed: int = 0) -> dict:
    """Env-class accuracy of a logistic probe on frozen features (6 classes, chance = 1/6), speaker-disjoint split."""
    rng = np.random.default_rng(seed)
    us = np.unique(spk)
    te = set(rng.choice(us, size=max(1, len(us) // 4), replace=False))
    m = np.array([s in te for s in spk])
    out = {"chance": round(1 / len(np.unique(env)), 4)}
    for name, Z in (("raw_z", Z_raw), ("P_M3", Z_proj)):
        mu, sd = Z[~m].mean(0), Z[~m].std(0) + 1e-6
        clf = LogisticRegression(max_iter=1000).fit((Z[~m] - mu) / sd, env[~m])
        out[name] = round(float(clf.score((Z[m] - mu) / sd, env[m])), 4)
    return out


def tsne_plot(Z_raw, Z_proj, env, path, n=2000, seed=0):  # pragma: no cover - plotting
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE

    ix = np.random.default_rng(seed).choice(len(env), min(n, len(env)), replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(10, 4))
    for a, Z, t in zip(ax, (Z_raw, Z_proj), ("raw WavLM z", "invariance head P(z)")):
        E = TSNE(2, random_state=seed, init="pca").fit_transform(Z[ix])
        a.scatter(E[:, 0], E[:, 1], c=env[ix], cmap="tab10", s=4); a.set_title(t)
    fig.savefig(path, dpi=150, bbox_inches="tight")


# ------------------------------------------------------------------ E3
def pseudo_sessions(X: np.ndarray, size: int = 3) -> np.ndarray:
    """Mean of `size` consecutive windows (~24 s). X: [n_windows, d]."""
    n = len(X) // size
    return X[: n * size].reshape(n, size, -1).mean(1)


def e3_false_alerts(clean: dict, noisy: dict, snr: dict, train_spk, test_spk, k=16, base_n=5, quality_gate_snr=12.0,
                    seed=0) -> dict:
    """clean[s]: [n_win, d] features of control speaker s (already in the space under test: raw z or P(z));
    noisy[s]: [n_win, 3, d] test-corrupt copies; snr: the 3 nominal SNRs (e.g. [20, 10, 5]).
    PCA, sigma_pop and tau come from TRAINING controls only. Sessions 1-5 = baseline (clean); 6+ are scored twice:
    clean, and with a random corrupt copy, with and without the quality gate (SNR < gate => REJECT)."""
    rng = np.random.default_rng(seed)
    sess = {s: pseudo_sessions(clean[s]) for s in clean}
    ok = [s for s in train_spk if s in sess and len(sess[s]) > base_n + 1]
    allv = np.concatenate([sess[s] for s in ok])
    mu_pca = allv.mean(0)
    U, S, Vt = np.linalg.svd(allv - mu_pca, full_matrices=False)
    comp = Vt[:k]
    proj = lambda v: (v - mu_pca) @ comp.T
    # sigma_pop: pooled within-speaker SD in PCA space
    within = np.concatenate([proj(sess[s]) - proj(sess[s]).mean(0) for s in ok])
    sigma_pop = within.std(0)
    # tau: 90th percentile of D_t over clean new sessions of training controls
    Dtr = []
    for s in ok:
        P = proj(sess[s]); mu, sg = D.fit_baseline(P[:base_n], sigma_pop)
        Dtr += [D.drift_score(p, mu, sg) for p in P[base_n:]]
    tau = float(np.percentile(Dtr, 90))
    rows = {"clean": [], "noisy": [], "noisy_gated": []}
    alerts = {"clean": [], "noisy": [], "noisy_gated": []}
    for s in test_spk:
        if s not in sess or len(sess[s]) <= base_n:
            continue
        P = proj(sess[s]); mu, sg = D.fit_baseline(P[:base_n], sigma_pop)
        Nz = noisy[s][: len(sess[s]) * 3].reshape(len(sess[s]), 3, 3, -1)  # [session, window, copy, d]
        hist = {"clean": [], "noisy": [], "noisy_gated": []}
        for i in range(base_n, len(P)):
            c = rng.integers(0, 3)
            xn = proj(Nz[i, :, c].mean(0))
            rows["clean"].append(D.drift_score(P[i], mu, sg) > tau)
            rows["noisy"].append(D.drift_score(xn, mu, sg) > tau)
            hist["clean"].append((rows["clean"][-1], 1.0))
            hist["noisy"].append((rows["noisy"][-1], 1.0))
            if snr[c] >= quality_gate_snr:                       # gate: REJECTed sessions are never scored
                rows["noisy_gated"].append(rows["noisy"][-1])
                hist["noisy_gated"].append((rows["noisy"][-1], 1.0))
        for key in alerts:
            alerts[key].append(any(D.confirmed_drift(hist[key][: j + 1]) for j in range(len(hist[key]))))
    return {"tau": tau, "exceed_rate": {k: round(float(np.mean(v)), 4) if v else None for k, v in rows.items()},
            "alert_rate_per_speaker": {k: round(float(np.mean(v)), 4) if v else None for k, v in alerts.items()},
            "n_test_speakers": len(alerts["clean"]),
            "note": "Pseudo-sessions share one day and one room, so these false-alert rates are a lower bound."}


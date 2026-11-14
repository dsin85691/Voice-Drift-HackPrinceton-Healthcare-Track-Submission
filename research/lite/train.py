"""B0 / B1 / P1 for the lightweight projection experiment (projection-head spec v0.1). CPU or GPU, minutes.

  B0  frozen WavLM layer embedding, no projection (baseline)
  B1  projection trained with invariance + variance + covariance
  P1  B1 + acoustic preservation (pause fraction, pitch variability, speech-run rate)

Speakers: the extraction split gives TRAIN and EVAL (held-out) speakers. Early stopping uses an inner validation
split of the TRAIN speakers (composite loss on training-noise views), so evaluation speakers, held-out noise
and the synthetic pause views never influence training or stopping.

Evaluation on held-out speakers (cosine distance; each model judged against ITS OWN normal variation):
  normal variation  distances between different clean windows of the same speaker; tau_m = 90th percentile
  Experiment A      clean vs held-out-noise view of the same window (nuisance-only change)
  Experiment B      clean vs synthetic extra-pause view of the same window (controlled delivery change; SYNTHETIC)
  combined          AUROC(pause pairs vs noise pairs), nuisance false-alert rate (> tau), sensitivity (> tau),
                    sensitivity at a 10% nuisance false-alert rate; speaker-bootstrap 95% CIs
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..heads import AcousticHead, ProjectionHead
from ..stats import roc_auc_score
from . import losses as Lf
from .acoustic import TARGETS
from .extract import VIEWS

WEIGHTS = {"B1": {"inv": 1.0, "var": 1.0, "cov": 0.04, "pres": 0.0},
           "P1": {"inv": 1.0, "var": 1.0, "cov": 0.04, "pres": 0.5}}
OPT = {"lr": 1e-3, "weight_decay": 1e-4, "batch": 128, "epochs": 30, "patience": 5, "seed": 42, "gamma": 1.0}


@dataclass
class LiteData:
    H: np.ndarray          # [n_windows, 5 views, 768] float32 at the chosen layer
    spk: np.ndarray
    split: np.ndarray      # "train" / "val" (val = held-out EVALUATION speakers)
    window_id: np.ndarray
    targets: np.ndarray    # [n, 3] raw acoustic targets (clean view)
    mask: np.ndarray       # [n, 3] bool validity
    layer: int


def load_lite(feat_dir: str | Path, layer: int = 6) -> LiteData:
    import pandas as pd

    feat_dir = Path(feat_dir)
    idx, F = [], []
    for pq in sorted(feat_dir.glob("L1_*.parquet")):
        npy = pq.with_suffix(".npy")
        if npy.exists():
            df = pd.read_parquet(pq)
            f = np.load(npy, mmap_mode="r")
            assert len(df) == len(f), pq.name
            idx.append(df); F.append(np.asarray(f[:, layer, :], dtype=np.float32))
    if not idx:
        raise FileNotFoundError(f"no complete L1 shards in {feat_dir}")
    df = pd.concat(idx, ignore_index=True)
    F = np.concatenate(F)
    pos = {(w, v): i for i, (w, v) in enumerate(zip(df.window_id, df.view))}
    wins = df[df.view == "clean"].drop_duplicates("window_id")
    wins = wins[[all((w, v) in pos for v in VIEWS) for w in wins.window_id]]
    H = np.stack([np.stack([F[pos[(w, v)]] for v in VIEWS]) for w in wins.window_id])
    T = wins[[f"t_{k}" for k in TARGETS]].to_numpy(float)
    M = wins[[f"m_{k}" for k in TARGETS]].to_numpy(bool)
    return LiteData(H, wins.speaker_id.to_numpy(str), wins.split.to_numpy(str), wins.window_id.to_numpy(str), T, M, layer)


def inner_split(spk_train: np.ndarray, frac: float = 0.15, seed: int = 42) -> tuple[set, set]:
    rng = np.random.default_rng(seed)
    us = sorted(set(spk_train))
    rng.shuffle(us)
    k = max(1, round(len(us) * frac))
    return set(us[k:]), set(us[:k])


def assert_disjoint(d: LiteData, tr: set, va: set) -> None:
    ev = set(d.spk[d.split == "val"])
    assert not (tr & va) and not (tr & ev) and not (va & ev), "speaker overlap between train / inner-val / eval"
    # all 5 views of a window share its row, so clean and augmented copies can never cross partitions
    assert len(set(d.window_id)) == len(d.window_id)


# ------------------------------------------------------------------ training
def train_variant(d: LiteData, kind: str, device: str = "cpu", opt: dict = OPT, inv_reduction: str = "mean",
                  log=print) -> dict:
    assert kind in WEIGHTS
    w = WEIGHTS[kind]
    torch.manual_seed(opt["seed"]); rng = np.random.default_rng(opt["seed"])
    tr_spk, va_spk = inner_split(d.spk[d.split == "train"], seed=opt["seed"])
    assert_disjoint(d, tr_spk, va_spk)
    tr = np.where(np.isin(d.spk, list(tr_spk)))[0]
    va = np.where(np.isin(d.spk, list(va_spk)))[0]
    H = torch.from_numpy(d.H).to(device)
    # acoustic target normalization (train speakers, valid entries only) - saved with the checkpoint
    t_mu = np.array([d.targets[tr][d.mask[tr][:, k], k].mean() if d.mask[tr][:, k].any() else 0 for k in range(3)])
    t_sd = np.array([d.targets[tr][d.mask[tr][:, k], k].std() + 1e-6 if d.mask[tr][:, k].any() else 1 for k in range(3)])
    T = torch.from_numpy(((d.targets - t_mu) / t_sd).astype(np.float32)).to(device)
    M = torch.from_numpy(d.mask).to(device)
    head = ProjectionHead()
    head.mean.copy_(torch.from_numpy(d.H[tr, 0].mean(0)))
    head.std.copy_(torch.from_numpy(d.H[tr, 0].std(0) + 1e-6))
    head = head.to(device)
    ahead = AcousticHead().to(device)
    params = list(head.parameters()) + (list(ahead.parameters()) if w["pres"] > 0 else [])
    optim = torch.optim.AdamW(params, lr=opt["lr"], weight_decay=opt["weight_decay"])
    hist, best, best_state, bad, t0 = [], float("inf"), None, 0, time.time()

    def batch_loss(ix, view):
        z, zt = head(H[ix, 0]), head(H[ix, view])
        return Lf.composite(z, zt, w, ahead, T[ix], M[ix], opt["gamma"], inv_reduction)

    for ep in range(opt["epochs"]):
        head.train(); ahead.train()
        perm = rng.permutation(tr)
        acc = {}
        for b in range(0, len(perm), opt["batch"]):
            ix = torch.from_numpy(perm[b:b + opt["batch"]]).to(device)
            view = torch.from_numpy(rng.choice([1, 2], len(ix))).to(device)   # independently generated aug views
            loss, parts = Lf.composite(head(H[ix, 0]), head(H[ix, view]), w, ahead, T[ix], M[ix], opt["gamma"],
                                       inv_reduction)
            assert torch.isfinite(loss), f"non-finite loss at epoch {ep}: {parts}"
            optim.zero_grad(); loss.backward(); optim.step()
            for k, v in parts.items():
                acc[k] = acc.get(k, 0) + v * len(ix) / len(perm)
        head.eval(); ahead.eval()
        with torch.no_grad():
            vix = torch.from_numpy(va).to(device)
            vloss, vparts = batch_loss(vix, 1)
        hist.append({"epoch": ep, "train": acc, "val": vparts, "val_total": float(vloss)})
        log(f"  {kind} ep {ep:2d} | train " + " ".join(f"{k}={v:.4f}" for k, v in acc.items())
            + f" | val total={float(vloss):.4f}")
        if float(vloss) < best - 1e-6:
            best, bad = float(vloss), 0
            best_state = ({k: v.detach().clone() for k, v in head.state_dict().items()},
                          {k: v.detach().clone() for k, v in ahead.state_dict().items()})
        else:
            bad += 1
            if bad >= opt["patience"]:
                break
    head.load_state_dict(best_state[0]); ahead.load_state_dict(best_state[1])
    return {"kind": kind, "head": head.cpu().eval(), "acoustic_head": ahead.cpu().eval(), "history": hist,
            "best_val": best, "epochs_run": len(hist), "runtime_s": round(time.time() - t0, 1),
            "params": sum(p.numel() for p in head.parameters()), "target_norm": {"mean": t_mu.tolist(), "std": t_sd.tolist()},
            "weights": w, "inv_reduction": inv_reduction, "inner_val_speakers": sorted(va_spk)}


# ------------------------------------------------------------------ evaluation
def _cos(A, B):
    A, B = np.asarray(A, float), np.asarray(B, float)
    return 1 - np.sum(A * B, 1) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(B, axis=1), 1e-12)


def embed(d: LiteData, model) -> np.ndarray:
    """[n, 5, dim] embeddings for B0 (model=None: raw WavLM layer) or a trained head."""
    if model is None:
        return d.H.astype(np.float64)
    with torch.no_grad():
        n, v, k = d.H.shape
        return model(torch.from_numpy(d.H.reshape(n * v, k))).numpy().reshape(n, v, -1).astype(np.float64)


def effective_rank(Z: np.ndarray) -> float:
    s = np.linalg.svd(Z - Z.mean(0), compute_uv=False)
    p = s / max(s.sum(), 1e-12)
    p = p[p > 0]
    return float(np.exp(-(p * np.log(p)).sum()))


def _metrics(nv, dn, dp):
    tau = np.quantile(nv, 0.9) if len(nv) else np.nan
    thr10 = np.quantile(dn, 0.9)
    y = np.r_[np.ones(len(dp)), np.zeros(len(dn))]
    return {"auroc_change_vs_nuisance": roc_auc_score(y, np.r_[dp, dn]),
            "nuisance_false_alert_rate": float(np.mean(dn > tau)),
            "change_sensitivity": float(np.mean(dp > tau)),
            "sensitivity_at_10pct_nuisance_fa": float(np.mean(dp > thr10)),
            "noise_dist_rel_normal": float(np.mean(dn) / np.median(nv)),
            "pause_dist_rel_normal": float(np.mean(dp) / np.median(nv))}


def evaluate(d: LiteData, models: dict, B: int = 1000, seed: int = 0) -> dict:
    """models: {"B0": None, "B1": head, "P1": head}. Held-out (val) speakers only."""
    ev = np.where(d.split == "val")[0]
    spk = d.spk[ev]
    us = sorted(set(spk))
    out = {}
    for name, m in models.items():
        E = embed(d, m)[ev]
        per = {}
        for s in us:   # per-speaker pieces so the bootstrap can resample speakers
            k = np.where(spk == s)[0]
            C = E[k, 0]
            iu = np.triu_indices(len(k), 1)
            G = 1 - (C @ C.T) / np.maximum(np.outer(np.linalg.norm(C, axis=1), np.linalg.norm(C, axis=1)), 1e-12)
            per[s] = (G[iu], _cos(E[k, 0], E[k, 3]), _cos(E[k, 0], E[k, 4]))
        cat = lambda ss, j: np.concatenate([per[s][j] for s in ss])
        point = _metrics(cat(us, 0), cat(us, 1), cat(us, 2))
        rng = np.random.default_rng(seed)
        boot = {k: [] for k in point}
        for _ in range(B):
            ss = list(rng.choice(us, len(us), replace=True))
            for k, v in _metrics(cat(ss, 0), cat(ss, 1), cat(ss, 2)).items():
                boot[k].append(v)
        Zc = E[:, 0]
        out[name] = {**{k: round(v, 4) for k, v in point.items()},
                     **{k + "_ci": [round(float(np.percentile(v, 2.5)), 4), round(float(np.percentile(v, 97.5)), 4)]
                        for k, v in boot.items()},
                     "feature_std_mean": round(float(Zc.std(0).mean()), 5), "feature_std_min": round(float(Zc.std(0).min()), 6),
                     "effective_rank": round(effective_rank(Zc), 2), "dim": int(Zc.shape[1])}
    out["_meta"] = {"eval_speakers": len(us), "eval_windows": int(len(ev)),
                    "experiment_B": "SYNTHETIC extra pauses (room tone inserted after each speech run)",
                    "experiment_A": "held-out noise files (noise/babble) + reverb/mic simulation"}
    return out


# ------------------------------------------------------------------ report + export
def collapsed(m: dict) -> bool:
    return m["effective_rank"] < 3 or m["feature_std_mean"] < 1e-3


def report(results: dict, runs: dict, d: LiteData, path: str | Path) -> str:
    rows = ["| Model | AUROC change vs nuisance | Nuisance false-alert rate | Change sensitivity | Sens. @10% nuisance FA | Eff. rank |",
            "|---|---|---|---|---|---|"]
    for k in ("B0", "B1", "P1"):
        m = results[k]
        rows.append(f"| {k} | {m['auroc_change_vs_nuisance']} {m['auroc_change_vs_nuisance_ci']} | "
                    f"{m['nuisance_false_alert_rate']} {m['nuisance_false_alert_rate_ci']} | "
                    f"{m['change_sensitivity']} {m['change_sensitivity_ci']} | {m['sensitivity_at_10pct_nuisance_fa']} | "
                    f"{m['effective_rank']} |")
    b0, b1, p1 = results["B0"], results["B1"], results["P1"]
    q = [
        f"1. **Training completed:** B1 {runs['B1']['epochs_run']} epochs ({runs['B1']['runtime_s']} s), "
        f"P1 {runs['P1']['epochs_run']} epochs ({runs['P1']['runtime_s']} s); losses finite throughout.",
        f"2. **Collapse:** B1 {'YES' if collapsed(b1) else 'no'} (eff. rank {b1['effective_rank']}), "
        f"P1 {'YES' if collapsed(p1) else 'no'} (eff. rank {p1['effective_rank']}).",
        f"3. **More robust to nuisance?** Nuisance false-alert rate B0 {b0['nuisance_false_alert_rate']} vs "
        f"P1 {p1['nuisance_false_alert_rate']} (B1 {b1['nuisance_false_alert_rate']}); compare the CIs before claiming it.",
        f"4. **Sensitivity to the (synthetic) pause change preserved?** B0 {b0['change_sensitivity']} vs "
        f"P1 {p1['change_sensitivity']} (B1 {b1['change_sensitivity']}).",
        f"5. **Did preservation help over invariance-only?** AUROC P1 {p1['auroc_change_vs_nuisance']} vs "
        f"B1 {b1['auroc_change_vs_nuisance']}.",
        "6. **Larger experiment justified?** Only if P1's AUROC CI sits above B0's and nothing collapsed; this is a "
        f"smoke run on {results['_meta']['eval_speakers']} held-out speakers with SYNTHETIC speech changes.",
    ]
    text = "\n".join([
        "# Lightweight projection: smoke-run report", "",
        f"Frozen `microsoft/wavlm-base`, layer {d.layer}; {len(d.H)} windows, {len(set(d.spk))} speakers "
        f"({results['_meta']['eval_speakers']} held out for evaluation). Experiment B is **synthetic** "
        "(inserted pauses); no real speech-change recordings were used. No health inference is made.", "",
        *rows, "", "Brackets: speaker-bootstrap 95% CIs. Thresholds: each model's own 90th percentile of same-speaker "
        "clean-window distances (normal variation).", "", *q, ""])
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(text, encoding="utf-8")
    return text


def plot(runs: dict, results: dict, path: str | Path) -> None:  # pragma: no cover - plotting
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(14, 3.6))
    for k, r in runs.items():
        for part in ("inv", "var", "cov", "pres"):
            ys = [h["train"].get(part) for h in r["history"]]
            if any(y is not None for y in ys):
                ax[0].plot(ys, label=f"{k} {part}")
        ax[1].plot([h["val_total"] for h in r["history"]], label=f"{k} val")
    ax[0].set_yscale("log"); ax[0].set_title("train loss components"); ax[0].legend(fontsize=7)
    ax[1].set_title("inner-val composite loss"); ax[1].legend()
    names = ["B0", "B1", "P1"]
    fa = [results[n]["nuisance_false_alert_rate"] for n in names]
    se = [results[n]["change_sensitivity"] for n in names]
    x = np.arange(3)
    ax[2].bar(x - 0.2, fa, 0.4, label="nuisance false alerts"); ax[2].bar(x + 0.2, se, 0.4, label="pause-change detected")
    ax[2].set_xticks(x, names); ax[2].set_ylim(0, 1); ax[2].legend(fontsize=8); ax[2].set_title("held-out speakers")
    fig.tight_layout(); fig.savefig(path, dpi=140)


def export_for_dashboard(d: LiteData, run: dict, out_dir: str | Path, version: int, results: dict) -> Path:
    """artifacts/vN that the server loads: projection_head.pt + drift constants + population window thresholds."""
    from ..evaluate import pseudo_sessions
    from ..export import K, drift_constants, fit_pca, window_thresholds

    out = Path(out_dir) / f"v{version}"
    out.mkdir(parents=True, exist_ok=True)
    head = run["head"]
    torch.save(head.state_dict(), out / "projection_head.pt")
    with torch.no_grad():
        Z = head(torch.from_numpy(d.H[:, 0])).numpy().astype(np.float64)
    R = d.H[:, 0].astype(np.float64)
    spks = sorted(set(d.spk))
    sess_z = {s: pseudo_sessions(Z[d.spk == s]) for s in spks}
    sess_r = {s: pseudo_sessions(R[d.spk == s]) for s in spks}
    mu, comp, var = fit_pca(np.concatenate(list(sess_z.values())))
    mu_r, comp_r, var_r = fit_pca(np.concatenate(list(sess_r.values())))
    sp, tau = drift_constants(sess_z, mu, comp)
    sp_r, tau_r = drift_constants(sess_r, mu_r, comp_r)
    wz = window_thresholds({s: Z[d.spk == s] for s in spks})
    wr = window_thresholds({s: R[d.spk == s] for s in spks})
    np.savez(out / "pca.npz", mean=mu, components=comp)
    np.savez(out / "pca_raw.npz", mean=mu_r, components=comp_r)
    cfg = {"layer": d.layer, "head": f"lite-{run['kind']}", "k": K, "tau": tau, "sigma_pop": sp.tolist(),
           "pca_variance_retained": var, "wavlm": "microsoft/wavlm-base",
           "raw": {"tau": tau_r, "sigma_pop": sp_r.tolist(), "pca_variance_retained": var_r},
           "window": {"metric": "cosine", "quantile": 0.9, "base_windows": 15, "n_windows": wz["n_windows"],
                      "proj": {"tau": wz["tau"]}, "raw": {"tau": wr["tau"]}},
           "note": "lightweight smoke run: constants fitted on a small speaker set (no diagnosis labels)"}
    (out / "drift_config.json").write_text(json.dumps(cfg, indent=1))
    (out / "metrics.json").write_text(json.dumps({"lite_eval": results, "target_norm": run["target_norm"],
                                                  "weights": run["weights"]}, indent=1, default=float))
    return out


def run_all(feat_dir, out_dir="artifacts", version=1, layer=6, device=None, reports_dir="reports",
            inv_reduction="mean", deploy="P1"):
    """One call for the notebook: load, train B1 + P1, evaluate B0/B1/P1, report, plots, checkpoints, export."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    d = load_lite(feat_dir, layer)
    print(f"{len(d.H)} windows, {len(set(d.spk))} speakers, eval speakers {len(set(d.spk[d.split == 'val']))}, "
          f"finite: {np.isfinite(d.H).all()}, dim {d.H.shape[-1]}")
    assert np.isfinite(d.H).all(), "non-finite embeddings in the cache"
    runs = {k: train_variant(d, k, device, inv_reduction=inv_reduction) for k in ("B1", "P1")}
    ck = Path(reports_dir) / "checkpoints"; ck.mkdir(parents=True, exist_ok=True)
    for k, r in runs.items():
        torch.save({"head": r["head"].state_dict(), "acoustic_head": r["acoustic_head"].state_dict(),
                    "layer": layer, "target_norm": r["target_norm"], "weights": r["weights"]}, ck / f"{k}.pt")
    res = evaluate(d, {"B0": None, "B1": runs["B1"]["head"], "P1": runs["P1"]["head"]})
    (Path(reports_dir) / "lite_metrics.json").write_text(json.dumps(
        {"results": res, "history": {k: r["history"] for k, r in runs.items()},
         "params": {k: r["params"] for k, r in runs.items()}}, indent=1, default=float))
    text = report(res, runs, d, Path(reports_dir) / "lite_report.md")
    try:
        plot(runs, res, Path(reports_dir) / "lite_plots.png")
    except Exception as e:  # matplotlib missing: report still written
        print("plots skipped:", e)
    adir = export_for_dashboard(d, runs[deploy], out_dir, version, res)
    print(text, "\nexported", adir)
    return res, runs, adir

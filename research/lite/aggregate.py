"""Aggregate experiment (evidence spec): raw WavLM (B0) vs the projection heads (P1, B1) at a MATCHED operating point.

Models
  B0  frozen WavLM-base layer, no projection
  B1  projection trained for noise invariance only (invariance + variance + covariance)
  P1  B1 + acoustic preservation (also predicts pause fraction, pitch variability, speech-run rate)

Unit: a session = 3 consecutive 8 s windows (~24 s), scored leave-one-session-out against the mean of the same
speaker's OTHER sessions' clean windows (cosine distance), as the app scores a check-in.

Operating points (both reported; the PRIMARY one is fixed below, before the rerun):
  matched         each evaluation speaker's boundary = 90th percentile of the clean session distances of the OTHER
                  evaluation speakers (leave-one-speaker-out cross-validation), separately for every model, so every
                  model sits at about the same clean alert rate. No speaker's own data sets its own boundary.
  train_boundary  the original rule: 90th percentile of TRAINING speakers' clean distances (kept for transparency;
                  it gave unequal clean rates, which is why it is not the primary comparison).

Threshold-free: AUC for "changed speech vs noise-only" (session distances; changed = synthetic pauses or slower
delivery, noise-only = the same sessions with held-out noise at 10 dB).
Paired differences: P1-B0, P1-B1, B1-B0, bootstrapped over evaluation SPEAKERS with the same resample for both models.

Run (CPU, about a minute):
  python -m research.lite.aggregate --feats lite_feats --checkpoints reports/checkpoints
Needs L1 shards; uses L2 (noise sweep) and L3 (delivery change, change + noise) shards when present.
Writes results/aggregate.json, which the app's Results and Home pages read.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

SESSION = 3
Q = 0.9
PRIMARY = {"methods": ["P1", "B0"], "level": "session", "condition": "noise_10dB", "operating_point": "matched",
           "fixed_on": "2026-11-14",
           "note": "Fixed before the matched-operating-point rerun; not changed after seeing its results."}
MODEL_DESC = {
    "B0": "Raw WavLM-base layer embedding, no projection (baseline).",
    "B1": "Projection trained for noise invariance only (invariance, variance and covariance terms). Simplest head.",
    "P1": "B1 plus acoustic preservation: the head must also predict pause fraction, pitch variability and speech-run "
          "rate, so that real delivery changes are not projected away."}
CHANGE_CONDS = ("pause_change", "delivery_change")
NOISE_REF = "noise_10dB"


# ------------------------------------------------------------------ distances
def _cos(A, mu):
    A = np.atleast_2d(A)
    return 1 - (A @ mu) / np.maximum(np.linalg.norm(A, axis=1) * np.linalg.norm(mu), 1e-12)


def loso_session_distances(E_clean: np.ndarray, E_test: np.ndarray, size: int = SESSION) -> np.ndarray:
    """One speaker. E_*: [n_windows, d] in window order. Session i: reference = mean of the clean windows of all
    OTHER sessions; distance of the session-mean of E_test over session i's windows. NaN rows (missing view) -> NaN."""
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


# ------------------------------------------------------------------ boundaries
def matched_boundaries(clean: dict, q: float = Q) -> dict:
    """{speaker: tau} with tau = q-quantile of the OTHER evaluation speakers' clean distances (leave-one-speaker-out)."""
    out = {}
    for s in clean:
        rest = np.concatenate([v for k, v in clean.items() if k != s and len(v)])
        out[s] = float(np.quantile(rest, q))
    return out


# ------------------------------------------------------------------ statistics
def _auc(pos: np.ndarray, neg: np.ndarray) -> float:
    """Mann-Whitney AUC: P(changed distance > noise-only distance), ties count half."""
    pos, neg = pos[~np.isnan(pos)], neg[~np.isnan(neg)]
    if not len(pos) or not len(neg):
        return float("nan")
    allv = np.r_[pos, neg]
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv)); ranks[order] = np.arange(1, len(allv) + 1)
    for v in np.unique(allv):              # average ranks for ties
        m = allv == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    return float((ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def _rate(dist: dict, tau: dict, spks) -> float:
    h = n = 0
    for s in spks:
        v = dist.get(s)
        if v is None or not len(v):
            continue
        v = v[~np.isnan(v)]
        h += float((v > tau[s]).sum()); n += len(v)
    return h / n if n else float("nan")


def _ci(boot) -> list:
    b = np.asarray(boot, float)
    b = b[~np.isnan(b)]
    if not len(b):
        return [None, None]
    return [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def _r(x, k: int = 4):
    """Round; NaN -> None (json.dumps would write NaN, which browsers' JSON.parse rejects)."""
    return None if x is None or np.isnan(x) else round(float(x), k)


def _n(dist: dict, spks) -> int:
    return int(sum(int((~np.isnan(dist[s])).sum()) for s in spks if s in dist))


def summarize(D: dict, taus: dict, spks: list, B: int, seed: int) -> dict:
    """D[method][cond][spk] -> distances; taus[method][spk] -> boundary. One shared speaker bootstrap for every
    method, condition, AUC and paired difference (so differences are paired)."""
    methods = list(D)
    conds = sorted({c for m in methods for c in D[m]})
    point = {m: {c: _rate(D[m][c], taus[m], spks) for c in D[m]} for m in methods}
    auc_pt = {m: {c: _auc(np.concatenate([D[m][c][s] for s in spks]), np.concatenate([D[m][NOISE_REF][s] for s in spks]))
                  for c in CHANGE_CONDS if c in D[m] and NOISE_REF in D[m]} for m in methods}
    rng = np.random.default_rng(seed)
    boot_rate = {m: {c: [] for c in D[m]} for m in methods}
    boot_auc = {m: {c: [] for c in auc_pt[m]} for m in methods}
    for _ in range(B):
        ss = list(rng.choice(spks, len(spks), replace=True))
        for m in methods:
            for c in D[m]:
                boot_rate[m][c].append(_rate(D[m][c], taus[m], ss))
            for c in auc_pt[m]:
                boot_auc[m][c].append(_auc(np.concatenate([D[m][c][s] for s in ss]),
                                           np.concatenate([D[m][NOISE_REF][s] for s in ss])))
    rates = {m: {c: {"rate": _r(point[m][c]), "ci": _ci(boot_rate[m][c]), "n": _n(D[m][c], spks)}
                 for c in D[m]} for m in methods}
    auc = {m: {c: {"value": _r(auc_pt[m][c]), "ci": _ci(boot_auc[m][c]),
                   "n_changed": _n(D[m][c], spks), "n_noise_only": _n(D[m][NOISE_REF], spks)}
               for c in auc_pt[m]} for m in methods}
    pairs = [p for p in (("P1", "B0"), ("P1", "B1"), ("B1", "B0")) if p[0] in D and p[1] in D]
    diffs = {}
    for a, b in pairs:
        key = f"{a}-{b}"
        diffs[key] = {"rate": {}, "auc": {}}
        for c in conds:
            if c in D[a] and c in D[b]:
                bd = np.array(boot_rate[a][c]) - np.array(boot_rate[b][c])
                ci = _ci(bd)
                diffs[key]["rate"][c] = {"diff": _r(point[a][c] - point[b][c]), "ci": ci,
                                         "excludes_zero": ci[0] is not None and (ci[0] > 0 or ci[1] < 0)}
        for c in auc_pt.get(a, {}):
            if c in auc_pt.get(b, {}):
                bd = np.array(boot_auc[a][c]) - np.array(boot_auc[b][c])
                ci = _ci(bd)
                diffs[key]["auc"][c] = {"diff": _r(auc_pt[a][c] - auc_pt[b][c]), "ci": ci,
                                        "excludes_zero": ci[0] is not None and (ci[0] > 0 or ci[1] < 0)}
    return {"rates": rates, "auc": auc, "paired_differences": diffs}


def sentence(primary_rates: dict, a: str, b: str, change: str) -> str:
    """The section-2 sentence, filled only from computed numbers. Every available speech change is reported, and a
    change detected at about the clean (unchanged) rate is said to be undetected rather than presented as detection."""
    p = lambda r: f"{round(r['rate'] * 100)}%" if r and r.get("rate") is not None else "n/a"
    ra, rb = primary_rates[a], primary_rates[b]
    clean = f"{p(rb['clean'])} ({b}) and {p(ra['clean'])} ({a})"
    names = {"pause_change": "synthetic pauses", "delivery_change": "20% slower delivery"}
    parts = []
    for c in ("pause_change", "delivery_change"):
        if c not in ra or c not in rb:
            continue
        txt = f"{names[c]} were detected in {p(rb[c])} versus {p(ra[c])}" if c == "pause_change" else \
              f"{names[c]} in {p(rb[c])} versus {p(ra[c])}"
        near = all(r[c]["rate"] is not None and r["clean"]["rate"] is not None and r[c]["rate"] - r["clean"]["rate"] <= 0.10
                   for r in (ra, rb))
        if near:
            txt += " (about the same rate as on unchanged speech, so neither model detects it)"
        parts.append(txt)
    changes = " and ".join(parts) if parts else f"a controlled change was detected in {p(rb.get(change))} versus {p(ra.get(change))}"
    return (f"At a matched clean-speech alert rate ({clean}, target 10%), added noise at 10 dB triggered alerts in "
            f"{p(rb.get(NOISE_REF))} of unchanged sessions with raw WavLM and {p(ra.get(NOISE_REF))} with the projection; "
            f"{changes}.")


# ------------------------------------------------------------------ feature loading (needs torch / pandas)
def load_views(feat_dir, prefix: str, layer: int) -> dict:
    """{(window_id, view): vec} from <prefix>_*.parquet/.npy shards. Empty if that stage wasn't run."""
    import pandas as pd

    out = {}
    for pq in sorted(Path(feat_dir).glob(f"{prefix}_*.parquet")):
        npy = pq.with_suffix(".npy")
        if npy.exists():
            df = pd.read_parquet(pq)
            f = np.load(npy, mmap_mode="r")
            for i, (w, v) in enumerate(zip(df.window_id, df.view)):
                out[(w, v)] = np.asarray(f[i, layer], dtype=np.float32)
    return out


def embedders(ckpt_dir) -> dict:
    import torch

    from ..heads import ProjectionHead

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
        return np.asarray(X, np.float64)
    import torch

    with torch.no_grad():
        return model(torch.from_numpy(np.asarray(X, np.float32))).numpy().astype(np.float64)


EXTRA = {  # condition -> (stage prefix, view name)
    "noise_20dB": ("L2", "noise20"), "noise_10dB": ("L2", "noise10"), "noise_5dB": ("L2", "noise5"),
    "noise_0dB": ("L2", "noise0"),
    "delivery_change": ("L3", "stretch"), "change_plus_noise": ("L3", "pause_noise10"),
    "delivery_plus_noise": ("L3", "stretch_noise10")}
CONDITION_TEXT = {
    "clean": "Unchanged clean speech (should alert about 10% by design)",
    "noise_mix": "Held-out noise or babble at 10-30 dB, sometimes with reverb or mic coloring",
    "noise_20dB": "Held-out noise or babble added at 20 dB SNR", "noise_10dB": "Held-out noise or babble added at 10 dB SNR",
    "noise_5dB": "Held-out noise or babble added at 5 dB SNR", "noise_0dB": "Held-out noise or babble added at 0 dB SNR",
    "pause_change": "SYNTHETIC change: room-tone pauses inserted after each phrase (should alert)",
    "delivery_change": "SYNTHETIC change: 20% slower delivery, pitch unchanged (should alert)",
    "change_plus_noise": "Inserted pauses AND held-out noise at 10 dB (should alert)",
    "delivery_plus_noise": "Slower delivery AND held-out noise at 10 dB (should alert)"}
SHOULD_ALERT = {"pause_change", "delivery_change", "change_plus_noise", "delivery_plus_noise"}


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5,
                              cwd=Path(__file__).resolve().parents[2]).stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def run(feat_dir="lite_feats", ckpt_dir="reports/checkpoints", layer: int = 6, out_path="results/aggregate.json",
        B: int = 2000, seed: int = 0, q: float = Q, target_rate: float | None = None) -> dict:
    """target_rate: target clean false-alert rate (boundary = its (1 - target) quantile). Default 10%: with ~6 clean
    sessions per held-out speaker, 5% would be decided by one or two sessions per fold."""
    from .train import load_lite

    if target_rate is not None:
        q = 1.0 - float(target_rate)

    d = load_lite(feat_dir, layer)
    extra = {**load_views(feat_dir, "L2", layer), **load_views(feat_dir, "L3", layer)}
    models = embedders(ckpt_dir)
    train_spk = sorted(set(d.spk[d.split == "train"]))
    eval_spk = sorted(set(d.spk[d.split == "val"]))
    idx = {s: np.where(d.spk == s)[0] for s in set(d.spk)}
    ev = np.where(d.split == "val")[0]
    dist = {"session": {}, "window": {}}
    tau_train = {}
    for name, m in models.items():
        E = {c: _embed(m, d.H[:, i]) for c, i in {"clean": 0, "noise_mix": 3, "pause_change": 4}.items()}
        for c, (prefix, view) in EXTRA.items():
            keys = [(d.window_id[i], view) for i in ev]
            if all(k in extra for k in keys):
                arr = np.full_like(E["clean"], np.nan)
                arr[ev] = _embed(m, np.stack([extra[k] for k in keys]))
                E[c] = arr
        tr_s = np.concatenate([loso_session_distances(E["clean"][idx[s]], E["clean"][idx[s]]) for s in train_spk])
        tr_w = np.concatenate([loso_window_distances(E["clean"][idx[s]], E["clean"][idx[s]]) for s in train_spk])
        tau_train[name] = {"session": float(np.quantile(tr_s, q)), "window": float(np.quantile(tr_w, q))}
        for lvl, fn in (("session", loso_session_distances), ("window", loso_window_distances)):
            dist[lvl][name] = {c: {s: fn(E["clean"][idx[s]], Ec[idx[s]]) for s in eval_spk} for c, Ec in E.items()}

    out_levels, matched, trainb = {}, {}, {}
    for lvl in ("session", "window"):
        D = dist[lvl]
        t_match = {m: matched_boundaries(D[m]["clean"], q) for m in D}
        t_train = {m: {s: tau_train[m][lvl] for s in eval_spk} for m in D}
        out_levels[lvl] = {"matched": summarize(D, t_match, eval_spk, B, seed),
                           "train_boundary": summarize(D, t_train, eval_spk, B, seed)}
        for m in D:
            cr = out_levels[lvl]["matched"]["rates"][m]["clean"]
            matched.setdefault(m, {})[lvl] = {
                "boundary_by_speaker": {s: round(t, 5) for s, t in t_match[m].items()},
                "boundary_mean": round(float(np.mean(list(t_match[m].values()))), 5),
                "clean_alert_rate": cr,
                "within_tolerance": cr["rate"] is not None and bool(abs(cr["rate"] - (1 - q)) <= 0.05 + 1e-9)}
            trainb.setdefault(m, {})[lvl] = {"boundary": round(tau_train[m][lvl], 5),
                                             "clean_alert_rate": out_levels[lvl]["train_boundary"]["rates"][m]["clean"]}

    S = out_levels["session"]["matched"]
    a, b = PRIMARY["methods"]
    proj = a if a in models else ("B1" if "B1" in models else None)
    change = "delivery_change" if "delivery_change" in S["rates"].get("B0", {}) else "pause_change"
    head = None
    if proj and NOISE_REF in S["rates"][b]:
        pd_ = S["paired_differences"].get(f"{proj}-{b}", {"rate": {}, "auc": {}})
        head = {"level": "session (3 windows, ~24 s)", "operating_point": "matched", "projection_model": proj,
                "noise_condition": "held-out noise or babble added at 10 dB SNR", "change_key": change,
                "change_condition": CONDITION_TEXT[change],
                "sentence": sentence(S["rates"], proj, b, change),
                "false_alert_raw": S["rates"][b][NOISE_REF], "false_alert_projection": S["rates"][proj][NOISE_REF],
                "change_sensitivity_raw": S["rates"][b][change], "change_sensitivity_projection": S["rates"][proj][change],
                "pause_sensitivity_raw": S["rates"][b]["pause_change"], "pause_sensitivity_projection": S["rates"][proj]["pause_change"],
                "clean_false_alert_raw": S["rates"][b]["clean"], "clean_false_alert_projection": S["rates"][proj]["clean"],
                "primary_difference": pd_["rate"].get(NOISE_REF), "change_difference": pd_["rate"].get(change),
                "auc_raw": S["auc"].get(b, {}).get(change), "auc_projection": S["auc"].get(proj, {}).get(change)}
    if head and not all(matched.get(m, {}).get("session", {}).get("within_tolerance") for m in (proj, b)):
        # honest wording: the procedure is fixed in advance; if a model lands outside the tolerance, say so
        head["sentence"] = head["sentence"].replace("At a matched", "At an approximately matched", 1)
    sweep_levels = [c for c in ("noise_20dB", "noise_10dB", "noise_5dB", "noise_0dB") if c in S["rates"].get("B0", {})]
    res = {"generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "illustrative": False, "schema": 2,
           "primary_comparison": PRIMARY, "headline": head,
           "matched_operating_point": {"procedure": "Per evaluation speaker, the boundary is the 90th percentile of the "
                                                    "OTHER evaluation speakers' clean session distances "
                                                    "(leave-one-speaker-out), computed separately for each model.",
                                       "target_clean_rate": 0.10, "tolerance": 0.05, "by_method": matched},
           "train_boundary": {"procedure": "90th percentile of training speakers' clean distances (original rule).",
                              "by_method": trainb},
           "results": {lvl: {op: out_levels[lvl][op]["rates"] for op in ("matched", "train_boundary")}
                       for lvl in out_levels},
           "auc": S["auc"], "paired_differences": S["paired_differences"],
           "window_level": {"auc": out_levels["window"]["matched"]["auc"],
                            "paired_differences": out_levels["window"]["matched"]["paired_differences"]},
           "noise_sweep": {"levels_db": [int(c.split("_")[1][:-2]) for c in sweep_levels],
                           "by_method": {m: [S["rates"][m][c] for c in sweep_levels] for m in S["rates"]}},
           "conditions": {c: CONDITION_TEXT[c] for c in S["rates"]["B0"]},
           "should_alert": sorted(SHOULD_ALERT & set(S["rates"]["B0"])),
           "models": {m: MODEL_DESC[m] for m in models},
           "meta": {"wavlm_layer": layer, "eval_speakers": len(eval_spk), "train_speakers": len(train_spk),
                    "session_windows": SESSION, "boundary_quantile": q, "distance": "cosine",
                    "bootstrap": f"{B} resamples over evaluation speakers, shared across models (paired)",
                    "models": list(models)},
           "limitations": [
               f"Small: {len(eval_spk)} held-out speakers, all from one Italian interview corpus (Androids).",
               ("Both speech changes are synthetic (inserted pauses; 20% time-stretch), not recordings of people "
                "actually speaking more slowly or flatly." if "delivery_change" in S["rates"].get("B0", {}) else
                "The speech change is synthetic (inserted pauses), not recordings of people actually speaking more "
                "slowly or flatly. The slower-delivery condition was not available for this run."),
               "Sessions are pseudo-sessions cut from one interview: same day, same room, same microphone. Real "
               "day-to-day variation is larger, so real false-alert rates will be higher than these.",
               "The matched boundary is set from the other evaluation speakers' clean audio (cross-validated), not "
               "from each person's own baseline as in the app.",
               "Speech-change sensitivity says nothing about detecting any health condition."]}
    res["matched_operating_point"]["target_clean_rate"] = round(1 - q, 4)
    SR_ = S["rates"]
    if all("delivery_change" in SR_.get(m, {}) for m in ("B0", proj or "P1")) and all(
            SR_[m]["delivery_change"]["rate"] - SR_[m]["clean"]["rate"] <= 0.10 for m in ("B0", proj or "P1")):
        res["limitations"].insert(2, "Neither the raw nor the projected embedding detected 20% slower delivery: alert "
                                     "rates matched those on unchanged speech. Averaging each 8 s window removes most "
                                     "speaking-rate information, so the overall voice-pattern score should not be "
                                     "described as detecting slowed speech; speaking rate is reported separately.")
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    # VD-07 traceable artifacts, written together with ONE timestamp so nothing stale can pass as new
    ts = res["generated_at"]
    n_clean = min((v["session"]["clean_alert_rate"]["n"] for v in matched.values()), default=0)
    per_spk = n_clean / max(1, len(eval_spk))
    cal = {"generated_at": ts, "target_clean_false_alert_rate": round(1 - q, 4), "level": "session",
           "method": res["matched_operating_point"]["procedure"],
           "frozen_before_test": "each speaker's boundary is fixed from OTHER speakers' clean sessions before any "
                                 "of that speaker's noisy or changed sessions are scored",
           "sample_size_note": f"{n_clean} clean sessions across {len(eval_spk)} speakers (about {per_spk:.0f} each): "
                               f"one session is about {100 / max(per_spk, 1):.0f} points of a speaker's rate, so the "
                               f"target rate can only be approximated.",
           "methods": {m: {"thresholds_by_speaker": v["session"]["boundary_by_speaker"],
                           "threshold_mean": v["session"]["boundary_mean"],
                           "observed_clean_false_alert_rate": v["session"]["clean_alert_rate"],
                           "within_tolerance": v["session"]["within_tolerance"],
                           "train_speaker_threshold": trainb[m]["session"]["boundary"]}
                       for m, v in matched.items()}}
    metrics = {"generated_at": ts, "operating_point": "matched", "rates": res["results"]["session"]["matched"],
               "auc": res["auc"], "paired_differences": res["paired_differences"],
               "noise_sweep": res["noise_sweep"], "unmatched_reference": res["results"]["session"]["train_boundary"]}
    meta = {"generated_at": ts, "source_commit": _git_commit(), "feature_dir": str(feat_dir),
            "checkpoints": {m: str(Path(ckpt_dir) / f"{m}.pt") for m in models if m != "B0"},
            "encoder": "microsoft/wavlm-base (frozen)", "wavlm_layer": layer,
            "dataset": "Androids corpus, interview task (lite selection)",
            "seeds": {"bootstrap": seed, "window_selection": 42, "training": 42},
            "splits": {"train_speakers": train_spk, "eval_speakers": eval_spk,
                       "rule": "speaker-disjoint, fixed by research.lite.extract.select_windows(seed=42)"},
            "calibration_method": "leave-one-evaluation-speaker-out quantile of clean session distances",
            "target_clean_false_alert_rate": round(1 - q, 4), "bootstrap_resamples": B,
            "conditions": list(res["conditions"]), "primary_comparison": PRIMARY}
    for name, obj in (("calibration.json", cal), ("evaluation_metrics.json", metrics),
                      ("evaluation_metadata.json", meta)):
        (out.parent / name).write_text(json.dumps(obj, indent=1))
    return res


def main() -> None:  # pragma: no cover
    ap = argparse.ArgumentParser()
    ap.add_argument("--feats", default="lite_feats")
    ap.add_argument("--checkpoints", default="reports/checkpoints")
    ap.add_argument("--layer", type=int, default=6)
    ap.add_argument("--out", default="results/aggregate.json")
    ap.add_argument("--target-rate", type=float, default=0.10, help="target clean false-alert rate")
    a = ap.parse_args()
    o = run(a.feats, a.checkpoints, a.layer, a.out, target_rate=a.target_rate)
    f = lambda r: f"{r['rate']:.0%} [{r['ci'][0]:.0%}-{r['ci'][1]:.0%}]" if r and r.get("ci", [None])[0] is not None else "n/a"
    for m, v in o["matched_operating_point"]["by_method"].items():
        s = v["session"]
        print(f"{m}: matched clean rate {f(s['clean_alert_rate'])}  within 10±5: {s['within_tolerance']}")
    if o["headline"]:
        h = o["headline"]
        print(h["sentence"])
        pdiff = h["primary_difference"]
        if pdiff:
            print(f"PRIMARY {h['projection_model']}-B0 noise 10 dB: {pdiff['diff']:+.0%} "
                  f"[{pdiff['ci'][0]:+.0%}, {pdiff['ci'][1]:+.0%}]  excludes zero: {pdiff['excludes_zero']}")
    for m, cs in o["auc"].items():
        for c, r in cs.items():
            print(f"AUC {m} {c} vs noise-only: {r['value']:.2f} [{r['ci'][0]:.2f}-{r['ci'][1]:.2f}]")
    print("wrote", a.out)


if __name__ == "__main__":
    main()

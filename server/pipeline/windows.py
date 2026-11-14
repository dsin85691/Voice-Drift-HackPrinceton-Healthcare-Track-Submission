"""Window-level raw-vs-projected drift for ONE recording (Voice Drift visualization spec v0.1).

For each 8 s window t (the same windows the session vector is built from):
    D_raw(t)  = cosine_distance(h_t, mu_raw)       h_t = raw encoder vector of window t
    D_proj(t) = cosine_distance(z_t, mu_proj)      z_t = projected / invariant vector of window t
    R(t)      = D(t) / tau_model                   each model normalized by its OWN window-level threshold
so both curves share one fixed threshold line at R = 1.

- mu_* = mean of the patient's baseline WINDOW vectors (first 5 PASS check-ins), so the reference is built from
  the same window-level features it is compared with.
- tau_* = window-level thresholds, never the session tau. Preferred source: the patient's own baseline,
  leave-one-session-out (each baseline session's windows vs the mean of the OTHER baseline sessions; 90th
  percentile). Fallback: population window thresholds exported with the artifacts (Androids controls).
  Both come only from quiet baseline / normal-variation data, never from the noise events being shown.
- The analysis is computed once per check-in and stored, so its thresholds stay frozen for that recording.
- It never changes the longitudinal (session) drift, the status, or alerts.
"""
from __future__ import annotations

import hashlib

import numpy as np

from . import quality as Q

SR = 16000
FRAME = 480          # VAD frame (30 ms)
ENV_HZ = 10
STATES = {
    "A": "Personal baseline not ready.",
    "B": "Window thresholds not calibrated yet: showing uncalibrated distances, no threshold.",
    "C": "Calibrated: both models normalized to their own window threshold (fixed line at 1).",
    "D": "Recording quality insufficient for reliable drift analysis.",
    "E": "Partially valid recording: invalid windows are shown as gaps.",
}


# ------------------------------------------------------------------ primitives
def _sig(v: float, n: int = 6) -> float:
    """Round to n significant figures (cosine distances can be ~1e-4; fixed decimals would lose them)."""
    return float(f"{float(v):.{n}g}")



def cosine_distance(A: np.ndarray, mu: np.ndarray) -> np.ndarray:
    A = np.atleast_2d(np.asarray(A, dtype=np.float64))
    mu = np.asarray(mu, dtype=np.float64)
    den = np.linalg.norm(A, axis=1) * np.linalg.norm(mu)
    return 1.0 - (A @ mu) / np.maximum(den, 1e-12)


def envelope(x: np.ndarray, mask: np.ndarray | None, hz: int = ENV_HZ) -> dict:
    """Coarse amplitude envelope (dBFS RMS) + speech flags at `hz`. ~10 numbers/s: cannot be turned back into audio."""
    step = SR // hz
    n = len(x) // step
    fr = x[: n * step].reshape(n, step) if n else np.zeros((0, step))
    db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + 1e-9) if n else np.zeros(0)
    speech = []
    if mask is not None and len(mask):
        for i in range(n):
            a, b = (i * step) // FRAME, ((i + 1) * step) // FRAME + 1
            speech.append(bool(mask[a:b].any()))
    return {"hz": hz, "db": [round(float(v), 1) for v in np.maximum(db, -90)], "speech": speech}


def local_mask(x: np.ndarray, a: int, b: int) -> np.ndarray:
    """VAD run on the window ALONE. A whole-recording VAD sets one noise floor for the entire clip, so noise
    added at 30 s would change which frames count as speech at 5 s; then windows that never saw the noise
    would change too. Window-local VAD keeps each window's result independent of the others."""
    return Q.vad_mask(x[a:b])


def window_quality(x: np.ndarray, mask: np.ndarray | None, a: int, b: int, wcfg: dict) -> dict:
    """Per-window quality from a window-local VAD. The SNR is a LOCAL estimate. (`mask` is unused; kept for API.)"""
    m = local_mask(x, a, b)
    speech_s = float(m.sum() * FRAME / SR)
    snr = float(Q.estimate_snr_db(x[a:b], m)) if m.sum() else None
    reasons = []
    if speech_s < wcfg["reject_speech_s"]:
        reasons.append("too little speech in window")
    if snr is not None and snr < wcfg["reject_snr_db"]:
        reasons.append("speech buried in noise")
    verdict = "REJECT" if reasons else ("WARN" if snr is not None and snr < wcfg["warn_snr_db"] else "PASS")
    return {"quality": verdict, "speech_seconds": round(speech_s, 2),
            "snr_estimate_db": None if snr is None else round(snr, 1), "reasons": reasons}


# ------------------------------------------------------------------ calibration
def calibrate_patient(sessions_proj: list[np.ndarray], sessions_raw: list[np.ndarray], q: float = 0.9,
                      min_sessions: int = 3, min_windows: int = 10) -> dict | None:
    """Leave-one-session-out window thresholds from the patient's baseline windows (quiet, normal variation)."""
    if len(sessions_proj) < min_sessions or sum(len(w) for w in sessions_proj) < min_windows:
        return None
    out = {}
    for name, S in (("proj", sessions_proj), ("raw", sessions_raw)):
        d = []
        for i in range(len(S)):
            others = np.concatenate([S[j] for j in range(len(S)) if j != i])
            d.extend(cosine_distance(S[i], others.mean(0)).tolist())
        tau = float(np.quantile(d, q))
        if not np.isfinite(tau) or tau <= 0:
            return None
        out[name] = tau
    return {"tau_proj": out["proj"], "tau_raw": out["raw"], "quantile": q, "metric": "cosine",
            "source": "patient-baseline (leave-one-session-out)", "n_windows": int(sum(len(w) for w in sessions_proj))}


def population_calibration(meta: dict | None) -> dict | None:
    w = (meta or {}).get("window")
    if not w or w.get("proj", {}).get("tau", 0) <= 0 or w.get("raw", {}).get("tau", 0) <= 0:
        return None
    return {"tau_proj": float(w["proj"]["tau"]), "tau_raw": float(w["raw"]["tau"]), "quantile": w.get("quantile", 0.9),
            "metric": w.get("metric", "cosine"), "source": "population (Androids controls, exported)",
            "n_windows": w.get("n_windows")}


# ------------------------------------------------------------------ the analysis
def clean_interventions(items, duration: float) -> list[dict]:
    out = []
    for it in items or []:
        try:
            s, e = float(it["start_seconds"]), float(it.get("end_seconds", duration))
        except (KeyError, TypeError, ValueError):
            continue
        s, e = max(0.0, s), min(duration, e)
        if e > s:
            iv = {"type": str(it.get("type", "controlled_background_noise"))[:40],
                  "start_seconds": round(s, 2), "end_seconds": round(e, 2),
                  "source": str(it.get("source", "marker"))[:20]}
            try:
                if it.get("snr_db") is not None:
                    iv["snr_db"] = round(float(it["snr_db"]), 1)
            except (TypeError, ValueError):
                pass
            out.append(iv)
    return out


def analyze(*, x: np.ndarray, quality, emb, baseline: list, encoder: str, wcfg: dict, interventions=None,
            population_meta: dict | None = None, baseline_n: int = 5) -> dict:
    """baseline: the patient's first-5-PASS check-ins (store.Checkin rows), oldest first.
    emb: acoustic.Embedding for this recording (None if it was not embedded, e.g. REJECT)."""
    from ..store import blob_to_mat

    from research.extract import window_bounds

    dur = round(len(x) / SR, 2)
    mask = quality.speech_mask
    bounds = window_bounds(len(x))
    wins = []
    for a, b in bounds:
        wq = window_quality(x, mask, a, b, wcfg)
        wins.append({"start_seconds": round(a / SR, 2), "end_seconds": round(b / SR, 2), **wq,
                     "raw_distance": None, "projected_distance": None, "raw_ratio": None, "projected_ratio": None})
    # speech shading from the same window-local VAD as the scores (so it can't shift when other windows change)
    local = np.zeros(len(x) // FRAME + 1, bool)
    for a, b in bounds:
        m = local_mask(x, a, b)
        local[a // FRAME: a // FRAME + len(m)] = m
    out = {"duration_seconds": dur, "window_seconds": 8, "metric": "cosine", "encoder": encoder,
           "interventions": clean_interventions(interventions, dur), "envelope": envelope(x, local),
           "recording_quality": quality.verdict, "recording_reasons": list(quality.reasons),
           "baseline_ready": False, "baseline_progress": [0, baseline_n], "window_calibration_ready": False,
           "raw_threshold": None, "projected_threshold": None, "calibration": None, "windows": wins}

    usable = [c for c in baseline if c.window_vecs is not None and c.window_vecs_raw is not None
              and (c.versions or {}).get("encoder") == encoder]
    out["baseline_progress"] = [len(usable), baseline_n]
    # D = nothing usable: the recording wasn't embedded (live check-ins are not scored when the whole recording
    # is rejected) or no single window passed. Otherwise windows decide: valid ones are scored, invalid are gaps.
    if emb is None or emb.W is None or all(w["quality"] == "REJECT" for w in wins):
        return _finish(out, "D")
    if len(usable) < baseline_n:
        out["baseline_note"] = ("baseline recorded with a different encoder or before window analysis existed: "
                                "re-record the baseline" if len(baseline) >= baseline_n else "")
        return _finish(out, "A")
    usable = usable[:baseline_n]
    P, R = [], []
    for c in usable:                      # drop baseline windows that failed their own window quality gate
        p, r = blob_to_mat(c.window_vecs), blob_to_mat(c.window_vecs_raw)
        qs = [w.get("quality") for w in ((c.extra or {}).get("window_analysis") or {}).get("windows", [])]
        if len(qs) == len(p):
            keep = np.array([q != "REJECT" for q in qs])
            p, r = p[keep], r[keep]
        if len(p):
            P.append(p); R.append(r)
    if not P:
        out["baseline_note"] = "no usable baseline windows"
        return _finish(out, "A")
    if P[0].shape[1] != emb.W.shape[1] or R[0].shape[1] != emb.W_raw.shape[1]:
        out["baseline_note"] = "feature dimensions changed since the baseline: re-record the baseline"
        return _finish(out, "A")
    out["baseline_ready"] = True
    mu_p, mu_r = np.concatenate(P).mean(0), np.concatenate(R).mean(0)
    dp, dr = cosine_distance(emb.W, mu_p), cosine_distance(emb.W_raw, mu_r)

    cal = calibrate_patient(P, R) or population_calibration(population_meta)
    if cal:
        cal["version"] = hashlib.sha1(("|".join(c.id for c in usable) + encoder + cal["source"]).encode()).hexdigest()[:10]
        cal["baseline_checkins"] = [c.id for c in usable]
        out.update(window_calibration_ready=True, raw_threshold=_sig(cal["tau_raw"]),
                   projected_threshold=_sig(cal["tau_proj"]), calibration=cal)
    for w, a, b in zip(wins, dr, dp):
        if w["quality"] == "REJECT":
            continue                                   # null, never zero
        w["raw_distance"], w["projected_distance"] = _sig(a), _sig(b)
        if cal:
            w["raw_ratio"] = round(float(a) / cal["tau_raw"], 4)
            w["projected_ratio"] = round(float(b) / cal["tau_proj"], 4)
    if not cal:
        return _finish(out, "B")
    return _finish(out, "E" if any(w["quality"] == "REJECT" for w in wins) else "C")


def _finish(out: dict, state: str) -> dict:
    out["state"] = state
    out["state_message"] = STATES[state]
    valid = [w for w in out["windows"] if w["raw_ratio"] is not None]
    raw_only = sum(1 for w in valid if w["raw_ratio"] > 1 >= w["projected_ratio"])
    proj_only = sum(1 for w in valid if w["projected_ratio"] > 1 >= w["raw_ratio"])
    out["summary"] = {
        "valid_windows": sum(1 for w in out["windows"] if w["quality"] != "REJECT"),
        "total_windows": len(out["windows"]),
        "max_raw_ratio": max((w["raw_ratio"] for w in valid), default=None),
        "max_projected_ratio": max((w["projected_ratio"] for w in valid), default=None),
        "raw_only_crossings": raw_only if valid else None,
        "projected_only_crossings": proj_only if valid else None,
        "both_crossings": sum(1 for w in valid if w["raw_ratio"] > 1 and w["projected_ratio"] > 1) if valid else None,
    }
    return out

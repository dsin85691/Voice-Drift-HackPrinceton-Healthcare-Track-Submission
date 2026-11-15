"""Stateful scoring shared by the live pipeline and the seeded scenarios (one code path, spec 6)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
from sqlmodel import Session, select

from . import protocol as P
from .config import get_config
from .pipeline import drift as D
from .pipeline import evidence as EV
from .store import Alert, Baseline, Checkin, blob_to_vec, new_id, utcnow, vec_to_blob

TIMING_KEYS = {"speech_rate": "speech_rate_wpm", "pause_rate": "pause_rate_pm", "mean_pause": "mean_pause_s"}
DISCLAIMER = "Screening and monitoring aid — not a diagnosis."
SAFETY_MESSAGE = ("If you are thinking about harming yourself, you are not alone. "
                  "Call or text 988 (Suicide & Crisis Lifeline), any time. Please also contact your clinician today.")


def history(s: Session, pid: str, before: datetime | None = None) -> list[Checkin]:
    q = select(Checkin).where(Checkin.patient_id == pid).order_by(Checkin.created_at, Checkin.id)
    rows = list(s.exec(q))
    return [r for r in rows if before is None or r.created_at < before]


def _timing_view(t: dict | None) -> dict:
    t = t or {}
    return {k: t.get(v) for k, v in TIMING_KEYS.items()}


def baseline_rows(prev: list[Checkin], n: int) -> list[Checkin]:
    """The first n PASS check-ins that carry a session vector (safety check-ins never count)."""
    ok = [c for c in prev if c.quality.get("verdict") == "PASS" and c.session_vec and c.status != "SAFETY"]
    return ok[:n]


def apply_scoring(s: Session, ck: Checkin, *, vec: np.ndarray | None, vec_raw: np.ndarray | None,
                  timing: dict | None, safety: dict, dm: D.DriftModel) -> list[Alert]:
    """Fill status/drift/timing on `ck` (already holding quality, reference_risk...) and create alerts.
    `ck` must not be in the DB history yet, or must be excluded by created_at ordering."""
    cfg = get_config().drift
    verdict = ck.quality.get("verdict", "REJECT")
    prev = [c for c in history(s, ck.patient_id) if c.id != ck.id and c.created_at <= ck.created_at]
    alerts: list[Alert] = []
    ck.extra = dict(ck.extra or {})
    ck.extra["safety"] = safety

    if safety["triggered"]:
        ck.status = "SAFETY"
        ck.drift = None
        ck.reference_risk = None
        ck.extra.pop("cohort_risk", None)      # no scores of any kind are stored for SAFETY check-ins
        ck.extra.pop("window_analysis", None)
        ck.window_vecs = ck.window_vecs_raw = None
        ck.timing = None
        ck.session_vec = ck.session_vec_raw = None
        alerts.append(Alert(id=new_id("a"), patient_id=ck.patient_id, checkin_id=ck.id, kind="SAFETY"))
        for a in alerts:
            s.add(a)
        return alerts

    for v in (vec, vec_raw):     # zero-norm or non-finite embeddings would give NaN cosine/z scores: refuse them
        if v is not None and verdict != "REJECT" and (not np.isfinite(v).all() or float(np.linalg.norm(v)) == 0.0):
            raise ValueError("invalid embedding (zero norm or non-finite)")
    ck.session_vec = vec_to_blob(vec) if verdict != "REJECT" else None
    ck.session_vec_raw = vec_to_blob(vec_raw) if verdict != "REJECT" else None
    if verdict == "REJECT":
        ck.window_vecs = ck.window_vecs_raw = None
    bl = s.get(Baseline, ck.patient_id)
    base_n = int(cfg["baseline_n"])
    # a verification repeat belongs to its original's episode: it never adds to the sustained-difference count
    scored_hist = [(c.drift["exceeds"], c.drift["weight"]) for c in prev
                   if c.drift and not (c.extra or {}).get("verification_of")]
    drift_json, confirmed, exceeds = None, False, False
    incompatible = None
    if bl is not None and bl.frozen_at is not None and verdict in ("PASS", "WARN"):
        rows = baseline_rows(prev, base_n)
        bp, cp = (P.of_checkin(rows[0]) if rows else None), P.of_checkin(ck)
        if bp is not None and not P.compatible(bp, cp):
            incompatible = {"status": "INCOMPATIBLE_PROTOCOL", "baseline_prompt": bp, "current_prompt": cp,
                            "message": "This recording used a different speaking task from your baseline, so it "
                                       "was not compared. Record with the same task as your baseline, or start a "
                                       "new baseline."}
        elif ck.extra.get("acoustic") is not None:
            cmp = EV.compare(ck.extra["acoustic"], [(c.extra or {}).get("acoustic") for c in rows])
            ck.extra["acoustic_features"] = cmp
            ck.extra["acoustic_summary"] = EV.describe(cmp)

    if bl is None or bl.frozen_at is None:
        # still building the baseline: only PASS check-ins count, no drift yet
        passes = baseline_rows(prev, base_n) + ([ck] if verdict == "PASS" and vec is not None else [])
        ck.extra["baseline_progress"] = [len(passes), base_n]
        if len(passes) >= base_n:
            xs = np.stack([dm.project(blob_to_vec(c.session_vec)) for c in passes[:base_n]])
            mu, sg = D.fit_baseline(xs, dm.sigma_pop, int(cfg["shrink_n0"]))
            trows = [_timing_view(c.timing) for c in passes[:base_n]]
            tmu, tsg = D.fit_timing_baseline(trows, dict(cfg["timing_sigma_pop"]), int(cfg["shrink_n0"]))
            s.merge(Baseline(patient_id=ck.patient_id, n=base_n, mu=vec_to_blob(mu), sigma=vec_to_blob(sg),
                             timing_mu=tmu, timing_sigma=tsg,
                             # the baseline is "set" when the recording that completed it was made (for seeded
                             # history that is in the past, not the moment the seed was loaded)
                             frozen_at=ck.created_at or utcnow()))
    elif incompatible is not None:
        ck.extra["comparison"] = incompatible              # withheld: no unqualified drift score
    elif verdict in ("PASS", "WARN") and vec is not None:
        mu, sg = blob_to_vec(bl.mu), blob_to_vec(bl.sigma)
        d = D.drift_score(dm.project(vec), mu, sg)
        if not np.isfinite(d):
            raise ValueError("non-finite drift score")       # never stored; the check-in fails visibly instead
        # ONE comparison, on the stored values, so the shown score, boundary and status can never disagree.
        # Rule: strictly greater than the boundary (a score exactly at the boundary is within range).
        d_s, tau_s = round(float(d), 6), round(float(dm.tau), 6)
        exceeds = d_s > tau_s
        w = 1.0 if verdict == "PASS" else float(cfg["warn_weight"])
        confirmed = D.confirmed_drift(scored_hist + [(exceeds, w)], float(cfg["confirm_sum"]), int(cfg["confirm_last"]))
        wc = int(ck.extra.get("window_count", 0))
        drift_json = {"d": d_s, "tau": tau_s, "exceeds": bool(exceeds), "weight": w,
                      "window_count": wc, "baseline_n": bl.n, "confirmed": bool(confirmed)}
        if timing and any(v is not None for v in _timing_view(timing).values()):
            cur = _timing_view(timing)
            z = D.timing_z(cur, bl.timing_mu, bl.timing_sigma)
            ck.timing = {**timing, "z": z}
            ck.extra["timing_chips"] = D.timing_chips(cur, bl.timing_mu, z)
            mu_t = bl.timing_mu or {}
            ck.extra["timing_change_pct"] = {
                k: (round(100 * (cur[k] - mu_t[k]) / mu_t[k], 1) if cur.get(k) is not None and mu_t.get(k) else None)
                for k in ("speech_rate", "mean_pause")}
    ck.drift = drift_json
    if ck.timing is None and timing:
        ck.timing = {**timing, "z": {"speech_rate": None, "pause_rate": None, "mean_pause": None}}

    verdicts = [c.quality.get("verdict") for c in prev] + [verdict]
    run = D.reject_streak(verdicts)
    dam_high = (ck.reference_risk or {}).get("level") == "high"
    baseline_ready = bl is not None and bl.frozen_at is not None
    ck.status = D.decide_status(safety=False, baseline_ready=baseline_ready, confirmed=confirmed,
                                exceeds=exceeds, dam_high=dam_high, reject_run=run)
    if incompatible is not None:
        ck.status = "INCOMPATIBLE"

    # Speech-change review requests are created by the persistence-based engine in server/monitoring.py (alerting
    # spec v1.0), evaluated after the check-in is saved. Only the SAFETY path (above) creates an immediate alert.
    return alerts


def compare_raw_vs_head(s: Session, ck: Checkin, dm: D.DriftModel, dm_raw: D.DriftModel) -> dict:
    """Drift for one check-in using raw z vs the invariance head (robustness toggle)."""
    cfg = get_config().drift
    prev = [c for c in history(s, ck.patient_id) if c.id != ck.id and c.created_at <= ck.created_at]
    base = baseline_rows(prev, int(cfg["baseline_n"]))
    if len(base) < int(cfg["baseline_n"]) or not ck.session_vec:
        return {"available": False, "reason": "baseline not ready or check-in not scored"}
    out = {"available": True, "baseline_n": len(base)}
    for name, model, field in (("invariant", dm, "session_vec"), ("raw", dm_raw, "session_vec_raw")):
        xs = np.stack([model.project(blob_to_vec(getattr(c, field))) for c in base if getattr(c, field)])
        mu, sg = D.fit_baseline(xs, model.sigma_pop, int(cfg["shrink_n0"]))
        d = D.drift_score(model.project(blob_to_vec(getattr(ck, field))), mu, sg)
        out[name] = {"d": round(d, 3), "tau": round(model.tau, 3), "exceeds": bool(d > model.tau)}
    return out


def result_json(ck: Checkin, patient_id: str | None = None) -> dict:
    """CheckinResult (spec 7.2). SAFETY check-ins carry no drift/reference_risk keys at all."""
    ex = ck.extra or {}
    r = {
        "checkin_id": ck.id,
        "patient_id": ck.patient_id,
        "created_at": ck.created_at.isoformat() + "Z",
        "status": ck.status,
        "quality": ck.quality,
        "timing": ck.timing,
        "evidence": ck.evidence or [],
        "evidence_dropped": ex.get("evidence_dropped", 0),
        "evidence_available": ex.get("evidence_available", True),
        "summary": ex.get("summary", ""),
        "timing_chips": ex.get("timing_chips", []),
        "safety": ex.get("safety", {"triggered": False, "source": None}),
        "transcript": ck.transcript,
        "baseline_progress": ex.get("baseline_progress"),
        "simulated": ex.get("simulated", False),
        "versions": ck.versions,
        "latency_ms": ex.get("latency_ms", {}),
        "disclaimer": DISCLAIMER,
        "protocol": P.of_checkin(ck),
        "checkin_type": ex.get("checkin_type", "standard"),
        "verification_of": ex.get("verification_of"),
        "verification": ex.get("verification"),
        "comparison": ex.get("comparison"),
        "context": ex.get("context"),
    }
    if ck.status == "SAFETY":
        r["timing"] = None
        r["evidence"] = []
        r["summary"] = ""
        r["transcript"] = ck.transcript if ex.get("keep_transcript", True) else ""
        return r
    r["drift"] = ck.drift
    r["reference_risk"] = ck.reference_risk
    r["cohort_risk"] = ex.get("cohort_risk")   # clinician-only; patient_view never includes it
    r["has_window_analysis"] = bool(ex.get("window_analysis"))
    r["acoustic_features"] = ex.get("acoustic_features")
    r["acoustic_summary"] = ex.get("acoustic_summary", [])
    d = ck.drift or {}
    r["analysis"] = ({"status": "COMPARED", "normalized_drift": round(d["d"] / d["tau"], 4),
                      "session_distance": d["d"], "session_threshold": d["tau"],
                      "threshold_source": "session-level drift boundary (not the window-level visualization threshold)",
                      "model_version": (ck.versions or {}).get("encoder"),
                      "projection_version": (ck.versions or {}).get("artifacts")}
                     if d.get("tau") else
                     {"status": "INCOMPATIBLE_PROTOCOL" if ex.get("comparison") else
                      "NOT_SCORED" if (ck.quality or {}).get("verdict") == "REJECT" else "BASELINE"})
    r.update(contract(ck))
    return r


def _pct(x):
    return None if x is None else round(100 * x, 1)


def contract(ck: Checkin) -> dict:
    """Backlog VD-02 output contract, derived ONLY from the stored analysis (never recomputed elsewhere).
    review_required is added by the API from the review-flag engine's stored evaluation."""
    ex, d, q = ck.extra or {}, ck.drift or {}, ck.quality or {}
    af = ex.get("acoustic_features") or {}
    tc = ex.get("timing_change_pct") or {}
    sr = (af.get("speaking_rate_wpm") or {}).get("relative")
    pf = (af.get("pause_fraction") or {}).get("relative")
    status = ("unavailable" if not d.get("tau") else "elevated" if d.get("exceeds") else "within_range")
    return {
        "source": ex.get("source") or ("seeded" if ex.get("simulated") else "live"),
        "synthetic_voice": bool(ex.get("synthetic_voice")),
        "quality_status": "fail" if q.get("verdict") == "REJECT" else "warning" if q.get("verdict") == "WARN" else "pass",
        "speech_duration_seconds": q.get("speech_seconds", q.get("speech_s")),
        "speaking_rate_change_pct": _pct(sr) if sr is not None else tc.get("speech_rate"),
        "pause_duration_change_pct": _pct(pf) if pf is not None else tc.get("mean_pause"),
        "pause_measure": "pause_fraction" if pf is not None else ("mean_pause_s" if tc.get("mean_pause") is not None else None),
        "drift_score": round(d["d"] / d["tau"], 4) if d.get("tau") else None,
        "drift_score_units": "x session boundary (1.0 = boundary)",
        "drift_threshold": 1.0 if d.get("tau") else None,
        "drift_status": status,
        "drift_rule": "elevated when the session score is strictly greater than the boundary",
        "model_version": f"{(ck.versions or {}).get('encoder', 'unknown')}|{(ck.versions or {}).get('artifacts', 'unknown')}",
    }


def patient_view(r: dict) -> dict:
    """What a patient may see: no scores, no evidence, no transcript. Messages only."""
    from .pipeline.quality import patient_message

    q = r["quality"]
    out = {"checkin_id": r["checkin_id"], "status": "SAFETY" if r["status"] == "SAFETY" else "OK",
           "quality": {"verdict": q.get("verdict"), "message": patient_message(q.get("reasons", []))
                       if q.get("verdict") == "REJECT" else ""},
           "baseline_progress": r.get("baseline_progress") if r["status"] == "BASELINE" else None,
           "simulated": r.get("simulated", False), "disclaimer": r["disclaimer"]}
    reasons = q.get("reasons", [])
    out["outcome"] = ("safety" if r["status"] == "SAFETY" else
                      "failed" if any(x in ("processing error", "unreadable audio") for x in reasons) else
                      "rejected" if q.get("verdict") == "REJECT" else "accepted")
    out["has_window_analysis"] = r.get("has_window_analysis", False) if out["outcome"] == "accepted" else False
    out["quality"].update(status=q.get("status", "REJECT" if q.get("verdict") == "REJECT" else "PASS"),
                          reason_codes=q.get("reason_codes", []))
    out["protocol"] = r.get("protocol")
    out["checkin_type"] = r.get("checkin_type", "standard")
    out["verification_of"] = r.get("verification_of")
    if out["outcome"] == "accepted":     # self-monitoring: people see their own descriptive measurements
        out["quality"]["message"] = "Recording quality passed."
        out["acoustic_features"] = r.get("acoustic_features")
        out["acoustic_summary"] = r.get("acoustic_summary", [])
        out["verification"] = r.get("verification")
        out["context"] = r.get("context")
    if r["status"] == "SAFETY":
        out["safety"] = {"triggered": True, "message": SAFETY_MESSAGE}
    elif q.get("verdict") == "REJECT":
        out["message"] = out["quality"]["message"]
    elif r["status"] == "INCOMPATIBLE":
        out["message"] = "Check-in recorded. " + (r.get("comparison") or {}).get("message", "")
    elif r["status"] in ("RED", "AMBER"):
        out["message"] = "Check-in recorded. Your latest recording differs from your personal speech baseline."
    elif r["status"] == "BASELINE":
        n, tot = r.get("baseline_progress") or [0, 5]
        out["message"] = f"Check-in recorded. Building your baseline ({n}/{tot})."
    else:
        out["message"] = "Check-in recorded."
    return out

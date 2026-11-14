"""Stateful scoring shared by the live pipeline and the seeded scenarios (one code path, spec 6)."""
from __future__ import annotations

from datetime import datetime

import numpy as np
from sqlmodel import Session, select

from .config import get_config
from .pipeline import drift as D
from .store import Alert, Baseline, Checkin, blob_to_vec, new_id, utcnow, vec_to_blob

TIMING_KEYS = {"speech_rate": "speech_rate_wpm", "pause_rate": "pause_rate_pm", "mean_pause": "mean_pause_s"}
DISCLAIMER = "Screening and monitoring aid — not a diagnosis."


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

    ck.session_vec = vec_to_blob(vec) if verdict != "REJECT" else None
    ck.session_vec_raw = vec_to_blob(vec_raw) if verdict != "REJECT" else None
    if verdict == "REJECT":
        ck.window_vecs = ck.window_vecs_raw = None
    bl = s.get(Baseline, ck.patient_id)
    base_n = int(cfg["baseline_n"])
    scored_hist = [(c.drift["exceeds"], c.drift["weight"]) for c in prev if c.drift]
    drift_json, confirmed, exceeds = None, False, False

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
                             timing_mu=tmu, timing_sigma=tsg, frozen_at=utcnow()))
    elif verdict in ("PASS", "WARN") and vec is not None:
        mu, sg = blob_to_vec(bl.mu), blob_to_vec(bl.sigma)
        d = D.drift_score(dm.project(vec), mu, sg)
        exceeds = d > dm.tau
        w = 1.0 if verdict == "PASS" else float(cfg["warn_weight"])
        confirmed = D.confirmed_drift(scored_hist + [(exceeds, w)], float(cfg["confirm_sum"]), int(cfg["confirm_last"]))
        wc = int(ck.extra.get("window_count", 0))
        drift_json = {"d": round(d, 3), "tau": round(dm.tau, 3), "exceeds": bool(exceeds), "weight": w,
                      "window_count": wc, "baseline_n": bl.n, "confirmed": bool(confirmed)}
        if timing and any(v is not None for v in _timing_view(timing).values()):
            cur = _timing_view(timing)
            z = D.timing_z(cur, bl.timing_mu, bl.timing_sigma)
            ck.timing = {**timing, "z": z}
            ck.extra["timing_chips"] = D.timing_chips(cur, bl.timing_mu, z)
    ck.drift = drift_json
    if ck.timing is None and timing:
        ck.timing = {**timing, "z": {"speech_rate": None, "pause_rate": None, "mean_pause": None}}

    verdicts = [c.quality.get("verdict") for c in prev] + [verdict]
    run = D.reject_streak(verdicts)
    dam_high = (ck.reference_risk or {}).get("level") == "high"
    baseline_ready = bl is not None and bl.frozen_at is not None
    ck.status = D.decide_status(safety=False, baseline_ready=baseline_ready, confirmed=confirmed,
                                exceeds=exceeds, dam_high=dam_high, reject_run=run)

    def add(kind: str):
        a = Alert(id=new_id("a"), patient_id=ck.patient_id, checkin_id=ck.id, kind=kind)
        s.add(a)
        alerts.append(a)

    if ck.status == "RED":
        add("DRIFT")
    elif ck.status == "AMBER":
        if run >= 2:
            open_missed = s.exec(select(Alert).where(Alert.patient_id == ck.patient_id, Alert.kind == "MISSED",
                                                     Alert.acked_at == None)).first()  # noqa: E711
            if not open_missed:
                add("MISSED")
        if exceeds or dam_high:
            add("REVIEW")
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
    return r


def patient_view(r: dict) -> dict:
    """What a patient may see: no scores, no evidence, no transcript. Messages only."""
    from .pipeline.quality import patient_message

    q = r["quality"]
    out = {"checkin_id": r["checkin_id"], "status": "SAFETY" if r["status"] == "SAFETY" else "OK",
           "quality": {"verdict": q.get("verdict"), "message": patient_message(q.get("reasons", []))
                       if q.get("verdict") == "REJECT" else ""},
           "baseline_progress": r.get("baseline_progress") if r["status"] == "BASELINE" else None,
           "simulated": r.get("simulated", False), "disclaimer": r["disclaimer"]}
    if r["status"] == "SAFETY":
        out["safety"] = {"triggered": True,
                         "message": "If you are thinking about harming yourself, you are not alone. "
                                    "Call or text 988 (Suicide & Crisis Lifeline), any time. "
                                    "Please also contact your clinician today."}
    elif q.get("verdict") == "REJECT":
        out["message"] = out["quality"]["message"]
    elif r["status"] in ("RED", "AMBER"):
        out["message"] = "Check-in recorded. Your clinician will follow up."
    elif r["status"] == "BASELINE":
        n, tot = r.get("baseline_progress") or [0, 5]
        out["message"] = f"Check-in recorded. Building your baseline ({n}/{tot})."
    else:
        out["message"] = "Check-in recorded."
    return out

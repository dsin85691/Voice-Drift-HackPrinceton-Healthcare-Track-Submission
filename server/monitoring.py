"""Speech-change review flags (alerting spec v1.0): eligibility gates, persistence rule, flag lifecycle,
deduplication, cooldown, idempotency, audit trail and a demo-only notification outbox.

Session score R_t = D_t / tau_t, the app's existing calibrated session drift (documented deviation: the spec
proposes cosine distance; the app's session score is an RMS z-distance in PCA space with its own calibrated boundary,
so Trends, statuses and alerts stay on one scale). Window-level scores never feed this engine.

A review flag is an experimental, NONURGENT request for human review. It is not a risk score, diagnosis or alert of
deterioration, and nothing is ever sent outside the app (notification_mode: demo).
"""
from __future__ import annotations

import hashlib
import math
from datetime import timedelta

from sqlmodel import Session, select

from . import scoring, store
from .config import get_config

ACTIVE = ("OPEN", "ACKNOWLEDGED", "REVIEWED")      # same episode still open: dedups new flags
PENDING = ("OPEN", "ACKNOWLEDGED")                 # still waiting for a reviewer: what the queue counts
TRANSITIONS = {  # lifecycle (spec 6.2)
    "acknowledge": ({"OPEN"}, "ACKNOWLEDGED"),
    "review": ({"ACKNOWLEDGED"}, "REVIEWED"),
    "close": ({"OPEN", "REVIEWED", "ACKNOWLEDGED"}, "CLOSED"),
    "dismiss": ({"OPEN", "ACKNOWLEDGED"}, "DISMISSED"),
}
EXPLANATION = ("{n} of the last {w} quality-passing recordings exceeded the person's experimental speech-change "
               "reference boundary on {d} separate days.")


def cfg() -> dict:
    return dict(get_config().monitoring)


def rule_version() -> str:
    c = cfg()
    return f"{c['rule_id']}@{c['rule_version']}"


def _audit(s: Session, actor: str, etype: str, eid: str, action: str, reason: str | None = None, **detail):
    s.add(store.AuditEvent(event_id=store.new_id("ev"), actor_id=actor, entity_type=etype, entity_id=eid,
                           action=action, reason_code=reason, detail=detail))


# ------------------------------------------------------------------ per-check-in score + gates
def data_mode(ck: store.Checkin) -> str:
    return "synthetic_demo" if (ck.extra or {}).get("simulated") else "real_recording"


def baseline_version(prev: list[store.Checkin]) -> str | None:
    rows = scoring.baseline_rows(prev, int(get_config().drift["baseline_n"]))
    if len(rows) < int(get_config().drift["baseline_n"]):
        return None
    return "bl_" + hashlib.sha1("|".join(c.id for c in rows).encode()).hexdigest()[:10]


def local_day(ck: store.Checkin):
    """The calendar day the person recorded on, in their own time zone. Check-ins are stamped in UTC; the browser
    sends its offset (JavaScript getTimezoneOffset: minutes, UTC minus local). Without one (seeded data, older
    check-ins) the UTC date is used. Two recordings one evening either side of midnight UTC are one day, not two."""
    off = (ck.extra or {}).get("tz_offset_min")
    if isinstance(off, (int, float)) and -840 <= off <= 840:
        return (ck.created_at - timedelta(minutes=off)).date()
    return ck.created_at.date()


def flag_id_for(patient_id: str, rule_id: str, checkin_id: str) -> str:
    """Stable id from the inputs that define the flag: loading the same demo data again gives the same id, so an open
    tab or bookmark on a review keeps working. One check-in can trigger at most one flag per rule."""
    return "flag_" + hashlib.sha1(f"{patient_id}|{rule_id}|{checkin_id}".encode()).hexdigest()[:12]


def session_eval(ck: store.Checkin, prev: list[store.Checkin]) -> dict:
    """Score + eligibility of ONE check-in (gates E1-E3). `prev` = the patient's earlier check-ins."""
    v = ck.versions or {}
    mode = data_mode(ck)
    reasons = []
    q = (ck.quality or {}).get("verdict")
    if ck.status == "SAFETY":
        reasons.append("SAFETY_PATH")                       # handled by the separate, immediate safety path
    if (ck.extra or {}).get("verification_of"):
        reasons.append("VERIFICATION_REPEAT")               # same episode as its original: never counted twice
    if (ck.extra or {}).get("comparison"):
        reasons.append("PROMPT_INCOMPATIBLE")
    if q not in ("PASS", "WARN"):
        reasons.append("QUALITY_NOT_PASSED")
    if not ck.session_vec and "SAFETY_PATH" not in reasons and "QUALITY_NOT_PASSED" not in reasons:
        reasons.append("EMBEDDING_MISSING")
    bver = baseline_version(prev + [ck]) if ck.drift else None
    if not ck.drift or bver is None:
        reasons.append("BASELINE_PENDING")
    d, tau = (ck.drift or {}).get("d"), (ck.drift or {}).get("tau")
    encoder, artifacts = v.get("encoder", "unknown"), v.get("artifacts", "unknown")
    calibrated = mode == "synthetic_demo" or (artifacts not in ("fallback", "seed", "unknown", None)
                                               and "fallback" not in encoder)
    if not reasons and (tau is None or not math.isfinite(tau) or tau <= 0):
        reasons.append("THRESHOLD_INVALID")
    if not reasons and not calibrated:
        reasons.append("CALIBRATION_PENDING")             # fallback boundary = simulated constant, not a calibration
    calib = (f"illustrative-seed:tau={tau:.4f}" if mode == "synthetic_demo" else f"{artifacts}:tau={tau:.4f}") if tau else None
    context = "|".join([encoder, artifacts, bver or "-", calib or "-"])
    r = d / tau if (d is not None and tau) else None
    return {"checkin_id": ck.id, "recorded_at": ck.created_at, "local_day": local_day(ck),
            "eligible": not reasons, "reasons": reasons,
            "distance": d, "threshold": tau, "normalized_score": round(r, 4) if r is not None else None,
            "exceeded": (r > 1) if r is not None else None, "context": context, "encoder": encoder,
            "projection": artifacts, "baseline_id": bver, "calibration_id": calib, "data_mode": mode,
            "calibration_kind": "illustrative (synthetic data)" if mode == "synthetic_demo" else
                                ("calibrated" if calibrated else "not calibrated")}


def _state_from_reasons(reasons: list[str]) -> str:
    if "VERIFICATION_REPEAT" in reasons:
        return "VERIFICATION_REPEAT"
    if "PROMPT_INCOMPATIBLE" in reasons:
        return "INCOMPATIBLE_PROTOCOL"
    if "BASELINE_PENDING" in reasons:
        return "BASELINE_PENDING"
    if "CALIBRATION_PENDING" in reasons or "THRESHOLD_INVALID" in reasons:
        return "CALIBRATION_PENDING"
    return "INELIGIBLE"


# ------------------------------------------------------------------ the rule engine
def evaluate(s: Session, patient_id: str, checkin_id: str, actor: str = "system") -> dict:
    """Idempotent: a repeated call for the same patient + check-in + rule version returns the stored result."""
    c = cfg()
    rv = rule_version()
    key = f"{patient_id}:{checkin_id}:{rv}"
    prior = s.get(store.MonitoringEvaluation, key)
    if prior:
        return {**prior.result, "replayed": True}
    p = s.get(store.Patient, patient_id)
    ck = s.get(store.Checkin, checkin_id)
    if p is None or ck is None or ck.patient_id != patient_id:
        raise LookupError("unknown patient or check-in, or check-in belongs to another patient")
    hist = [x for x in scoring.history(s, patient_id) if x.created_at <= ck.created_at]
    evals, prev = [], []
    for x in hist:                                           # deterministic: recomputed from stored history
        evals.append(session_eval(x, prev)); prev.append(x)
    me = evals[-1] if evals and evals[-1]["checkin_id"] == ck.id else session_eval(ck, prev)

    s.merge(store.SessionScore(checkin_id=ck.id, patient_id=patient_id, session_distance=me["distance"],
                               session_threshold=me["threshold"], normalized_score=me["normalized_score"],
                               threshold_exceeded=me["exceeded"], score_status="scored" if me["normalized_score"] is not None else "not_scored",
                               monitoring_context_id=me["context"], model_version=me["encoder"],
                               projection_version=me["projection"], baseline_id=me["baseline_id"],
                               calibration_id=me["calibration_id"]))
    out = {"patient_id": patient_id, "checkin_id": ck.id, "rule": rv, "eligible": me["eligible"],
           "normalized_score": me["normalized_score"], "threshold_exceeded": me["exceeded"],
           "reason_codes": list(me["reasons"]), "data_mode": me["data_mode"],
           "calibration": me["calibration_kind"], "monitoring_context_id": me["context"],
           "persistence": None, "flag": {"created": False, "flag_id": None, "status": None},
           "notification": {"mode": c["notification_mode"], "external_delivery": False},
           "config": {k: c[k] for k in ("window_sessions", "min_sessions", "min_elevated", "min_distinct_days",
                                        "lookback_days", "cooldown_hours")}}

    if not me["eligible"]:
        out["monitoring_state"] = _state_from_reasons(me["reasons"])
        return _store(s, key, patient_id, ck.id, out)

    # persistence window: latest eligible sessions in the SAME compatible context, within lookback of this check-in
    lo = ck.created_at - timedelta(days=int(c["lookback_days"]))
    window = [e for e in evals if e["eligible"] and e["context"] == me["context"] and e["recorded_at"] >= lo]
    window = window[-int(c["window_sessions"]):]
    elevated = [e for e in window if e["exceeded"]]
    days = {e["local_day"] for e in elevated}
    out["persistence"] = {"eligible_sessions": len(window), "elevated_sessions": len(elevated),
                          "distinct_days": len(days), "window_checkins": [e["checkin_id"] for e in window],
                          "rule_satisfied": False}
    if len(window) < int(c["min_sessions"]):
        out["monitoring_state"] = "INSUFFICIENT_DATA"
        out["reason_codes"].append("INSUFFICIENT_HISTORY")
        return _store(s, key, patient_id, ck.id, out)
    satisfied = len(elevated) >= int(c["min_elevated"]) and len(days) >= int(c["min_distinct_days"])
    out["persistence"]["rule_satisfied"] = satisfied
    if not satisfied:
        out["monitoring_state"] = "ELEVATED_OBSERVATION" if elevated else "MONITORING"
        return _store(s, key, patient_id, ck.id, out)

    # ---- review-eligible: dedup, cooldown, consent, then create
    rid = c["rule_id"]
    active = s.exec(select(store.ReviewFlag).where(store.ReviewFlag.patient_id == patient_id,
                                                   store.ReviewFlag.rule_id == rid,
                                                   store.ReviewFlag.monitoring_context_id == me["context"],
                                                   store.ReviewFlag.status.in_(ACTIVE))).first()
    if active:                                   # same episode: attach evidence, never a second flag
        have = {e.checkin_id for e in s.exec(select(store.FlagEvidence).where(store.FlagEvidence.flag_id == active.flag_id))}
        for e in window:
            if e["checkin_id"] not in have:
                s.add(store.FlagEvidence(flag_id=active.flag_id, checkin_id=e["checkin_id"],
                                         role="qualifying" if e["exceeded"] else "supporting"))
        _audit(s, actor, "review_flag", active.flag_id, "evidence_added", "ACTIVE_FLAG_EXISTS", checkin_id=ck.id)
        out.update(monitoring_state="REVIEW_ELIGIBLE",
                   flag={"created": False, "flag_id": active.flag_id, "status": active.status})
        out["reason_codes"].append("ACTIVE_FLAG_EXISTS")
        return _store(s, key, patient_id, ck.id, out)
    recent = s.exec(select(store.ReviewFlag).where(store.ReviewFlag.patient_id == patient_id,
                                                   store.ReviewFlag.rule_id == rid,
                                                   store.ReviewFlag.monitoring_context_id == me["context"],
                                                   store.ReviewFlag.status.in_(("CLOSED", "DISMISSED")))).all()
    cool = timedelta(hours=float(c["cooldown_hours"]))
    if any(f.closed_at and timedelta(0) <= ck.created_at - f.closed_at < cool for f in recent):
        out["monitoring_state"] = "SUPPRESSED"
        out["reason_codes"].append("COOLDOWN")
        return _store(s, key, patient_id, ck.id, out)
    if (p.monitoring_enrollment_status != "enrolled"
            or p.review_consent_status not in ("consented", "demo_consent")):
        out["monitoring_state"] = "SUPPRESSED"
        out["reason_codes"].append("NOT_ENROLLED_OR_NO_REVIEW_CONSENT")
        return _store(s, key, patient_id, ck.id, out)

    flag = store.ReviewFlag(
        flag_id=flag_id_for(patient_id, rid, ck.id), patient_id=patient_id, rule_id=rid, rule_version=c["rule_version"],
        monitoring_context_id=me["context"], status="OPEN", data_mode=me["data_mode"],
        created_at=ck.created_at, assigned_reviewer_id=p.assigned_reviewer_id or c["default_reviewer"],
        summary={"explanation": EXPLANATION.format(n=len(elevated), w=len(window), d=len(days)),
                 "first_elevated": min(e["recorded_at"] for e in elevated).isoformat() + "Z",
                 "last_elevated": max(e["recorded_at"] for e in elevated).isoformat() + "Z",
                 "qualifying": len(elevated), "window": len(window), "distinct_days": len(days),
                 "encoder": me["encoder"], "projection": me["projection"], "baseline_id": me["baseline_id"],
                 "calibration_id": me["calibration_id"], "calibration": me["calibration_kind"],
                 "config": out["config"]})
    s.add(flag)
    for e in window:
        s.add(store.FlagEvidence(flag_id=flag.flag_id, checkin_id=e["checkin_id"],
                                 role="qualifying" if e["exceeded"] else "supporting"))
    _audit(s, actor, "review_flag", flag.flag_id, "created", "PERSISTENCE_RULE_MET", checkin_id=ck.id)
    _notify(s, flag, c)
    out.update(monitoring_state="REVIEW_ELIGIBLE", flag={"created": True, "flag_id": flag.flag_id, "status": "OPEN"})
    return _store(s, key, patient_id, ck.id, out)


def _notify(s: Session, flag: store.ReviewFlag, c: dict) -> None:
    """Outbox record (spec 8.2). Demo mode: delivered to the in-app queue only; external delivery never happens.
    The payload is minimal: no patient identifier, score, transcript or interpretation."""
    n = store.NotificationOutbox(
        notification_id=store.new_id("nt"), flag_id=flag.flag_id, recipient_id=flag.assigned_reviewer_id,
        channel="in_app", external_delivery=False,
        payload={"subject": "Voice Drift: New speech-change review available",
                 "body": "A new nonurgent speech-monitoring review is available in your secure dashboard."})
    if c["notification_mode"] == "demo":
        n.delivery_status, n.attempt_count, n.sent_at = "delivered_in_app", 1, store.utcnow()
    else:  # pragma: no cover - no other adapter exists; never send anything external
        n.delivery_status = "suppressed"
    s.add(n)
    _audit(s, "system", "notification", flag.flag_id, "queued", n.delivery_status, notification_id=n.notification_id)


def _store(s: Session, key: str, pid: str, cid: str, out: dict) -> dict:
    s.add(store.MonitoringEvaluation(key=key, patient_id=pid, checkin_id=cid, result=out))
    return out


# ------------------------------------------------------------------ reviewer actions + read models
def act(s: Session, flag_id: str, action: str, actor: str, note: str = "") -> store.ReviewFlag:
    f = s.get(store.ReviewFlag, flag_id)
    if f is None:
        raise LookupError("unknown flag")
    if action == "note":
        _audit(s, actor, "review_flag", flag_id, "note", None, note=note[:500])
        return f
    if action == "request_repeat":
        # asks the person for another check-in (in-app only). Not a treatment directive; status is unchanged.
        if f.status not in ACTIVE:
            raise ValueError(f"cannot request a repeat on a flag that is {f.status}")
        n = store.NotificationOutbox(
            notification_id=store.new_id("nt"), flag_id=flag_id, recipient_id=f.patient_id, channel="in_app_patient",
            external_delivery=False, delivery_status="delivered_in_app", attempt_count=1, sent_at=store.utcnow(),
            payload={"subject": "Repeat check-in requested",
                     "body": "Your reviewer asked for another check-in using the same reading passage."})
        s.add(n)
        _audit(s, actor, "review_flag", flag_id, "request_repeat", None, note=note[:500],
               notification_id=n.notification_id)
        return f
    if action not in TRANSITIONS:
        raise ValueError(f"unknown action {action!r}")
    allowed, new = TRANSITIONS[action]
    if f.status not in allowed:
        raise ValueError(f"cannot {action} a flag that is {f.status}")
    if action in ("close", "dismiss") and not note.strip():
        raise ValueError(f"{action} needs a documented reason (note)")
    now = store.utcnow()
    f.status = new
    if new == "ACKNOWLEDGED":
        f.acknowledged_at = now
    elif new == "REVIEWED":
        f.reviewed_at = now
    elif new in ("CLOSED", "DISMISSED"):
        f.closed_at = now
    s.add(f)
    _audit(s, actor, "review_flag", flag_id, action, None, from_status=list(allowed), to_status=new, note=note[:500])
    return f


def flag_json(f: store.ReviewFlag, s: Session | None = None, detail: bool = False) -> dict:
    iso = lambda t: t.isoformat() + "Z" if t else None
    out = {"flag_id": f.flag_id, "patient_id": f.patient_id, "flag": "Speech Change Review", "rule_id": f.rule_id,
           "rule_version": f.rule_version, "status": f.status, "data_mode": f.data_mode,
           "created_at": iso(f.created_at), "acknowledged_at": iso(f.acknowledged_at),
           "reviewed_at": iso(f.reviewed_at), "closed_at": iso(f.closed_at),
           "assigned_reviewer_id": f.assigned_reviewer_id, "summary": f.summary,
           "monitoring_context_id": f.monitoring_context_id}
    out["review_state"] = "pending" if f.status in PENDING else "reviewed"
    if s is not None:
        ev = list(s.exec(select(store.FlagEvidence).where(store.FlagEvidence.flag_id == f.flag_id)))
        out["supporting_checkins"] = len(ev)                       # every check-in the rule looked at (kept for API)
        out["considered_checkins"] = len(ev)
        out["qualifying_checkins"] = sum(1 for e in ev if e.role == "qualifying")   # the ones above the boundary
        # the most recent qualifying check-in: the score and boundary that triggered (or last supported) the review
        trig = None
        for e in ev:
            if e.role != "qualifying":
                continue
            ck = s.get(store.Checkin, e.checkin_id)
            if ck is not None and (trig is None or ck.created_at > trig.created_at):
                trig = ck
        if trig is not None:
            d = trig.drift or {}
            out["trigger"] = {"checkin_id": trig.id, "recorded_at": iso(trig.created_at),
                              "normalized_score": round(d["d"] / d["tau"], 4) if d.get("tau") else None,
                              "threshold": 1.0, "rule": "session score strictly above the boundary (1.0)",
                              "reason": EXPLANATION.format(n=f.summary.get("qualifying"), w=f.summary.get("window"),
                                                           d=f.summary.get("distinct_days")),
                              "acoustic_summary": (trig.extra or {}).get("acoustic_summary", []),
                              "timing_chips": (trig.extra or {}).get("timing_chips", [])}
        p = s.get(store.Patient, f.patient_id)
        out["patient_display_name"] = p.display_name if p else f.patient_id
        if detail:
            rows = []
            for e in ev:
                ck, sc = s.get(store.Checkin, e.checkin_id), s.get(store.SessionScore, e.checkin_id)
                if ck is None:
                    continue
                rows.append({"checkin_id": ck.id, "role": e.role, "recorded_at": iso(ck.created_at),
                             "quality": (ck.quality or {}).get("verdict"),
                             "normalized_score": sc.normalized_score if sc else None,
                             "threshold_exceeded": sc.threshold_exceeded if sc else None,
                             "timing_chips": (ck.extra or {}).get("timing_chips", []),
                             "acoustic_features": (ck.extra or {}).get("acoustic_features"),
                             "acoustic_summary": (ck.extra or {}).get("acoustic_summary", []),
                             "context": (ck.extra or {}).get("context"),
                             "has_window_analysis": bool((ck.extra or {}).get("window_analysis"))})
            # verification repeats of the evidence check-ins: shown, never counted as extra evidence
            ids = {r["checkin_id"] for r in rows}
            rows_v = [{"checkin_id": c.id, "verification_of": c.extra["verification_of"],
                       "recorded_at": iso(c.created_at), "result": (c.extra.get("verification") or {}).get("result")}
                      for c in scoring.history(s, f.patient_id) if (c.extra or {}).get("verification_of") in ids]
            out["verifications"] = rows_v
            out["evidence"] = sorted(rows, key=lambda r: r["recorded_at"])
            out["audit"] = [{"timestamp": iso(a.timestamp), "actor_id": a.actor_id, "action": a.action,
                             "reason_code": a.reason_code, "detail": a.detail}
                            for a in s.exec(select(store.AuditEvent).where(store.AuditEvent.entity_id == f.flag_id)
                                            .order_by(store.AuditEvent.timestamp))]
            out["notifications"] = [{"channel": n.channel, "delivery_status": n.delivery_status,
                                     "external_delivery": n.external_delivery, "payload": n.payload}
                                    for n in s.exec(select(store.NotificationOutbox).where(store.NotificationOutbox.flag_id == f.flag_id))]
            out["limitations"] = [
                "This persistence rule is a configurable experimental heuristic, not a validated clinical decision rule.",
                "A crossing of the session boundary describes speech and recording measurements, not a health condition.",
                "Five baseline recordings allow a demonstration of personalization but not reliable alert performance.",
                "A dismissed flag is not evidence that a speech change was clinically insignificant."]
    return out

"""Guided verification check-in (feature F3): repeat an unusual check-in with the same passage, same pipeline, same
baseline and boundary, and report whether the elevated measurement was reproduced.

Results describe experimental reproducibility only (never "healthy"/"unhealthy"):
  REPRODUCED      both valid sessions exceed the same compatible session boundary
  NOT_REPRODUCED  only the original exceeds it
  INCONCLUSIVE    the repeat failed quality, was not scored, or is not comparable with the original
  UNCALIBRATED    no calibrated session boundary exists, so no threshold-crossing is classified

A repeat is part of its original's monitoring episode: it is excluded from persistence counting (monitoring.py) and
from the sustained-difference count (scoring.py), so a same-day retry can never inflate a multi-day rule.
"""
from __future__ import annotations

from sqlmodel import Session

from . import protocol, scoring, store


def is_calibrated(ck: store.Checkin) -> bool:
    """A real, trained session boundary (not the fallback constant, not synthetic seed data)."""
    v = ck.versions or {}
    if (ck.extra or {}).get("simulated"):
        return False
    return v.get("artifacts") not in ("fallback", "seed", "unknown", None) and "fallback" not in (v.get("encoder") or "")


def _norm(ck: store.Checkin) -> float | None:
    d = ck.drift or {}
    return round(d["d"] / d["tau"], 4) if d.get("tau") else None


def _side(ck: store.Checkin) -> dict:
    return {"checkin_id": ck.id, "recorded_at": ck.created_at.isoformat() + "Z",
            "quality": (ck.quality or {}).get("verdict"), "quality_reasons": (ck.quality or {}).get("reasons", []),
            "normalized_drift": _norm(ck), "exceeded": (ck.drift or {}).get("exceeds")}


def classify(*, repeat_quality_ok: bool, comparable: bool, repeat_scored: bool, calibrated: bool,
             original_exceeded: bool, repeat_exceeded: bool) -> tuple[str, str | None]:
    """Deterministic rule; returns (result, reason_code)."""
    if not repeat_quality_ok:
        return "INCONCLUSIVE", "REPEAT_FAILED_QUALITY"
    if not comparable:
        return "INCONCLUSIVE", "NOT_COMPARABLE"
    if not repeat_scored:
        return "INCONCLUSIVE", "REPEAT_NOT_SCORED"
    if not calibrated:
        return "UNCALIBRATED", "NO_CALIBRATED_BOUNDARY"
    if not original_exceeded:
        return "INCONCLUSIVE", "ORIGINAL_NOT_ELEVATED"
    return ("REPRODUCED", None) if repeat_exceeded else ("NOT_REPRODUCED", None)


def evaluate(s: Session, rep: store.Checkin) -> dict:
    oid = (rep.extra or {}).get("verification_of")
    orig = s.get(store.Checkin, oid) if oid else None
    if orig is None or orig.patient_id != rep.patient_id:
        return {"verification_of": oid, "result": "INCONCLUSIVE", "reason": "ORIGINAL_NOT_FOUND"}
    ov, rv = orig.versions or {}, rep.versions or {}
    od, rd = orig.drift or {}, rep.drift or {}
    comparable = (protocol.compatible(protocol.of_checkin(orig), protocol.of_checkin(rep))
                  and ov.get("encoder") == rv.get("encoder") and ov.get("artifacts") == rv.get("artifacts")
                  and (not rd or od.get("tau") == rd.get("tau")))
    result, reason = classify(
        repeat_quality_ok=(rep.quality or {}).get("verdict") in ("PASS", "WARN"), comparable=comparable,
        repeat_scored=bool(rd.get("tau")), calibrated=is_calibrated(orig) and is_calibrated(rep),
        original_exceeded=bool(od.get("exceeds")), repeat_exceeded=bool(rd.get("exceeds")))
    return {"verification_of": orig.id, "result": result, "reason": reason, "original": _side(orig),
            "repeat": _side(rep), "session_threshold": rd.get("tau") or od.get("tau"),
            "note": "Describes whether the measurement repeated under the same conditions. It is not a health "
                    "assessment."}


def offer(s: Session, ck: store.Checkin) -> dict:
    """Should the person be offered a repeat of this check-in?"""
    ex = ck.extra or {}
    no = lambda why: {"available": False, "kind": None, "reason": why}
    if ex.get("verification_of"):
        return no("this check-in is itself a repeat")
    if ex.get("simulated"):
        return no("synthetic demo data cannot be repeated with a live recording")
    if ck.status == "SAFETY" or (ck.quality or {}).get("verdict") not in ("PASS", "WARN"):
        return no("not an accepted check-in")
    if not (ck.drift or {}).get("exceeds"):
        return no("within the session boundary")
    if any((c.extra or {}).get("verification_of") == ck.id for c in scoring.history(s, ck.patient_id)):
        return no("already repeated")
    if is_calibrated(ck):
        return {"available": True, "kind": "VERIFY", "reason": None,
                "message": "Your latest recording differs from your personal speech baseline."}
    return {"available": True, "kind": "MANUAL_REPEAT", "reason": "NO_CALIBRATED_BOUNDARY",
            "message": "You can repeat this check-in. No calibrated boundary is loaded, so the repeat will not be "
                       "classified as confirming or not confirming a change."}


def linked(s: Session, ck: store.Checkin) -> dict:
    """Original + its repeats, for the comparison view (works from either side)."""
    ex = ck.extra or {}
    orig = s.get(store.Checkin, ex["verification_of"]) if ex.get("verification_of") else ck
    reps = [c for c in scoring.history(s, ck.patient_id) if (c.extra or {}).get("verification_of") == orig.id]
    return {"original": _side(orig), "offer": offer(s, orig),
            "repeats": [{**_side(r), "verification": (r.extra or {}).get("verification") or evaluate(s, r)}
                        for r in reps]}

"""SQLite via SQLModel (spec 7.4). JSON columns hold dicts; vectors are fp32 BLOBs."""
from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone

import numpy as np
from pydantic import NaiveDatetime
from sqlalchemy import Column, LargeBinary
from sqlalchemy.types import JSON
from sqlalchemy.pool import StaticPool
from sqlmodel import Field, Session, SQLModel, create_engine, delete, select

from .config import get_config

_engine = None


def utcnow() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class Patient(SQLModel, table=True):
    id: str = Field(primary_key=True)
    display_name: str
    clinician_id: str = "dr_demo"
    consent_at: NaiveDatetime | None = None
    created_at: NaiveDatetime = Field(default_factory=utcnow)
    simulated: bool = False
    # review workflow (alerting spec E5). "demo_consent" is a hackathon stand-in, never real authorization.
    monitoring_enrollment_status: str | None = None     # enrolled / not_enrolled
    review_consent_status: str | None = None            # demo_consent / consented / none
    assigned_reviewer_id: str | None = None


class Checkin(SQLModel, table=True):
    id: str = Field(primary_key=True)
    patient_id: str = Field(index=True)
    created_at: NaiveDatetime = Field(default_factory=utcnow, index=True)
    status: str = "BASELINE"
    quality: dict = Field(default_factory=dict, sa_column=Column(JSON))
    drift: dict | None = Field(default=None, sa_column=Column(JSON))
    timing: dict | None = Field(default=None, sa_column=Column(JSON))
    reference_risk: dict | None = Field(default=None, sa_column=Column(JSON))
    evidence: list = Field(default_factory=list, sa_column=Column(JSON))
    transcript: str = ""
    session_vec: bytes | None = Field(default=None, sa_column=Column(LargeBinary))      # 128 x fp32
    session_vec_raw: bytes | None = Field(default=None, sa_column=Column(LargeBinary))  # raw-z (robustness toggle)
    # per-window vectors (npy bytes, shape [n_windows, d]); used for window-level calibration. Not audio.
    window_vecs: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    window_vecs_raw: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    versions: dict = Field(default_factory=dict, sa_column=Column(JSON))
    extra: dict = Field(default_factory=dict, sa_column=Column(JSON))  # safety, latency, evidence_dropped, simulated, chips...


class Baseline(SQLModel, table=True):
    patient_id: str = Field(primary_key=True)
    n: int = 0
    mu: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    sigma: bytes | None = Field(default=None, sa_column=Column(LargeBinary))
    timing_mu: dict = Field(default_factory=dict, sa_column=Column(JSON))
    timing_sigma: dict = Field(default_factory=dict, sa_column=Column(JSON))
    frozen_at: NaiveDatetime | None = None


class Alert(SQLModel, table=True):
    id: str = Field(primary_key=True)
    patient_id: str = Field(index=True)
    checkin_id: str
    kind: str  # SAFETY / DRIFT / REVIEW / MISSED
    created_at: NaiveDatetime = Field(default_factory=utcnow)
    acked_by: str | None = None
    acked_at: NaiveDatetime | None = None
    note: str | None = None


class SessionScore(SQLModel, table=True):
    """Session-level score as seen by the alert engine (provenance frozen at evaluation time)."""
    checkin_id: str = Field(primary_key=True)
    patient_id: str = Field(index=True)
    session_distance: float | None = None
    session_threshold: float | None = None
    normalized_score: float | None = None
    threshold_exceeded: bool | None = None
    score_status: str = "none"              # scored / not_scored
    monitoring_context_id: str | None = None
    model_version: str | None = None
    projection_version: str | None = None
    baseline_id: str | None = None
    calibration_id: str | None = None
    calculated_at: NaiveDatetime = Field(default_factory=utcnow)


class ReviewFlag(SQLModel, table=True):
    flag_id: str = Field(primary_key=True)
    patient_id: str = Field(index=True)
    rule_id: str
    rule_version: str
    monitoring_context_id: str
    status: str = "OPEN"                    # OPEN / ACKNOWLEDGED / REVIEWED / CLOSED / DISMISSED
    data_mode: str = "real_recording"
    created_at: NaiveDatetime = Field(default_factory=utcnow)
    acknowledged_at: NaiveDatetime | None = None
    reviewed_at: NaiveDatetime | None = None
    closed_at: NaiveDatetime | None = None
    assigned_reviewer_id: str | None = None
    summary: dict = Field(default_factory=dict, sa_column=Column(JSON))


class FlagEvidence(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    flag_id: str = Field(index=True)
    checkin_id: str
    role: str                               # qualifying / supporting
    added_at: NaiveDatetime = Field(default_factory=utcnow)


class NotificationOutbox(SQLModel, table=True):
    notification_id: str = Field(primary_key=True)
    flag_id: str = Field(index=True)
    recipient_id: str | None = None
    channel: str = "in_app"
    delivery_status: str = "pending"        # delivered_in_app / suppressed / failed
    external_delivery: bool = False
    attempt_count: int = 0
    payload: dict = Field(default_factory=dict, sa_column=Column(JSON))   # minimal: no ids, scores or audio
    created_at: NaiveDatetime = Field(default_factory=utcnow)
    sent_at: NaiveDatetime | None = None
    read_at: NaiveDatetime | None = None


class AuditEvent(SQLModel, table=True):
    event_id: str = Field(primary_key=True)
    actor_id: str
    entity_type: str
    entity_id: str = Field(index=True)
    action: str
    timestamp: NaiveDatetime = Field(default_factory=utcnow)
    reason_code: str | None = None
    detail: dict = Field(default_factory=dict, sa_column=Column(JSON))


class MonitoringEvaluation(SQLModel, table=True):
    """Idempotency: one stored result per patient_id + checkin_id + rule_version."""
    key: str = Field(primary_key=True)
    patient_id: str = Field(index=True)
    checkin_id: str
    result: dict = Field(default_factory=dict, sa_column=Column(JSON))
    created_at: NaiveDatetime = Field(default_factory=utcnow)


class EventLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    ts: NaiveDatetime = Field(default_factory=utcnow)
    checkin_id: str = Field(index=True)
    stage: str
    ms: int = 0
    error: str | None = None


def vec_to_blob(v: np.ndarray | None) -> bytes | None:
    return None if v is None else np.asarray(v, dtype="<f4").tobytes()


def blob_to_vec(b: bytes | None) -> np.ndarray | None:
    return None if b is None else np.frombuffer(b, dtype="<f4").astype(np.float64)


def mat_to_blob(m: np.ndarray | None) -> bytes | None:
    """2-D float32 matrix with its shape (npy format)."""
    if m is None:
        return None
    import io
    buf = io.BytesIO()
    np.save(buf, np.asarray(m, dtype="<f4"), allow_pickle=False)
    return buf.getvalue()


def blob_to_mat(b: bytes | None) -> np.ndarray | None:
    if b is None:
        return None
    import io
    return np.load(io.BytesIO(b), allow_pickle=False).astype(np.float64)


def _make(url: str):
    kw = {"poolclass": StaticPool} if url in ("sqlite://", "sqlite:///:memory:") else {}
    return create_engine(url, connect_args={"check_same_thread": False, "timeout": 30}, **kw)


def _migrate(eng) -> None:
    """create_all() never adds columns to an existing table: add any new nullable ones (SQLite ALTER TABLE)."""
    from sqlalchemy import inspect, text
    insp = inspect(eng)
    for table in SQLModel.metadata.sorted_tables:
        if not insp.has_table(table.name):
            continue
        have = {c["name"] for c in insp.get_columns(table.name)}
        for col in table.columns:
            if col.name not in have and col.nullable:
                with eng.begin() as conn:
                    conn.execute(text(f'ALTER TABLE "{table.name}" ADD COLUMN "{col.name}" '
                                      f'{col.type.compile(eng.dialect)}'))


def _init(url: str):
    eng = _make(url)
    SQLModel.metadata.create_all(eng)
    _migrate(eng)
    return eng


def engine():
    global _engine
    if _engine is None:
        _engine = _init(get_config().server["db_url"])
    return _engine


def reset_engine(url: str | None = None):
    """Tests: point at a fresh DB."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = _init(url or get_config().server["db_url"])
    return _engine


def session() -> Session:
    return Session(engine(), expire_on_commit=False)


def log_event(checkin_id: str, stage: str, ms: int = 0, error: str | None = None) -> None:
    with session() as s:
        s.add(EventLog(checkin_id=checkin_id, stage=stage, ms=ms, error=error))
        s.commit()


def purge_patient(pid: str) -> None:
    with session() as s:
        cids = [c.id for c in s.exec(select(Checkin).where(Checkin.patient_id == pid))]
        fids = [f.flag_id for f in s.exec(select(ReviewFlag).where(ReviewFlag.patient_id == pid))]
        if fids:
            for model, col in ((FlagEvidence, FlagEvidence.flag_id), (NotificationOutbox, NotificationOutbox.flag_id),
                               (AuditEvent, AuditEvent.entity_id)):
                s.exec(delete(model).where(col.in_(fids)))
        for model, col in ((Alert, Alert.patient_id), (Checkin, Checkin.patient_id), (SessionScore, SessionScore.patient_id),
                           (ReviewFlag, ReviewFlag.patient_id), (MonitoringEvaluation, MonitoringEvaluation.patient_id),
                           (Baseline, Baseline.patient_id), (Patient, Patient.id)):
            s.exec(delete(model).where(col == pid))
        if cids:
            s.exec(delete(EventLog).where(EventLog.checkin_id.in_(cids)))
        s.commit()




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


def _make(url: str):
    kw = {"poolclass": StaticPool} if url in ("sqlite://", "sqlite:///:memory:") else {}
    return create_engine(url, connect_args={"check_same_thread": False, "timeout": 30}, **kw)


def engine():
    global _engine
    if _engine is None:
        url = get_config().server["db_url"]
        _engine = _make(url)
        SQLModel.metadata.create_all(_engine)
    return _engine


def reset_engine(url: str | None = None):
    """Tests: point at a fresh DB."""
    global _engine
    if _engine is not None:
        _engine.dispose()
    _engine = _make(url or get_config().server["db_url"])
    SQLModel.metadata.create_all(_engine)
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
        for model, col in ((Alert, Alert.patient_id), (Checkin, Checkin.patient_id),
                           (Baseline, Baseline.patient_id), (Patient, Patient.id)):
            s.exec(delete(model).where(col == pid))
        if cids:
            s.exec(delete(EventLog).where(EventLog.checkin_id.in_(cids)))
        s.commit()




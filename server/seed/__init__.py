"""Scenario loading: seeded check-ins go through the same scoring path as live ones."""
from __future__ import annotations

import json
from datetime import timedelta
from pathlib import Path

import numpy as np

from .. import scoring, store
from ..pipeline import drift

SEED_DIR = Path(__file__).parent


def load_scenario(name: str, dm: drift.DriftModel, base_time=None) -> str:
    data = json.loads((SEED_DIR / f"{name}.json").read_text(encoding="utf-8"))
    pid = data["patient"]["id"]
    store.purge_patient(pid)
    # 15:00 UTC reads as daytime across US time zones (10 AM Eastern / 7 AM Pacific in winter, an hour later in
    # summer); 09:00 UTC showed as 4-5 AM.
    now = base_time or store.utcnow().replace(hour=15, minute=0, second=0, microsecond=0)
    if base_time is None and now > store.utcnow():
        now -= timedelta(days=1)      # before 15:00 UTC "today 15:00" is in the future: never seed future check-ins
    with store.session() as s:
        from ..pipeline.run import demo_enrollment
        s.add(store.Patient(id=pid, display_name=data["patient"]["display_name"], consent_at=store.utcnow(),
                            simulated=True, **demo_enrollment()))
        s.commit()
        for i, d in enumerate(data["days"]):
            ck = store.Checkin(
                id=f"c_{pid}_{i:02d}", patient_id=pid, created_at=now + timedelta(days=d["day_offset"]),
                quality=d["quality"], transcript=d["transcript"], evidence=d["evidence"],
                reference_risk={"source": "unavailable", "level": None, "raw": None},
                versions={"artifacts": "seed", "git": "seed", "asr": "seed", "encoder": "simulated"},
                extra={"simulated": True, "acted": d["acted"], "environment": d["environment"],
                       "summary": d["summary"], "window_count": 7, "evidence_available": True,
                       "latency_ms": {}},
            )
            scoring.apply_scoring(s, ck, vec=np.array(d["vec"]), vec_raw=np.array(d["vec_raw"]),
                                  timing=d["timing"], safety={"triggered": False, "source": None}, dm=dm)
            s.add(ck)
            s.commit()
            from .. import monitoring                       # same review-flag engine as live check-ins
            monitoring.evaluate(s, pid, ck.id, actor="seed")
            s.commit()
    return pid

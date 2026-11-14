"""FastAPI app: /api endpoints, SSE, and the static frontend. Binds to 127.0.0.1 (spec 8)."""
from __future__ import annotations

import asyncio
import json
import logging
import threading
from contextlib import asynccontextmanager
from datetime import timedelta
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from sqlmodel import select

from . import scoring, store
from .config import ROOT, get_config
from .pipeline.run import Runtime
from .seed import load_scenario

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")
log = logging.getLogger("vd.app")

PROMPTS = {"default": "Tell me about your day so far - how you slept, how your energy has been, "
                      "and what you have been looking forward to."}
SEVERITY = {"SAFETY": 0, "RED": 1, "AMBER": 2, "GREEN": 3, "BASELINE": 4}


class AckBody(BaseModel):
    clinician: str
    note: str = ""


class ResetBody(BaseModel):
    scenario: str  # stable | drift | all | blank


class CompareBody(BaseModel):
    checkin_id: str


def create_app(runtime: Runtime | None = None) -> FastAPI:
    holder: dict = {"rt": runtime}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        store.engine()
        if holder["rt"] is None:
            holder["rt"] = Runtime()
        cfg = get_config()
        log.info("resolved config: %s", json.dumps(cfg, default=str)[:1500])
        yield

    app = FastAPI(title="Voice Drift Monitor", lifespan=lifespan)
    rt = lambda: holder["rt"]
    app.state.holder = holder

    # ------------------------------------------------------------------ health
    @app.get("/api/health")
    def health():
        return rt().health()

    @app.get("/api/prompt")
    def prompt():
        return {"prompt_id": "default", "text": PROMPTS["default"]}

    # ------------------------------------------------------------------ check-ins
    @app.post("/api/patients/{pid}/checkins", status_code=202)
    async def post_checkin(pid: str, audio: UploadFile = File(...), prompt_id: str = Form("default")):
        r = rt()
        with store.session() as s:
            if s.get(store.Patient, pid) is None:
                s.add(store.Patient(id=pid, display_name=pid, consent_at=store.utcnow()))
                s.commit()
        cid = store.new_id("c")
        suffix = Path(audio.filename or "a.webm").suffix or ".webm"
        path = r.tmp / f"{cid}{suffix}"
        try:
            path.write_bytes(await audio.read())
        except Exception:
            path.unlink(missing_ok=True)
            raise
        r.bus.emit(cid, "stage", {"stage": "received", "ms": 0})
        threading.Thread(target=r.process, args=(cid, pid, path, prompt_id), daemon=True).start()
        return {"checkin_id": cid}

    def _strip(ev: dict, role: str) -> dict:
        if role != "patient":
            return ev
        if ev["event"] == "result":
            return {"event": "result", "data": scoring.patient_view(ev["data"])}
        return {"event": ev["event"], "data": {"stage": ev["data"].get("stage"), "ms": ev["data"].get("ms")}}

    @app.get("/api/checkins/{cid}/events")
    async def events(cid: str, role: str = "clinician"):
        r = rt()

        async def gen():
            seen, idle = 0, 0
            loop = asyncio.get_running_loop()
            while True:
                new = await loop.run_in_executor(None, r.bus.wait, cid, seen, 2.0)
                if not new:
                    idle += 1
                    if idle > 60 and seen == 0:
                        yield "event: error\ndata: {\"error\": \"unknown check-in\"}\n\n"
                        return
                    yield ": keepalive\n\n"
                    continue
                for ev in new:
                    ev = _strip(ev, role)
                    yield f"event: {ev['event']}\ndata: {json.dumps(ev['data'])}\n\n"
                    seen += 1
                    if ev["event"] == "result":
                        return

        return StreamingResponse(gen(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.get("/api/checkins/{cid}")
    def get_checkin(cid: str, role: str = "clinician"):
        with store.session() as s:
            ck = s.get(store.Checkin, cid)
            if not ck:
                raise HTTPException(404, "unknown check-in")
            res = scoring.result_json(ck)
        return scoring.patient_view(res) if role == "patient" else res

    # ------------------------------------------------------------------ clinician
    @app.get("/api/patients")
    def patients():
        out = []
        with store.session() as s:
            for p in s.exec(select(store.Patient)):
                cks = scoring.history(s, p.id)
                last = cks[-1] if cks else None
                open_alerts = list(s.exec(select(store.Alert).where(store.Alert.patient_id == p.id,
                                                                    store.Alert.acked_at == None)))  # noqa: E711
                status = last.status if last else "BASELINE"
                if any(a.kind == "SAFETY" for a in open_alerts):
                    status = "SAFETY"
                out.append({"id": p.id, "display_name": p.display_name, "simulated": p.simulated, "status": status,
                            "last_checkin": last.created_at.isoformat() + "Z" if last else None,
                            "open_alerts": len(open_alerts), "checkins": len(cks)})
        out.sort(key=lambda r: (SEVERITY.get(r["status"], 9), r["display_name"]))
        return out

    @app.get("/api/patients/{pid}/timeline")
    def timeline(pid: str, days: int = Query(30, ge=1, le=365)):
        with store.session() as s:
            p = s.get(store.Patient, pid)
            if not p:
                raise HTTPException(404, "unknown patient")
            cks = scoring.history(s, pid)
            cutoff = store.utcnow() - timedelta(days=days)
            bl = s.get(store.Baseline, pid)
            alerts = list(s.exec(select(store.Alert).where(store.Alert.patient_id == pid)
                                 .order_by(store.Alert.created_at)))
            rows = []
            for c in cks:
                if c.created_at < cutoff:
                    continue
                full = scoring.result_json(c)
                rows.append({k: full.get(k) for k in ("checkin_id", "created_at", "status", "quality", "drift",
                                                      "timing", "timing_chips", "reference_risk", "evidence",
                                                      "summary", "simulated", "transcript", "safety")})
                rows[-1]["acted"] = (c.extra or {}).get("acted", False)
                rows[-1]["environment"] = (c.extra or {}).get("environment")
            tau = rt().dm.tau
            baseline = None
            if bl and bl.frozen_at:
                baseline = {"n": bl.n, "frozen_at": bl.frozen_at.isoformat() + "Z", "timing_mu": bl.timing_mu,
                            "timing_sigma": bl.timing_sigma, "tau": tau}
            return {"patient": {"id": p.id, "display_name": p.display_name, "simulated": p.simulated},
                    "baseline": baseline, "tau": tau, "checkins": rows,
                    "alerts": [{"id": a.id, "checkin_id": a.checkin_id, "kind": a.kind,
                                "created_at": a.created_at.isoformat() + "Z", "acked_by": a.acked_by,
                                "acked_at": a.acked_at.isoformat() + "Z" if a.acked_at else None,
                                "note": a.note} for a in alerts]}

    @app.post("/api/alerts/{aid}/ack")
    def ack(aid: str, body: AckBody):
        with store.session() as s:
            a = s.get(store.Alert, aid)
            if not a:
                raise HTTPException(404, "unknown alert")
            a.acked_by, a.acked_at, a.note = body.clinician, store.utcnow(), body.note
            s.add(a)
            s.commit()
            return {"id": a.id, "kind": a.kind, "checkin_id": a.checkin_id, "acked_by": a.acked_by,
                    "acked_at": a.acked_at.isoformat() + "Z", "note": a.note}

    @app.delete("/api/patients/{pid}")
    def delete_patient(pid: str):
        store.purge_patient(pid)
        return {"deleted": pid}

    # ------------------------------------------------------------------ demo
    @app.post("/api/demo/reset")
    def demo_reset(body: ResetBody):
        r = rt()
        names = {"stable": ["stable"], "drift": ["drift"], "all": ["stable", "drift"], "blank": []}
        if body.scenario not in names:
            raise HTTPException(422, "scenario must be stable|drift|all|blank")
        done = [load_scenario(n, r.dm) for n in names[body.scenario]]
        if body.scenario in ("blank", "all"):
            store.purge_patient("p_demo")
            with store.session() as s:
                s.add(store.Patient(id="p_demo", display_name="Demo speaker (live)", consent_at=store.utcnow()))
                s.commit()
            done.append("p_demo")
        return {"patients": done}

    @app.post("/api/demo/compare")
    def demo_compare(body: CompareBody):
        r = rt()
        with store.session() as s:
            ck = s.get(store.Checkin, body.checkin_id)
            if not ck:
                raise HTTPException(404, "unknown check-in")
            if ck.status == "SAFETY":
                raise HTTPException(409, "no scores for safety check-ins")
            return scoring.compare_raw_vs_head(s, ck, r.dm, r.dm_raw)

    # ------------------------------------------------------------------ frontend
    web = ROOT / "web"
    if web.exists():
        app.mount("/static", StaticFiles(directory=web), name="static")

        @app.get("/")
        def index():
            return FileResponse(web / "index.html")

    return app


app = create_app()

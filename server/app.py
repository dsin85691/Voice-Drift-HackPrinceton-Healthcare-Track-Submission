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
from .pipeline import drift as drift_mod
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

    def drift_space(simulated: bool):
        """(head, raw) drift models: the fallback space for SIMULATED data, the loaded model for real recordings."""
        if simulated:
            fb = drift_mod.fallback_model()
            return fb, fb
        return rt().dm, rt().dm_raw
    app.state.holder = holder

    # ------------------------------------------------------------------ health
    @app.get("/api/health")
    def health():
        return rt().health()

    @app.get("/api/calibration")
    def calibration():
        """Clinician dashboard: how trustworthy the cohort reference is (reliability diagram, ECE, Brier, AUC)."""
        cal = rt().cohort.cal
        if not cal:
            return {"available": False, "reason": "no calibration.json in the loaded artifacts yet"}
        keep = ("model", "method", "session_windows", "n_sessions", "n_speakers", "source_prevalence", "bands",
                "metrics", "gate_passed", "label", "caveats")
        return {"available": True, **{k: cal[k] for k in keep if k in cal}}

    @app.get("/api/prompt")
    def prompt():
        return {"prompt_id": "default", "text": PROMPTS["default"]}

    # ------------------------------------------------------------------ check-ins
    @app.post("/api/patients/{pid}/checkins", status_code=202)
    async def post_checkin(pid: str, audio: UploadFile = File(...), prompt_id: str = Form("default"),
                           interventions: str = Form("")):
        try:   # noise events marked live during recording (JSON list); malformed input is ignored, never fatal
            ivs = json.loads(interventions) if interventions else []
            ivs = ivs if isinstance(ivs, list) else []
        except ValueError:
            ivs = []
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
        threading.Thread(target=r.process, args=(cid, pid, path, prompt_id, ivs[:20]), daemon=True).start()
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
                            "data_mode": "synthetic_demo" if p.simulated else "real_recording",
                            "last_checkin": last.created_at.isoformat() + "Z" if last else None,
                            "last_quality": (last.quality or {}).get("verdict") if last else None,
                            "open_alerts": len(open_alerts), "checkins": len(cks),
                            "valid_checkins": sum(1 for c in cks if (c.quality or {}).get("verdict") in ("PASS", "WARN")
                                                  and c.status != "SAFETY")})
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
                                                      "timing", "timing_chips", "reference_risk", "cohort_risk", "has_window_analysis", "evidence",
                                                      "summary", "simulated", "transcript", "safety")})
                rows[-1]["acted"] = (c.extra or {}).get("acted", False)
                rows[-1]["environment"] = (c.extra or {}).get("environment")
                rows[-1]["data_mode"] = "synthetic_demo" if (c.extra or {}).get("simulated") else "real_recording"
                rows[-1]["baseline_progress"] = (c.extra or {}).get("baseline_progress")
            tau = (drift_space(p.simulated)[0]).tau
            baseline = None
            if bl and bl.frozen_at:
                baseline = {"n": bl.n, "frozen_at": bl.frozen_at.isoformat() + "Z", "timing_mu": bl.timing_mu,
                            "timing_sigma": bl.timing_sigma, "tau": tau}
            return {"patient": {"id": p.id, "display_name": p.display_name, "simulated": p.simulated,
                                "data_mode": "synthetic_demo" if p.simulated else "real_recording"},
                    "baseline": baseline, "tau": tau, "checkins": rows,
                    "baseline_n_required": int(get_config().drift["baseline_n"]),
                    "drift_level": "session",
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
        # simulated scenarios live in the fallback drift space they were generated in (never the loaded model's)
        done = [load_scenario(n, drift_space(True)[0]) for n in names[body.scenario]]
        if body.scenario in ("blank", "all"):
            store.purge_patient("p_demo")
            with store.session() as s:
                s.add(store.Patient(id="p_demo", display_name="Demo speaker (live)", consent_at=store.utcnow()))
                s.commit()
            done.append("p_demo")
        return {"patients": done}

    @app.get("/api/checkins/{cid}/windows")
    def checkin_windows(cid: str):
        """Window-level raw-vs-projected analysis of one stored check-in (clinician view)."""
        with store.session() as s:
            ck = s.get(store.Checkin, cid)
            if not ck:
                raise HTTPException(404, "unknown check-in")
            if ck.status == "SAFETY":
                raise HTTPException(409, "no scores for safety check-ins")
            wa = (ck.extra or {}).get("window_analysis")
            if not wa:
                raise HTTPException(404, "no window analysis for this check-in (seeded or recorded before the feature)")
            from .pipeline.windows import contract
            sim = (ck.extra or {}).get("simulated", False)
            return {"checkin_id": cid, "patient_id": ck.patient_id, "mode": "live_recording",
                    "simulated": sim, "created_at": ck.created_at.isoformat() + "Z",
                    "versions": ck.versions, **contract(wa, "synthetic_demo" if sim else "real_recording")}

    @app.post("/api/demo/replay")
    async def demo_replay(patient_id: str = Form(...), audio: UploadFile = File(...),
                          kind: str = Form("babble_self"), start_seconds: float = Form(...),
                          end_seconds: float = Form(...), snr_db: float = Form(5.0), seed: int = Form(0),
                          noise_file: UploadFile | None = File(None)):
        """Controlled replay: analyse the same clean speech with and without a known noise interval.
        Analysis only, nothing is stored; the uploaded audio is deleted after processing."""
        from .pipeline import noise as N
        r = rt()
        if kind not in N.KINDS:
            raise HTTPException(422, f"kind must be one of {N.KINDS}")
        if not (end_seconds > start_seconds >= 0) or not (-10 <= snr_db <= 40):
            raise HTTPException(422, "need 0 <= start < end and -10 <= snr_db <= 40")
        if kind == "file" and noise_file is None:
            raise HTTPException(422, "kind=file needs a noise_file upload")
        cid = store.new_id("replay")
        clean = r.tmp / f"{cid}_clean{Path(audio.filename or 'a.wav').suffix or '.wav'}"
        clean.write_bytes(await audio.read())
        npath = None
        if noise_file is not None and kind == "file":
            npath = r.tmp / f"{cid}_noise{Path(noise_file.filename or 'n.wav').suffix or '.wav'}"
            npath.write_bytes(await noise_file.read())
        loop = asyncio.get_running_loop()
        try:
            return await loop.run_in_executor(None, lambda: r.replay(
                patient_id, clean, kind=kind, start_s=start_seconds, end_s=end_seconds, snr_db=snr_db,
                seed=seed, noise_path=npath))
        except Exception as e:
            for p in (clean, npath):
                if p:
                    p.unlink(missing_ok=True)
            raise HTTPException(422, f"replay failed: {e}") from e

    @app.get("/api/results")
    def results():
        """Evidence for the Results/Home pages. Nothing here is computed in the browser or invented: the aggregate
        numbers come from research/lite/aggregate.py, the demo from the last real one-click run."""
        cfg = get_config()
        def load(key):
            p = ROOT / cfg.results[key]
            return json.loads(p.read_text()) if p.exists() else None
        d = cfg.demo
        clip = ROOT / d["clip"]
        prof_ready = False
        with store.session() as s:
            bl = s.get(store.Baseline, d["profile"])
            prof_ready = bool(bl and bl.frozen_at)
        return {"aggregate": load("aggregate"), "demo_snapshot": load("demo_snapshot"),
                "demo": {"clip_present": clip.exists(), "profile": d["profile"], "profile_baseline_ready": prof_ready,
                         "noise": d["noise"], "snr_db": d["snr_db"], "start_s": d["start_s"], "end_s": d["end_s"]},
                "model": rt().acoustic.encoder_name,
                "team": list((get_config().get("about") or {}).get("team") or [])}

    @app.post("/api/demo/run")
    async def demo_run():
        """One click: the bundled clip, analysed clean and with the configured noise interval, against the configured
        profile's baseline. Same code path as the Replay lab. The last run is saved as a snapshot (real output)."""
        import shutil
        r, d = rt(), get_config().demo
        clip = ROOT / d["clip"]
        if not clip.exists():
            raise HTTPException(409, f"no demo clip at {d['clip']}: add a consented 40-60 s recording there")
        tmp = r.tmp / f"{store.new_id('demo')}{clip.suffix}"
        shutil.copyfile(clip, tmp)                      # replay deletes its input; never touch the bundled file
        loop = asyncio.get_running_loop()
        out = await loop.run_in_executor(None, lambda: r.replay(
            d["profile"], tmp, kind=d["noise"], start_s=float(d["start_s"]), end_s=float(d["end_s"]),
            snr_db=float(d["snr_db"]), seed=int(d["seed"])))
        out["ran_at"] = store.utcnow().isoformat() + "Z"
        out["model"] = r.acoustic.encoder_name
        snap = ROOT / get_config().results["demo_snapshot"]
        snap.parent.mkdir(parents=True, exist_ok=True)
        snap.write_text(json.dumps(out))
        return out

    @app.post("/api/demo/compare")
    def demo_compare(body: CompareBody):
        r = rt()
        with store.session() as s:
            ck = s.get(store.Checkin, body.checkin_id)
            if not ck:
                raise HTTPException(404, "unknown check-in")
            if ck.status == "SAFETY":
                raise HTTPException(409, "no scores for safety check-ins")
            dm, dmr = drift_space((ck.extra or {}).get("simulated", False))
            return scoring.compare_raw_vs_head(s, ck, dm, dmr)

    # ------------------------------------------------------------------ frontend
    web = ROOT / "web"
    if web.exists():
        app.mount("/static", StaticFiles(directory=web), name="static")

        @app.get("/")
        def index():
            return FileResponse(web / "index.html")

    return app


app = create_app()

"""Fixture server with exactly the real API shapes (spec 7), so the frontend can be built without models.
Run: uvicorn server.mock_server:app --port 8001"""
from __future__ import annotations

import asyncio
import json

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import StreamingResponse

app = FastAPI(title="Voice Drift Monitor (mock)")

RESULT = {
    "checkin_id": "c_mock", "patient_id": "p_demo", "created_at": "2026-11-14T15:02:11Z", "status": "GREEN",
    "quality": {"verdict": "PASS", "speech_s": 41.2, "snr_db": 23.5, "clipping": 0.0, "reasons": []},
    "drift": {"d": 0.94, "tau": 1.31, "exceeds": False, "weight": 1.0, "window_count": 7, "baseline_n": 5,
              "confirmed": False},
    "timing": {"speech_rate_wpm": 148, "pause_rate_pm": 11.2, "mean_pause_s": 0.61,
               "z": {"speech_rate": -0.4, "pause_rate": 0.3, "mean_pause": 0.2}},
    "reference_risk": {"source": "kintsugi-dam-3.1", "level": "low", "raw": 0.21},
    "evidence": [{"item": "sleep", "phq9": 3, "direction": "worse",
                  "quote": "I've been waking up at four every night", "verified": True}],
    "evidence_dropped": 0, "safety": {"triggered": False, "source": None},
    "transcript": "I've been waking up at four every night but otherwise fine.",
    "versions": {"artifacts": "v3", "git": "a1b2c3d", "asr": "small.en-int8"},
    "latency_ms": {"decode": 310, "asr": 3900, "acoustic": 2700, "llm": 4400, "total": 9200},
    "disclaimer": "Screening and monitoring aid — not a diagnosis.",
}


@app.get("/api/health")
def health():
    return {"models_loaded": True, "gpu": False, "artifacts_version": "mock", "llm_reachable": False}


@app.post("/api/patients/{pid}/checkins", status_code=202)
async def post(pid: str, audio: UploadFile = File(...), prompt_id: str = Form("default")):
    return {"checkin_id": "c_mock"}


@app.get("/api/checkins/{cid}/events")
async def events(cid: str):
    async def gen():
        for st in ("quality", "transcript", "scores", "evidence"):
            await asyncio.sleep(0.4)
            yield f"event: stage\ndata: {json.dumps({'stage': st, 'ms': 400})}\n\n"
        yield f"event: result\ndata: {json.dumps(RESULT)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream")


@app.get("/api/checkins/{cid}")
def get(cid: str):
    return RESULT


@app.get("/api/patients/{pid}/timeline")
def timeline(pid: str, days: int = 30):
    return {"baseline": {"n": 5}, "checkins": [{k: RESULT[k] for k in ("checkin_id", "created_at", "status", "quality",
            "drift")}], "alerts": []}


@app.post("/api/alerts/{aid}/ack")
def ack(aid: str, body: dict):
    return {"id": aid, "kind": "DRIFT", "acked_by": body.get("clinician"), "note": body.get("note")}


@app.post("/api/demo/reset")
def reset(body: dict):
    return {"patients": ["p_demo"]}


@app.post("/api/demo/compare")
def compare(body: dict):
    return {"available": True, "raw": {"d": 1.9, "tau": 1.31, "exceeds": True},
            "invariant": {"d": 0.94, "tau": 1.31, "exceeds": False}}

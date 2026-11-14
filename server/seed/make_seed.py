"""Generates the SIMULATED seed scenarios (session vectors + transcripts, no audio).

Run: python -m server.seed.make_seed
These are synthetic stand-ins for the "10 days of real check-ins" in spec 9.2 until the demo speaker's
venue recordings exist. Everything is flagged simulated; the last two `drift` days are flagged acted.
Vectors live in the fallback scale (within-speaker SD ~ drift.sigma_pop). Replace with real session
vectors from the pipeline by re-recording and exporting (see README).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from ..pipeline.language import verify_quote

OUT = Path(__file__).parent
K, D = 16, 128
SD = 0.4

NEUTRAL = [
    "Today was pretty good. I went for a walk after lunch and then caught up on some reading. "
    "I slept okay, maybe seven hours, and I've been looking forward to the weekend with my family.",
    "Work was busy but fine. I had coffee with a friend this morning and we talked about the trip we are planning. "
    "I feel like I have a decent amount of energy and I finished most of what I wanted to get done.",
    "Not much to report. I cooked dinner last night and really enjoyed it. I've been keeping up with my exercise "
    "and I'm sleeping through the night most days.",
]
ACTED = [
    "I guess it's been okay. I've been waking up at four every night and I can't get back to sleep. "
    "I'm so tired all the time. I just don't feel like doing anything, even the things I usually enjoy.",
    "Honestly I haven't really felt like talking to anyone. I have no energy and everything takes so much effort. "
    "I don't have much of an appetite either, I just push the food around.",
]
ACTED_EVIDENCE = [
    [("sleep", 3, "worse", "waking up at four every night"),
     ("energy", 4, "worse", "I'm so tired all the time"),
     ("anhedonia", 1, "worse", "I just don't feel like doing anything")],
    [("energy", 4, "worse", "I have no energy and everything takes so much effort"),
     ("appetite", 5, "worse", "I don't have much of an appetite")],
]
NEUTRAL_EVIDENCE = [("energy", 4, "better", "I feel like I have a decent amount of energy")]


def _ev(items, transcript):
    rows = []
    for label, phq, direction, quote in items:
        if verify_quote(quote, transcript):
            rows.append({"item": label, "phq9": phq, "direction": direction, "quote": quote,
                         "verified": True, "confidence": 0.8})
    return rows


def build(kind: str) -> dict:
    rng = np.random.default_rng(7 if kind == "stable" else 11)
    speaker = rng.normal(0, 1.0, D)           # speaker identity (kept on purpose)
    env_dir = {e: rng.normal(0, 1, D) for e in ("hallway", "dining hall")}
    shift = np.zeros(D)
    shift[:K] = rng.choice([-1, 1], K) * rng.uniform(0.9, 1.5, K)   # "low energy, flat" direction
    days = []
    n_days = 10
    for i in range(n_days):
        env = "quiet room"
        acted = False
        if kind == "stable" and i >= 5:
            env = ["quiet room", "hallway", "dining hall", "hallway", "quiet room"][i - 5]
        if kind == "drift" and i >= 8:
            acted = True
        v = speaker + rng.normal(0, SD, D)
        raw = v.copy()
        if env != "quiet room":
            e = env_dir[env] / np.linalg.norm(env_dir[env][:K]) * np.sqrt(K)
            raw = raw + 0.55 * e          # noise shifts raw features strongly...
            v = v + 0.10 * e              # ...and the invariance head only a little
        if acted:
            v = v + shift * (1.0 if i == 8 else 1.25)
            raw = raw + shift * (1.0 if i == 8 else 1.25)
        snr = {"quiet room": 31, "hallway": 17, "dining hall": 14}[env] + rng.normal(0, 1.2)
        verdict = "PASS" if snr >= 18 else "WARN"
        speech = 42 + rng.normal(0, 3)
        sr, pr, mp = 148 + rng.normal(0, 6), 11.2 + rng.normal(0, 1.0), 0.61 + rng.normal(0, 0.04)
        if acted:
            sr, pr, mp = sr * (0.78 if i == 8 else 0.74), pr * 1.5, mp * 1.45
        if kind == "stable" and i == 7:   # one unusable check-in (dining hall too loud)
            verdict, snr = "REJECT", 9.4
        transcript = (ACTED[i - 8] if acted else NEUTRAL[i % 3])
        ev = _ev(ACTED_EVIDENCE[i - 8], transcript) if acted else (_ev(NEUTRAL_EVIDENCE, transcript) if i % 3 == 1 else [])
        days.append({
            "day_offset": i - n_days + 1,
            "environment": env, "acted": acted,
            "quality": {"verdict": verdict, "speech_s": round(float(speech), 1), "snr_db": round(float(snr), 1),
                        "clipping": 0.0, "reasons": ["too much background noise"] if verdict == "REJECT" else
                        (["some background noise"] if verdict == "WARN" else [])},
            "vec": [round(float(a), 4) for a in v], "vec_raw": [round(float(a), 4) for a in raw],
            "timing": {"speech_rate_wpm": round(float(sr), 1), "pause_rate_pm": round(float(pr), 1),
                       "mean_pause_s": round(float(mp), 2)},
            "transcript": transcript, "evidence": ev,
            "summary": "Reports disrupted sleep, low energy and reduced interest." if acted else "",
        })
    return {"scenario": kind, "patient": {"id": f"p_{kind}",
            "display_name": "Alex (simulated, stable)" if kind == "stable" else "Sam (simulated, drift)"},
            "days": days}


if __name__ == "__main__":
    for k in ("stable", "drift"):
        (OUT / f"{k}.json").write_text(json.dumps(build(k)), encoding="utf-8")
        print("wrote", k)

"""Interpretable acoustic evidence (feature F4): measurements that say HOW speech differs, next to the embedding
distance. Same extraction for baseline and follow-up recordings; a missing measurement is None, never 0.

pause_fraction     share of the speaking interval (first to last detected speech) spent in pauses between speech
                   runs. A pause is a non-speech gap of at least `min_pause_s` (default 200 ms), so short unvoiced
                   sounds inside words are not counted; leading set-up silence and trailing silence are excluded.
speaking_rate_wpm  recognised words per minute of detected speech, only when a transcript with word timings exists
                   (the existing ASR timing features; no new recogniser is added for this).
pitch_variability  not computed: no validated pitch extractor is part of the app, so it is reported as unavailable.

These describe the recording. They are not symptoms and are not claimed to explain the embedding difference.
"""
from __future__ import annotations

import math

import numpy as np

FRAME_S = 0.03            # quality.vad_mask frames are 30 ms
MIN_INTERVAL_S = 4.0
FEATURES = ("pause_fraction", "speaking_rate_wpm", "pitch_variability")
UNITS = {"pause_fraction": "share of speaking time", "speaking_rate_wpm": "words per minute",
         "pitch_variability": "semitones"}
LABELS = {"pause_fraction": "Pause time", "speaking_rate_wpm": "Speaking rate", "pitch_variability": "Pitch variability"}


def pause_fraction(mask: np.ndarray | None, min_pause_s: float = 0.2) -> float | None:
    if mask is None or len(mask) == 0 or not mask.any():
        return None
    idx = np.flatnonzero(mask)
    a, b = int(idx[0]), int(idx[-1]) + 1                     # speaking interval: first to last speech frame
    if (b - a) * FRAME_S < MIN_INTERVAL_S:
        return None
    seg = mask[a:b]
    min_frames = max(1, math.ceil(min_pause_s / FRAME_S - 1e-9))
    pause, run = 0, 0
    for v in seg:
        if not v:
            run += 1
        else:
            if run >= min_frames:
                pause += run
            run = 0
    return round(pause / len(seg), 4)


def measure(mask: np.ndarray | None, timing: dict | None, transcript_available: bool) -> dict:
    """Current-recording values for each feature: {name: {"value": float|None, "status": ..., "reason": ...}}."""
    pf = pause_fraction(mask)
    out = {"pause_fraction": {"value": pf, "status": "VALID" if pf is not None else "UNAVAILABLE",
                              "reason": None if pf is not None else "speaking interval too short"}}
    wpm = (timing or {}).get("speech_rate_wpm") if transcript_available else None
    out["speaking_rate_wpm"] = {"value": wpm, "status": "VALID" if wpm is not None else "UNAVAILABLE",
                                "reason": None if wpm is not None else
                                ("no transcript" if not transcript_available else "too few recognised words")}
    out["pitch_variability"] = {"value": None, "status": "UNAVAILABLE",
                                "reason": "no validated pitch extractor in this app"}
    return out


def compare(current: dict | None, baseline: list[dict]) -> dict:
    """Baseline mean vs current, per feature. `baseline` = measure() outputs of the baseline check-ins; only those
    with a valid value count. Relative difference only where the baseline mean is meaningfully above zero."""
    out = {}
    for f in FEATURES:
        cur = ((current or {}).get(f) or {}).get("value")
        vals = [b[f]["value"] for b in baseline if b and (b.get(f) or {}).get("value") is not None]
        base = float(np.mean(vals)) if vals else None
        row = {"label": LABELS[f], "unit": UNITS[f], "baseline": None if base is None else round(base, 4),
               "baseline_n": len(vals), "current": cur, "delta": None, "relative": None}
        if cur is None:
            row["status"] = "UNAVAILABLE"
            row["reason"] = ((current or {}).get(f) or {}).get("reason") or "not measured"
        elif base is None:
            row["status"] = "NO_BASELINE"
            row["reason"] = "baseline recordings have no comparable measurement"
        else:
            row["status"] = "VALID"
            row["delta"] = round(cur - base, 4)
            row["relative"] = round((cur - base) / base, 4) if abs(base) > 1e-6 else None
        out[f] = row
    return out


def describe(cmp: dict) -> list[str]:
    """Plain descriptive sentences, no interpretation. Small differences are reported as similar."""
    s = []
    pf = cmp.get("pause_fraction") or {}
    if pf.get("status") == "VALID":
        d = pf["delta"]
        s.append("Your latest recording contained more pause time than your baseline recordings." if d > 0.03 else
                 "Your latest recording contained less pause time than your baseline recordings." if d < -0.03 else
                 "Pause time was similar to your baseline recordings.")
    sr = cmp.get("speaking_rate_wpm") or {}
    if sr.get("status") == "VALID":
        r = sr["relative"]
        if r is not None:
            s.append("You spoke more slowly than in your baseline recordings." if r < -0.08 else
                     "You spoke faster than in your baseline recordings." if r > 0.08 else
                     "Your speaking rate was similar to your baseline recordings.")
    return s

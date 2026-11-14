"""ASR (faster-whisper small.en, int8) + timing features. Degrades to 'unavailable' without the package."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from ..config import get_config

log = logging.getLogger("vd.asr")


@dataclass
class Transcript:
    text: str = ""
    words: list[tuple[str, float, float]] = field(default_factory=list)  # (word, start, end)
    available: bool = True
    engine: str = "small.en-int8"


class ASR:
    def __init__(self):
        self.model = None
        self.engine = "unavailable"
        cfg = get_config()
        try:
            from faster_whisper import WhisperModel

            for dev, ct in (("cuda", "int8"), ("cpu", "int8")):  # CPU fallback (spec 11)
                try:
                    self.model = WhisperModel(cfg.models["asr"], device=dev, compute_type=ct)
                    self.engine = f"{cfg.models['asr']}-{ct}-{dev}"
                    break
                except Exception as e:
                    log.warning("ASR on %s failed: %s", dev, e)
        except ImportError:
            log.warning("faster-whisper not installed: transcripts unavailable")

    def transcribe(self, x: np.ndarray) -> Transcript:
        if self.model is None:
            return Transcript(available=False, engine="unavailable")
        segs, _ = self.model.transcribe(x, word_timestamps=True, vad_filter=True, language="en")
        words, parts = [], []
        for s in segs:
            parts.append(s.text.strip())
            for w in s.words or []:
                words.append((w.word.strip(), float(w.start), float(w.end)))
        return Transcript(" ".join(parts), words, True, self.engine)


def timing_features(words: list[tuple[str, float, float]], speech_s: float, pause_min_s: float = 0.25) -> dict:
    """speech_rate_wpm (words / speech-minute), pause_rate_pm (pauses > 250 ms / minute),
    mean_pause_s. Returns None values when there are too few words."""
    if len(words) < 10 or speech_s <= 0:
        return {"speech_rate_wpm": None, "pause_rate_pm": None, "mean_pause_s": None}
    gaps = [b[1] - a[2] for a, b in zip(words, words[1:])]
    pauses = [g for g in gaps if g > pause_min_s]
    span_min = max((words[-1][2] - words[0][1]) / 60.0, 1e-6)
    return {
        "speech_rate_wpm": round(len(words) / (speech_s / 60.0), 1),
        "pause_rate_pm": round(len(pauses) / span_min, 1),
        "mean_pause_s": round(float(np.mean(pauses)), 2) if pauses else 0.0,
    }

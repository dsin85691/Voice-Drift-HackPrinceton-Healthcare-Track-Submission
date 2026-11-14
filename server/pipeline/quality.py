"""Quality gate: VAD, SNR estimate, clipping -> PASS / WARN / REJECT (spec 5.1 step 3)."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

SR = 16000
FRAME = 480  # 30 ms


@dataclass
class Quality:
    verdict: str
    speech_s: float
    snr_db: float
    clipping: float
    reasons: list[str] = field(default_factory=list)
    speech_mask: np.ndarray | None = None  # per-frame bool, not serialized
    duration_s: float = 0.0
    rms: float = 0.0

    def to_json(self) -> dict:
        frac = self.speech_s / self.duration_s if self.duration_s > 0 else 0.0
        return {"verdict": self.verdict,
                "status": "REJECT" if self.verdict == "REJECT" else "PASS",   # PASS includes usable-with-warning
                "duration_seconds": round(self.duration_s, 2), "speech_seconds": round(self.speech_s, 2),
                "speech_fraction": round(frac, 3), "clipping_fraction": round(self.clipping, 4),
                "rms_amplitude": round(self.rms, 5),
                "speech_s": round(self.speech_s, 1), "snr_db": round(self.snr_db, 1), "clipping": round(self.clipping, 4),
                "snr_method": "estimate: speech frames vs non-speech frames of the same recording",
                "reasons": self.reasons, "reason_codes": [REASON_CODES.get(r, r) for r in self.reasons]}


def _frames(x: np.ndarray) -> np.ndarray:
    n = len(x) // FRAME
    return x[: n * FRAME].reshape(n, FRAME)


def vad_mask(x: np.ndarray) -> np.ndarray:
    """Per-30ms-frame speech mask.

    Silero VAD (ONNX) is used when the `silero_vad` package is installed; otherwise an adaptive
    energy VAD (floor = low percentile of frame energy, threshold = floor + 9 dB, with hangover).
    """
    fr = _frames(x)
    if len(fr) == 0:
        return np.zeros(0, bool)
    try:  # pragma: no cover - optional dependency
        from silero_vad import get_speech_timestamps, load_silero_vad

        ts = get_speech_timestamps(_to_tensor(x), load_silero_vad(onnx=True), sampling_rate=SR)
        m = np.zeros(len(fr), bool)
        for t in ts:
            m[t["start"] // FRAME: t["end"] // FRAME + 1] = True
        return m
    except Exception:
        pass
    e = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    floor = np.percentile(e, 10)
    peak = np.percentile(e, 95)
    # margin above the floor adapts to the dynamic range: 9 dB on clean audio, down to 3 dB when noise
    # compresses the range (speech at 5-10 dB SNR is only ~4-10 dB above the noise floor). Frames of pure noise
    # that slip through are caught by the SNR check, which rejects the window/recording.
    margin = float(np.clip(0.5 * (peak - floor), 3.0, 9.0))
    thr = max(floor + margin, min(peak - 25.0, floor + 20.0))
    m = e > thr
    # bridge gaps up to 90 ms (hangover): close then drop isolated single-frame blips
    closed = np.convolve(m.astype(float), np.ones(4), mode="same") > 0
    dens = np.convolve(m.astype(float), np.ones(3), mode="same")
    return closed & (dens >= 2)


def _to_tensor(x):  # pragma: no cover
    import torch

    return torch.from_numpy(x)


def estimate_snr_db(x: np.ndarray, mask: np.ndarray) -> float:
    fr = _frames(x)
    mask = mask[: len(fr)]
    if mask.sum() == 0:
        return -20.0
    p_speech = float(np.mean(fr[mask] ** 2))
    if (~mask).sum() < 3:
        # no non-speech frames: use the quietest 10% of speech frames as the noise proxy
        e = np.sort(np.mean(fr ** 2, axis=1))
        p_noise = float(np.mean(e[: max(1, len(e) // 10)]))
    else:
        p_noise = float(np.mean(fr[~mask] ** 2))
    return 10 * np.log10((p_speech + 1e-12) / (p_noise + 1e-12))


def assess(x: np.ndarray, cfg) -> Quality:
    q = cfg.quality
    mask = vad_mask(x)
    speech_s = float(mask.sum() * FRAME / SR)
    snr = estimate_snr_db(x, mask)
    clipping = float(np.mean(np.abs(x) >= 0.99)) if len(x) else 0.0
    dur = len(x) / SR
    rms = float(np.sqrt(np.mean(np.asarray(x, np.float64) ** 2))) if len(x) else 0.0
    rej, warn = [], []
    if speech_s < max(q["reject_speech_s"], q.get("min_speech_seconds", 0)):
        rej.append("not enough speech")
    elif dur > 0 and speech_s / dur < q.get("min_speech_fraction", 0):
        rej.append("too much silence")
    if snr < q["reject_snr_db"]:
        rej.append("too much background noise")
    if clipping > q["reject_clipping"]:
        rej.append("audio is distorted (too loud)")
    if not rej:
        if speech_s < q["warn_speech_s"]:
            warn.append("short speech")
        if snr < q["warn_snr_db"]:
            warn.append("some background noise")
        if clipping > q["warn_clipping"]:
            warn.append("slight distortion")
    verdict = "REJECT" if rej else "WARN" if warn else "PASS"
    return Quality(verdict, speech_s, snr, clipping, rej or warn, mask, dur, rms)


REASON_CODES = {
    "not enough speech": "INSUFFICIENT_SPEECH", "too much silence": "EXCESSIVE_SILENCE",
    "too much background noise": "LOW_SNR_ESTIMATE", "audio is distorted (too loud)": "CLIPPING",
    "recording too short": "TOO_SHORT", "recording too long": "TOO_LONG", "unreadable audio": "UNREADABLE_AUDIO",
    "processing error": "PROCESSING_ERROR", "short speech": "SHORT_SPEECH", "some background noise": "SOME_NOISE",
    "slight distortion": "SLIGHT_CLIPPING"}

PATIENT_MESSAGES = {   # actionable, non-alarming (NHS error-message style); never about the person's health
    "not enough speech": "Not enough speech was detected. Check that your microphone is working and read the whole passage.",
    "too much silence": "Most of the recording was silent. Start reading soon after you press Start, and stop when you finish.",
    "recording too short": "The recording was too short for analysis. Please record for at least 20 seconds.",
    "recording too long": "The recording was longer than 90 seconds. Please keep it to about a minute.",
    "too much background noise": "Recording quality was insufficient for this comparison. Try again somewhere quieter.",
    "audio is distorted (too loud)": "The audio level was too high and parts may be distorted. Move the microphone back a little.",
    "unreadable audio": "We couldn't read this recording. It was not saved; please record again.",
    "processing error": "Something went wrong while analysing this recording. It was not analysed; please try again.",
}


def patient_message(reasons: list[str]) -> str:
    return " ".join(PATIENT_MESSAGES.get(r, r) for r in reasons)

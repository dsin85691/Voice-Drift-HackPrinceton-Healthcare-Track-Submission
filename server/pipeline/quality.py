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

    def to_json(self) -> dict:
        return {"verdict": self.verdict, "speech_s": round(self.speech_s, 1),
                "snr_db": round(self.snr_db, 1), "clipping": round(self.clipping, 4),
                "reasons": self.reasons}


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
    thr = max(floor + 9.0, min(peak - 25.0, floor + 20.0))
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
    rej, warn = [], []
    if speech_s < q["reject_speech_s"]:
        rej.append("not enough speech")
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
    return Quality(verdict, speech_s, snr, clipping, rej or warn, mask)


PATIENT_MESSAGES = {
    "not enough speech": "We couldn't hear enough speech - try speaking for the full minute.",
    "too much background noise": "Too much background noise - try a quieter spot.",
    "audio is distorted (too loud)": "The recording was too loud - move the microphone back a little.",
}


def patient_message(reasons: list[str]) -> str:
    return " ".join(PATIENT_MESSAGES.get(r, r) for r in reasons)

"""Decode uploaded audio (webm/opus, wav, ...) to 16 kHz mono float32. No loudness normalization."""
from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import numpy as np

SR = 16000


class DecodeError(Exception):
    pass


def _decode_av(path: Path) -> np.ndarray:
    import av  # PyAV bundles ffmpeg libs, so no system ffmpeg is needed

    chunks = []
    with av.open(str(path)) as container:
        stream = next((s for s in container.streams if s.type == "audio"), None)
        if stream is None:
            raise DecodeError("no audio stream")
        resampler = av.AudioResampler(format="flt", layout="mono", rate=SR)
        for frame in container.decode(stream):
            for f in resampler.resample(frame):
                chunks.append(f.to_ndarray().reshape(-1))
        for f in resampler.resample(None) or []:
            chunks.append(f.to_ndarray().reshape(-1))
    if not chunks:
        raise DecodeError("empty audio")
    return np.concatenate(chunks).astype(np.float32)


def _decode_ffmpeg(path: Path) -> np.ndarray:
    exe = shutil.which("ffmpeg")
    if not exe:
        raise DecodeError("ffmpeg not installed")
    out = subprocess.run(
        [exe, "-v", "error", "-i", str(path), "-f", "f32le", "-ac", "1", "-ar", str(SR), "-"],
        capture_output=True, check=False,
    )
    if out.returncode != 0:
        raise DecodeError(out.stderr.decode(errors="ignore")[:200])
    return np.frombuffer(out.stdout, dtype=np.float32).copy()


def decode_audio(path: str | Path) -> np.ndarray:
    path = Path(path)
    try:
        x = _decode_av(path)
    except Exception as e_av:
        try:
            x = _decode_ffmpeg(path)
        except Exception:
            raise DecodeError(f"unreadable audio: {e_av}") from None
    return np.nan_to_num(x)


def write_wav(path: str | Path, x: np.ndarray, sr: int = SR) -> None:
    """Minimal 16-bit PCM writer (tests / seeds)."""
    import wave

    pcm = (np.clip(x, -1, 1) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


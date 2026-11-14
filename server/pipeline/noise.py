"""Deterministic noise interventions for the controlled-replay demo (visualization spec section 8, Mode 1).

The same clean speech is analysed twice: as recorded, and with a known noise added over a known interval at a
known SNR. Every noise is generated from a seed, so the same inputs always give the same result.
"""
from __future__ import annotations

import numpy as np

SR = 16000
KINDS = ("pink", "speech_shaped", "babble_self", "file")


def _shape(white: np.ndarray, mag: np.ndarray) -> np.ndarray:
    X = np.fft.rfft(white)
    return np.fft.irfft(X * mag[: len(X)], len(white))


def make_noise(kind: str, n: int, seed: int, ref: np.ndarray | None = None, noise_file: np.ndarray | None = None):
    """kind:
    - pink:          1/f noise (broadband, steady)
    - speech_shaped: white noise with the long-term spectrum of `ref` (steady, speech-like spectrum, no words)
    - babble_self:   6 time-shifted, time-reversed copies of `ref` summed (speech-like, unintelligible babble
                     made from the same recording; no third-party voices needed)
    - file:          an uploaded noise recording, looped / cropped deterministically"""
    rng = np.random.default_rng(seed)
    if kind == "pink":
        f = np.fft.rfftfreq(n, 1 / SR)
        return _shape(rng.standard_normal(n), 1 / np.sqrt(np.maximum(f, 20.0)))
    if kind == "speech_shaped":
        if ref is None or len(ref) < 1024:
            raise ValueError("speech_shaped noise needs the reference recording")
        mag = np.abs(np.fft.rfft(np.resize(ref, n)))
        mag = np.convolve(mag, np.ones(64) / 64, mode="same")             # smooth: keep the envelope only
        return _shape(rng.standard_normal(n), mag / (mag.max() + 1e-12))
    if kind == "babble_self":
        if ref is None or len(ref) < SR:
            raise ValueError("babble_self needs at least 1 s of reference speech")
        r = ref[::-1]
        out = np.zeros(n)
        for _ in range(6):
            out += np.roll(np.resize(r, n), int(rng.integers(0, len(r))))
        return out
    if kind == "file":
        if noise_file is None or not len(noise_file):
            raise ValueError("noise file is empty")
        start = int(rng.integers(0, max(1, len(noise_file))))
        return np.resize(np.roll(noise_file, -start), n)
    raise ValueError(f"unknown noise kind {kind!r}; use one of {KINDS}")


def mix_interval(clean: np.ndarray, noise: np.ndarray, start_s: float, end_s: float, snr_db: float,
                 speech_mask: np.ndarray | None = None, ramp_s: float = 0.05) -> np.ndarray:
    """Add `noise` to `clean` only inside [start_s, end_s), scaled so that speech power / noise power = snr_db
    over that interval (speech frames from the VAD mask when available). 50 ms raised-cosine ramps, no clicks."""
    x = clean.astype(np.float64).copy()
    a, b = int(start_s * SR), min(len(x), int(end_s * SR))
    if b <= a:
        return x.astype(np.float32)
    seg = x[a:b]
    if speech_mask is not None and len(speech_mask):
        m = np.repeat(speech_mask, 480)[a:b]
        m = np.pad(m, (0, len(seg) - len(m)))[: len(seg)].astype(bool)
        p_sig = float(np.mean(seg[m] ** 2)) if m.sum() > 480 else float(np.mean(seg ** 2))
    else:
        p_sig = float(np.mean(seg ** 2))
    nz = np.resize(noise, len(seg)).astype(np.float64)
    p_n = float(np.mean(nz ** 2)) + 1e-20
    nz *= np.sqrt(p_sig / (p_n * 10 ** (snr_db / 10)))
    r = min(int(ramp_s * SR), len(seg) // 2)
    if r > 0:
        w = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, r))
        nz[:r] *= w
        nz[-r:] *= w[::-1]
    x[a:b] = seg + nz
    return np.clip(x, -1.0, 1.0).astype(np.float32)

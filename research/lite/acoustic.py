"""Interpretable acoustic targets, synthetic speech-change view, and mild microphone simulation.

Targets (computed on the CLEAN 16 kHz segment, with a validity flag each):
  pause_fraction   - share of 30 ms frames the VAD calls non-speech            valid if segment >= 4 s
  pitch_var_st     - SD of F0 in semitones over voiced frames (autocorrelation)  valid if >= 0.5 s voiced
  run_rate         - speech-run onsets per second (VAD runs >= 90 ms): a reliable
                     voiced-activity timing measure, used instead of a speaking-rate
                     estimate (no ASR on Androids)                               valid if >= 2 s of speech
"""
from __future__ import annotations

import numpy as np

SR = 16000
FRAME = 480          # 30 ms VAD frames
TARGETS = ("pause_fraction", "pitch_var_st", "run_rate")


def _vad(x: np.ndarray) -> np.ndarray:
    from server.pipeline.quality import vad_mask
    return vad_mask(x)


def _runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) frame index pairs of True runs."""
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            out.append((i, j))
            i = j
        else:
            i += 1
    return out


def f0_track(x: np.ndarray, fmin: float = 70.0, fmax: float = 400.0, win: int = 640, hop: int = 160,
             thresh: float = 0.5) -> np.ndarray:
    """Normalized-autocorrelation F0 per 40 ms frame (10 ms hop); NaN where unvoiced."""
    n = 1 + max(0, (len(x) - win) // hop)
    lo, hi = int(SR / fmax), int(SR / fmin)
    f0 = np.full(n, np.nan)
    e_all = np.array([np.sum(x[i * hop:i * hop + win] ** 2) for i in range(n)])
    gate = np.percentile(e_all, 30) if n else 0
    for i in range(n):
        fr = x[i * hop:i * hop + win]
        if len(fr) < win or e_all[i] <= gate:
            continue
        fr = fr - fr.mean()
        ac = np.correlate(fr, fr, mode="full")[win - 1:]
        if ac[0] <= 0:
            continue
        ac = ac / ac[0]
        lag = lo + int(np.argmax(ac[lo:hi]))
        if ac[lag] >= thresh:
            f0[i] = SR / lag
    return f0


def acoustic_targets(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Returns (values[3], valid_mask[3]) for one clean 16 kHz segment."""
    vals, ok = np.zeros(3), np.zeros(3, bool)
    m = _vad(x)
    if len(x) >= 4 * SR and len(m):
        vals[0], ok[0] = 1.0 - float(m.mean()), True
    f0 = f0_track(x)
    v = f0[~np.isnan(f0)]
    if len(v) >= 50:
        st = 12 * np.log2(v / 100.0)
        vals[1], ok[1] = float(np.std(st)), True
    speech_s = m.sum() * FRAME / SR
    if speech_s >= 2.0:
        runs = [r for r in _runs(m) if r[1] - r[0] >= 3]
        vals[2], ok[2] = len(runs) / (len(x) / SR), True
    return vals, ok


def insert_pauses(x: np.ndarray, extra_s: float = 0.4) -> np.ndarray:
    """Synthetic 'more pauses' delivery: after every speech run, insert `extra_s` of the recording's OWN room tone
    (its quietest frames), never digital silence, so the pause sounds like the room, not like padding.
    Same speaker, same words, same conditions; only the pausing changes. Labelled synthetic in every report."""
    m = _vad(x)
    runs = [r for r in _runs(m) if r[1] - r[0] >= 3]
    if not runs:
        return x.copy()
    fr = x[: len(x) // FRAME * FRAME].reshape(-1, FRAME)
    quiet = fr[np.argsort(np.mean(fr ** 2, axis=1))[: max(3, len(fr) // 10)]].reshape(-1)
    n_extra = int(extra_s * SR)
    tone = np.resize(quiet, n_extra)
    fade = np.minimum(1, np.minimum(np.arange(n_extra), np.arange(n_extra)[::-1]) / 160)   # 10 ms fades
    tone = tone * fade
    out, prev = [], 0
    for _, e in runs:
        cut = min(len(x), e * FRAME)
        out.append(x[prev:cut]); out.append(tone)
        prev = cut
    out.append(x[prev:])
    return np.concatenate(out).astype(np.float32)


def mic_sim(x: np.ndarray, rng, max_db: float = 6.0) -> np.ndarray:
    """Mild random microphone frequency response: smooth gain curve through 6 log-spaced control points
    (+-max_db), applied in the frequency domain. Length and timing are unchanged."""
    X = np.fft.rfft(x)
    f = np.fft.rfftfreq(len(x), 1 / SR)
    ctrl_f = np.geomspace(100, 7000, 6)
    g_db = np.array([rng.uniform(-max_db, max_db) for _ in ctrl_f])
    gain = 10 ** (np.interp(np.log(np.maximum(f, 50)), np.log(ctrl_f), g_db) / 20)
    return np.fft.irfft(X * gain, len(x)).astype(np.float32)

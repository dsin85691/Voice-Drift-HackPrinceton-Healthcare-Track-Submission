"""Seeded GPU augmentation (spec 3.2). Mixing and RIR convolution run in torch (FFT convolution).
No pitch/speed/tempo changes (they would alter depression-related acoustics).

Env classes: 0 clean, 1 stationary noise, 2 music, 3 babble, 4 reverb, 5 phone. Test-corrupt copies use
HELD-OUT noise files and real RIRs at 20/10/5 dB and carry env = -1 (never used in training).
"""
from __future__ import annotations

import functools
import json
import random

import numpy as np
import torch

from .extract import seed_for

SR = 16000
ENV = {"clean": 0, "noise": 1, "music": 2, "babble": 3, "reverb": 4, "phone": 5}
TRAIN_CLASSES = ["noise", "music", "babble", "reverb", "phone"]
TEST_SNRS = [20, 10, 5]
N_TRAIN_AUG = 4


def fft_convolve(x: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    n = x.shape[-1] + h.shape[-1] - 1
    nf = 1 << (n - 1).bit_length()
    y = torch.fft.irfft(torch.fft.rfft(x, nf) * torch.fft.rfft(h, nf), nf)
    return y[..., : x.shape[-1]]


def mix_at_snr(sig: torch.Tensor, noise: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Scale `noise` so that 10*log10(P_sig / P_noise) == snr_db, then add."""
    if noise.shape[-1] < sig.shape[-1]:
        noise = noise.repeat(int(np.ceil(sig.shape[-1] / max(1, noise.shape[-1]))))
    noise = noise[: sig.shape[-1]]
    ps, pn = sig.pow(2).mean(), noise.pow(2).mean().clamp_min(1e-12)
    return sig + noise * torch.sqrt(ps / (pn * 10 ** (snr_db / 10)))


def simulated_rir(rng: random.Random, sr: int = SR) -> torch.Tensor:
    """Exponentially decaying noise RIR, RT60 ~ U(0.2, 0.8) s."""
    rt60 = rng.uniform(0.2, 0.8)
    n = int(rt60 * sr)
    g = torch.Generator().manual_seed(rng.getrandbits(31))
    t = torch.arange(n) / sr
    h = torch.randn(n, generator=g) * torch.exp(-6.9 * t / rt60)
    h[0] = 1.0
    return h / h.norm()


def phone_sim(x: torch.Tensor, sr: int = SR) -> torch.Tensor:
    """300-3400 Hz band-pass (equivalent to 8 kHz resample + band-pass) then 8-bit mu-law companding."""
    X = torch.fft.rfft(x)
    f = torch.fft.rfftfreq(x.shape[-1], 1 / sr).to(x.device)
    y = torch.fft.irfft(X * ((f >= 300) & (f <= 3400)).to(X.dtype), x.shape[-1])
    peak = y.abs().max().clamp_min(1e-6)
    v = y / peak
    mu = 255.0
    comp = torch.sign(v) * torch.log1p(mu * v.abs()) / np.log1p(mu)
    q = torch.round(comp * 127) / 127
    return torch.sign(q) * ((1 + mu) ** q.abs() - 1) / mu * peak


@functools.lru_cache(maxsize=4096)
def _duration(path: str) -> float:
    """sf.info opens the file; on a Drive mount that costs ~0.1-0.5 s, so do it once per noise file."""
    import soundfile as sf
    return sf.info(path).duration


def _load(path: str, n: int, rng: random.Random, device) -> torch.Tensor:
    """A random n-sample (16 kHz) excerpt of a noise file, at any stored sample rate."""
    from .extract import read_16k

    dur = _duration(path)
    need = n / SR
    t0 = rng.uniform(0, max(0.0, dur - need))
    x = read_16k(path, t0, min(dur, t0 + need))[:n]
    return torch.from_numpy(np.ascontiguousarray(x)).to(device)


class Augmenter:
    def __init__(self, noise_index: str | dict, device: str = "cpu"):
        self.idx = json.load(open(noise_index)) if isinstance(noise_index, str) else noise_index
        self.device = device

    def copy_plan(self, kinds: list[str]):
        plan = []
        if "clean" in kinds:
            plan.append(("clean", 0, None))
        if "train" in kinds:
            plan += [("train", i, None) for i in range(1, 1 + N_TRAIN_AUG)]
        if "test" in kinds:
            plan += [("test", 100 + i, snr) for i, snr in enumerate(TEST_SNRS)]
        return plan

    def apply(self, wavs: list[torch.Tensor], meta: list[tuple]):
        out, envs = [], []
        for w, (wid, kind, idx, snr) in zip(wavs, meta):
            rng = random.Random(seed_for(wid, idx))     # hash(window_id, copy_idx)
            if kind == "clean":
                out.append(w); envs.append(0)
            elif kind == "train":
                out.append(self._train_aug(w, rng)); envs.append(self._last_env)
            else:
                out.append(self._test_corrupt(w, rng, snr)); envs.append(-1)
        return out, envs

    def _noise_clip(self, pool: str, n: int, rng: random.Random):
        return _load(rng.choice(self.idx[pool]), n, rng, self.device)

    def _train_aug(self, w: torch.Tensor, rng: random.Random) -> torch.Tensor:
        cls = rng.choice(TRAIN_CLASSES)
        self._last_env = ENV[cls]
        snr = rng.uniform(5, 20)
        n = w.shape[-1]
        if cls == "noise":
            y = mix_at_snr(w, self._noise_clip("noise_train", n, rng), snr)
        elif cls == "music":
            y = mix_at_snr(w, self._noise_clip("music_train", n, rng), snr)
        elif cls == "babble":
            b = sum(self._noise_clip("speech_train", n, rng) for _ in range(rng.randint(3, 5)))
            y = mix_at_snr(w, b, snr)
        elif cls == "reverb":
            y = fft_convolve(w, simulated_rir(rng).to(w.device))
        else:
            y = phone_sim(w)   # channel effects (reverb, phone) carry no additive noise
        return y * 10 ** (rng.uniform(-6, 6) / 20)        # volume +-6 dB

    def _test_corrupt(self, w: torch.Tensor, rng: random.Random, snr: float) -> torch.Tensor:
        n = w.shape[-1]
        if self.idx.get("rir_test") and rng.random() < 0.5:
            from .extract import read_16k
            h = torch.from_numpy(read_16k(rng.choice(self.idx["rir_test"]))).to(w.device)
            w = fft_convolve(w, h / h.norm())
        return mix_at_snr(w, self._noise_clip("noise_test", n, rng), snr)

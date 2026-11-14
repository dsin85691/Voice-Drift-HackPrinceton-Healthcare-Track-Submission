"""Acoustic branch: audio -> session vector v (128-d) via WavLM + P_M3, plus the "raw z" vector for
the robustness toggle. If WavLM / artifacts are not available it falls back to a documented
spectral embedding so the whole app still runs (versions.encoder says which one was used)."""
from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass

import numpy as np

from ..config import ROOT, get_config, resolve

log = logging.getLogger("vd.acoustic")
SR = 16000
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from research.extract import window_audio, window_bounds  # noqa: E402  (shared with training: parity)


@dataclass
class Embedding:
    v: np.ndarray        # head / invariant session vector, (128,)
    v_raw: np.ndarray    # raw-z session vector (for /demo/compare)
    window_count: int
    encoder: str
    logit: float | None = None   # mean window logit of the disease head (None without trained artifacts)
    # window level (same 8 s windows the session vector is built from): seconds + one vector per window
    win_bounds: list[tuple[float, float]] | None = None
    W: np.ndarray | None = None        # projected / invariant, [n_windows, d]
    W_raw: np.ndarray | None = None    # raw, [n_windows, d_raw]


# ----------------------------------------------------------------------------- spectral fallback
_N_MEL, _NFFT, _HOP = 40, 512, 160


def _mel_fb() -> np.ndarray:
    def hz2mel(f): return 2595 * np.log10(1 + f / 700)
    def mel2hz(m): return 700 * (10 ** (m / 2595) - 1)
    pts = mel2hz(np.linspace(hz2mel(50), hz2mel(7600), _N_MEL + 2))
    bins = np.floor((_NFFT + 1) * pts / SR).astype(int)
    fb = np.zeros((_N_MEL, _NFFT // 2 + 1))
    for i in range(_N_MEL):
        l, c, r = bins[i], bins[i + 1], bins[i + 2]
        for j in range(l, c):
            fb[i, j] = (j - l) / max(1, c - l)
        for j in range(c, r):
            fb[i, j] = (r - j) / max(1, r - c)
    return fb


_FB = _mel_fb()
_RNG = np.random.default_rng(1234)
_PROJ = np.linalg.qr(_RNG.standard_normal((128, 128)))[0][:, :80]  # fixed 80 -> 128 orthonormal map


def _mel_power(x: np.ndarray) -> np.ndarray:
    n = 1 + (len(x) - _NFFT) // _HOP
    idx = np.arange(_NFFT)[None, :] + _HOP * np.arange(n)[:, None]
    fr = x[idx] * np.hanning(_NFFT)[None, :]
    P = np.abs(np.fft.rfft(fr, axis=1)) ** 2
    return P @ _FB.T  # (frames, mel)


def spectral_embeddings(x: np.ndarray, speech_mask: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
    """(invariant v, raw v), both (128,). Raw = log-mel mean/std. Invariant = same after subtracting a
    per-band noise floor estimated from non-speech frames (a classical channel/noise normalization)."""
    P = _mel_power(x.astype(np.float64)) + 1e-10
    if speech_mask is not None and len(speech_mask):
        # 30 ms mask -> 10 ms frames
        m = np.repeat(speech_mask, 3)[: len(P)]
        m = np.pad(m, (0, len(P) - len(m)), constant_values=False)
    else:
        m = np.ones(len(P), bool)
    if m.sum() < 5:
        m = np.ones(len(P), bool)

    def stats(Pm: np.ndarray) -> np.ndarray:
        L = np.log(Pm[m])
        f = np.concatenate([L.mean(0), L.std(0)])
        return f

    raw = stats(P)
    floor = P[~m].mean(0) if (~m).sum() >= 5 else np.percentile(P, 10, axis=0)
    inv = stats(np.maximum(P - floor[None, :], 0.03 * P))
    # fixed centering constants so vectors are O(1); within-speaker SD is set by sigma_pop in config
    mk = lambda f: _PROJ @ ((f - np.concatenate([np.full(_N_MEL, -8.0), np.full(_N_MEL, 2.0)])) / 2.5)
    return mk(inv), mk(raw)


# ----------------------------------------------------------------------------- WavLM path
def load_projection(adir):
    """The deployed projection: the lightweight head (projection_head.pt, no diagnosis labels) if present,
    else the M3 projector (projector.pt). Always loaded with weights_only=True."""
    import torch

    from research.heads import ProjectionHead, Projector

    if (adir / "projection_head.pt").exists():
        m, f = ProjectionHead(), adir / "projection_head.pt"
    else:
        m, f = Projector(), adir / "projector.pt"
    m.load_state_dict(torch.load(f, map_location="cpu", weights_only=True))
    return m.eval()


class _WavLMEncoder:
    def __init__(self, adir, dm_meta: dict):
        import torch

        from research.extract import load_wavlm
        from research.heads import Projector

        cfg = get_config()
        self.torch = torch
        self.dev = "cuda" if torch.cuda.is_available() else "cpu"
        self.layer = int(dm_meta["layer"])
        self.model = load_wavlm(cfg.models["wavlm"], cfg.models["wavlm_revision"], self.dev)
        self.proj = load_projection(adir)
        self.f_d = None
        if (adir / "disease_head.pt").exists():
            self.f_d = torch.nn.Linear(128, 1)
            self.f_d.load_state_dict(torch.load(adir / "disease_head.pt", map_location="cpu", weights_only=True))
            self.f_d.eval()

    def embed(self, wins: list[np.ndarray]):
        """Returns per-window (projected [n,128], raw z [n,768]) and the mean window logit."""
        from research.extract import layer_means

        z = layer_means(self.model, wins, device=self.dev)[:, self.layer, :]   # fp32 on the GTX 1650
        with self.torch.no_grad():
            ht = self.proj(self.torch.from_numpy(z))
            lg = float(self.f_d(ht).squeeze(-1).mean()) if self.f_d is not None else None   # mean WINDOW logit
        return ht.numpy(), z, lg


class AcousticBranch:
    def __init__(self):
        self.encoder_name = "spectral-fallback"
        self._enc: _WavLMEncoder | None = None
        cfg = get_config()
        adir = resolve(cfg.artifacts["dir"])
        cands = sorted(adir.glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1)
        if cands and ((cands[-1] / "projector.pt").exists() or (cands[-1] / "projection_head.pt").exists()):
            try:
                meta = json.loads((cands[-1] / "drift_config.json").read_text())
                self._enc = _WavLMEncoder(cands[-1], meta)
                self.encoder_name = f"wavlm-base+{meta.get('head', 'P_M3')}@{cands[-1].name}"
            except Exception as e:  # missing torch/transformers/weights
                log.warning("WavLM encoder unavailable (%s) - using spectral fallback", e)

    def embed(self, x: np.ndarray, speech_mask: np.ndarray | None) -> Embedding:
        bounds = window_bounds(len(x))
        wins = [x[a:b] for a, b in bounds]
        secs = [(round(a / SR, 3), round(b / SR, 3)) for a, b in bounds]
        if self._enc is not None and wins:
            W, W_raw, lg = self._enc.embed(wins)
            # session vector = mean of the window vectors (unchanged behaviour)
            return Embedding(W.mean(0), W_raw.mean(0), len(wins), self.encoder_name, lg, secs, W, W_raw)
        v, raw = spectral_embeddings(x, speech_mask)      # session vector: whole recording (unchanged)
        W, W_raw = [], []
        from .quality import vad_mask
        for a, b in bounds:                               # window vectors: same recipe on each 8 s window, with a
            wi, wr = spectral_embeddings(x[a:b], vad_mask(x[a:b]))   # window-LOCAL VAD (independent windows)
            W.append(wi); W_raw.append(wr)
        return Embedding(v, raw, max(1, len(wins)), self.encoder_name, None, secs,
                         np.array(W) if W else None, np.array(W_raw) if W_raw else None)


# ----------------------------------------------------------------------------- cohort reference (calibrated)
class CohortRisk:
    """Platt-calibrated output of our own disease head, as an Androids-COHORT reference for the clinician.
    Shown only if calibration.json exists AND its E1 gate passed (session AUC CI above 0.5). Never shown to the
    patient, never combined with drift, never used to set the status."""

    def __init__(self, adir=None):
        self.cal = None
        if adir is None:
            cands = sorted(resolve(get_config().artifacts["dir"]).glob("v*"),
                           key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else -1)
            adir = cands[-1] if cands else None
        if adir is not None and (adir / "calibration.json").exists():
            self.cal = json.loads((adir / "calibration.json").read_text())

    @property
    def available(self) -> bool:
        return bool(self.cal and self.cal.get("gate_passed"))

    def score(self, session_logit: float | None) -> dict:
        base = {"source": "androids-cohort-" + (self.cal or {}).get("model", "M3"),
                "label": (self.cal or {}).get("label", "Androids-cohort reference")}
        if self.cal is None or session_logit is None:
            return {**base, "available": False, "reason": "no trained classifier / calibration yet"}
        if not self.cal.get("gate_passed"):
            return {**base, "available": False,
                    "reason": "hidden: the classifier did not beat chance on held-out Androids speakers (E1 gate)"}
        p = float(1.0 / (1.0 + np.exp(-(self.cal["a"] * session_logit + self.cal["b"]))))
        t = self.cal["bands"]
        level = "elevated" if p >= t["elevated"] else "moderate" if p >= t["moderate"] else "low"
        m = self.cal["metrics"]
        return {**base, "available": True, "band": level, "prob": round(p, 3),
                "calibration": {"method": self.cal.get("method", "platt"), "ece": m["ece"], "brier": m["brier"],
                                "auc": m["auc"], "auc_ci": m["auc_ci"]}}


# ----------------------------------------------------------------------------- DAM reference risk
class ReferenceRisk:
    """Kintsugi DAM wrapper. The exact DAM API/thresholds are an open item (spec 13): this loads the
    model only when its package is importable and otherwise reports 'unavailable' (spec 11)."""

    def __init__(self):
        self.model = None
        try:  # pragma: no cover - requires the DAM checkpoint + deps
            import torch
            from huggingface_hub import hf_hub_download

            cfg = get_config()
            ckpt = hf_hub_download(cfg.models["dam_repo"], cfg.models["dam_ckpt"],
                                   revision=cfg.models["dam_revision"])
            state = torch.load(ckpt, map_location="cpu", weights_only=True)  # never a full unpickle
            from featex import DAM  # type: ignore  # vendored from the model card
            self.model = DAM(state)
        except Exception as e:
            log.info("DAM reference risk unavailable: %s", e)

    def score(self, x: np.ndarray) -> dict:
        if self.model is None:
            return {"source": "unavailable", "level": None, "raw": None}
        raw = float(self.model.predict(x))  # pragma: no cover
        return {"source": "kintsugi-dam-3.1", "level": _dam_level(raw), "raw": round(raw, 3)}


def _dam_level(raw: float) -> str:
    # Replace with DAM's published ordinal thresholds once verified (open item, spec 13).
    return "low" if raw < 0.33 else "moderate" if raw < 0.66 else "high"

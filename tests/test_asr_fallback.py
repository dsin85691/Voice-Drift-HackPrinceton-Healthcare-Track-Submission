"""ASR must never fail a check-in: CUDA errors at inference (e.g. libcublas missing) fall back to CPU, then to no transcript."""
import numpy as np

from server.pipeline import asr as asr_mod


class _Seg:
    text, words = " hello", []


def _asr(engine, run):
    a = asr_mod.ASR.__new__(asr_mod.ASR)
    a.model, a.engine = object(), engine
    a._run = run
    return a


def test_cuda_inference_error_switches_to_cpu(monkeypatch):
    calls = []

    def run(x):
        calls.append(a.engine)
        if a.engine.endswith("-cuda"):
            raise RuntimeError("Library libcublas.so.12 is not found or cannot be loaded")
        return asr_mod.Transcript("hello", [], True, a.engine)

    a = _asr("small.en-int8-cuda", run)

    def load(devices):
        a.model, a.engine = object(), "small.en-int8-cpu"
    a._load = load
    t = a.transcribe(np.zeros(16000, np.float32))
    assert t.available and t.text == "hello" and calls == ["small.en-int8-cuda", "small.en-int8-cpu"]


def test_asr_error_degrades_to_unavailable():
    def run(x):
        raise RuntimeError("boom")
    a = _asr("small.en-int8-cpu", run)
    t = a.transcribe(np.zeros(16000, np.float32))
    assert t.available is False

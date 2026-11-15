import sys
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from server import store  # noqa: E402
from server.config import get_config  # noqa: E402
from server.pipeline.decode import write_wav  # noqa: E402

SR = 16000


def speechlike(seconds=30, snr_db=30.0, seed=0, burst=(1.5, 0.5), amp=0.2):
    """Bursty harmonic signal (voiced 'syllables' separated by short gaps) in white noise at ~snr_db."""
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    f0 = 140 + 15 * np.sin(2 * np.pi * 0.3 * t)
    sig = sum(np.sin(2 * np.pi * h * np.cumsum(f0) / SR) / h for h in range(1, 12))
    period = sum(burst)
    gate = ((t % period) < burst[0]).astype(float)
    gate = np.convolve(gate, np.ones(320) / 320, mode="same")
    sig = sig * gate
    sig = amp * sig / (np.sqrt(np.mean(sig[gate > 0.5] ** 2)) + 1e-9)
    p_sig = np.mean(sig[gate > 0.5] ** 2)
    noise = rng.standard_normal(len(t)) * np.sqrt(p_sig / 10 ** (snr_db / 10))
    return (sig + noise).astype(np.float32)


@pytest.fixture()
def tmp_env(tmp_path, monkeypatch):
    cfg = get_config()
    monkeypatch.setitem(cfg["server"], "tmp_dir", str(tmp_path / "tmp"))
    url = f"sqlite:///{(tmp_path / 't.db').as_posix()}"
    monkeypatch.setitem(cfg["server"], "db_url", url)
    # never depend on whatever trained artifacts happen to be on this machine (artifacts/ is git-ignored)
    monkeypatch.setitem(cfg["artifacts"], "dir", str(tmp_path / "no_artifacts"))
    # demos: no background warm-up during tests, and snapshots never written into the repo's results/
    monkeypatch.setitem(cfg["demo"], "warm_on_start", False)
    monkeypatch.setitem(cfg["results"], "demo_snapshot", str(tmp_path / "demo_snapshot.json"))
    monkeypatch.setitem(cfg["results"], "sample_snapshot", str(tmp_path / "sample_snapshot.json"))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "")
    store.reset_engine(url)
    return tmp_path


@pytest.fixture()
def client(tmp_env):
    from fastapi.testclient import TestClient

    from server.app import create_app

    with TestClient(create_app()) as c:
        c.tmp_path = tmp_env / "tmp"
        yield c


def wait_result(c, cid, timeout=30):
    t0 = time.time()
    while time.time() - t0 < timeout:
        r = c.get(f"/api/checkins/{cid}")
        if r.status_code == 200:
            return r.json()
        time.sleep(0.1)
    raise TimeoutError(cid)


@pytest.fixture()
def wav_bytes(tmp_path):
    def make(**kw):
        p = tmp_path / "x.wav"
        write_wav(p, speechlike(**kw))
        return p.read_bytes()
    return make



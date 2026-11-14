"""Train/serve parity (spec 10): the same wav through research/extract.py and server/pipeline/acoustic.py
must give cosine >= 0.999. Needs torch + transformers + the cached WavLM weights + exported artifacts, so it
is skipped on machines without them (it runs on the demo laptop: `pytest tests/test_parity.py -rs`)."""
import json

import numpy as np
import pytest

pytest.importorskip("torch")
pytest.importorskip("transformers")

from server.config import get_config, resolve  # noqa: E402
from tests.conftest import speechlike  # noqa: E402


def test_server_matches_research_extract():
    adir = resolve(get_config().artifacts["dir"])
    cands = sorted(adir.glob("v*"))
    if not cands or not (cands[-1] / "projector.pt").exists():
        pytest.skip("no exported artifacts yet")
    import torch

    from research.extract import layer_means, load_wavlm, window_audio
    from research.heads import Projector
    from server.pipeline.acoustic import AcousticBranch

    x = speechlike(24, snr_db=25)
    br = AcousticBranch()
    if br._enc is None:
        pytest.skip("WavLM encoder could not load")
    layer = json.loads((cands[-1] / "drift_config.json").read_text())["layer"]
    model = load_wavlm(get_config().models["wavlm"], get_config().models["wavlm_revision"], "cpu")
    z = layer_means(model, window_audio(x), device="cpu")[:, layer]
    P = Projector(); P.load_state_dict(torch.load(cands[-1] / "projector.pt", weights_only=True)); P.eval()
    with torch.no_grad():
        ref = P(torch.from_numpy(z)).numpy().mean(0)
    got = br.embed(x, None).v
    cos = float(ref @ got / (np.linalg.norm(ref) * np.linalg.norm(got)))
    assert cos >= 0.999

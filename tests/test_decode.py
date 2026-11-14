import numpy as np
import pytest

from server.pipeline.decode import decode_audio, write_wav
from tests.conftest import SR, speechlike


def _encode_webm(x, path):
    av = pytest.importorskip("av")
    out = av.open(str(path), "w", format="webm")
    st = out.add_stream("libopus", rate=48000)
    st.layout = "mono"
    import fractions
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)[None, :]
    frame = av.AudioFrame.from_ndarray(pcm, format="s16", layout="mono")
    frame.sample_rate = SR
    frame.time_base = fractions.Fraction(1, SR)
    rs = av.AudioResampler(format="fltp", layout="mono", rate=48000, frame_size=st.codec_context.frame_size or 960)
    for f in rs.resample(frame):
        for p in st.encode(f):
            out.mux(p)
    for p in st.encode(None):
        out.mux(p)
    out.close()


def test_decode_wav_roundtrip(tmp_path):
    x = speechlike(5)
    write_wav(tmp_path / "a.wav", x)
    y = decode_audio(tmp_path / "a.wav")
    assert abs(len(y) - len(x)) < 10 and np.corrcoef(x, y[: len(x)])[0, 1] > 0.999


def test_decode_webm_opus(tmp_path):
    x = speechlike(5)
    _encode_webm(x, tmp_path / "a.webm")
    y = decode_audio(tmp_path / "a.webm")
    assert 4.8 * SR < len(y) < 5.3 * SR and y.dtype == np.float32

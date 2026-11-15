"""Extra coverage for things the main suites don't pin: frontend integrity, wording rules, protocol/quality edge cases,
recent regressions (seed dates, baseline date), results-file consistency, and the offline WavLM loader."""
import json
import re
from datetime import timedelta
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
APP_JS = (ROOT / "web" / "app.js").read_text(encoding="utf-8")


# ------------------------------------------------------------------ frontend integrity
def test_frontend_script_parses():
    """A syntax error in app.js blanks the whole app. Parse it (newer syntax down-levelled for the ES2017 parser)."""
    esprima = pytest.importorskip("esprima")
    src = APP_JS.replace("?.", ".").replace("??", "||")
    src = re.sub(r"catch\s*\{", "catch (_e) {", src)
    esprima.parseModule(src)


def test_every_icon_the_frontend_uses_is_defined():
    """Regression: icon("chart") rendered an empty tile because "chart" was missing from the icon set."""
    block = APP_JS[APP_JS.index("const IC = {"): APP_JS.index("};", APP_JS.index("const IC = {"))]
    defined = set(re.findall(r"^\s+(\w+):\s*'", block, re.M))
    used = set(re.findall(r'icon\("(\w+)"', APP_JS))
    assert used <= defined, f"icons used but not defined: {sorted(used - defined)}"


def test_frontend_never_hardcodes_old_minimum_or_stale_hints():
    assert "demo/clip.wav" not in APP_JS                      # outdated hint removed
    assert re.search(r"const MIN_S = (\d+)", APP_JS).group(1) == "23"   # Stop unlocks with headroom over min_s


def test_recording_minimum_has_headroom_over_server_minimum():
    from server.config import get_config
    ui_min = int(re.search(r"const MIN_S = (\d+)", APP_JS).group(1))
    assert ui_min - float(get_config().audio["min_s"]) >= 3        # browser audio comes out up to ~2 s short


# ------------------------------------------------------------------ wording rules (never contradict the numbers)
@pytest.mark.parametrize("delta,rel,expect", [
    (0.03, 0.22, "more pause time"), (-0.03, -0.20, "less pause time"),
    (0.005, 0.04, "similar"), (0.019, 0.12, "more pause time"),          # relative change alone is enough
])
def test_pause_sentence_matches_the_shown_change(delta, rel, expect):
    from server.pipeline import evidence
    s = evidence.describe({"pause_fraction": {"status": "VALID", "delta": delta, "relative": rel}})
    assert expect in s[0]
    if expect != "similar":
        assert f"{'+' if rel > 0 else ''}{round(rel * 100)}%" in s[0]


@pytest.mark.parametrize("rel,expect", [(-0.34, "more slowly"), (0.10, "faster"), (0.03, "similar")])
def test_speaking_rate_sentence(rel, expect):
    from server.pipeline import evidence
    s = evidence.describe({"speaking_rate_wpm": {"status": "VALID", "relative": rel, "delta": 1}})
    assert expect in s[0]


def test_unavailable_features_produce_no_sentence():
    from server.pipeline import evidence
    assert evidence.describe({"pause_fraction": {"status": "UNAVAILABLE"}}) == []


# ------------------------------------------------------------------ quality / protocol edge cases
def test_quality_status_and_reason_codes():
    from server.pipeline import quality
    q = quality.Quality("WARN", 20.0, 15.0, 0.0, ["some background noise"], None, 30.0, 0.05).to_json()
    assert q["status"] == "PASS" and q["reason_codes"] == ["SOME_NOISE"] and q["speech_fraction"] == pytest.approx(0.667, abs=1e-3)
    r = quality.Quality("REJECT", 5.0, 3.0, 0.02, ["not enough speech", "audio is distorted (too loud)"], None, 30, 0.1).to_json()
    assert r["status"] == "REJECT" and r["reason_codes"] == ["INSUFFICIENT_SPEECH", "CLIPPING"]
    assert all(k in quality.PATIENT_MESSAGES for k in quality.REASON_CODES if k not in ("short speech", "some background noise", "slight distortion"))


def test_protocol_legacy_and_compatibility():
    from server import protocol, store
    legacy = store.Checkin(id="x", patient_id="p", extra={})
    assert protocol.of_checkin(legacy)["prompt_id"] == protocol.LEGACY_ID
    std = protocol.meta(protocol.STANDARD_ID)
    assert protocol.compatible(std, protocol.meta(protocol.STANDARD_ID))
    assert not protocol.compatible(std, protocol.of_checkin(legacy))
    assert protocol.get("nope")["prompt_id"] == protocol.LEGACY_ID      # unknown ids never crash


# ------------------------------------------------------------------ evidence headline: undetected changes said so
def test_headline_marks_a_change_detected_at_the_clean_rate_as_undetected():
    from research.lite.aggregate import sentence
    r = lambda x: {"rate": x}
    rates = {"P1": {"clean": r(0.11), "noise_10dB": r(0.19), "pause_change": r(0.53), "delivery_change": r(0.11)},
             "B0": {"clean": r(0.17), "noise_10dB": r(0.86), "pause_change": r(1.0), "delivery_change": r(0.17)}}
    s = sentence(rates, "P1", "B0", "delivery_change")
    assert "synthetic pauses were detected in 100% versus 53%" in s
    assert "20% slower delivery in 17% versus 11% (about the same rate as on unchanged speech" in s
    rates["P1"]["delivery_change"] = r(0.60)
    assert "neither model detects it" not in sentence(rates, "P1", "B0", "delivery_change")


# ------------------------------------------------------------------ regressions: dates
def test_seeded_checkins_are_never_in_the_future_and_baseline_is_dated_at_its_fifth_recording(client):
    from server import store
    client.post("/api/demo/reset", json={"scenario": "all"})
    now = store.utcnow()
    for pid in ("p_drift", "p_stable"):
        tl = client.get(f"/api/patients/{pid}/timeline", params={"days": 365}).json()
        times = [c["created_at"] for c in tl["checkins"]]
        assert all(t[:19] <= (now + timedelta(seconds=1)).isoformat()[:19] for t in times)
        fifth = [c for c in tl["checkins"] if c["quality"]["verdict"] == "PASS"][4]
        assert tl["baseline"]["frozen_at"] == fifth["created_at"]


# ------------------------------------------------------------------ results file consistency
def test_results_flags_calibration_files_from_a_different_run(client, tmp_path, monkeypatch):
    from server.config import get_config
    cfg = get_config()
    monkeypatch.setitem(cfg["results"], "aggregate", str(tmp_path / "aggregate.json"))
    (tmp_path / "aggregate.json").write_text(json.dumps({"generated_at": "A", "headline": None}))
    for n in ("calibration", "evaluation_metrics", "evaluation_metadata"):
        (tmp_path / f"{n}.json").write_text(json.dumps({"generated_at": "A", "source_commit": "abc"}))
    ca = client.get("/api/results").json()["calibration_artifacts"]
    assert ca["present"] and ca["consistent"] and ca["source_commit"] == "abc"
    (tmp_path / "calibration.json").write_text(json.dumps({"generated_at": "B"}))   # stale file from another run
    assert client.get("/api/results").json()["calibration_artifacts"]["consistent"] is False
    (tmp_path / "evaluation_metrics.json").unlink()
    assert client.get("/api/results").json()["calibration_artifacts"]["present"] is False


# ------------------------------------------------------------------ server odds and ends
def test_unknown_prompt_and_unknown_checkin_are_clear_errors(client, wav_bytes):
    assert client.get("/api/prompt", params={"prompt_id": "nope"}).status_code == 404
    assert client.get("/api/checkins/nope").status_code == 404
    assert client.get("/api/checkins/nope/verification").status_code == 404
    assert client.get("/api/patients/nope/timeline").status_code == 404


def test_patient_view_never_leaks_review_or_risk_fields(client):
    client.post("/api/demo/reset", json={"scenario": "drift"})
    cid = client.get("/api/patients/p_drift/timeline").json()["checkins"][-1]["checkin_id"]
    pv = client.get(f"/api/checkins/{cid}", params={"role": "patient"}).json()
    for k in ("review_flag", "review_required", "cohort_risk", "reference_risk", "drift", "evidence", "transcript"):
        assert k not in pv, k


# ------------------------------------------------------------------ offline WavLM loader (runs where transformers is installed)
def test_offline_wavlm_loader_uses_cached_safetensors(tmp_path, monkeypatch):
    pytest.importorskip("transformers")
    pytest.importorskip("safetensors")
    import os
    from research.extract import _wavlm_from_cache
    hub = Path(os.environ.get("HF_HUB_CACHE") or Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    if not list((hub / "models--microsoft--wavlm-base").glob("snapshots/*/model.safetensors")):
        pytest.skip("wavlm-base safetensors not cached on this machine")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    m = _wavlm_from_cache("microsoft/wavlm-base", "main")
    assert type(m).__name__ == "WavLMModel" and sum(p.numel() for p in m.parameters()) > 90_000_000

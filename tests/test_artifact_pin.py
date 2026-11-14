"""Regression: every loader must use the SAME artifact version (config artifacts.version). Before this fix the encoder/
projection and the cohort calibration took the newest artifacts/v* folder while the drift constants honoured the pin,
so a v2 projector could be scored against v1 drift constants."""
import json

from server.config import get_config


def _mk(root, name, files):
    d = root / name
    d.mkdir(parents=True)
    for f, txt in files.items():
        (d / f).write_text(txt)
    return d


def test_pinned_version_is_used_by_every_loader(tmp_path, monkeypatch):
    from server.pipeline import acoustic, drift
    cfg = get_config()
    _mk(tmp_path, "v1", {"drift_config.json": json.dumps({"tau": 1.0})})
    _mk(tmp_path, "v2", {"drift_config.json": json.dumps({"tau": 2.0}), "calibration.json": json.dumps({"gate_passed": True})})
    monkeypatch.setitem(cfg["artifacts"], "dir", str(tmp_path))
    monkeypatch.setitem(cfg["artifacts"], "version", "v1")
    assert drift.artifact_dir().name == "v1"
    assert acoustic.CohortRisk().cal is None                    # v2's calibration must NOT leak into a v1 deployment
    monkeypatch.setitem(cfg["artifacts"], "version", "auto")
    assert drift.artifact_dir().name == "v2"
    assert acoustic.CohortRisk().cal == {"gate_passed": True}

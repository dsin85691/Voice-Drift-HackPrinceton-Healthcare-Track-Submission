import json

import numpy as np
import pytest

pytest.importorskip("torch")

from research import calibration as cal  # noqa: E402


def _synth(n=4000, a=1.5, b=-0.5, seed=0):
    rng = np.random.default_rng(seed)
    z = rng.normal(0, 2, n)
    y = (rng.random(n) < cal.sigmoid(a * z + b)).astype(int)
    return z, y


def test_platt_recovers_true_parameters():
    z, y = _synth()
    a, b = cal.platt_fit(z, y)
    assert abs(a - 1.5) < 0.15 and abs(b + 0.5) < 0.15


def test_calibration_fixes_overconfident_logits():
    z, y = _synth()
    z_over = 3.0 * z                                    # an overconfident model: same ranking, 3x the logit
    a, b = cal.platt_fit(z_over, y)
    p = cal.platt_apply(z_over, a, b)
    assert cal.ece(p, y) < 0.03 < cal.ece(cal.sigmoid(z_over), y)
    assert cal.brier(p, y) < cal.brier(cal.sigmoid(z_over), y)


def test_metrics_basics():
    y = np.array([0, 1, 0, 1])
    assert cal.brier(y.astype(float), y) == 0 and cal.ece(y.astype(float), y) == 0
    rel = cal.reliability(np.linspace(0, 1, 50), np.zeros(50))
    assert sum(r["n"] for r in rel) == 50 and len(rel) == cal.N_BINS
    assert cal.prior_shift(0.0, 0.5, 0.5) == 0.0 and cal.prior_shift(0.0, 0.55, 0.1) < 0


def test_session_logits_group_three_windows_per_speaker():
    spk = np.array(["a"] * 7 + ["b"] * 3)
    y = np.array([1] * 7 + [0] * 3)
    fold = np.array([1] * 7 + [2] * 3)
    z = np.arange(10, dtype=float)
    S, Y, F, Z = cal.session_logits(z, spk, y, fold)
    assert list(S) == ["a", "a", "b"] and list(Z) == [1.0, 4.0, 8.0]   # 7 windows -> 2 sessions, tail dropped
    assert list(Y) == [1, 1, 0] and list(F) == [1, 1, 2]


def _sessions(signal: float, n_spk=60, per=8, seed=0):
    rng = np.random.default_rng(seed)
    S, Y, F, Z = [], [], [], []
    for i in range(n_spk):
        lab = i % 2
        mu = rng.normal(signal * (lab - 0.5), 0.5)       # speaker-level effect (sessions are correlated)
        for _ in range(per):
            S.append(f"s{i}"); Y.append(lab); F.append(i % 5 + 1); Z.append(mu + rng.normal(0, 0.7))
    return [np.array(v) for v in (S, Y, F, Z)]


def test_fit_calibration_gate_bands_and_cis():
    c = cal.fit_calibration(*_sessions(3.0), B=300)
    m = c["metrics"]
    assert c["gate_passed"] and m["auc"] > 0.8 and m["auc_ci"][0] <= m["auc"] <= m["auc_ci"][1]
    assert 0 <= c["bands"]["moderate"] <= c["bands"]["elevated"] <= 1
    assert c["n_speakers"] == 60 and c["session_windows"] == 3 and c["method"] == "platt"
    assert json.loads(json.dumps(c, default=float))                       # JSON-serializable
    # a classifier with no signal must be hidden by the gate
    assert not cal.fit_calibration(*_sessions(0.0, seed=3), B=300)["gate_passed"]


def test_cohort_risk_server_mapping(tmp_path):
    from server.pipeline.acoustic import CohortRisk
    c = cal.fit_calibration(*_sessions(3.0), B=200)
    (tmp_path / "calibration.json").write_text(json.dumps(c, default=float))
    cr = CohortRisk(tmp_path)
    assert cr.available
    lo, hi = cr.score(-5.0), cr.score(5.0)
    assert lo["available"] and lo["band"] == "low" and hi["band"] == "elevated" and lo["prob"] < hi["prob"]
    assert cr.score(None)["available"] is False                           # fallback encoder: no logit
    c["gate_passed"] = False
    (tmp_path / "calibration.json").write_text(json.dumps(c, default=float))
    out = CohortRisk(tmp_path).score(5.0)
    assert out["available"] is False and "E1 gate" in out["reason"] and "prob" not in out

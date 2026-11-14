import itertools

import numpy as np
import pytest

from server.config import get_config
from server.pipeline import drift as D
from server.pipeline import language, quality, safety
from tests.conftest import SR, speechlike


# ---------------------------------------------------------------- quality gate
@pytest.mark.parametrize("snr", [10, 20, 30])
def test_snr_estimate_within_3db(snr):
    x = speechlike(40, snr_db=snr, seed=snr)
    q = quality.assess(x, get_config())
    assert abs(q.snr_db - snr) <= 3.0, q.snr_db


def test_silence_rejects():
    q = quality.assess(np.zeros(SR * 30, np.float32) + 1e-5 * np.random.randn(SR * 30).astype(np.float32), get_config())
    assert q.verdict == "REJECT"


def test_clipped_sine_rejects():
    t = np.arange(SR * 30) / SR
    x = np.clip(3.0 * np.sin(2 * np.pi * 220 * t), -1, 1).astype(np.float32)
    q = quality.assess(x, get_config())
    assert q.verdict == "REJECT" and q.clipping > 0.01


def test_clean_speechlike_passes():
    assert quality.assess(speechlike(40, snr_db=30), get_config()).verdict == "PASS"


# ---------------------------------------------------------------- drift math
def test_drift_hand_computed():
    mu = np.array([0.0, 1.0, 2.0])
    sg = np.array([1.0, 0.5, 2.0])
    x = np.array([1.0, 2.0, 0.0])            # z = 1, 2, -1  -> mean sq = 6/3 = 2
    assert D.drift_score(x, mu, sg) == pytest.approx(np.sqrt(2.0))


def test_shrinkage_formula_and_zero_sd():
    sp = np.array([0.4, 0.4])
    s = np.array([0.0, 0.8])
    out = D.shrunk_sigma(s, sp, n=5, n0=5)
    assert out[0] == pytest.approx(np.sqrt(0.5 * 0.16))       # s_j = 0 -> never collapses to 0
    assert out[1] == pytest.approx(np.sqrt((5 * 0.64 + 5 * 0.16) / 10))
    mu, sg = D.fit_baseline(np.ones((5, 2)), sp)
    assert np.all(sg > 0) and np.allclose(mu, 1)


def test_alert_rule_truth_table():
    """All 27 PASS/WARN/REJECT triples x all 8 exceed patterns, against a brute-force reference."""
    w = {"PASS": 1.0, "WARN": 0.5}
    for verdicts in itertools.product(["PASS", "WARN", "REJECT"], repeat=3):
        for ex in itertools.product([False, True], repeat=3):
            hist = [(e, w[v]) for v, e in zip(verdicts, ex) if v != "REJECT"]   # REJECTs never scored
            got = D.confirmed_drift(hist)
            exp = sum(w[v] for v, e in zip(verdicts, ex) if v != "REJECT" and e) >= 1.5
            assert got == exp, (verdicts, ex)


def test_alert_examples():
    assert D.confirmed_drift([(True, 1.0), (True, 0.5)])        # 1.5
    assert not D.confirmed_drift([(True, 1.0), (True, 0.0)] [:1])
    assert not D.confirmed_drift([(True, 0.5), (True, 0.5)])     # 1.0
    assert not D.confirmed_drift([(True, 1.0), (False, 1.0), (False, 1.0), (False, 1.0)])  # outside last 3


def test_reject_streak_and_status():
    assert D.reject_streak(["PASS", "REJECT", "REJECT"]) == 2
    assert D.reject_streak(["REJECT", "PASS"]) == 0
    st = lambda **k: D.decide_status(**{**dict(safety=False, baseline_ready=True, confirmed=False, exceeds=False,
                                               dam_high=False, reject_run=0), **k})
    assert st() == "GREEN" and st(exceeds=True) == "AMBER" and st(dam_high=True) == "AMBER"
    assert st(reject_run=2) == "AMBER" and st(confirmed=True, exceeds=True) == "RED"
    assert st(baseline_ready=False) == "BASELINE" and st(safety=True, confirmed=True) == "SAFETY"


# ---------------------------------------------------------------- quote verifier
def test_quote_verifier():
    t = "I've been   waking up at four\nevery night and I'm so Tired."
    assert language.verify_quote("waking up at four every night", t)       # whitespace
    assert language.verify_quote("SO TIRED", t)                            # case
    assert not language.verify_quote("I wake up at four a.m. each night", t)  # paraphrase
    assert not language.verify_quote("", t)


def test_llm_parse_drops_unverified_and_item9():
    t = "I just don't feel like doing anything lately."
    raw = ('{"safety_flag": false, "items": ['
           '{"phq9_item":1,"label":"anhedonia","direction":"worse","quotes":["don\'t feel like doing anything","made up"],"confidence":0.9},'
           '{"phq9_item":9,"label":"self-harm","direction":"worse","quotes":["lately"],"confidence":0.9}],'
           '"summary_for_clinician":"x"}')
    r = language.LanguageBranch.parse("```json\n" + raw + "\n```", t)
    assert [e["quote"] for e in r["evidence"]] == ["don't feel like doing anything"]
    assert r["evidence_dropped"] == 1


# ---------------------------------------------------------------- safety
def test_every_lexicon_phrase_triggers():
    samples = ["I want to kill myself", "thinking about ending it all", "I feel suicidal", "I want to die",
               "I wish I was dead", "everyone is better off without me", "I keep hurting myself",
               "some self-harm", "I might take my own life", "there is no point in living",
               "I don't want to be here", "I can't go on", "I want to end my life", "I'd rather not be alive"]
    for s in samples:
        assert safety.lexicon_match(s), s
    assert len(safety.lexicon_patterns()) == len(samples)
    assert not safety.lexicon_match("I went for a walk and ate lunch")
    assert safety.screen("fine", llm_flag=True) == {"triggered": True, "source": "llm"}
    assert safety.screen("I want to die")["source"] == "lexicon"


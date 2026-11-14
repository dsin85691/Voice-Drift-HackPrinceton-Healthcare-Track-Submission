"""Lightweight projection-head experiment (projection-head spec v0.1): no diagnosis labels.

acoustic.py  - interpretable targets (pause fraction, pitch variability, speech-run rate) + validity masks,
               synthetic pause insertion (Experiment B view), mild microphone simulation
extract.py   - GPU stage L1: frozen WavLM on 5 views per window (clean, 2 train-aug, held-out-noise, pause)
losses.py    - invariance / variance / covariance (VICReg-style) + masked acoustic preservation
train.py     - B0 / B1 / P1, speaker-disjoint validation, Experiments A + B, diagnostics, report, export
"""

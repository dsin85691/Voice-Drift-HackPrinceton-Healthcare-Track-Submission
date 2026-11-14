# Voice Drift Monitor

60-second voice check-in -> quality verdict, drift vs. the person's own baseline, reference risk, and PHQ-9
evidence quoted from the transcript. **Screening and monitoring aid - not a diagnosis.**
Implements `Voice Drift Monitor — Engineering Specification.md`.

## Run it

```
pip install -r requirements.txt
python -m uvicorn server.app:app --host 127.0.0.1 --port 8000     # or: make run
```

- Patient check-in: http://127.0.0.1:8000/  (mic works on localhost)
- Clinician dashboard: http://127.0.0.1:8000/#/clinician
- Judging demo (reset buttons + raw-vs-invariance panel): add `?demo=1`, then press **Reset: all**
- Tests: `python -m pytest -q` (~15 s)

## What runs with and without models

The app degrades per component instead of failing (spec section 11). `/api/health` reports what is live.

| Component | Full | Fallback (what you get with only `requirements.txt`) |
| --- | --- | --- |
| ASR | faster-whisper small.en (`pip install faster-whisper`) | transcript "unavailable"; no timing chips, no evidence, **lexicon safety can't run** |
| Encoder | WavLM + P_M3 from `artifacts/vN/` | documented spectral embedding (log-mel stats, noise-floor-subtracted for the "invariant" path) |
| Evidence + LLM safety flag | `ANTHROPIC_API_KEY` in `.env` (transcript only is sent) | "evidence unavailable" |
| Reference risk | Kintsugi DAM | "unavailable" (hidden from the verdict) |
| VAD | `silero-vad` | adaptive energy VAD |

**Important:** without ASR the lexicon safety screen has nothing to scan. Install faster-whisper before any
live use.

## Lightweight projection head (fast path, no diagnosis labels)

`research/lite/` implements the projection-head spec v0.1. A small head on frozen WavLM learns to stay the
**same under recording nuisances** (invariance + variance + covariance terms, VICReg-style) while still
**predicting interpretable speech characteristics** (pause fraction, pitch variability, speech-run rate). That is
P1. B1 is the same without the acoustic term, and B0 is raw WavLM. It needs about 2,400 clips instead of G1/G2's
21,000, and trains in minutes.

- **Data:** 24 speakers × 20 consecutive windows. Speakers are split before anything is computed: 25% are held
  out for evaluation, and early stopping uses a separate inner split of the training speakers.
- **Five views per window:**
  1. clean;
  2. and 3. two independent nuisance views (noise or babble at 10–30 dB, simulated reverb, mild microphone EQ;
     no pitch or time changes);
  4. a view with held-out noise files (Experiment A);
  5. a **synthetic** extra-pauses view, made by inserting the recording's own room tone after each speech run
     (Experiment B).
- **Evaluation (held-out speakers):** each model is judged against its own normal variation, i.e. distances
  between a speaker's clean windows. The report gives the nuisance false-alert rate, change sensitivity, AUROC for
  change vs nuisance, and sensitivity at a 10% nuisance false-alert rate, with speaker-bootstrap CIs. It also
  covers effective rank and feature spread (collapse checks), per-term loss curves, parameter count and runtime.
- **Output:** `reports/lite_report.md` answers the spec's six questions from the numbers. `artifacts/vN/` holds
  `projection_head.pt` and drift and window thresholds, which the server loads automatically.
- **Deviations from the spec:**
  - The invariance term is averaged over the 128 dimensions instead of summed (`inv_reduction="sum"` restores
    it). Summed, it is ~128× the variance term, the imbalance that stopped M3 from learning.
  - The head has an input standardizer.
  - Experiment B is synthetic until real slow or paused recordings exist.
  - "Speaking rate" is replaced by speech-run rate, since Androids has no transcripts.

**Run it:** see section L in `research/colab_run.py`.

## Check-in workflow (final feature spec v1.0)

- **F1, fixed passage.** `standard_reading_v1` (about 120 words, about 40 s; `server/protocol.py`). It is served
  by `GET /api/checkin/protocol` and stored on every check-in. A check-in whose prompt or protocol differs from the
  baseline's is **not compared**: its status is `INCOMPATIBLE` and it shows a message, not a drift score. Check-ins
  from before this change count as the legacy free-speech task.
- **F2, quality.** The quality result also records duration, speech seconds, speech fraction, clipping fraction,
  RMS and reason codes. It adds a rule rejecting recordings that are more than 65% silence. The existing stricter
  thresholds still apply. Rejected recordings never update the baseline, never get a score and never count
  toward review flags.
- **F3, repeat check-in.** A check-in above the session boundary offers "Repeat Check-In"
  (`GET /api/checkins/{id}/verification`). The repeat is posted with `verification_of`, using the same passage.
  It is classified as REPRODUCED, NOT_REPRODUCED, INCONCLUSIVE or UNCALIBRATED; without a trained boundary the
  result is always UNCALIBRATED. A repeat belongs to its original's episode, so it never counts toward
  persistence or the sustained-difference count.
- **F4, acoustic evidence.** Pause fraction comes from the voice detector: gaps of at least 200 ms between the
  first and last speech, with leading and trailing silence excluded. Speaking rate comes from the existing ASR
  word timings when a transcript exists. Pitch variability is reported as unavailable, because the app has no
  validated pitch tracker. Each recording is compared with the mean of its baseline recordings, using the same
  method.
- **F5, context.** `PATCH /api/checkins/{id}/context` takes tags and a short note. Context is shown to reviewers
  and never changes a score.
- **F6, review.** The review flag engine from alerting spec v1.0 gains a `request_repeat` action: an in-app
  request the person sees on Home; the flag's status is unchanged. Closing or dismissing a flag needs a written
  reason. Nothing is ever delivered outside the app.

## Aggregate experiment (the Results page)

`research/lite/aggregate.py` writes `results/aggregate.json`, which the Results and Home pages read. It compares B0,
B1 and P1 at a **matched operating point**. Each evaluation speaker's boundary is the 90th percentile of the
*other* evaluation speakers' clean session distances (leave-one-speaker-out), set separately for each model, so
every model flags about 10% of clean sessions.

It reports:
- alert rates with speaker-bootstrap intervals, under both the matched boundary and the original training-speaker
  boundary;
- the AUC for changed speech against noise-only (no threshold);
- **paired** differences (P1−B0, P1−B1, B1−B0) on a shared speaker bootstrap;
- the noise sweep.

The primary comparison was fixed on 2026-11-14, before the rerun: P1 vs B0, session level, held-out noise at
10 dB, matched operating point.

Inputs:
- **L1:** clean, noise mix and synthetic pauses.
- **L2:** additive held-out noise at 20, 10, 5 and 0 dB.
- **L3** (optional): 20% slower delivery without a pitch change (WSOLA time-stretch), plus pauses + 10 dB noise and
  slower delivery + 10 dB noise.

The extra extraction is about 360 clips, eval speakers only, and `--cpu` runs it without a GPU or ledger entry:

```
python -m research.lite.extract --manifest manifest.parquet --noise-index noise_index.json --out lite_feats \
    --change-sweep --cpu --path-map <drive prefix>=<local prefix>
python -m research.lite.aggregate --feats lite_feats --checkpoints reports/checkpoints
```

## Inside a recording: raw vs projected, window by window

The main demo visualization (`server/pipeline/windows.py`, `#/windows/<checkin>` and the `?demo=1#/lab` Replay
lab). For each 8 s window of one recording it plots raw-encoder drift and projected (invariance-head) drift, each
divided by **its own** calibrated window threshold, so one **fixed** line at 1 serves both and never moves.

- **Distance:** cosine distance to the patient's baseline reference, which is the mean of their baseline
  *window* vectors.
- **Thresholds:** window-level only, never the session τ.
  - First choice: calibrated from the patient's own quiet baseline, leave-one-session-out, 90th percentile.
  - Fallback: population thresholds exported with the artifacts.
  - Each check-in's analysis is stored when it is computed, so its thresholds are frozen.
- **Windows are independent.** Voice activity detection runs inside each window, so noise added at 30 s cannot
  change the score at 5 s. A test enforces this.
- **Window quality:** a window with less than 3 s of speech, or a local SNR below 5 dB, is a **gap**, never a zero.
- **States:**

  | State | Meaning | Shown |
  | --- | --- | --- |
  | A | Baseline not ready | Progress only |
  | B | Not calibrated | Separate uncalibrated panels, no threshold |
  | C | Calibrated | Both normalized curves and the fixed line at 1 |
  | D | Nothing usable | Rejection reason |
  | E | Partially valid | Valid windows scored, invalid windows as gaps |

- **Shading** is labelled "Model disagreement": an observation, not proof that noise was removed.
- **Replay lab (Mode 1):**
  - The same clean speech is analysed as recorded, and again with a known, seeded noise (babble, speech-shaped,
    pink, or an uploaded file) over a known interval at a known SNR.
  - It is analysis only: nothing is stored, and the audio is deleted.
- **Live mode (Mode 2):** with `?demo=1`, the check-in screen has a "Mark noise start/end" button. The marks are
  stored with the check-in and drawn on its timeline.
- **It never changes** the longitudinal (session) drift, the status, or alerts. It is never shown to the patient
  and is dropped for SAFETY check-ins.

**Expect honest results.** On the **fallback** features (no trained artifacts), the "projected" path is classical
noise-floor subtraction, and in testing it is **not** more robust than raw. Both curves jump inside the noise
interval. The contrast the demo is about needs the trained head from `artifacts/vN`. Patient baselines recorded
before this feature, or with a different encoder, show state A until they are re-recorded. For the demo, use
**about 10 dB SNR**: at 0–5 dB, most noisy windows are (correctly) rejected as gaps.

## Cohort reference risk (calibrated classifier)

Next to within-person **drift**, the clinician view can show a **cohort reference**. It is the output of our own
disease head (M3 on frozen WavLM features), calibrated with Platt scaling and shown as a band: low / moderate /
elevated. It answers "how does this session compare with the Androids research cohort?", **not** "does this
person have depression?".

**How it is built** (`research/calibration.py`, step 6 in `research/colab_run.py`):

1. **Out-of-fold logits only.** `run_cv(..., return_oof=True)` scores every window with the fold model that held
   it out. Calibration is never fitted on training predictions.
2. **Session level, at live length.** Window logits are averaged over 3-window pseudo-sessions (~24 s), because a
   live check-in has 2–7 windows, not a whole 5-minute interview.
3. **Platt scaling** (`p = sigmoid(a * logit + b)`, 2 parameters). 116 speakers is too few for isotonic regression.
4. **Metrics with speaker-bootstrap CIs:** AUC, Brier score, expected calibration error (ECE), before and after
   calibration, plus a reliability diagram. They are stored in `calibration.json` and `metrics.json`, and shown
   in the dashboard under "How reliable is the cohort reference?".
5. **Bands:** elevated ≥ the 80th percentile of control sessions, moderate ≥ the control median.
6. **Gate:** if the session AUC's 95% CI includes 0.5 (spec §11), the card is hidden and says why.

**Rules:** clinician only. The patient view never includes it, it is never combined with drift, it never sets
the status, and it is dropped entirely for SAFETY check-ins. Without trained artifacts it shows "unavailable".

**Known limitations (state these in the pitch):**

- Calibrated to the **source cohort**: Italian interview speech, Androids recording conditions, ~55% depressed.
  It is not corrected for language, microphone or task shift, which no data in this project can measure.
- **No prevalence correction.** `calibration.prior_shift()` exists, but picking a target prevalence is a clinical
  decision, so it is not applied.
- Calibration is fitted on the 5 CV fold models and applied to the all-folds model that is shipped (standard
  practice; the shipped model saw ~25% more speakers and may be slightly sharper).

**Ways to improve on this after the hackathon:**

- **Target-domain recalibration:** refit (or temperature-scale) on labeled English speech recorded the way the
  app records, then apply prevalence (label-shift) correction for the intended screening population.
- **Per-check-in uncertainty:** bootstrap the Platt parameters, or use conformal prediction to report a
  calibrated set or interval instead of one band.
- **Richer calibrators** once there is more data: isotonic regression, Venn-Abers, or beta calibration.
- **Calibrated LLM evidence confidence:** the PHQ-9 evidence items carry an LLM-reported `confidence` that is not
  calibrated. RLCR-style training (reinforcement learning with a calibration-aware reward such as the Brier score),
  or post-hoc recalibration against clinician-labeled transcripts, would make that number meaningful.
- **Calibrating drift alerts:** report the observed false-alert rate per environment (E3) next to τ in the UI.

### Deviation: consistency-loss scale in M2/M3

The spec writes the consistency term as α‖P(z_c) − P(z_n)‖² with α = 1. Summed over the 128 LayerNorm'd
dimensions, that term is about 100× the disease cross-entropy, and in testing M2/M3 **never learned the disease
task** (train AUC ≈ 0.5; early stopping returned near-untrained weights). The implementation uses the **mean** over
the 128 dimensions (equivalent to α = 1/128 on the spec's formula). A regression test guards it. If M3 still
looks unstable on real data, the spec's §11 fallbacks still apply (lower β, or ship M2).

## Check-in length: why 20 seconds

A check-in must be at least **20 s of recording** with at least **15 s of actual speech**. This is a calibration
choice, not a hard limit of the model.

| Setting | Value | Effect |
| --- | --- | --- |
| `audio.min_s` (`config.yaml`) and `MIN_S` (`web/app.js`) | 20 s | Stop is disabled before this; the server rejects shorter uploads. |
| `quality.reject_speech_s` | 15 s | Less speech than this (after removing silence) = REJECT, not scored. |
| `quality.warn_speech_s` | 25 s | Less than this = WARN, scored at half weight in the alert rule. |

**Why not shorter?**

- The encoder cuts audio into 8 s windows and keeps a final partial window only if it is at least 4 s.
  So about 4 s is the technical minimum.
- The drift threshold τ is calibrated so that about 10% of normal sessions exceed it, using sessions of about
  24 s (3 windows). A shorter check-in averages fewer windows, so its session vector is noisier. Normal check-ins
  would then cross τ more often, causing more false alerts and false RED statuses.
- The transcript side needs length too: timing chips need at least 10 words, and PHQ-9 evidence needs the person
  to say something.

**To change it:** edit the three settings above together. For a much shorter check-in (about 8–10 s), τ should
also be recalibrated on 1-window sessions in `research/export.py`, otherwise expect more false alerts.

**Cancel vs Stop:** *Stop & submit* is enabled after 20 s and sends the recording. *Cancel* works at any time and
discards the recording in the browser: nothing is uploaded or stored.

## Layout

```
server/        FastAPI app, SSE, pipeline/ (decode, quality, asr, safety, acoustic, language, drift), store, seeds, mock_server
web/           static frontend (no build step)
research/      Colab-side code: data_prep, augment, extract, train_heads, evaluate, export, budget, heads, stats
safety/        lexicon.yaml
tests/         pytest suite
tools/         budget CLI, artifact fetcher, pre-commit hook
config.yaml    every threshold, model id and timeout
```

## Offline pipeline (Colab)

1. `research/data_prep.py` -> `manifest.parquet`, noise split by file. **Verify the Androids layout first** (spec 13).
2. `research/extract.py --stage G0 ...` (smoke), then G1, G2. Each stage is wrapped by `research/budget.py`
   (ledger, watchdog, `runtime.unassign()` in `finally`). Run `python tools/budget.py --drive ...` before each.
3. On CPU: `train_heads.run_cv` -> `evaluate.e1_table / e2_noise_probe / e3_false_alerts` -> `export.export(...)`
   writes `artifacts/vN/` (projector.pt, pca.npz, pca_raw.npz, drift_config.json, metrics.json). Keep it out of this repo.
4. Copy the bundle to `artifacts/` on the laptop (or `make artifacts ARTIFACT_URL=...`); the server picks the newest `vN`.
   Pin `models.wavlm_revision` and `models.dam_revision` in `config.yaml` to commit hashes.

## Deviations from the spec (and why)

- **Frontend is dependency-free ES modules + inline SVG, not React/Vite/Tailwind/Recharts.** The build machine had
  no Node. Behavior, screens and routes follow section 9; porting to React later doesn't touch the API.
- `checkin` table has an extra `session_vec_raw` column (and `extra` JSON) to support `/demo/compare` after the audio is deleted.
- Added `GET /api/patients`, `GET /api/prompt`, `DELETE /api/patients/{pid}`, and a `?role=patient` view that strips
  every score, transcript and evidence field (the spec's "patient never sees a number").
- `research/stats.py` replaces scikit-learn's LogisticRegression/AUC (the dev machine's app-control policy blocked sklearn).
- Default `tau` / `sigma_pop` in `config.yaml` are for the fallback features (tau = simulated 90th percentile of the
  null); real values come from `drift_config.json`.

## Not done / needs the team

- **Nothing here has been run on real Androids data or a GPU.** The research code is tested on planted synthetic
  features only (CV, leakage assertions, bootstrap, probe, E3, export -> server load). No E1-E3 numbers exist yet.
- Seeded `stable`/`drift` scenarios are **synthetic** (labeled simulated); replace with real venue recordings (spec 9.2).
- Kintsugi DAM wrapper is a stub until its API/thresholds are verified (spec 13). Quality-gate thresholds need tuning
  on venue recordings. Golden-run clips (3 consented recordings) and the train/serve parity test
  (`tests/test_parity.py`, skips without WavLM + artifacts) still need to be run on the demo laptop.
- Consent forms, LLM data-retention terms, and the 988/clinician copy should be reviewed by a human before live use.

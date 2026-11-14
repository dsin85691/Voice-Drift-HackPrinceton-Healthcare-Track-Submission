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

# Voice Drift Monitor — Engineering Specification

Nov 13, 2026 · @Tae

## 1. Summary and scope

We ship a laptop-hosted web app: a 60-second voice check-in gives back a quality verdict, a drift score against the person's own baseline, a reference risk score, and PHQ-9 evidence quoted from the transcript. Total Colab GPU time is capped at **5.0 hours**. To fit that cap, environment invariance is trained as a small projection head on frozen WavLM features rather than by LoRA fine-tuning of the encoder.

**What changed from the research spec**

| Area | Research spec | This build | Why |
| --- | --- | --- | --- |
| Encoder | WavLM + wav2vec2, LoRA on top 4 layers | WavLM-base frozen; all 13 hidden states mean-pooled and cached once | One GPU pass. Layer choice happens on CPU afterwards. |
| Invariance | LoRA + consistency + GRL, 5 folds × 4 models | Projection head P: 768→128 on cached features, trained on CPU in minutes | Same three losses, about 1% of the compute. Five folds cost nothing extra. |
| Model ladder | M0, M1, M2, M3, L0 | L0 (linear probe on raw z), M1 (P + aug), M3 (P + aug + consistency + GRL) | M2 is optional and runs only if time allows (it's a flag). |
| Absolute risk | Androids population model | Kintsugi DAM (Apache-2.0, whisper-small.en backbone, English) | Trained on English speech, no training needed. Shown as a reference, never a diagnosis. |
| Cross-lingual EATD, distillation, team multi-room study | In scope | Cut | Doesn't serve the demo or the minimum success level. |
| Sensitivity demo | None | Acted "flat affect" sessions from a team member, labeled as simulation | Shows that drift *can* fire, without claiming clinical validity. |

**Claims we will make:** (1) frozen SSL features separate depressed from control on Androids (E1, with CI); (2) the invariance head lowers noise-probe accuracy and false drift alerts under noise (E2, E3); (3) the live system runs end to end in under 15 s with safety handling. **Claims we won't make:** that drift tracks real changes in depression severity.

## 2. System architecture

Training runs once on Colab and produces a model bundle under 5 MB. The laptop runs everything at demo time; the only network call is the transcript sent to the LLM.

&#91;embedded content: system architecture · offline training on Colab, online inference on the laptop\]

The safety screen gates every score. Audio goes to the acoustic branch and the transcript to the language branch; both write to SQLite, and the clinician dashboard reads from it.

## 3. Offline pipeline (Colab)

The GPU does one job: one batched WavLM forward pass over every clean, augmented and test-corrupted window, cached to Drive. Everything after that (layer choice, the three models, five-fold CV, drift calibration, evaluation) runs on CPU from the cache.

### 3.1 Data preparation (CPU runtime)

1. Download Androids (3.69 GB) to Drive. Verify it contains the presegmented interview clips, the turn-timing file and `fold-list.csv`. Keep **interviewee turns only**, never interviewer speech.
2. Window each speaker's interview speech into **8.0 s windows, no overlap**. Drop tails shorter than 4 s. Cap at **40 windows per speaker**, chosen with a seed, which gives about 4,600 windows. The live app uses the same windowing, so training and serving stay consistent.
3. Write `manifest.parquet`: `window_id, speaker_id, label, fold, src_file, t0, t1`.
4. Noise sources: MUSAN (noise, music, speech for babble) and the OpenSLR-28 RIR database. Split noise **by file**. Hold out the RIR database's point-source noises and real RIRs for test corruption only.

### 3.2 Augmentation

| Copy | Count per window | Recipe | Env class e |
| --- | --- | --- | --- |
| Clean | 1 | as recorded | 0 clean |
| Train-aug | 4 | One class sampled from {stationary noise, music, babble (3–5 MUSAN speakers summed), reverb (simulated RIR), phone (8 kHz resample + 300–3400 Hz band-pass + µ-law)}. Additive SNR \~ U(5, 20) dB; volume ±6 dB. | 1–5 |
| Test-corrupt | 3 | **Held-out** noise files and real RIRs at 20, 10 and 5 dB | not used in training |

No pitch, speed or tempo changes, because those alter depression-related acoustics. Every augmentation seed is derived from `hash(window_id, copy_idx)`, so reruns reproduce exactly. Mixing and RIR convolution run on the GPU in torch (FFT convolution), so CPU workers don't become the bottleneck.

### 3.3 Feature extraction (GPU, see the §4 ledger)

- Model: `microsoft/wavlm-base`, revision pinned. bf16 autocast, eval mode, batch 64 × 8 s.
- Output per copy: mean-pooled hidden states from **all 13 layers** (embedding output + 12 transformer layers), masked over valid frames. Shape `[13, 768]`, fp16.
- Volume: about 4,600 windows × 8 copies ≈ 36,800 forward passes (≈ 82 h of audio), giving ≈ 0.74 GB on Drive as `feats/{shard}.npy` plus an index parquet.
- Write a checkpoint every shard (1,000 windows), so a Colab disconnect loses less than one shard.
- Optional: score Kintsugi DAM on 5 windows per Androids speaker as an external E1 baseline (Italian input, so this is a reference only).

### 3.4 Models (CPU)

All models share the official 5 folds. Inside each outer fold, 15% of training speakers are held out as an inner validation set for early stopping, layer choice and C. Speaker score = mean of window probabilities.

| Model | Input | Trained on | Losses |
| --- | --- | --- | --- |
| L0 | z at layer ℓ\* (standardized) | clean windows | Logistic regression, C from inner CV |
| M1 | P(z) | clean + train-aug pairs | Disease CE on both views |
| M3 | P(z) | clean + train-aug pairs | Disease CE + α consistency + β GRL env CE |
| M2 (optional) | P(z) | clean + train-aug pairs | Disease CE + α consistency |

- **ℓ\*** = the layer with the best inner-CV L0 AUC, chosen separately per outer fold. Report which layers were picked.
- **Projector P:** Linear 768→256, GELU, dropout 0.2, Linear 256→128, LayerNorm. **Heads:** f\_d = Linear 128→1; f\_e = Linear 128→64→6 behind a GRL.
- **Training:** AdamW (lr 1e-3, weight decay 1e-2), batch 256 pairs, 40 epochs max, early stopping on inner-val AUC. α = 1.0; β ramps from 0 to 0.5 over the first 30% of steps. 3 seeds per fold, averaged. Expected cost: under 2 CPU-minutes per fold per seed.

```latex
L = \tfrac{1}{2}\left[\mathrm{CE}(f_d(P(z_c)), y) + \mathrm{CE}(f_d(P(z_n)), y)\right] + \alpha \lVert P(z_c) - P(z_n) \rVert_2^2 + \beta\, \tfrac{1}{2}\left[\mathrm{CE}(f_e(\mathrm{GRL}(P(z_c))), 0) + \mathrm{CE}(f_e(\mathrm{GRL}(P(z_n))), e)\right]
```

There is no speaker-adversarial term anywhere: the drift model keeps speaker identity on purpose.

### 3.5 Evaluation

- **E1, AUC vs SNR:** L0, M1, M3 (and DAM) on held-out folds at clean, 20, 10 and 5 dB with held-out noise. Report speaker-level AUC, balanced accuracy and sensitivity at 80% specificity, with speaker-bootstrap 95% CIs (2,000 resamples). The key number is the **paired** bootstrap CI of (M3 − L0) at each SNR.
- **E2, noise probe:** logistic regression predicting the env class from frozen features (raw z vs P\_M3(z)), using a speaker-disjoint split. Chance is 1/6. Add a t-SNE plot for the deck.
- **E3, false drift alerts:** for each held-out control speaker, pseudo-session = mean of 3 consecutive windows (≈ 24 s). Sessions 1–5 form the baseline (clean); sessions 6+ are new. Score new sessions twice: clean, and with a randomly chosen test-corrupt copy. Report alerts per speaker (2-of-3 rule) and per-session exceedance rate, for raw z vs P\_M3(z), with and without the quality gate. State that pseudo-sessions share one day and one room, so this is a lower bound.
- **E3b (real audio, small):** the demo speaker's 5 quiet baselines plus 3 noisy-room recordings, scored with the same pipeline as live.

### 3.6 Export for deployment

Retrain P\_M3 on **all** folds with the median-chosen ℓ\*, then fit PCA and the drift constants (§6) on all control speakers. Artifacts go in `artifacts/v{n}/`: `projector.pt`, `pca.npz`, `drift_config.json` (ℓ\*, k, τ, σ\_pop, WavLM revision, git SHA), and `metrics.json` (E1–E3 tables). Total size is under 5 MB and is committed to the private artifacts store, never the public repo with Androids-derived data.

## 4. GPU budget ledger (hard cap 5.0 h)

Planned GPU use is 2.0 h, which leaves a 3.0 h reserve. Only stages G0–G3 may hold a GPU runtime. Everything else runs on a CPU runtime or the laptop.

| Stage | What | GPU | Budget | Kill rule |
| --- | --- | --- | --- | --- |
| G0 | Smoke: 200 windows × 8 copies through the full extract path; measure clips/s | L4 | 0.25 h | If throughput is under 20 clips/s, fix the dataloader before G1 |
| G1 | Extract clean + train-aug copies (≈ 23k passes) | L4 | 0.75 h | Stop at 1.0 h; continue from the shard checkpoint |
| G2 | Extract test-corrupt copies (≈ 14k passes) | L4 | 0.5 h | Stop at 0.75 h; drop 20 dB if needed |
| G3 | DAM reference scores on Androids (optional) | L4 | 0.25 h | Skip if G1 + G2 used more than 1.75 h |
| G4 | Re-extract after a bug fix (contingency) | L4 | 0.25 h | One use only |
| — | Reserve | — | 3.0 h | Needs a team decision to spend |

**Enforcement**

- Cell 1 of every GPU notebook writes `{stage, start_utc, gpu_name}` to `budget/ledger.csv` on Drive. A `finally:` block writes the end time and calls `google.colab.runtime.unassign()`.
- A watchdog thread unassigns the runtime when `stage_elapsed > kill_rule` or `ledger_total > 4.75 h`.
- Before any GPU stage, run `python tools/budget.py` to print hours used and hours remaining. The team lead signs off on any spend from the reserve.
- The laptop GTX 1650 doesn't count toward the cap. It serves inference only, never training.

## 5. Online inference pipeline (laptop)

A 60-second check-in returns in about 10 s, against the 15 s target. Transcription runs first and gates safety. The acoustic and LLM branches run in parallel. All models load once at startup into about 2 GB of the GTX 1650's 4 GB.

### 5.1 Stages

1. **Capture (browser).** MediaRecorder records webm/opus. The screen shows the fixed prompt, a 60 s countdown, a live level meter, and a "too quiet / too loud" hint. The user can stop after 20 s.
2. **Decode.** ffmpeg converts to 16 kHz mono float32. Reject anything under 20 s or over 90 s. There is no loudness normalization, to match training (volume perturbation of ±6 dB covers gain differences).
3. **Quality gate.** Silero VAD (ONNX, CPU) gives speech seconds. Estimated SNR = 10·log10(mean power in speech frames / mean power in non-speech frames). Clipping = fraction of samples with |x| ≥ 0.99. Outcomes: **PASS**; **WARN** (scored, but given low weight in drift); **REJECT** (speech < 15 s, SNR < 12 dB, or clipping > 1%). Thresholds come from `config.yaml` and get tuned on Saturday's venue recordings. A rejected check-in is still stored as an event.
4. **Transcribe.** faster-whisper `small.en`, int8 on CUDA (falls back to int8 on CPU), with word timestamps and `vad_filter=True`.
5. **Safety screen, which blocks everything after it.** A lexicon regex (self-harm and suicidal phrases, kept in `safety/lexicon.yaml`) runs on the transcript. A positive match sets status to `SAFETY`. The LLM call also returns a `safety_flag`, and either source is enough to trigger. In `SAFETY`, no scores are shown, the 988 Suicide & Crisis Lifeline panel and a "contact your clinician" prompt appear, and a high-priority clinician alert is created. If the LLM is down, the lexicon still works.
6. **Acoustic branch, in parallel with step 7.** Window into 8 s windows exactly as in training. WavLM-base (fp32 on GPU) → layer ℓ\* → mean-pool → P\_M3 → average over windows → session vector v (128-d). Drift is computed as in §6. Kintsugi DAM gives the reference risk on the same audio.
7. **Language branch.** Timing features come from word timestamps plus VAD: speech rate (words per speech-minute), pause rate (pauses > 250 ms per minute), and mean pause length. One LLM call (temperature 0, JSON schema, 8 s timeout) returns PHQ-9 item evidence plus `safety_flag`. Every quote is checked verbatim against the transcript after normalizing whitespace and case. Quotes that fail are dropped and counted.
8. **Compose and persist.** Write the check-in record (§7), update the alert state, and delete the temporary audio file in a `finally:` block.

### 5.2 Models at runtime

| Component | Model | Device / precision | Memory |
| --- | --- | --- | --- |
| VAD | Silero VAD v5 (ONNX) | CPU | < 50 MB |
| ASR | faster-whisper small.en | CUDA int8 | ≈ 0.5 GB |
| Encoder | microsoft/wavlm-base (pinned) | CUDA fp32 (no fp16 on the 1650) | ≈ 0.4 GB |
| Invariance head | P\_M3 + PCA | CPU | < 5 MB |
| Reference risk | KintsugiHealth/dam (dam3.1.ckpt, loaded with `weights_only=True`) | CUDA fp32, CPU fallback | ≈ 1 GB |
| Evidence + safety | Hosted LLM API (fast tier, configurable) | Network | — |

### 5.3 Latency budget (60 s clip)

| Step | Target | Runs |
| --- | --- | --- |
| Upload + decode + VAD | 0.8 s | serial |
| ASR | 4.0 s | serial |
| Acoustic (WavLM + DAM + drift) | 3.0 s | parallel with LLM |
| LLM evidence + safety | 5.0 s | parallel with acoustic |
| Persist + respond | 0.2 s | serial |
| **End to end** | **≈ 10 s** | under the 15 s target |

The UI streams progress over Server-Sent Events (`quality → transcript → scores → evidence`), so the user sees steps finish instead of a single 10 s spinner.

## 6. Drift and risk scoring

Drift is an RMS z-score in a 16-d PCA space. The per-dimension spread is shrunk toward population within-speaker spread, because five baseline sessions are too few to estimate a covariance. A full Mahalanobis distance is not used.

**Projection.** x = PCA\_k(v), with k = 16 fitted on control-speaker session vectors. Report the variance retained in `metrics.json`.

**Baseline.** The first 5 PASS check-ins give μ\_j and s\_j for each dimension j. The SD is shrunk toward σ\_pop,j, the pooled within-speaker SD of controls, with n = 5 and n₀ = 5:

```latex
\hat\sigma_j^2 = \frac{n\, s_j^2 + n_0\, \sigma_{\mathrm{pop},j}^2}{n + n_0}
```

**Session drift.**

```latex
D_t = \sqrt{\frac{1}{k}\sum_{j=1}^{k}\left(\frac{x_{t,j}-\mu_j}{\hat\sigma_j}\right)^2}
```

**Threshold.** τ = the 90th percentile of D\_t over clean new sessions of training-fold controls, so the per-session exceedance rate is 10% on clean audio. If sessions were independent, the 2-of-3 rule would then fire on about 2.8% of stable windows. E3 reports what actually happens.

**Alert rule.** Each scored check-in carries a weight w (PASS = 1.0, WARN = 0.5). A **confirmed drift** fires when the sum of w over check-ins with D\_t > τ, among the last 3 scored check-ins, is at least 1.5. REJECT check-ins are never scored. Two or more REJECTs in a row raise a separate "missed or unusable check-ins" flag.

**Timing features.** Speech rate, pause rate and mean pause length get z-scores against their own 5-session baseline, using the same shrinkage. They appear as evidence chips ("speech 18% slower than baseline") and are **not** added into D\_t, to avoid double counting with the acoustic embedding.

**Reference risk.** Kintsugi DAM's output is mapped to `low / moderate / high` using its published ordinal thresholds. It is shown as "population reference" and never combined numerically with drift.

**Status per check-in**

| Status | Condition | Who sees what |
| --- | --- | --- |
| SAFETY | Lexicon or LLM safety flag | Patient: 988 panel and clinician prompt. Clinician: high-priority alert. No scores shown anywhere. |
| RED | Confirmed drift | Clinician: alert with drift trend, timing chips and evidence quotes. Patient: neutral "your clinician will follow up". |
| AMBER | Single exceedance, or DAM = high, or 2+ REJECTs in a row | Clinician: "review" badge. Patient: neutral. |
| GREEN | None of the above | Both: "check-in recorded" |
| BASELINE | Fewer than 5 PASS check-ins | Both: "building your baseline (3/5)" |

The patient never sees a risk number. Alerts route only to the clinician view.

## 7. API contract and data schemas

Freeze this contract by **Saturday 02:00**. The frontend builds against `mock_server.py`, which returns fixtures of exactly these shapes, until the real backend lands.

### 7.1 Endpoints (FastAPI, `/api`)

| Method | Path | Body / query | Returns |
| --- | --- | --- | --- |
| GET | `/health` | — | `{models_loaded, gpu, artifacts_version, llm_reachable}` |
| POST | `/patients/{pid}/checkins` | multipart `audio` (webm/wav), `prompt_id` | `202 {checkin_id}` |
| GET | `/checkins/{cid}/events` | SSE | `stage` events, then `result` (a `CheckinResult`) |
| GET | `/checkins/{cid}` | — | `CheckinResult` |
| GET | `/patients/{pid}/timeline` | `?days=30` | `{baseline, checkins: [CheckinSummary], alerts: [Alert]}` |
| POST | `/alerts/{aid}/ack` | `{clinician, note}` | `Alert` |
| POST | `/demo/reset` | \`{scenario: "stable" | "drift" |
| POST | `/demo/compare` | `{checkin_id}` | Drift computed with raw z and with P\_M3, for the robustness toggle |

### 7.2 CheckinResult

```json
{
  "checkin_id": "c_01H...",
  "patient_id": "p_demo",
  "created_at": "2026-11-14T15:02:11Z",
  "status": "GREEN",
  "quality": {"verdict": "PASS", "speech_s": 41.2, "snr_db": 23.5, "clipping": 0.0, "reasons": []},
  "drift": {"d": 0.94, "tau": 1.31, "exceeds": false, "weight": 1.0,
            "window_count": 7, "baseline_n": 5, "confirmed": false},
  "timing": {"speech_rate_wpm": 148, "pause_rate_pm": 11.2, "mean_pause_s": 0.61,
             "z": {"speech_rate": -0.4, "pause_rate": 0.3, "mean_pause": 0.2}},
  "reference_risk": {"source": "kintsugi-dam-3.1", "level": "low", "raw": 0.21},
  "evidence": [{"item": "sleep", "phq9": 3, "direction": "worse",
                 "quote": "I've been waking up at four every night", "verified": true}],
  "evidence_dropped": 0,
  "safety": {"triggered": false, "source": null},
  "transcript": "...",
  "versions": {"artifacts": "v3", "git": "a1b2c3d", "asr": "small.en-int8"},
  "latency_ms": {"decode": 310, "asr": 3900, "acoustic": 2700, "llm": 4400, "total": 9200},
  "disclaimer": "Screening and monitoring aid — not a diagnosis."
}
```

### 7.3 LLM output schema (enforced, validated with Pydantic)

```json
{
  "safety_flag": false,
  "items": [{"phq9_item": 1, "label": "anhedonia", "direction": "worse | better | neutral",
             "quotes": ["exact substring of transcript"], "confidence": 0.0}],
  "summary_for_clinician": "<= 2 sentences, no diagnosis words"
}
```

The prompt instructs the model to quote verbatim only, to output nothing for items with no evidence, and never to diagnose. PHQ-9 item 9 (self-harm) is never shown as evidence. It only sets `safety_flag`.

### 7.4 Storage tables (SQLite via SQLModel)

| Table | Key columns |
| --- | --- |
| `patient` | id, display\_name, clinician\_id, consent\_at, created\_at |
| `checkin` | id, patient\_id, created\_at, status, quality JSON, drift JSON, timing JSON, reference\_risk JSON, evidence JSON, transcript TEXT, session\_vec BLOB (128 × fp32), versions JSON |
| `baseline` | patient\_id, n, mu BLOB, sigma BLOB, timing\_mu JSON, timing\_sigma JSON, frozen\_at |
| `alert` | id, patient\_id, checkin\_id, kind (SAFETY / DRIFT / REVIEW / MISSED), created\_at, acked\_by, acked\_at, note |
| `event_log` | ts, checkin\_id, stage, ms, error |

## 8. Storage, privacy and security

Raw audio is never stored. Only derived features, scores and the transcript persist, and only team members' voices ever enter the live system.

- **Audio lifecycle:** written to `tmp/` on upload, deleted in a `finally:` block once the pipeline finishes. A startup sweep deletes anything older than 5 minutes. A test asserts that `tmp/` is empty after 100 check-ins, including ones that hit errors.
- **Third parties:** only the transcript goes to the LLM API, never audio. A banner in the check-in UI says so. Live check-ins come from consenting team members only. Judges and passers-by don't record.
- **Consent:** each team member signs a one-paragraph consent covering recording, transcript storage, the LLM API and demo display. Store `consent_at` on the patient record. The delete endpoint purges every row for that patient.
- **Datasets:** Androids audio and Androids-derived features stay on Drive. `.gitignore` blocks `data/`, `feats/` and `*.npy`. A pre-commit hook rejects files over 5 MB and paths containing `androids`. Kintsugi DAM is Apache-2.0: keep its LICENSE in `third_party/`.
- **Model loading:** the DAM checkpoint is a pickle. Load it only with `torch.load(..., weights_only=True)` from a pinned revision. Never load it with `trust_remote_code` or unpickle it fully.
- **Network:** the backend binds to `127.0.0.1`. The frontend is served from the same FastAPI app. The demo laptop runs no public tunnel. API keys come from `.env`, which is git-ignored.
- **Labels:** every screen carries "Screening and monitoring aid — not a diagnosis". Seeded or acted data is marked "simulated" wherever it appears.

## 9. Frontend and demo

There are two views: a patient check-in sized for a phone, and a clinician dashboard for a laptop. A hidden robustness panel exists for the judging demo. Stack: React + Vite + TypeScript + Tailwind + Recharts, built as static files and served by FastAPI.

### 9.1 Screens

| Screen | Contents | Done when |
| --- | --- | --- |
| Check-in | Fixed prompt, record button, 60 s ring timer, live level meter, quiet/loud hint, stop after 20 s | Works in Chrome on the laptop and on a phone over localhost via USB |
| Processing | SSE step list: Quality → Transcript → Voice → Evidence, each with a tick and a duration | All four steps tick in about 10 s |
| Patient result | Quality verdict in plain words ("Too much background noise — try a quieter spot"), "Check-in recorded", baseline progress, no scores. SAFETY replaces everything with the 988 panel. | No number visible to the patient |
| Clinician list | Patients with a status badge, last check-in, and open alerts | Sorted by severity |
| Patient detail | Drift line chart with the τ line, baseline band and markers for REJECT and WARN; timing chips; evidence quotes highlighted inside the transcript; DAM reference level; alert acknowledge with note | Every alert can be traced to its check-ins and quotes |
| Robustness panel (`?demo=1`) | The same check-in scored with raw WavLM and with the invariance head, side by side against τ | Shows both numbers from one recording |

### 9.2 Seeded scenarios (`/demo/reset`)

- **stable:** 10 days of real check-ins from the demo speaker, recorded Saturday in a quiet room, a hallway and a dining hall, at the venue.
- **drift:** 5 quiet baseline days, 3 normal days, then 2 days of **acted** low-energy, slow, flat delivery, labeled "simulated" on the chart. This shows that drift can fire. It doesn't claim to detect real deterioration.

### 9.3 Two-minute demo script

1. **0:00–0:15, problem.** Patients go weeks between appointments, and deterioration goes unseen.
2. **0:15–0:45, live check-in.** The demo speaker records at the table with a headset mic. Result: PASS, GREEN, plus evidence quotes on the clinician side.
3. **0:45–1:10, robustness.** Play a babble track from a phone next to the mic, or use a pre-recorded hallway clip as the fallback. The robustness panel shows raw WavLM crossing τ while the invariance head stays below it.
4. **1:10–1:35, drift scenario.** Switch to the `drift` patient: RED alert, trend line, "speech 22% slower" chip, quotes. Mention that it's acted.
5. **1:35–2:00, evidence and honesty.** Show the E3 bar chart of false alerts under noise (raw vs invariant), the safety design, and what isn't validated yet.

## 10. Repo, config and testing

One public monorepo holds code only. Data, features and model artifacts live on Drive and get pulled by `make artifacts`.

```
voice-drift/
  research/            # Colab notebooks + importable code
    data_prep.py       # manifest, windowing, noise splits
    augment.py         # GPU mixing, RIR, phone sim (seeded)
    extract.py         # WavLM 13-layer cache, sharded, resumable
    train_heads.py     # L0, M1, M3 (+M2), 5-fold CV
    evaluate.py        # E1, E2, E3 + bootstrap CIs, plots
    export.py          # artifacts/vN
    budget.py          # GPU ledger + watchdog
  server/
    app.py             # FastAPI, SSE, static frontend
    pipeline/          # decode, quality, asr, safety, acoustic, language, drift
    store.py           # SQLModel tables
    mock_server.py     # fixture responses for frontend dev
    seed/              # scenario JSON (session vectors + transcripts, no audio)
  web/                 # React + Vite
  safety/lexicon.yaml
  config.yaml          # thresholds, model ids, timeouts
  tests/
  third_party/LICENSES
```

**Configuration.** `config.yaml` holds every threshold: quality limits, τ (from `drift_config.json`, overridable), the WARN weight, LLM model and timeout, and pinned model revisions. The server logs the resolved config and artifact version at startup.

**Tests (pytest, run in under 60 s on the laptop)**

- [ ] Quality gate: synthetic tone in white noise at known SNR is estimated within ±3 dB; silence gives REJECT; a clipped sine gives REJECT.
- [ ] Drift math: a hand-computed 3-dim example; shrinkage when s\_j = 0; alert rule truth table over all 27 PASS/WARN/REJECT × exceed combinations.
- [ ] Quote verifier: exact match passes, a paraphrase fails, whitespace and case differences pass.
- [ ] Safety: every lexicon phrase triggers; with the LLM mocked as down, the lexicon still triggers; in SAFETY, the response has no `drift` or `reference_risk` fields.
- [ ] Train/serve parity: the same wav run through `research/extract.py` and `server/pipeline/acoustic.py` gives cosine ≥ 0.999.
- [ ] Audio deletion: `tmp/` is empty after an error path.
- [ ] Golden run: 3 committed team-member clips (consented, under 1 MB total) produce stable statuses; latency is logged.

**Leakage checks (assertions in `train_heads.py`):** no speaker appears in both train and test of a fold; augmented and test copies share their source's fold; ℓ\*, C, τ and early stopping touch only training-fold speakers.

## 11. Failure modes and fallbacks

Each risk has a pre-agreed fallback, so nobody has to debate it at 4 a.m.

| Failure | Detect by | Fallback |
| --- | --- | --- |
| L0 AUC CI includes 0.5 on every layer | E1 at Gate 1 | Ship with drift + DAM only. Report H1 as not supported. The robustness story (E2, E3) still stands, because it doesn't need disease signal. |
| GRL makes M3 unstable (env-probe accuracy oscillates, disease AUC collapses) | Inner-val curves | Lower β to 0.1, or ship M2 (consistency only) as the deployed head |
| M3 clean AUC drops more than 0.05 vs L0 | E1 paired CI | Deploy M3 for drift only and keep L0 for research tables. That's a valid split: drift needs invariance, not disease AUC. |
| E3 shows no false-alert reduction | E3 | Report it honestly and keep the quality gate as the robustness mechanism in the demo |
| GPU ledger reaches 4.75 h | `budget.py` | Stop extraction. Run E1 at clean + 10 dB only, from whatever shards exist. |
| Colab can't get an L4 | G0 | T4 at about 2× the time still fits within 2.0 h planned. Never use an A100 (about 3× the unit burn). |
| DAM won't load or runs too slowly on the 1650 | Startup health | Move DAM to CPU (+2 s), or hide the reference risk with an "unavailable" label |
| LLM API slow or down | 8 s timeout | Return the result without evidence, marked "evidence unavailable". The safety lexicon still runs. |
| Venue noise ruins the live check-in | Quality verdict | Headset mic first; then a pre-recorded clip through the same pipeline; then the backup video |
| Laptop GPU out of memory | Startup health | ASR to CPU int8 (+3 s) |

## 12. Team roles and timeline

Three owners, one per workstream. Two gates decide whether to cut scope. Submit by 07:30 Sunday, an hour before the deadline.

| Role | Owns | Also on the hook for |
| --- | --- | --- |
| Research | §3, §4, §6 constants, E1–E3 numbers | GPU ledger, leakage assertions |
| Backend | §5, §7, §8, tests | Venue recordings, latency, seeded scenarios |
| Frontend + demo | §9, pitch, backup video | Rehearsals, Devpost page |

&#91;embedded content: build plan · 3 workstreams, 5 phases, 2 gates (EST, not to scale)\]

**Gate 1 (Sat 12:00).** Proceed with M3 only if the L0 E1 table exists and one live check-in has run end to end with drift stubbed. Otherwise, apply the first two fallbacks in §11. **Gate 2 (Sat 20:00).** The demo must run on real artifacts. After this, there are no new features, only fixes.

## 13. Open items to verify

All of these need checking before Saturday 02:00. Each can change a number in this spec, but none changes the architecture.

- [ ] Submission deadline and judging slot on the HackPrinceton Discord. A teammate's note says Sunday 08:30 EST.
- [ ] Androids layout: interview clip folders, turn-timing file format, and whether the clips are already interviewee-only. Also the `fold-list.csv` columns.
- [ ] Kintsugi DAM: required input length and sample rate, output format and ordinal thresholds, `featex.py` dependencies, and peak memory on the GTX 1650. The model card lists Apache-2.0 and a whisper-small.en base ([model card](https://huggingface.co/KintsugiHealth/dam/blob/main/README.md)).
- [ ] MUSAN is a single tarball of roughly 11 GB. Fetch it on a Colab CPU runtime and extract only `noise/`, `music/` and `speech/`.
- [ ] LLM provider, API key, and its data-retention terms for transcripts.
- [ ] Quality-gate thresholds, from 10 venue recordings made on Saturday morning.
- [ ] Names against the roles in §12, and signed consent from every member who records.

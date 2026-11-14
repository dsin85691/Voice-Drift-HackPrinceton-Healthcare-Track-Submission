# Test audio

Clean, noisy and deliberately bad recordings for trying Voice Drift without a microphone. Built by `python tools/make_test_audio.py`.

**All speech here is synthetic** (the bakery passage read by the offline Windows text-to-speech voice "Microsoft David", from `server/demo_assets/`). Noise comes from the app's own generator (`server/pipeline/noise.py`): pink noise, speech-shaped noise, and "babble" made from the same voice played reversed. No real person's voice and no dataset audio is included.

## How to use

- **Noise-robustness lab:** Results -> "Try your own recording": upload `clean/reading_normal.wav` and either pick a built-in noise or upload a file from `noise_only/`. Use the profile **Sample speaker (synthetic voice)** (click "Analyze sample recording" once first so its baseline exists).
- **Pre-mixed noisy versions:** `noisy/` holds the same reading with noise already added, for comparing against `clean/reading_normal.wav`.
- **Quality gate:** the files in `quality/` should each be rejected with a clear reason.

## Files (verdicts measured by the app's quality gate when the files were generated)

| File | Length | Quality verdict | Reasons | SNR estimate (dB) | What it is |
|---|---|---|---|---|---|
| `clean/reading_normal.wav` | 43 s | PASS | - | 98 | Standard passage, normal pace (same synthetic voice as the sample baseline). |
| `clean/reading_slower.wav` | 65 s | PASS | - | 100 | Same passage, read noticeably slower. |
| `noisy/reading_normal_pink_20dB.wav` | 43 s | PASS | - | 19 | Pink noise over the whole reading, 20 dB SNR (mild). |
| `noisy/reading_normal_pink_15dB.wav` | 43 s | WARN | some background noise | 14 | Pink noise over the whole reading, 15 dB SNR (noisy but usable). |
| `noisy/reading_normal_babble_15dB.wav` | 43 s | WARN | some background noise | 15 | Babble over the whole reading, 15 dB SNR (noisy but usable). |
| `noisy/reading_normal_pink_10dB.wav` | 43 s | REJECT | too much background noise | 10 | Pink noise over the whole reading, 10 dB SNR. |
| `noisy/reading_normal_babble_10dB.wav` | 43 s | REJECT | too much background noise | 10 | Babble (reversed same voice) over the whole reading, 10 dB SNR. |
| `noisy/reading_normal_babble_5dB.wav` | 43 s | REJECT | too much background noise | 7 | Babble over the whole reading, 5 dB SNR (heavy). |
| `noisy/reading_normal_speechshaped_10dB.wav` | 43 s | REJECT | too much background noise | 10 | Speech-shaped steady noise, 10 dB SNR. |
| `noisy/reading_normal_babble_10dB_14-30s.wav` | 43 s | PASS | - | 57 | Babble at 10 dB only between 14 s and 30 s (the noise-demo setting). |
| `noise_only/pink_30s.wav` | 30 s | REJECT | not enough speech, too much background noise | 2 | Pink noise alone, for the Lab's 'noise file' option. |
| `noise_only/babble_30s.wav` | 30 s | REJECT | too much background noise | 4 | Babble alone (reversed synthetic voice), for the Lab's 'noise file' option. |
| `quality/silence_30s.wav` | 30 s | REJECT | not enough speech, too much background noise | -20 | Near-silence: should be rejected (no speech). |
| `quality/too_short_10s.wav` | 10 s | REJECT | recording too short | 74 | 10 s of speech: should be rejected (minimum 20 s). |
| `quality/mostly_silent_45s.wav` | 45 s | REJECT | not enough speech | 100 | 12 s of speech then 33 s of silence: should be rejected (too little speech). |
| `quality/clipped.wav` | 43 s | REJECT | audio is distorted (too loud) | 112 | Reading amplified until it clips: should be rejected (distorted). |

SNR in the file names is the mixing ratio over speech frames; the estimate column is the app's own rough estimate from the recording, so the two differ.

Noise over the **whole** reading at 10 dB or below is rejected by the quality gate (it rejects recordings whose estimated SNR is under 12 dB). That is intended: the app refuses to score such recordings rather than guess. The noise-robustness demo therefore adds noise to an **interval** only (see the 14-30 s file), and the 15 dB files are the noisy-but-usable case.

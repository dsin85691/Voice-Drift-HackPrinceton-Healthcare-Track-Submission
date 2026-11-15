# Devpost media

Screenshots of the live app for the HackPrinceton Devpost gallery, in gallery order. Captured from the running app
with the real models (WavLM-base + projection head v1, Whisper small.en) and the seeded demo data, by:

```
python tools/capture_media.py          # server running, tools/warm_demo.py done, "Load demo: all" clicked
```

Every image is the real app; nothing is mocked. Sam is a **synthetic** demo profile and is labelled as such in the app.

| # | File | What it shows | Caption |
|---|---|---|---|
| 1 | `01_noise_demo_chart.png` | Results → noise demo (clinician view): the bundled reading with babble noise added at 10 dB from 14 to 30 s. | Same voice, same words, background noise added. The standard model (dashed) jumps to 4.4× its boundary; Voice Drift (solid) rises only to 1.3×. |
| 2 | `02_home_patient.png` | Home, patient view, Sam: hero and "At a glance" tiles. | A 40-second daily reading, compared with your own baseline. (Sam is a synthetic demo profile.) |
| 3 | `03_trend_chart.png` | Trends, Sam: one point per check-in, the boundary at 1, the last two above it. | Each point is one check-in. Two of the last three above the boundary, on separate days, raises a review. |
| 4 | `04_checkin_page.png` | Check-in, live profile: the reading passage and the recorder. | The same passage every time, so differences come from how you speak, not what you say. |
| 5 | `05_review_item.png` | Clinician view: Sam's open Speech Change Review with its rule, evidence chart and actions. | A persistent change becomes a nonurgent item for a clinician. Nothing is diagnosed. |
| 6 | `06_results_table.png` | Results: session-level table for raw WavLM, B1 and P1 with 95% intervals and paired differences. | Six held-out speakers: false alerts under 10 dB noise fell from 86% to 19%, at the cost of lower sensitivity to speech changes (pauses: 100% → 53%). |

Caption numbers checked against the data the screenshots came from:

- **Shot 1:** noisy windows 16–24 s and 24–32 s. Raw is 4.37× and 3.76× its boundary; projected is 1.30× and 1.34×, so it does cross its own boundary, slightly.
- **Shot 6:** `results/aggregate.json`.

## Inspiration

People often go weeks between appointments, and changes in how they're doing can go unseen until the next visit. 

Voice is a low-effort signal people could record at home in between. But a noisy kitchen or a new phone can change a recording as much as the speaker does, and population "depression detectors" answer a different question from "has this person changed?" 

So we set out to build a monitor that compares each person only with themselves, and that is deliberately hard to fool with background noise.


## What it does

A short at-home check-in: read the same passage aloud (about 40 seconds). Voice Drift checks recording quality, measures how far today's voice is from the person's own baseline, and sends a **nonurgent** item to a clinician's review queue only when the difference **persists**. Nothing is diagnosed, and the person never sees a risk label.

### From audio to one number

Each check-in is cut into 8-second windows. A frozen WavLM speech model turns each window into a vector, our small projection head cleans it of room noise, and the windows are averaged into one vector $x$ with 16 numbers for the whole check-in.

### The personal baseline

The person's first five good recordings set their baseline: an average $\mu$ and a spread for each of the 16 numbers. Five recordings are too few to trust on their own, so the person's spread $s$ is blended half and half with the typical spread $S$ of other speakers:

$$
\sigma^2 = \frac{s^2 + S^2}{2}
$$

### The drift score

For each of the 16 numbers, today's distance from baseline in units of the person's own variability is

$$
z = \frac{x - \mu}{\sigma}
$$

and the drift score $D$ is the root mean square of those 16 values. A score near 1 means a typical day.

### The boundary

The boundary $\tau$ is set in advance from other people's normal days: it is the 90th percentile of $D$ on held-out training speakers, about 1.71. The app shows

$$
r = \frac{D}{\tau}
$$

so $r > 1$ means "past the edge of normal day-to-day variation". About one ordinary day in ten crosses it by chance.

### When a human is asked to look

One crossing is only an observation. A review item is created only when $r > 1$ on at least 2 of the last 3 check-ins, on 2 different days within a week.

### Interpretable measurements, shown separately

Next to $r$, the app shows measurements a person can understand, such as speaking rate and

$$
\text{pause fraction} = \frac{\text{time in pauses of 200 ms or more}}{\text{speaking time}}
$$

These are never mixed into $D$.



## How we built it


- **Projection head without diagnosis labels.** A small network (768 → 256 → 128) on top of frozen WavLM, trained on clean speech and noisy or reverberant copies of the same speech. Its training loss has four parts:

  $$
  \text{loss} = \text{invariance} + \text{variance} + 0.04 \times \text{covariance} + 0.5 \times \text{acoustic}
  $$

  **Invariance** makes clean and noisy copies give the same output (ignore the room). **Variance** and **covariance** stop the outputs from collapsing or repeating each other (as in VICReg). **Acoustic** makes the output still predict pause fraction, pitch variability and speech rhythm, so real delivery changes are kept.
- **Data.** The Androids interview corpus, with MUSAN noise, music and babble plus simulated rooms and microphones for training. Evaluation uses **different speakers and different noise files**.
- **Compute.** One batched WavLM pass on Colab, under a strict 5-GPU-hour budget enforced by a ledger and an automatic kill switch; everything after that runs on CPU.
- **App.** FastAPI and SQLite, a plain-JavaScript frontend with no build step, Whisper for word timings, and a patient/clinician view switch. Clinician notifications stay inside the app in demo mode.

## Challenges we ran into

 **Finding what the score can't see.** When we tested 20% slower delivery, neither model detected it: averaging over each 8-second window throws away speaking rate. That is why rate and pauses are separate measurements.
- **Speed on a laptop CPU.** A live check-in first took 69 s. Running speech-to-text alongside WavLM and switching Whisper to greedy decoding brought it to about 20 s, with identical results.

## Accomplishments that we're proud of

At roughly matched clean-speech alert rates, with held-out noise added at 10 dB to unchanged speech:

| | Raw WavLM | With our projection |
|---|---|---|
| False alerts | 86% | **19%** |

The paired difference across 6 held-out speakers was $-67$ percentage points, with a 95% interval of $[-86, -47]$.

## What we learned

**Personal change and population classification are different problems.** A depression classifier we trained reached an AUC of about $0.9$ on the dataset. We chose not to ship it, because on this data that accuracy may reflect recording differences between groups rather than illness.
**Five recordings are a thin baseline.** It is a starting point, not a portrait of someone's normal.

## What's next for Voice Drift
- Longer, maturing baselines, with a boundary $\tau(n)$ calibrated for each baseline size.
- A longitudinal study with clinician-rated outcomes, to test whether a change from baseline actually tracks a change in mental health.
- Real (not synthetic) speech-change recordings, more speakers and languages, and a clinician-supervised pilot.


# Spec: seed an open review for Sam (synthetic drift profile)

## Goal

When the demo data loads, the review queue shows one open review for "Sam (simulated, drift)", raised by the app's own persistence rule. Alex stays clean.

## Current state

- Rule: at least 2 of the last 3 quality-passing check-ins above the person's boundary, on separate days, within 7 days.
- Sam's last three check-ins (boundary 1.708):

  | Check-in | Date | Distance | Above? |
  |---|---|---|---|
  | `c_p_drift_07` | Nov 12 | 0.833 | No |
  | `c_p_drift_08` | Nov 13 (acted) | 1.748 | Yes |
  | `c_p_drift_09` | Nov 14 (acted) | 1.436 | No |

- Only one is above, so `GET /api/v1/review-flags` returns an empty list.
- Check-in ids for Nov 12 to 14 are inferred from `c_p_drift_00` being Nov 5; confirm in the seed code.

## Changes

### 1. Seed data
- Regenerate `c_p_drift_09` (Nov 14) with a distance clearly above the boundary, about 1.9. Its timing values already show slower speech and longer pauses, so no other field needs to change.
- Its status should come out as above-boundary (amber), the same as Nov 13, by the normal scoring path. Do not set the status by hand.
- Leave Nov 13 as it is. Do not touch Alex.

### 2. Raise the flag through the rule
- At the end of `POST /api/demo/reset` for the scenarios that include Sam ("all" and "drift"), run the existing monitoring evaluation for Sam's latest check-in, the same call as `POST /api/v1/monitoring/evaluate` with `patient_id: p_drift` and `checkin_id: c_p_drift_09`.
- Do not insert a review-flag row directly. The flag must be a product of the rule so its supporting check-ins match the chart.
- Resetting twice must not create a second flag.

### 3. Boundary consistency
- Sam's baseline reports a boundary of 1.47 while each check-in is compared against 1.708. Decide which is intended and use one value in both places before choosing the Nov 14 distance.

### 4. What the user sees
- Review queue, Open tab: one row for Sam, status Open, supporting check-ins Nov 13 and Nov 14, Data column "Synthetic demo data".
- Home on Sam's profile: latest comparison no longer reads "Within your usual range"; it reads as above the usual range and links to the open review.
- Research workspace: Sam's "Latest comparison" matches Home. If "Open notes" is a separate concept from review flags, either merge them or make the counts agree.
- Trend chart caption: "2 above the boundary" in place of "1 above the boundary".

## Out of scope
- Any path that lets a user upload or type scores, transcripts or flags.
- An older, already-reviewed flag for the "All" tab (optional follow-up).
- A noisy-room profile that raises no flag (separate spec).

## Done when
- [ ] After "Load demo: all", `GET /api/v1/review-flags?status=OPEN` returns exactly one flag, for `p_drift`.
- [ ] The flag lists Nov 13 and Nov 14 as supporting check-ins.
- [ ] `p_stable` and `p_demo` have no flags.
- [ ] Loading the demo a second time still gives exactly one flag.
- [ ] Home, Trends, Research workspace and the review queue all describe Sam's latest check-in the same way.
- [ ] Acknowledging the flag in the queue (`POST /api/v1/review-flags/{id}/actions`) moves it out of Open, and a fresh demo load brings it back as Open.

# Agent log: `feat/bill-outlook`

## 2026-10-25 06:29 — feat(outlook): BillOutlook schema and assembler

- **What:** `schemas/outlook.py` (Pydantic `BillOutlook` with distribution, deadlines, weather, curve, pace, next milestone and all sentences in display order) and `outlook.build_outlook(bill_id, as_of, target_date)`. Every "chance by a date" (deadlines, weather, curve, what-if) goes through `calibrate()`, which applies the isotonic recalibration stored in `reports/deadline_calibration.json` when the calibration check calls for it (tracked, so fresh clones match); `p_royal_assent` stays the model's raw value. `model_version` = passage model set and fit date + `outlook-v1`.
- **Why:** output format of the outlook spec.
- **Checks:** `ruff check`: clean; bill 4035 on 2026-10-25 in 2.9 s: "Changeable: 72% chance of Royal Assent before the Christmas recess (17 Dec 2026, estimated)"; custom target 2027-06-30 -> "On track: 94% ...".

## 2026-10-25 06:26 — feat(outlook): next milestone and what-if

- **What:** `outlook.next_milestone_block`: the next stage and its date (a future scheduled sitting if one exists, else the median entry date of the simulated paths, marked expected), and a what-if: the chance by the target date from paths with no scheduled date vs paths with the next stage forced to that date (`run_paths(next_entry=...)`). Sentences: "If X happens on D as scheduled, the chance of law before ... rises from A to B"; for expected dates "only happens on D (its expected date)", since pinning the median date removes the faster paths. `calibrate()` hook applies a saved isotonic recalibration (identity until one is fitted).
- **Why:** section 7 of the outlook spec.
- **Checks:** `ruff check`: clean; on 2026-10-25, bill 4035 (Lords report and 3rd reading scheduled 26 Oct 2026): chance of law before the Christmas recess rises from 34% to 71%.

## 2026-10-25 06:25 — feat(outlook): pace compared with typical bills

- **What:** `outlook.pace_block`: sitting days (either House) from 1st reading to the entry of the bill's current stage, against bills from the same originating House that had finished before the forecast date and reached the same stage. Outputs `weeks_ahead_or_behind` (positive = behind; sitting days converted with the sitting days per sitting week of the past year), `pace_percentile` (share of past bills that were faster) and a sentence; "too early" for bills still at their first stage, "too few" under 10 comparison bills.
- **Why:** section 5 of the outlook spec.
- **Checks:** `ruff check`: clean; run on the 18 bills in Parliament on 2026-10-25 (e.g. bill 4019: 26 weeks behind at Lords committee stage, slower than 98%; carried-over bills are slow, as expected).

## 2026-10-25 06:25 — feat(outlook): distribution, deadlines, weather status and curve from simulated paths

- **What:** `model/forecast.py` gains `run_paths` (the simulation `forecast_bill` already ran, now reusable; `forecast_bill` calls it, predictions unchanged) with a what-if hook (`next_entry`) and `use_schedule`. New `model/outlook.py`: sections 0-3 of the outlook spec, all counted from the 10,000 simulated paths (no quantile interpolation needed): `Distribution` (p_royal_assent, failure risk, 5-95% Royal Assent quantiles, P(law by date)); deadlines (next Christmas / Easter / summer recess from Commons sitting days, estimated from the median of the last 12 years and snapped to a Thursday when not recorded; next 4 quarter ends; announced or simulated-median session end; custom date) with sentences; weather status (thresholds in `OUTLOOK_CONFIG`); weekly two-year curve with "most likely by / very likely by" months.
- **Why:** Roman asked for presentation-ready outlooks without changing the model.
- **Checks:** `ruff check`: clean; `pytest tests/model`: 117 passed (refactor changes nothing). Bill 4305 on 2026-10-25: p_royal_assent 0.886, 49% before the Easter recess (estimated 18 Mar 2027), "Most likely by March 2027". The database has no announced recess dates and no future sitting days, so every future recess date is estimated.

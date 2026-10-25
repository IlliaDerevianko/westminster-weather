# Agent log: `feat/bill-outlook`

## 2026-10-25 06:39 — feat(frontend): weather outlook panel on the bill page

- **What:** `frontend/src/lib/outlook.ts` (types for `GET /api/v1/bills/{id}/outlook`, fetch helper, weather icons/colours) and `frontend/src/app/BillOutlookPanel.tsx`: weather headline with chance of ever passing / failing, weekly two-year curve (SVG, target-date line), deadline list with bars, pace and next milestone with the what-if, model and calibration footnote, and a target-date picker. Shown above the three phase columns in `page.tsx` only when the bill is in Parliament on the snapshot date (404 hides it); the page's default snapshot forecasts as of today, a picked past snapshot as of that date. Backend: `calibrate(0)` stays 0, so the curve starts at 0% on the forecast date.
- **Why:** Roman asked to show the outlook JSON on the frontend.
- **Checks:** `npx tsc --noEmit`: clean; `ruff check`: clean; screenshots of bills 4035 and 4083 checked (4083: "Delays likely: 21% ... before the Christmas recess", live model 2026-10-25). Known: the session-end chance uses its own recalibration curve, so it can be higher than a later quarter end (4083: 94% by 10 Jun vs 80% by 30 Jun).

## 2026-10-25 06:33 — feat(outlook): bill_outlooks table, endpoint and generation for bills in Parliament

- **What:** `BillOutlookRow` (`bill_outlooks`, PK bill_id + forecast_date; key fields plus the full outlook as JSON) in `models.py`, `save_bill_outlooks_to_db` in `db.py`, DATA_CONTRACT §6.6. `GET /api/v1/bills/{bill_id}/outlook?target_date=YYYY-MM-DD[&as_of=...]` computes it on request (404 if the bill is not in Parliament). `model/outlook_generate.py` writes outlooks for every bill in Parliament; ran it for 2026-10-25: 18 bills saved to production. Tests in `tests/model/test_outlook.py` (weather thresholds and storm override, path counting, quarter ends, recess dates, curve sentence).
- **Why:** output format and generation steps of the outlook spec.
- **Checks:** `ruff check .`: clean; `pytest`: 384 passed. API: 4035, 4124, 4083 -> 200; 4035 with target_date 2027-03-31 -> "On track: 97% ..."; 3137 (an Act) -> 404.

## 2026-10-25 06:31 — feat(outlook): deadline calibration check and recalibration

- **What:** `model/outlook_calibration.py` re-simulates every rolling-backtest point (models refitted before each session, 4,000 paths) in the training era (sessions 27-30: 825 forecasts, 136 bills) and the test era (sessions 34-39: 1,281 forecasts, 222 bills), and scores the chance of Royal Assent before the next Christmas recess, next summer recess and session end against what happened. Outputs `backend/reports/deadline_calibration.png`, `deadline_calibration.json` (tables, Brier, isotonic curves) and `deadline_calibration_points.csv`. Raw test calibration is poor (underconfident: 40-50% happened 67% of the time; session end mean 71% vs 92%), so isotonic recalibration was fitted on training-era forecasts only: one curve for session end, one for date deadlines, chosen over a pooled curve by cross-validation inside the training era (Brier 0.110 vs 0.124 vs 0.137 none). `outlook.calibrate(p, group)` applies it.
- **Why:** calibration check required by the outlook spec before using the numbers in the pitch.
- **Checks:** `ruff check .`: clean. Test-era Brier: model 0.123, recalibrated 0.097, historical-average baseline 0.180. After recalibration, 70-80% forecasts happened 70% of the time (94); still off by >10 points: 40-50% (38%, n=121) and 50-60% (45%, n=148). Nothing tuned on test-era data.

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

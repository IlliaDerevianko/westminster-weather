# Agent log: `feat/passage-model-tasks-4-11`

## 2026-10-24 22:49 — feat(api): GET /api/v1/bills/{bill_id}/forecast

- **What:** added the forecast endpoint to `main.py` (`as_of` query, defaults to today; 404 for unknown bills or dates before the training cutoff).
- **Why:** BUILD_PLAN task 9 step 4.
- **Checks:** `ruff`: clean; `pytest`: 98 passed; manual TestClient calls: 200 for bill 3064 as of 2021-12-01, 404 before 2019-10-14, existing `/api/v1/bills/{bill_id}` still 200.

## 2026-10-24 22:49 — feat(model): Monte Carlo simulator and forecast_bill (task 9)

- **What:** added `model/simulate.py` (10,000 vectorised paths per bill, fixed seed per (bill, as_of); session end, carry-over, dissolution, withdrawal, defeat, same-day multi-state advances, ping-pong rounds; days to session end recomputed from each path's simulated end) and `model/forecast.py` (`forecast_bill`, start state via the interface, summaries, leak-free choice of `train`/`live` models, example chart). `cli train` now saves `hazard_train` and `hazard_live`; added `cli forecast` (`--bill-id`, `--examples`). Added `tests/model/test_simulate.py` and `reports/historical_forecasts.png`.
- **Why:** BUILD_PLAN task 9, MODEL_SPEC §5.
- **Checks:** `ruff`: clean; `pytest`: 98 passed. Forecasts take 0.7–3 s; outcome probabilities sum to 1 (tested).
- **Notes:** examples (train models): Online Safety Bill at Commons 1R 29% (became law 2023-10-26 after carry-over); Commercial Rent at Lords committee 66%, median 2022-03-30 (actual 2022-03-24); Renters (Reform) at Lords 1R/2R 46%, election risk 45% (fell at the 2024 dissolution). The simulator cannot anticipate a carry-over motion that has not happened yet.

## 2026-10-24 22:49 — feat(api): forecast response schemas

- **What:** added `TimelineQuantiles`, `StageForecastSchema` (DATA_CONTRACT §6.3 fields) and `BillForecastResponse` (MODEL_SPEC §5.1, original field names from the first backend commit plus `as_of`, `p_ra_before_session_end`, `stages`, `model_variant`, `n_paths`) to `schemas/bills.py`. Existing schemas unchanged.
- **Why:** BUILD_PLAN task 9 step 3; `BillForecastResponse` had been removed from `main` in an earlier refactor.
- **Checks:** `ruff`: clean; `pytest`: 98 passed.

## 2026-10-24 22:41 — feat(model): ping-pong as rounds (MODEL_SPEC §4.4)

- **What:** `model/pingpong.py` now fits rounds (`N − 1 ~ Poisson(λ)`, one round per stage 12/13/42 sitting), empirical gaps between rounds and from the last round to Royal Assent, using `get_stage_sittings`; `sample_duration` combines them. Tests updated; `cli train` re-run.
- **Why:** replaces the total-duration stand-in now that the interface exposes individual sittings.
- **Checks:** `ruff`: clean; `pytest`: 89 passed. Live fit: λ = 1.14 (median 2 rounds), round gap median 2 sitting days, final gap median 2; simulated total median within 3 days of observed (6).

## 2026-10-24 22:39 — feat(data): add get_stage_sittings(as_of) to the model interface

- **What:** added `get_stage_sittings(as_of)` to `data/interface.py` (method on `DatabaseInterface` plus module function): stage sittings dated ≤ `as_of` for bills introduced by `as_of`, with `model_state` from the stage map. Documented as DATA_CONTRACT §5.4a; added leakage and synthetic tests.
- **Why:** Aleks asked to fix the interface gap: MODEL_SPEC §4.4 counts ping-pong rounds, which need individual sittings; the four existing functions only give ping-pong entry and Royal Assent dates. Changing the teammate's interface code was approved by Aleks.
- **Checks:** `ruff`: clean; `pytest tests/model/test_interface.py`: 21 passed.

## 2026-10-24 22:37 — feat(model): failure routes and ping-pong (task 8)

- **What:** added `model/failure.py` (session-length distribution from prorogued sessions, conditional draws; dissolution hazard by years since election with a 5-year cap; carry-over rule; constant defeat fallback 0.002 per Commons vote stage; constant withdrawal hazard) and `model/pingpong.py` (ping-pong probability; total ping-pong → Royal Assent duration). `cli train` fits a training version (as of 2019-10-14, sessions 20–30) and a live version, saves them in `backend/models/`, writes `metrics.json` (`failure`) and `reports/failure_and_pingpong.png`. Added `tests/model/test_failure.py`.
- **Why:** BUILD_PLAN task 8, MODEL_SPEC §4.2–4.5. Task 7 skipped (optional), per Aleks's "keep going".
- **Checks:** `ruff`: clean; `pytest`: 83 passed.
- **Notes:** ping-pong rounds and round gaps (MODEL_SPEC §4.4) need individual ping-pong sittings, which the §5 interface does not expose; total ping-pong duration is used instead — asked Aleks. Only prorogued sessions feed the session-length distribution (dissolution is its own route). Dissolution bucket rates are smoothed (+0.5 events) because there are only 5 dissolutions.

## 2026-10-24 22:31 — feat(model): logistic advance hazard (task 6)

- **What:** added `model/hazard.py` (pooled L2 logistic hazard: state × time-in-state cells, end-of-session buckets, shared Tier 1 effects; C chosen by time-ordered validation; optional LightGBM comparison), `cli train` now also fits it and saves `backend/models/hazard.joblib`, `reports/hazard_coefficients.png` and `metrics.json` (`hazard`). Added `tests/model/test_hazard.py`. Installed `libomp` with Homebrew so the existing `lightgbm` dependency loads.
- **Why:** BUILD_PLAN task 6, MODEL_SPEC §4.1.
- **Checks:** `ruff`: clean; `pytest`: 75 passed. Validation log-loss (fit 20–26, score 27–30): constant 0.352, state × time only 0.304, logistic (C = 0.3) 0.292, LightGBM 0.278.
- **Notes:** `committee_whole_house` is 0 for every training bill: the Bills API only records stage 49 (Committee of the whole House) from session 37 (2022–23); earlier it is coded as stage 8. Its coefficient is therefore 0 — reported to Aleks. LightGBM beats logistic on validation; logistic is kept as the default (MODEL_SPEC: explainable) pending Aleks's decision.

## 2026-10-24 22:12 — feat(model): baselines B0 and B1 (task 5)

- **What:** added `model/splits.py` (train sessions 20–30, test 34–39, training data as of 2019-10-14), `model/baseline.py` (B0, B1, backtest forecast points, MAE), `cli train`, `tests/model/test_baseline.py`, `reports/metrics.json` (`baselines`) and `reports/baselines_mae_by_state.png`.
- **Why:** BUILD_PLAN task 5, MODEL_SPEC §7.
- **Checks:** `ruff`: clean; `pytest`: 70 passed. Result: 320 training Acts, 196 test Acts, 1,678 forecasts; B0 MAE 108.6 days, B1 MAE 78.1 days.
- **Notes:** B1 is by originating House and state (falls back to the state over both Houses below 5 training bills), because a state's position in the chain differs by House; this makes B1 harder to beat. B0 never forecasts a date before `as_of`. Training data is read as of 2019-10-14, the day before the first test session.

## 2026-10-24 22:10 — feat(model): merge renamed departments; note Task 9 session-end rule

- **What:** added `model/departments.py` (renamed or abolished departments → current successor; 40 names → 24) and applied it to the `department` feature and the top-10 one-hot columns. Added a TASK 9 REQUIREMENT note in `model/features.py`: the simulator must recompute `days_to_session_end` from each path's simulated session end, never use the fixed "41+". Rewrote the task 3 test that relied on bill 3064's old wrong label.
- **Why:** Aleks's follow-ups to task 4.
- **Checks:** `ruff`: clean; `pytest`: 65 passed.
- **Notes:** split departments map to one successor (BEIS/BIS/BERR/DTI/DIUS/DIT → Business and Trade; DCMS digital → Culture, Media and Sport). Mapping is model-side only; the database keeps the original names.

## 2026-10-24 22:07 — fix(data): label bills with a Royal Assent sitting as Acts

- **What:** in `data/bills_api.py` (`flatten_bills`), a bill is an Act (`is_act`, outcome `royal_assent`) if the API says so **or** it has a Royal Assent (stage 11) sitting; a warning is logged when this overrides the API. Added an offline test and a Neon DB test that fails if any non-Act bill has a Royal Assent sitting. Reloaded Neon with `cli fetch` (government bills only).
- **Why:** Aleks approved changing the DB code and data. Bill 3064 Commercial Rent (Coronavirus) Bill received Royal Assent on 2022-03-24 (legislation.gov.uk: Commercial Rent (Coronavirus) Act 2022, c. 12, enactment date 2022-03-24), but the Bills API still returns `isAct = False` while its `currentStage` is Royal Assent. Cause: the wrong API flag, not carry-over (the bill was only ever in session 36).
- **Checks:** `ruff`: clean; `pytest`: all passed after the reload. DB now: 522 royal_assent, 26 fell_dissolution, 17 fell_session_end, 18 in_progress, 5 withdrawn; sitting_days 4,274, sessions 19, scheduled 27 (unchanged).
- **Notes:** all 66 non-Act bills were checked against legislation.gov.uk for an Act with the same title enacted while the bill was live. Only one other hit: 2732 Non-Domestic Rating (Lists) Bill [HL], but that Act came from Commons bill 2774 (already an Act); 2732 was dropped after Lords 1st reading, so its label is correct.

## 2026-10-24 21:46 — feat(model): cli build for hazard training rows and advance-rate chart

- **What:** added `cli build [--as-of]` (prints row count, positive rate, rates by state, time in state and days to session end; saves `backend/data/processed/training_rows.parquet`) and `backend/reports/advance_rates.png`.
- **Why:** BUILD_PLAN task 4 "done when": row count and positive rate printed.
- **Checks:** `ruff`: clean; `pytest`: 61 passed. Result as of 2026-10-24: 45,600 rows, 586 bills, positive rate 0.095.
- **Notes:** fewer rows than MODEL_SPEC's 50k–150k estimate because the calendar now uses government-bill sittings only.

## 2026-10-24 21:45 — feat(model): Tier 1 feature registry and hazard training rows (task 4)

- **What:** added `model/features.py`: a registry of Tier 1 features (`(bill_id, as_of, data) -> value`) and `build_training_rows(as_of_cutoff)` producing bill × sitting-day rows with `state, tau_bucket, days_to_session_end_bucket`, features and `y`. Added `tests/model/test_features.py`, `pyarrow`, and restored `backend/data/` and `backend/models/` in `.gitignore`.
- **Why:** BUILD_PLAN task 4 / MODEL_SPEC §4.1, §6.
- **Checks:** `ruff`: clean; `pytest`: 61 passed, including the no-leakage tests (rows match registry values computed the day before; rows unchanged by a later cutoff).
- **Notes:** a τ = 0 row is added on every state's entry day (y = 1 if it moved on the same day), not only for same-day transitions, otherwise bucket 0 would be all positives. Time-varying features use information up to the previous day. Ping-pong and Royal Assent have no hazard rows (MODEL_SPEC §4.4). Department names are used as published, so renamed departments count separately.

## 2026-10-24 21:41 — Merge main into feat/passage-model-tasks-4-11

- **What:** merged latest `main` (government-bills-only rule and ingestion fix #14/#15, three-phase lifecycle view #13). Resolved one conflict in the `model/cli.py` docstring (kept `fetch` without `--gov-only` plus `states`).
- **Why:** Aleks asked to bring the branch up to date before task 4.
- **Checks:** `ruff`: clean; `pytest`: 52 passed.
- **Notes:** bill 3064 labelling issue is still open on `main`.

## 2026-10-24 21:33 — feat(model): task 3 sanity report, Royal Assent check and duration chart

- **What:** ran `cli states` on the Neon data (586 bills, sessions ≥ 20); added a check that every bill that reached Royal Assent is labelled an Act, with a test; saved `backend/reports/state_durations.png` (MODEL_SPEC §7 chart 5).
- **Why:** BUILD_PLAN task 3 "done when" checks and sanity report.
- **Checks:** `ruff`: clean; `pytest`: all passed. Task 3 checks: 4 OK, 1 FAIL (below).
- **Notes:** bill 3064 Commercial Rent (Coronavirus) Bill received Royal Assent on 2022-03-24 (stage 11 sitting in the DB) but the Bills API has `isAct = False`, so the data side labels it `fell_session_end` and `get_transitions` marks its `royal_assent` row as a failure. Data-side labelling issue; reported to Aleks, not worked around.

## 2026-10-24 21:14 — feat(model): states, transition checks and sanity report (task 3)

- **What:** added `model/states.py` (enum-labelled transitions read via `get_bills`/`get_transitions`/`get_calendar`, StopReason labels, session ≥ 20 population filter, task 3 checks, sanity report, duration chart), a `cli states` command and `tests/model/test_states.py`. Added `matplotlib`.
- **Why:** BUILD_PLAN task 3. The stage mapping, chains and skipped-state rule already live on the data side (`data/stage_map.py`, `get_transitions`), so the model reuses them through the interface instead of re-reading tables.
- **Checks:** `ruff`: clean; `pytest tests/model/test_states.py`: 5 passed.
- **Notes:** `build_transitions(as_of)` reads through the interface rather than taking raw `stage_sittings, bills` as BUILD_PLAN suggests, per Aleks's rule that model code never reads tables directly. A full `cli fetch` (all bill types) into Neon was run on Aleks's instruction; the sanity report follows in the next commit.

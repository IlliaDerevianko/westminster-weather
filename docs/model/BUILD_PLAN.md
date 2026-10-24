# Westminster Weather: Model Build Plan (for Claude Code)

Read this file and [`MODEL_SPEC.md`](./MODEL_SPEC.md) fully before writing any code.
Also read [`../../DATA_CONTRACT.md`](../../DATA_CONTRACT.md), [`../../CLAUDE.md`](../../CLAUDE.md) and the existing code in `backend/src/westminster_weather/`.

## 0. Working rules

- **The person directing you (Aleks) is not a programmer.** After each task, explain in 3–5 plain-English bullet points what was built, what the results show, and anything that looks wrong. Show numbers and save charts rather than describing code.
- **Follow the repo's git rules** in `CLAUDE.md`: work on branch `feat/passage-model`, never commit to `main`, commit after every task with a clear message, open a PR only when Aleks asks.
- **Use the existing stack:** Python 3.12, `uv`, packages already in `backend/pyproject.toml` (pandas, numpy, scipy, scikit-learn, lightgbm, lifelines, httpx). Ask before adding a dependency. Allowed additions without asking: `matplotlib`, `pyarrow`, `tqdm`.
- **Reuse the existing enums** `BillStage` and `StopReason` from `westminster_weather.domain.stages`. Do not invent parallel stage names.
- **No data leakage:** every function that builds data or features takes `as_of: date` and must not use anything dated after it. Write tests for this.
- **Run commands from `backend/`:** `uv sync`, `uv run pytest`, `uv run ruff check .`.
- **Stop and ask Aleks** if: an API endpoint does not behave as described here, a validation result is worse than baseline B1, a task will take much longer than its estimate, or a design decision is not covered by the spec.
- **Time budget:** about 12 hours total for tasks 1–10. If behind schedule, do Tier 1 features only and skip optional items marked *(optional)*.

## 1. Target layout

```
backend/
  src/westminster_weather/
    data/
      __init__.py
      bills_api.py        # task 1: download + cache raw Bills API data
      calendar.py         # task 2: sitting days, sessions, parliaments
      interface.py        # task 2: DATA_CONTRACT §5 functions, API-backed
      votes_api.py        # task 7 (optional): Commons/Lords divisions
      members_api.py      # task 7 (optional): state of the parties
    model/
      __init__.py
      states.py           # task 3: stage mapping, chains, transitions
      features.py         # task 4: feature registry + as_of feature builder
      baseline.py         # task 5: baselines B0, B1
      hazard.py           # task 6: advance hazard model
      failure.py          # task 8: session end, dissolution, defeat, withdrawal
      pingpong.py         # task 8: ping-pong model
      simulate.py         # task 9: Monte Carlo simulator
      forecast.py         # task 9: public forecast API + schemas
      evaluate.py         # task 10: backtest, metrics, charts
      alerts.py           # task 11: triggers and alerts
      cli.py              # entry point
  data/cache/             # raw API responses (gitignored)
  data/processed/         # parquet tables (gitignored)
  models/                 # fitted model files (gitignored)
  reports/                # charts + metrics.json (commit these)
  tests/model/            # unit tests
```

Add `backend/data/` and `backend/models/` to `.gitignore`.

CLI (used throughout):
```
uv run python -m westminster_weather.model.cli fetch       # task 1
uv run python -m westminster_weather.model.cli build       # tasks 2–4
uv run python -m westminster_weather.model.cli train       # tasks 5–8
uv run python -m westminster_weather.model.cli evaluate    # task 10
uv run python -m westminster_weather.model.cli forecast --bill-id <BILL_ID> [--as-of 2026-10-24]
uv run python -m westminster_weather.model.cli alerts      # task 11
```

---

## Task 1: Fetch and cache Bills API data (~45 min)

Facts already verified against the live API (2026-10-24):
- Base URL: `https://bills-api.parliament.uk/api/v1` (already in `config.settings`).
- List: `GET /Bills?BillType=1&Take=250&Skip=N` → `{items: [...], totalResults}`. 588 government bills.
- Filter by session uses parameter **`Session`** (not `SessionId`, which is silently ignored).
- Detail: `GET /Bills/{billId}` → keys include `billId, shortTitle, originatingHouse, introducedSessionId, includedSessionIds, isAct, isDefeated, billWithdrawn, sponsors[].organisation.name, lastUpdate, currentStage`.
- Stages: `GET /Bills/{billId}/Stages?Take=100` → `items[]` with `stageId, house, sessionId, sortOrder, stageSittings[] {date, stageId, billStageId}`.
- Reference: `GET /BillTypes`, `GET /Stages?take=100`.
- There is **no** `/Sessions` endpoint (returns 404).

Steps:
1. Extend `clients/parliament_bills.py` (or add `data/bills_api.py`) with a **synchronous** batch downloader using `httpx` with retries, a small concurrency limit (≤ 5 parallel requests) and polite backoff.
2. Download: all government bills (list + detail + stages), and **all bills of all types** list + stages *(used only to build the sitting-day calendar; optional if slow, then use government bills only)*.
3. Cache raw JSON per bill in `backend/data/cache/`. Re-running must skip bills whose `lastUpdate` has not changed.
4. Flatten into parquet: `bills.parquet`, `stage_sittings.parquet` (one row per sitting), matching `DATA_CONTRACT.md` §3.1–3.2 column names.

**Done when:** 588 government bills in `bills.parquet`; every bill has ≥ 1 stage sitting; a test checks the counts of Acts (≈ 521) and the presence of stage 11 for every Act.

## Task 2: Calendar and interface (~1 h)

1. `calendar.py`:
   - `sitting_days`: union of all stage-sitting dates (task 1) — fallback method from MODEL_SPEC §3. If the data teammate's `sitting_days` table exists, prefer it.
   - `sessions`: for each session ID, start = earliest sitting date, end = latest sitting date. Add a hand-maintained `backend/src/westminster_weather/data/sessions_override.csv` with columns `session_id, start_date, end_date, end_type` for corrections. Set `end_type = 'dissolution'` for sessions that ended with a general election (elections: 2010-05-06, 2015-05-07, 2017-06-08, 2019-12-12, 2024-07-04), otherwise `'prorogation'`. The current session has `end_date = NULL`.
   - `parliaments`: derived from election dates.
   - Future sitting-day generator (MODEL_SPEC §3).
2. `interface.py`: implement the four functions from `DATA_CONTRACT.md` §5 — `get_bills(as_of)`, `get_transitions(as_of)`, `get_covariates(as_of)`, `get_calendar(as_of)` — backed by the parquet files. Same signatures and columns as the contract so they can later be swapped for the database implementation without touching `model/`.
3. Update `DATA_CONTRACT.md` §2.5 so `model_state` values are the `BillStage` enum values (e.g. `commons_2nd_reading`) instead of `C_2R`, and the stage-ID mapping matches MODEL_SPEC §2.1.

**Done when:** tests prove that `get_transitions(as_of=d)` never returns an `entered_date` or `exited_date` after `d`, and that a bill in progress on `d` has `exit_type = NULL` for its current state.

## Task 3: States and transitions (~1 h)

`model/states.py`:
1. `STAGE_ID_TO_STATE` mapping from MODEL_SPEC §2.1.
2. `chain(originating_house) -> list[BillStage]` from §2.2.
3. `build_transitions(stage_sittings, bills, as_of)`: entry and exit date per state per bill, applying the skipped-state rule from §2.2, labelling failures with `StopReason` per §2.3, counting sitting days in each state.
4. Print a sanity report: count of bills per final state and outcome; median sitting days per state; number of bills with ping-pong.

**Done when:** every Act ends in `ROYAL_ASSENT`; every non-Act is either in progress (current session) or has a `StopReason`; no negative durations. Show Aleks the sanity report.

## Task 4: Features (~1.5 h)

`model/features.py`:
1. A **feature registry**: dict of name → function `(bill_id, as_of, data) -> value`, using the names in `DATA_CONTRACT.md` §5.5 and MODEL_SPEC §6.
2. `build_training_rows(as_of_cutoff)`: bill × sitting-day rows per MODEL_SPEC §4.1, with columns `bill_id, state, tau_bucket, days_to_session_end_bucket, <features>, y`.
3. Tier 1 features only in this task.

**Done when:** row count and positive rate are printed; a test checks that features for a given row use no data after that row's date.

## Task 5: Baselines (~30 min)

`model/baseline.py`: B0 and B1 from MODEL_SPEC §7. Save results to `reports/metrics.json` under `baselines`.

**Done when:** Aleks sees the B0 and B1 MAE in days on the test sessions.

## Task 6: Advance hazard (~2 h)

`model/hazard.py`:
1. Fit the pooled logistic model (MODEL_SPEC §4.1) on train sessions. Choose `C` by time-ordered CV (e.g. train on sessions ≤ 26, validate on 27–30).
2. Save the fitted model and the feature list to `backend/models/hazard.joblib`.
3. Produce the coefficient chart (`reports/hazard_coefficients.png`).
4. *(optional)* LightGBM version on the same rows; keep it only if it beats logistic on validation log-loss.

**Done when:** coefficients have sensible signs (e.g. `committee_whole_house` speeds things up, near-session-end buckets speed things up), shown to Aleks.

## Task 7: *(optional)* Votes and Members data (~1.5 h)

Only after tasks 1–6 work. Check endpoints in the Swagger docs listed at `https://developer.parliament.uk` before coding:
- Commons Votes API (`commonsvotes-api.parliament.uk`), Lords Votes API (`lordsvotes-api.parliament.uk`): divisions, linked to bills by title text and date; government position and rebels.
- Members API (`members-api.parliament.uk`): state of the parties by House and date → `working_majority`, `lords_government_share`.

Add Tier 1b features, retrain, and report whether validation improved. Keep features only if they help.

## Task 8: Failure routes and ping-pong (~1.5 h)

`model/failure.py` and `model/pingpong.py`, exactly as MODEL_SPEC §4.2–4.5:
- Session-length distribution, dissolution hazard, carry-over rule.
- Defeat probit (constant fallback if Task 7 skipped).
- Withdrawal constant hazard.
- Ping-pong occurrence, rounds, round gaps, gap to Royal Assent.

**Done when:** each fitted component prints its parameters in plain English (e.g. "average session lasts 152 sitting days").

## Task 9: Simulator and forecast API (~2 h)

1. `model/simulate.py`: vectorised Monte Carlo per MODEL_SPEC §5, 10,000 paths, fixed seed per (bill, `as_of`).
2. `model/forecast.py`: `forecast_bill(bill_id, as_of) -> BillForecastResponse` plus per-state results.
3. Add `StageForecastSchema` to `schemas/bills.py` (fields from `DATA_CONTRACT.md` §6.3) and an optional `stages: list[StageForecastSchema]` field on `BillForecastResponse`. Do not break existing fields or tests.
4. Add endpoint `GET /api/v1/bills/{bill_id}/forecast?as_of=YYYY-MM-DD` in `main.py`.

**Done when:** a forecast runs in under 5 s, probabilities across outcomes sum to 1 (test), and Aleks sees forecasts for 3 historical bills at different stages next to what actually happened.

## Task 10: Evaluation (~1.5 h)

`model/evaluate.py`: backtest and metrics from MODEL_SPEC §7, all five charts, write `reports/metrics.json` and `reports/EVALUATION.md` (plain-English summary: model vs baselines, what drives speed, limitations).

**Done when:** the report exists and the model beats B1 on MAE. If not, stop and discuss with Aleks before continuing.

## Task 11: Alerts (~1 h)

`model/alerts.py` per MODEL_SPEC §8:
- Store forecasts and alerts in `backend/data/processed/forecasts.parquet` and `alerts.parquet` (later: database tables from `DATA_CONTRACT.md` §6).
- `cli alerts`: refresh data for in-progress bills, detect triggers, re-forecast, write alerts, print them.
- Add endpoint `GET /api/v1/alerts?since=...` for the front end.

**Done when:** replaying a past bill day by day (e.g. through its whole passage) produces a sensible sequence of alerts, shown to Aleks.

## Task 12: Hand-off

- Update `README.md` with a short "Passage model" section: how to run each CLI command.
- Commit, push the branch, and tell Aleks it's ready for a PR.

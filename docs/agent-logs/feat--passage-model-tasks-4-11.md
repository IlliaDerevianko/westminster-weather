# Agent log: `feat/passage-model-tasks-4-11`

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

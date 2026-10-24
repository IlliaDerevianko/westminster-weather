## 2026-10-24 20:46 — perf(data): increase parallel downloads to 20 and verify 588 bills in Neon

- **What:** Increased `MAX_PARALLEL` to 20 in `data/bills_api.py` and ingested all 588 UK Government Bills (`bills`), 10,290 `stage_sittings`, 49 `stage_map` rows, 19 `sessions`, 4,274 `sitting_days`, and 27 `scheduled_sittings` into Neon Lakebase Postgres.
- **Why:** Populate Neon Lakebase Postgres with the full historical dataset of all 588 UK Government Bills from the UK Parliament Bills API.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -v`: 42 passed (including live Neon DB count & no-leakage checks).

## 2026-10-24 20:42 — test(data): add bills_api and interface tests against Neon Postgres

- **What:** Added `tests/model/test_bills_api.py` and `tests/model/test_interface.py` testing bill flattening, outcome labelling, state transitions, covariates, and live Neon Lakebase Postgres dataset counts (~588 Government Bills) and no-leakage guarantees.
- **Why:** Verify that all UK Government Bills and stage sittings in Neon Lakebase Postgres satisfy the `DATA_CONTRACT.md` invariants and §5 model interface.
- **Checks:** `uv run ruff check .`: clean; synthetic unit tests pass; Neon DB assertions run once `cli fetch` finishes populating Neon.

## 2026-10-24 20:38 — feat(data): add Neon DB adapters, bills_api, and interface

- **What:** Added `save_dataset_tables_to_db`, `fetch_bills_df`, `fetch_stage_sittings_df`, `fetch_scheduled_sittings_df`, `fetch_sessions_df`, and `fetch_sitting_days_df` in `db.py`, plus `data/bills_api.py`, `data/interface.py` (`DatabaseInterface`), and `model/cli.py`.
- **Why:** Ingest all UK Government Bills and stage sittings directly into Neon Lakebase Postgres and expose the `DATA_CONTRACT.md` §5 interface over Neon Postgres instead of Parquet/SQLite files.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 17 passed.

## 2026-10-24 20:34 — feat(db): add DATA_CONTRACT schema models and stage_map

- **What:** Added `StageMapRow` (`stage_map`), `ParliamentSessionRow` (`sessions`), `SittingDayRow` (`sitting_days`), `BillRow` (`bills`), `StageSittingRow` (`stage_sittings`), and `ScheduledSittingRow` (`scheduled_sittings`) to `models.py`, plus `data/stage_map.py`, full `BillStage`/`StopReason` enum values in `domain/stages.py`, and `DATA_CONTRACT.md`.
- **Why:** Prepare Neon Lakebase Postgres schema to store all ~588 UK Government Bills and their stage sittings as the single source of truth.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 17 passed.

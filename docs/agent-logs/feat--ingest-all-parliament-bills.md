## 2026-10-24 20:38 — feat(data): add Neon DB adapters, bills_api, and interface

- **What:** Added `save_dataset_tables_to_db`, `fetch_bills_df`, `fetch_stage_sittings_df`, `fetch_scheduled_sittings_df`, `fetch_sessions_df`, and `fetch_sitting_days_df` in `db.py`, plus `data/bills_api.py`, `data/interface.py` (`DatabaseInterface`), and `model/cli.py`.
- **Why:** Ingest all UK Government Bills and stage sittings directly into Neon Lakebase Postgres and expose the `DATA_CONTRACT.md` §5 interface over Neon Postgres instead of Parquet/SQLite files.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 17 passed.

## 2026-10-24 20:34 — feat(db): add DATA_CONTRACT schema models and stage_map

- **What:** Added `StageMapRow` (`stage_map`), `ParliamentSessionRow` (`sessions`), `SittingDayRow` (`sitting_days`), `BillRow` (`bills`), `StageSittingRow` (`stage_sittings`), and `ScheduledSittingRow` (`scheduled_sittings`) to `models.py`, plus `data/stage_map.py`, full `BillStage`/`StopReason` enum values in `domain/stages.py`, and `DATA_CONTRACT.md`.
- **Why:** Prepare Neon Lakebase Postgres schema to store all ~588 UK Government Bills and their stage sittings as the single source of truth.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 17 passed.

## 2026-10-24 20:34 — feat(db): add DATA_CONTRACT schema models and stage_map

- **What:** Added `StageMapRow` (`stage_map`), `ParliamentSessionRow` (`sessions`), `SittingDayRow` (`sitting_days`), `BillRow` (`bills`), `StageSittingRow` (`stage_sittings`), and `ScheduledSittingRow` (`scheduled_sittings`) to `models.py`, plus `data/stage_map.py`, full `BillStage`/`StopReason` enum values in `domain/stages.py`, and `DATA_CONTRACT.md`.
- **Why:** Prepare Neon Lakebase Postgres schema to store all ~588 UK Government Bills and their stage sittings as the single source of truth.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 17 passed.

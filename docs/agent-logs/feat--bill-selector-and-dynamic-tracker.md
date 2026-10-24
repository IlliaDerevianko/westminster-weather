## 2026-10-24 21:04 — feat(api): add /api/v1/bills list and dynamic DB evaluation for any bill_id

- **What:** Added `GET /api/v1/bills` (with `search`, `outcome`, and `house` filters) and extended `GET /api/v1/bills/{bill_id}` to evaluate any of the 588 UK Government Bills in Neon Lakebase Postgres at any `as_of_date`, returning database-driven `milestone_dates`, `stages`, `visible_events`, and `active_stop_reason`.
- **Why:** Allow the frontend to list and select any bill from Neon Postgres and render its stage-by-stage progression without hardcoded titles, dates, or presets.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest tests/test_health_and_stages.py`: 13 passed.

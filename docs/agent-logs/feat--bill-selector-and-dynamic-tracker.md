## 2026-10-24 21:08 — feat(frontend): add /bills selector page and dynamic DB-driven tracker page

- **What:** Added `/bills` directory page (`frontend/src/app/bills/page.tsx`) to search, filter, and select any of the 588 UK Government Bills in Neon Lakebase Postgres, and updated `/` (`frontend/src/app/page.tsx`) to load any selected bill (`?bill_id=<id>`) with database-driven stages, sittings, stop reasons, and milestone dates without hardcoded values.
- **Why:** Let users browse all bills in Neon Postgres and display any chosen bill on the main tracker page with zero hardcoded bill data or date presets.
- **Checks:** `npx tsc --noEmit`: clean; `GET /bills` and `GET /?bill_id=4030`: HTTP 200.

## 2026-10-24 21:04 — feat(api): add /api/v1/bills list and dynamic DB evaluation for any bill_id

- **What:** Added `GET /api/v1/bills` (with `search`, `outcome`, and `house` filters) and extended `GET /api/v1/bills/{bill_id}` to evaluate any of the 588 UK Government Bills in Neon Lakebase Postgres at any `as_of_date`, returning database-driven `milestone_dates`, `stages`, `visible_events`, and `active_stop_reason`.
- **Why:** Allow the frontend to list and select any bill from Neon Postgres and render its stage-by-stage progression without hardcoded titles, dates, or presets.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest tests/test_health_and_stages.py`: 13 passed.

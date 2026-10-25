# Agent log: `chore/repo-cleanup`

## 2026-10-25 08:49 — chore: remove finished research experiments; keep stage 3 self-contained

- **What:** Deleted the remaining-time model experiments (`analysis/remaining_time_dataset`, `_models`, `_v2`, `_tails`), `origin_comparison`, `analysis/stage_durations`, `stakeholder_chart`, `journey/journey_test`, `stage2_ra`, `journey/stage_durations`; their 5 test files; report folders `remaining_time_models`, `_v2`, `_tails`, `_tails_75_25`, `origin_comparison`, `stage_correlation`, the non-stage-3 files in `reports/journey`, the combined stage 2→3 chart; and `docs/model/REMAINING_TIME_MODELS.md`.
- **Kept working:** the live stage 3 model (`analysis/commencement_xgb`) read its tuned XGBoost settings from `reports/remaining_time_models_v2/metrics.json`; they are now the constant `XGB_PARAMS` (checked identical to what was read). Its chart and the deadline calibration chart took colours from `stakeholder_chart`; these now come from a new `analysis/chart_style.py`. Removed `commencement_xgb.stage2_to_3_chart` (it read the deleted experiment's predictions).
- **Why:** Category B of the user's cleanup.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 257 passed (19 experiment tests removed with their modules). All 45 remaining modules import; no imports of deleted modules. Live: forecast, outlook and bill endpoints 200; frontend 200.
- **Notes:** Everything is recoverable from git history.

## 2026-10-25 08:46 — chore: remove the take-up app, keep the pieces stage 3 reuses

- **What:** Removed the take-up API (`/api/v1/takeup/*` and its router in `main.py`), CLI, ingestion and NLP pipeline, evaluation and replays (12 modules, ~5,950 lines), their 6 test files, `reports/takeup/`, `docs/model/TAKEUP_PITCH.md` and `TAKEUP_EVALUATION.md`. Kept `takeup/model.py`, `conformal.py`, `signals.py`, `tables.py`, `db.py` and their 2 test files, because `journey/commencement.py` (stage 3, used by the forecast endpoint) imports them; noted this in the package docstring and at the top of `TAKEUP_PLAN.md`.
- **Why:** Category C of the user's cleanup; the promises pages had already been removed, so nothing called the take-up API.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 276 passed (117 take-up tests removed with their modules). Live: `/api/v1/takeup/ideas` 404; `/api/v1/bills/4305/forecast` 200.
- **Notes:** The forecast's stage 3 fields are null for bills 4305 and 3751, identically on `main` (checked in a temporary worktree), so that predates this change. Take-up database tables and `database_url_takeup` settings are kept (still used by `takeup.db`).

## 2026-10-25 08:43 — chore: remove the unused bill_outlooks generator and table definition

- **What:** Deleted `model/outlook_generate.py`, `db.save_bill_outlooks_to_db`, the `BillOutlookRow` ORM model and DATA_CONTRACT §6.6.
- **Why:** Category D of the user's cleanup. `GET /api/v1/bills/{id}/outlook` computes outlooks on request; nothing reads the `bill_outlooks` table.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 393 passed.
- **Notes:** The `bill_outlooks` table itself was not dropped from the database (if it exists, it is simply no longer written or created).

## 2026-10-25 08:41 — chore: remove unused API clients

- **What:** Deleted `clients/govuk_content.py`, `clients/parliament_bills.py` and `clients/legislation.py` (nothing imports them) and the two config settings only they used (`legislation_gov_uk_base_url`, `gov_uk_content_api_base_url`). `parliament_bills_api_base_url` stays (used by `data/bills_api.py`).
- **Why:** User asked to clean up unused code; chose categories A–D (dead code, research experiments, take-up backend, outlook generator). This is A.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 393 passed.

# Agent log: `feat/bill-view-first-signal-and-cleanup`

Combined from the stacked branches `feat/fast-tracked-bypassed-stages`, `feat/first-signal-date`, `feat/simplify-bill-view` and `feat/first-signal-milestone`, which were merged into this one branch.

## 2026-10-25 — docs(logs): combine agent logs into one file for the merged branch

- **What:** Renamed branch `feat/first-signal-milestone` to `feat/bill-view-first-signal-and-cleanup` and combined the four per-branch log files into this one, keeping all 10 entries newest first. Corrected the Checks line of the Neon-removal entry.
- **Why:** User asked to merge the stacked branches into one; AGENTS.md expects one log file per branch.
- **Checks:** none (docs only).
- **Notes:** The three older branches were deleted locally with `git branch -d` after confirming they were fully contained in this branch; none were on `origin`.

## 2026-10-25 — feat(domain): add first-signal milestone for every bill and merge same-date milestones

- **What:** Every bill now gets a "1. Signal appears" milestone on its `earliest_signal_date`, not only bills with pre-parliament records (#3137). Milestones that share a date are merged into one entry (labels joined with " · ", notes with "; ") instead of the later ones being dropped.
- **Why:** User asked why there was no milestone for when the first signal is detected. For bills without pre-parliament data the signal date equals the 1st reading date, and the old de-duplication by date dropped the signal milestone.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q`: 49 passed. Manual API check: #3703 → `2024-03-20 (1. Signal appears · 2. Commons 1st reading)`; #3137 unchanged at `2017-10-10 (1. Signal appears)`, and its 2022-03-17 entry now also lists Commons 1st reading.
- **Notes:** Merging also brings back same-day milestones that were hidden before, e.g. #4296 now shows Commons 2nd reading, committee, report and 3rd reading together on 2026-10-21. Branched from `feat/simplify-bill-view` (unmerged).

## 2026-10-25 — feat(ui): remove API links from stage cards

- **What:** Removed the "API ↗" link (to the Bills API `source_api_url`) shown next to each stage's dates on the bill view.
- **Why:** User asked to remove the API links.
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=3137` returns 200.
- **Notes:** `source_api_url` is still returned by the backend; only the UI link is gone. No other external links remain on the bill view.

## 2026-10-25 — feat(ui): remove phase overview boxes from bill view header

- **What:** Removed the row of three phase boxes (Before Parliament / Inside Parliament / After Royal Assent with "x/y done" counts) from the top of the single bill page.
- **Why:** User asked to remove the three boxes showing how many stages passed.
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=3137` returns 200 and no longer contains the "done" counts.
- **Notes:** Each phase column below still shows its own completed count.

## 2026-10-25 — feat(ui): remove Neon database mentions from the interface

- **What:** Removed every user-visible mention of Neon / Lakebase Postgres from the bill view and bills list: the selector label and aria-label, date-picker hint, milestone heading, loading and error messages, and the blue "Neon Lakebase Postgres Database" badge.
- **Why:** User said the database shouldn't be part of the interface.
- **Checks:** `npx tsc --noEmit`: clean; `grep` of `frontend/src` finds "Neon" only in two code comments; bill API responses contain no Neon text. (Correction: an earlier version of this entry cited a curl of `/?bill_id=3137`, but that page is rendered client-side, so the check could not see bill content.)
- **Notes:** Left code comments and backend-only text (FastAPI `/docs` description, `/health` `neon_branch`) as they are not shown in the app.

## 2026-10-25 — feat(ui): remove detailed stack layout and lifecycle label from bill view

- **What:** Removed the "3-Column Pipeline / Detailed Stack" layout toggle (and its `layoutMode` state) so the bill view always shows the 3-column pipeline. Removed the black "Full 3-Phase UK Government Bill Lifecycle" label from the bill header.
- **Why:** User said the detailed stack view gives no valuable information and asked for the label to be removed.
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=3137` and `/bills` return 200 on the dev server. Backend unchanged.
- **Notes:** Branched from `feat/first-signal-date` (unmerged) so the bill view keeps the First Signal change.

## 2026-10-25 — feat(ui): show first signal date instead of 1st reading on every bill

- **What:** The tracker header (`frontend/src/app/page.tsx`) and bill cards (`frontend/src/app/bills/page.tsx`) now show one **First Signal** date (`earliest_signal_date`) in place of the 1st Reading (Parliament) date and the separate Pre-Parliament Signal label.
- **Why:** User asked for every bill view to display the very first signal instead of the 1st reading in Parliament.
- **Checks:** `npx tsc --noEmit`: clean; backend unchanged in this commit.
- **Notes:** The 1st reading date is still visible on the Phase 2 stage cards and in the API response (`first_reading_date`).

## 2026-10-25 — feat(domain): derive earliest signal date from first pre-parliamentary stage

- **What:** `evaluate_parliament_bill_from_db` now sets `earliest_signal_date` to the date of the first Phase 1 stage (`signal_appears`). `list_all_bills_from_db` falls back to the 1st reading date for bills without pre-parliament data instead of returning `null`.
- **Why:** User asked for every bill view to show the very first signal instead of the 1st reading; for bills without pre-parliament records that is the first pre-parliamentary stage.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q`: 49 passed. Manual: #3137 → 2017-10-10, #3703 → 2024-03-20; 0 of 588 list rows have a null signal date.
- **Notes:** Only #3137 (Online Safety Bill) has real pre-parliament data, so for the other 587 bills the first signal equals the 1st reading date. Branched from `feat/fast-tracked-bypassed-stages` (unmerged) because it relies on its `earliest_signal_date` field.

## 2026-10-25 — Antigravity — `fix(ui,db): distinguish 2017 pre-parliament signal from 2022 1st reading and guard init_db`

- **Branch**: `feat/fast-tracked-bypassed-stages`
- **Summary**:
  - Updated `list_all_bills_from_db` and `BillSummarySchema` to return `earliest_signal_date` alongside `introduced_date` (Parliamentary 1st Reading date).
  - Updated `/bills` (`frontend/src/app/bills/page.tsx`) and the main tracker header/cards (`frontend/src/app/page.tsx`) to explicitly distinguish **Pre-Parliament Signal** (e.g. `2017-10-10` for Online Safety Act 2023 `#3137`) from **1st Reading (Parliament)** (`2022-03-17`), and display Phase 1 stage summaries directly on the cards.
  - Guarded `init_db()` in `backend/src/westminster_weather/db.py` with `_reference_tables_initialized` and `_init_db_lock` so DDL and reference table seeding run once per process instead of on every API request.
- **Files touched**:
  - `backend/src/westminster_weather/db.py`
  - `backend/src/westminster_weather/domain/bills.py`
  - `backend/src/westminster_weather/schemas/bills.py`
  - `frontend/src/app/bills/page.tsx`
  - `frontend/src/app/page.tsx`
  - `docs/agent-logs/feat--fast-tracked-bypassed-stages.md`
- **Verification**: `uv run ruff check .`, `uv run pytest -q`, and `npx tsc --noEmit` all passed.

## 2026-10-25 — Antigravity — `feat(ui): display red stop box only on stage where bill actually stopped`

- **Branch**: `feat/fast-tracked-bypassed-stages`
- **Summary**: Updated the 3-phase pipeline view in `frontend/src/app/page.tsx` so the red stop-risk box (`stopMeta`) on the right of a stage card is rendered **only** when that stage actually stopped (`stage.status === "stopped"` and `stage.active_stop_reason` is set). Bills that turn into law or are still in progress now render full-width (`col-span-12`) stage cards with no unused red stop boxes, while stopped bills (e.g. Bill `#3703` *Tobacco and Vapes Bill* at `commons_report_stage` due to `election_called`) display only the taken stop box on the exact stage where they stopped.
- **Files touched**:
  - `frontend/src/app/page.tsx`
  - `docs/agent-logs/feat--fast-tracked-bypassed-stages.md`
- **Verification**: `npx tsc --noEmit` and `uv run pytest -q` passed.

## 2026-10-25 — Antigravity — `feat(stages): mark bypassed pre-parliamentary and 0-sitting parliamentary stages as fast_tracked`

- **Branch**: `feat/fast-tracked-bypassed-stages`
- **Summary**: Added `StageStatus.FAST_TRACKED` (`"fast_tracked"`) so that when a bill bypasses formal pre-parliamentary stages (`signal_appears` .. `slot_in_programme` with no separate pre-legislative records prior to First Reading) or dispenses with a Parliamentary stage without formal sittings (e.g. `commons_report_stage` with 0 sittings on Bill `#4296` *Church Appointments (Repeal of Discriminatory Provision) Bill* when proceeding directly from Committee of the whole House to 3rd Reading), those stages are explicitly marked `"fast_tracked"` with a `Fast-Tracked` / `Fast-Tracked (0 sittings)` badge, `»` icon, and explanatory summary in the UI.
- **Files touched**:
  - `backend/src/westminster_weather/domain/bills.py`
  - `backend/tests/test_health_and_stages.py`
  - `frontend/src/app/page.tsx`
  - `docs/agent-logs/feat--fast-tracked-bypassed-stages.md`
- **Verification**: `uv run ruff check .`, `uv run pytest -q`, and `npx tsc --noEmit` all passed.

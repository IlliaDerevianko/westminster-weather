# Agent log: `feat/simplify-bill-view`

## 2026-10-25 — feat(ui): remove Neon database mentions from the interface

- **What:** Removed every user-visible mention of Neon / Lakebase Postgres from the bill view and bills list: the selector label and aria-label, date-picker hint, milestone heading, loading and error messages, and the blue "Neon Lakebase Postgres Database" badge.
- **Why:** User said the database shouldn't be part of the interface.
- **Checks:** `npx tsc --noEmit`: clean; rendered `/?bill_id=3137` has 0 matches for "neon"; bill API responses contain no Neon text.
- **Notes:** Left code comments and backend-only text (FastAPI `/docs` description, `/health` `neon_branch`) as they are not shown in the app.

## 2026-10-25 — feat(ui): remove detailed stack layout and lifecycle label from bill view

- **What:** Removed the "3-Column Pipeline / Detailed Stack" layout toggle (and its `layoutMode` state) so the bill view always shows the 3-column pipeline. Removed the black "Full 3-Phase UK Government Bill Lifecycle" label from the bill header.
- **Why:** User said the detailed stack view gives no valuable information and asked for the label to be removed.
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=3137` and `/bills` return 200 on the dev server. Backend unchanged.
- **Notes:** Branched from `feat/first-signal-date` (unmerged) so the bill view keeps the First Signal change.

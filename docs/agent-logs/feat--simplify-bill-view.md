# Agent log: `feat/simplify-bill-view`

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
- **Checks:** `npx tsc --noEmit`: clean; rendered `/?bill_id=3137` has 0 matches for "neon"; bill API responses contain no Neon text.
- **Notes:** Left code comments and backend-only text (FastAPI `/docs` description, `/health` `neon_branch`) as they are not shown in the app.

## 2026-10-25 — feat(ui): remove detailed stack layout and lifecycle label from bill view

- **What:** Removed the "3-Column Pipeline / Detailed Stack" layout toggle (and its `layoutMode` state) so the bill view always shows the 3-column pipeline. Removed the black "Full 3-Phase UK Government Bill Lifecycle" label from the bill header.
- **Why:** User said the detailed stack view gives no valuable information and asked for the label to be removed.
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=3137` and `/bills` return 200 on the dev server. Backend unchanged.
- **Notes:** Branched from `feat/first-signal-date` (unmerged) so the bill view keeps the First Signal change.

# Agent log: `chore/remove-promises-page`

## 2026-10-25 07:20 — feat(frontend): outlook chart y-axis at 0, 25, 50, 75 and 100%

- **What:** Replaced the 90% gridline and label with 25% and 75%; the inner lines (25/50/75%) are dashed.
- **Why:** User asked for 25% and 75% on the axis instead of 90%.
- **Checks:** `npx tsc --noEmit`: clean.
- **Notes:** The curve sentence still mentions when the chance passes 90% (from the API). The user's own uncommitted edit in this file was left out.

## 2026-10-25 07:19 — feat(frontend): outlook chart shows one year with monthly ticks and larger axis text

- **What:** The "Chance of becoming law by date" chart now ends one year after the forecast date (was 2 years), has a tick and gridline at the start of every month (short month name; year under January and the first tick) instead of yearly marks, and axis labels at 13px (was 10px). Bottom padding grows for the two-line labels.
- **Why:** User asked for bigger axis fonts, monthly marks and a one-year cutoff.
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=4305` loads (200). Not checked visually in a browser.
- **Notes:** Only the chart is cut; the API still returns 2 years of points, and the curve sentence above the chart can name a month beyond the one-year window. The target line is hidden if the target falls after the cutoff. The user's own uncommitted edit in this file was left out.

## 2026-10-25 07:17 — feat(frontend): summary boxes filter the bills directory by outcome

- **What:** The four summary boxes on `/bills` (Total, Royal Assent, In Progress, Fell / Withdrawn) are now toggle buttons that filter the list; "Total Government Bills" (all bills) is selected by default. The Outcome dropdown is removed and the search box widened to fill its space. Box definitions, counts and filtering share one `OUTCOME_GROUPS` list.
- **Why:** User asked to remove the outcome selector and make the summary boxes clickable, defaulting to all bills.
- **Checks:** `npx tsc --noEmit`: clean; `/bills` loads (200). Not clicked through in a browser.
- **Notes:** The dropdown's finer choices (fell at dissolution vs session end, withdrawn) are now one "Fell / Withdrawn" group, matching the box count. The house filter is unchanged. The user's own uncommitted edit in this file was left out.

## 2026-10-25 07:15 — chore(frontend): remove the target-date picker from the weather outlook panel

- **What:** Removed the "Target date" input and its "next recess" reset link from `BillOutlookPanel`, and the `targetDate` state; the panel always uses the API's default target (next recess more than 30 days away).
- **Why:** User asked to remove the date selector and "next recess" from the prediction window.
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=4305` loads (200).
- **Notes:** `fetchOutlook`'s optional `targetDate` argument and the API's `target_date` parameter are kept (now unused by the UI). The user's own uncommitted edit to this file (footnote removal) was left out.

## 2026-10-25 07:13 — Merge main (PR #24 remove-amended-stage, PR #25 bill-outlook) into chore/remove-promises-page

- **What:** Brought in main's weather outlook panel, outlook endpoint/generation/calibration and the removal of the "Amended or repealed" stage. No conflicts.
- **Why:** User asked where the predictions were; local branches predated PR #25.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 389 passed; `npx tsc --noEmit`: clean.
- **Notes:** The user's uncommitted UI edits were stashed before the merge and restored after it.

## 2026-10-25 07:10 — chore(frontend): remove the "Back to Tracker Page" button from the bills directory

- **What:** Removed the "← Back to Tracker Page" button and its now-empty wrapper from the bills directory header.
- **Why:** User asked to remove the back to tracker page button.
- **Checks:** `npx tsc --noEmit`: clean; `/bills` loads (200).
- **Notes:** Same nav clean-up as the previous commit, so kept on this branch. The user's own uncommitted edits were left out.

## 2026-10-25 07:08 — chore(frontend): remove the promise take-up forecasts pages and their buttons

- **What:** Deleted `/promises` and `/promises/[ideaId]`, the front-end take-up client `lib/takeup.ts` (used only by those pages), the "Promises → Bills" button on the tracker page and the "Promise Take-up Forecasts →" button on the bills directory.
- **Why:** User asked to remove the promise take-up forecasts page and the button leading to it.
- **Checks:** `npx tsc --noEmit`: clean (after clearing stale generated `.next/types` for the deleted route); dev server: `/` and `/bills` 200, `/promises` 404. No references to `/promises` or takeup left in `frontend/src`.
- **Notes:** Backend take-up API (`/api/v1/takeup/*`) and model untouched. The user's own uncommitted edits to `page.tsx` and `bills/page.tsx` were left out of this commit. Branched from `fix/outcome-badge-in-force`.

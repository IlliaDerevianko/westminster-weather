# Agent log: `chore/remove-promises-page`

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

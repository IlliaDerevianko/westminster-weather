# Agent log: `feat/simplify-bill-view`

## 2026-10-25 — feat(ui): remove detailed stack layout and lifecycle label from bill view

- **What:** Removed the "3-Column Pipeline / Detailed Stack" layout toggle (and its `layoutMode` state) so the bill view always shows the 3-column pipeline. Removed the black "Full 3-Phase UK Government Bill Lifecycle" label from the bill header.
- **Why:** User said the detailed stack view gives no valuable information and asked for the label to be removed.
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=3137` and `/bills` return 200 on the dev server. Backend unchanged.
- **Notes:** Branched from `feat/first-signal-date` (unmerged) so the bill view keeps the First Signal change.

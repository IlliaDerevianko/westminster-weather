# Agent log: `feat/first-signal-date`

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

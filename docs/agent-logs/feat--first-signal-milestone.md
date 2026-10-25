# Agent log: `feat/first-signal-milestone`

## 2026-10-25 — feat(domain): add first-signal milestone for every bill and merge same-date milestones

- **What:** Every bill now gets a "1. Signal appears" milestone on its `earliest_signal_date`, not only bills with pre-parliament records (#3137). Milestones that share a date are merged into one entry (labels joined with " · ", notes with "; ") instead of the later ones being dropped.
- **Why:** User asked why there was no milestone for when the first signal is detected. For bills without pre-parliament data the signal date equals the 1st reading date, and the old de-duplication by date dropped the signal milestone.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q`: 49 passed. Manual API check: #3703 → `2024-03-20 (1. Signal appears · 2. Commons 1st reading)`; #3137 unchanged at `2017-10-10 (1. Signal appears)`, and its 2022-03-17 entry now also lists Commons 1st reading.
- **Notes:** Merging also brings back same-day milestones that were hidden before, e.g. #4296 now shows Commons 2nd reading, committee, report and 3rd reading together on 2026-10-21. Branched from `feat/simplify-bill-view` (unmerged).

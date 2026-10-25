# Agent log: `fix/list-latest-stage`

## 2026-10-25 07:34 — feat(api): latest confirmed stage for Acts in the bill list

- **What:** `GET /api/v1/bills` returns `latest_stage` and `latest_stage_label` for Acts: the furthest stage completed after Royal Assent as of today (Law fully in force, Regulations laid, Regulations drafted, Commencement orders, Some sections start, or Royal Assent). Computed with three bulk queries; the in-force rule is moved into `fully_in_force_date()`, shared with the bill page. Tests for the rule and for list vs bill page agreement on 6 Acts.
- **Why:** User: most green bills in the list show "Royal Assent" though they are past that stage; show the latest confirmed stage instead. `bills.outcome` stops at Royal Assent.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 394 passed. One-off check of all 522 Acts: the list's stage equals the bill page's last completed stage (as of today) for every one. List endpoint still 0.7 s.
- **Notes:** Current spread: 356 fully in force, 110 regulations laid, 46 commencement orders, 7 some sections start, 3 regulations drafted. Bills that are not Acts get `null`. Branched from `feat/full-text-link`.

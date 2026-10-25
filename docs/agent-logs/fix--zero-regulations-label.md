# Agent log: `fix/zero-regulations-label`

## 2026-10-25 06:55 — fix(stages): say "none yet" for post-Royal-Assent stages with zero SIs

- **What:** When an Act has no Commencement Orders, Draft SIs or laid SIs and is not yet fully in force, those stages stay `upcoming` but now carry a summary ("None drafted yet — …", "None laid yet — …") and a legislation.gov.uk link. New test for bill 3751 (Water (Special Measures) Act 2025).
- **Why:** User asked why bill 3751's "Regulations drafted" and "Regulations laid" boxes were plain grey "Upcoming". Since PR #23 the live in-force check finds 2 of 33 sections still prospective, so zero-SI stages are `upcoming` rather than `skipped`, which looked the same as a bill before Royal Assent.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 380 passed.
- **Notes:** No API schema change; status values are unchanged. Not checked whether legislation.gov.uk has regulations under this Act that ingestion missed.

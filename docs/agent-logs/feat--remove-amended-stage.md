# Agent log: `feat/remove-amended-stage`

## 2026-10-25 06:20 — feat(stages): remove the "Amended or repealed" step after Royal Assent

- **What:** dropped `AMENDED_OR_REPEALED` from `POST_ROYAL_ASSENT_STAGE_INFO` (`domain/stages.py`), its stage-building branch and "3. Amended / Repealed" milestone in `domain/bills.py`, and its count label in `frontend/src/app/page.tsx`. The bill timeline now has 21 stages (6 after Royal Assent, ending at "Law fully in force"). The `BillStage` value and the amendment data in `post_royal_assent_events` / `act_commencements` are kept, only no longer shown. `tests/test_health_and_stages.py` expects 21 stages.
- **Why:** Roman asked to remove the last stage, amended or repealed.
- **Checks:** `ruff check .`: clean; `pytest`: 379 passed; `npx tsc --noEmit`: clean; API for bill 3137: 21 stages, no amended stage or milestone; page screenshot checked.

## 2026-10-24 21:26 — feat(frontend): display all 3 lifecycle phases side-by-side on tracker page

- **What:** Updated `frontend/src/app/page.tsx` and `frontend/src/app/bills/page.tsx` so every selected bill renders all 22 stages across all 3 lifecycle phases (`1. Before Parliament`, `2. Inside Parliament`, `3. After Royal Assent`) in a 3-column side-by-side pipeline (with a toggle for Detailed Stack view and a 3-phase summary stepper).
- **Why:** Give users an intuitive single-page overview of the entire legislative lifecycle for any bill in Neon Lakebase Postgres.
- **Checks:** `npx tsc --noEmit`: clean; `GET /` and `GET /api/v1/bills/3137`: HTTP 200 (`total_stages: 22` across all 3 phases).

## 2026-10-24 21:24 — feat(domain): evaluate all 3 lifecycle phases for every bill in Neon Postgres

- **What:** Extended `evaluate_parliament_bill_from_db` in `domain/bills.py` and `POST_ROYAL_ASSENT_STAGE_INFO` in `domain/stages.py` so `GET /api/v1/bills/{bill_id}` returns all 22 stages across all 3 lifecycle phases (`before_parliament`, `inside_parliament`, `after_royal_assent`) for every bill in Neon Lakebase Postgres.
- **Why:** Display the complete 3-phase legislative pipeline (Pre-Parliament, Inside Parliament, and After Royal Assent) for every bill on the tracker page.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest tests/test_health_and_stages.py`: 13 passed.

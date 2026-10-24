## 2026-10-24 21:24 — feat(domain): evaluate all 3 lifecycle phases for every bill in Neon Postgres

- **What:** Extended `evaluate_parliament_bill_from_db` in `domain/bills.py` and `POST_ROYAL_ASSENT_STAGE_INFO` in `domain/stages.py` so `GET /api/v1/bills/{bill_id}` returns all 22 stages across all 3 lifecycle phases (`before_parliament`, `inside_parliament`, `after_royal_assent`) for every bill in Neon Lakebase Postgres.
- **Why:** Display the complete 3-phase legislative pipeline (Pre-Parliament, Inside Parliament, and After Royal Assent) for every bill on the tracker page.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest tests/test_health_and_stages.py`: 13 passed.

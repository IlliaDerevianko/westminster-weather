# Agent log: `feat/stage3-frontend`

## 2026-10-25 05:50 — fix(stage3): no stage 3 forecast instead of an error when a source table is missing

- **What:** `journey/stage3_xgb.py` loads its data without caching failures: if a source table is missing or the database is unreachable it logs a warning and returns no stage 3 forecast, and retries on the next call. Test in `tests/journey/test_stage3_xgb.py`.
- **Why:** the production table `bill_texts` (Act texts; created outside this repo, no ingest here builds it) was dropped after PR #21 merged. The running API only worked from data cached before the drop; after a restart every Act's `/forecast` would have failed.
- **Checks:** `ruff check .`: clean; `pytest tests/journey`: 15 passed; restarted API without `bill_texts`: `/api/v1/bills/3137/forecast` and `/2700/forecast` return 200 with null stage 3 fields, `/api/v1/bills/2700` 200.

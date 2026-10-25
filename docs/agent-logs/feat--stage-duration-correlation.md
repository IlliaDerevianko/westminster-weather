# Agent log: `feat/stage-duration-correlation`

## 2026-10-25 — feat(analysis): measure stage durations and their correlation across a bill's journey

- **What:** New module `backend/src/westminster_weather/analysis/stage_durations.py`. For Commons-first Government Bills it measures 5 sampled segments (Commons 1st→2nd, Commons 2nd→3rd, Lords 1st→committee, Lords committee→report, Lords report→Royal Assent), plus whole-Commons and whole-Lords totals, in calendar and sitting days. It computes pairwise Spearman correlations for all bills and for regular bills (Commons > 14 days), and Royal Assent share by early-stage speed. Charts, CSV and `metrics.json` are written to `backend/reports/stage_correlation/`. Added `matplotlib` as an explicit dependency (was only transitive via `lifelines`; allowed by BUILD_PLAN §0).
- **Why:** User asked whether time spent in earlier stages correlates with time in later stages, using a sample of stages rather than every one, with an easy-to-read graph.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q`: 51 passed (3 new in `tests/model/test_stage_durations.py`).
- **Notes:** Lords-first bills (150) excluded so "earlier/later" means the same for every bill. 111 bills legitimately skipped Lords committee ("Committee negatived" / "Order of Commitment discharged") and are left out of the committee segments. `bills.size_pages` / `size_clauses` are empty, so bill size could not be controlled for.

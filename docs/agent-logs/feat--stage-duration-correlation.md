# Agent log: `feat/stage-duration-correlation`

## 2026-10-25 — feat(analysis): compare bill progress by originating House

- **What:** New module `backend/src/westminster_weather/analysis/origin_comparison.py`. For every Government Bill it measures time in the first House, the second House, the end-game (second House 3rd reading → Royal Assent) and the total to Royal Assent. It compares Commons-first with Lords-first bills (medians and Mann-Whitney U tests), for all bills and regular bills (first House > 14 days), plus outcome shares by origin (Fisher's exact test on Royal Assent rate). Outputs go to `backend/reports/origin_comparison/`.
- **Why:** User asked to see the differences in progress through the Commons, the Lords and to Royal Assent depending on where a bill starts.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q`: 52 passed (1 new in `tests/model/test_origin_comparison.py`).
- **Notes:** All 79 finance/supply-type bills start in the Commons and 108 Commons-first bills are fast-tracked vs 1 Lords-first, so the all-bills comparison overstates how much faster Commons-first bills are.

## 2026-10-25 — docs(reports): add stage duration correlation charts and findings

- **What:** Committed the generated outputs in `backend/reports/stage_correlation/`: 3 PNG charts, per-bill CSV, `metrics.json`, and a `README.md` with the method, findings and caveats.
- **Why:** User asked for an easy-to-read graph and where to find the plots. BUILD_PLAN says `backend/reports/` charts and metrics are committed.
- **Checks:** none (generated outputs and docs). Charts were checked visually for readability.
- **Notes:** Headline result: across all bills stage times correlate strongly (ρ 0.40–0.75), but excluding 108 fast-tracked bills Commons 1st → 2nd reading predicts later stages at only ρ ≈ 0.1, while committee-heavy stages still correlate moderately (ρ 0.35–0.47) and whole Commons vs whole Lords is ρ = 0.46.

## 2026-10-25 — feat(analysis): measure stage durations and their correlation across a bill's journey

- **What:** New module `backend/src/westminster_weather/analysis/stage_durations.py`. For Commons-first Government Bills it measures 5 sampled segments (Commons 1st→2nd, Commons 2nd→3rd, Lords 1st→committee, Lords committee→report, Lords report→Royal Assent), plus whole-Commons and whole-Lords totals, in calendar and sitting days. It computes pairwise Spearman correlations for all bills and for regular bills (Commons > 14 days), and Royal Assent share by early-stage speed. Charts, CSV and `metrics.json` are written to `backend/reports/stage_correlation/`. Added `matplotlib` as an explicit dependency (was only transitive via `lifelines`; allowed by BUILD_PLAN §0).
- **Why:** User asked whether time spent in earlier stages correlates with time in later stages, using a sample of stages rather than every one, with an easy-to-read graph.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q`: 51 passed (3 new in `tests/model/test_stage_durations.py`).
- **Notes:** Lords-first bills (150) excluded so "earlier/later" means the same for every bill. 111 bills legitimately skipped Lords committee ("Committee negatived" / "Order of Commitment discharged") and are left out of the committee segments. `bills.size_pages` / `size_clauses` are empty, so bill size could not be controlled for.

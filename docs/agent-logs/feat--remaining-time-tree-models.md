# Agent log: `feat/remaining-time-tree-models`

## 2026-10-25 — feat(analysis): point-in-time dataset for remaining days to Royal Assent

- **What:** New `backend/src/westminster_weather/analysis/remaining_time_dataset.py`. Builds one row per (finished bill, forecast point): 11 stage anchors (start of each stage in the 1st House, 2nd House and ping-pong) and 7 time anchors (30–365 days after introduction). Features use only sittings known at that point (stage anchors cut at the stage's first sitting in API sort order, so same-day later stages are not leaked). Target = calendar days to Royal Assent. Added `xgboost` as a dependency.
- **Why:** User asked for decision-tree-style models started from several points in the pipeline, with events that already happened as input features, trained only on finished, non-cancelled bills.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q tests/model/test_remaining_time_dataset.py`: 4 passed (leakage, finished-only and time-anchor tests).
- **Notes:** 522 Royal Assent bills → 7,040 rows (5,021 stage, 2,019 time). XGBoost and LightGBM need the OpenMP runtime on macOS; installed it with `brew install libomp`. Session end dates are deliberately not features (not known in advance). `size_clauses` and `kings_speech_flag` are empty in the database, so they are not used.

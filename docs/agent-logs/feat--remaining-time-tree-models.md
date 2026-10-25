# Agent log: `feat/remaining-time-tree-models`

## 2026-10-25 — docs(reports): add stakeholder summary chart of forecast error by stage

- **What:** New `backend/src/westminster_weather/analysis/stakeholder_chart.py`, which draws `0_summary_for_stakeholders.png` from the saved `mae_by_stage.csv` (no retraining): the chosen model (XGBoost) against the B1 rule of thumb, in plain language, with three call-outs (±111 days at introduction, the biggest gain of 34 days, ±17 days at the final stage). Listed it in the report README.
- **Why:** User asked for a noise-free summary graph of how the error changes, to show stakeholders.
- **Checks:** `uv run ruff check`: clean. Chart checked visually and label overlaps fixed; colours checked with the dataviz palette validator (only the intentionally gray baseline fails the chroma check).
- **Notes:** "±N days" is the mean absolute error at that stage.

## 2026-10-25 — docs(reports): add tree model evaluation charts and findings

- **What:** Committed the generated outputs in `backend/reports/remaining_time_models/`: 6 PNG charts, error tables by stage and by days since introduction, all test predictions, feature importance, `metrics.json` and a `README.md` with set-up, results and caveats. Also widened the ablation chart's x-axis so labels are not clipped.
- **Why:** User asked for a thorough evaluation showing how prediction error changes as the forecast starts later in the pipeline.
- **Checks:** none (generated outputs and docs). README numbers were checked against `metrics.json` and `mae_by_stage.csv`; a few hand-typed values were corrected.
- **Notes:** Test MAE, stage points: hist gradient boosting 54, LightGBM 55, XGBoost 55 (best on CV), random forest 56, B1 75, B0 106; XGBoost beats B1 by 16–24 days (95% CI). Error falls from 111 days at introduction to 17 at ping-pong. CV-to-test gap (37 → 55) is mostly era shift: B1 also worsens (56 → 75).

## 2026-10-25 — feat(analysis): tune and evaluate tree models for days left to Royal Assent

- **What:** New `backend/src/westminster_weather/analysis/remaining_time_models.py`. Time split per MODEL_SPEC §7 (train sessions 20–30, test 34–39). Tunes 6 tree-style models (decision tree, random forest, extra trees, hist gradient boosting, LightGBM, XGBoost) plus a ridge model with randomized search on 3 expanding-window folds inside the training sessions, on a log(1 + days) target. Compares against baselines B0 and B1, and evaluates by stage forecast point and by days since introduction, with bill-level bootstrap intervals, permutation importance and feature-group ablation.
- **Why:** User asked to tune and compare decision-tree-like models, start forecasts from several points in the pipeline, and report how the error changes later in the pipeline.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q`: 61 passed (5 new in `tests/model/test_remaining_time_models.py`).
- **Notes:** Removed `friedman_mse` from the decision-tree search: this scikit-learn version rejects it, and 24 of 60 trials had failed silently. Pace medians for training rows use all training sessions, so CV scores are slightly optimistic; test rows only use training medians.

## 2026-10-25 — feat(analysis): point-in-time dataset for remaining days to Royal Assent

- **What:** New `backend/src/westminster_weather/analysis/remaining_time_dataset.py`. Builds one row per (finished bill, forecast point): 11 stage anchors (start of each stage in the 1st House, 2nd House and ping-pong) and 7 time anchors (30–365 days after introduction). Features use only sittings known at that point (stage anchors cut at the stage's first sitting in API sort order, so same-day later stages are not leaked). Target = calendar days to Royal Assent. Added `xgboost` as a dependency.
- **Why:** User asked for decision-tree-style models started from several points in the pipeline, with events that already happened as input features, trained only on finished, non-cancelled bills.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q tests/model/test_remaining_time_dataset.py`: 4 passed (leakage, finished-only and time-anchor tests).
- **Notes:** 522 Royal Assent bills → 7,040 rows (5,021 stage, 2,019 time). XGBoost and LightGBM need the OpenMP runtime on macOS; installed it with `brew install libomp`. Session end dates are deliberately not features (not known in advance). `size_clauses` and `kings_speech_flag` are empty in the database, so they are not used.

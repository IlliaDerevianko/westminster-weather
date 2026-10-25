# Agent log: `feat/remaining-time-model-v2`

## 2026-10-25 — docs(model): guide to the days-to-Royal-Assent tree models

- **What:** New `docs/model/REMAINING_TIME_MODELS.md`: current results, files and commands, problem definition, evaluation rules (no test-set choices, time-ordered CV, no look-ahead, out-of-fold encodings, like-for-like comparisons, bill-level bootstrap), every feature group with its CV effect, every experiment tried with results (including the ones that failed), tuned hyperparameters, tuning infrastructure and runtime lessons, a step-by-step experiment recipe, and a ranked list of next steps.
- **Why:** User asked to document the whole process for future tuning and optimisation.
- **Checks:** none (docs only). Numbers taken from the v1 and v2 `metrics.json`, CSVs and run log.

## 2026-10-25 — docs(reports): v2 results, stakeholder comparison chart and faster tuning

- **What:** Ran the v2 pipeline and committed `backend/reports/remaining_time_models_v2/` (charts, error tables, test predictions, design experiments, metrics and README). Tuning now runs 8 Optuna trials in parallel (one thread per model) with median pruning after 3 of 5 folds, and logs live; trial counts and search spaces are unchanged. Added `plot_v2` to `stakeholder_chart.py`: new model vs previous v1 XGBoost vs rule of thumb, with a legend showing each line's average error.
- **Why:** User asked for the best possible tree model and the same stakeholder graph comparing it with the previous best XGBoost and the rule of thumb. The first full run was stopped at the user's request after 40+ minutes on LightGBM alone (all cores per model is slow on this small dataset).
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q`: all passed. Charts checked visually; README numbers taken from `metrics.json` and the CSVs.
- **Notes:** Final model = CV-weighted blend (51% LightGBM, 43% XGBoost, 6% hist GB), CV 42.1 vs v1 setup 46.4. Test: 53.8 vs v1 55.2 days (95% CI of the difference −2.9 to −0.0; barely significant) and vs rule of thumb 75.5. Gains are at the earliest forecast points (introduction −6.3 days, H1 2nd reading −3.6); later points are within ±2 days of v1, and time-based forecast points show no overall gain. Calendar features were dropped by backward elimination; residual target, sample weighting and monotonic constraints did not help on CV.

## 2026-10-25 — feat(analysis): v2 features, leak-free encodings and tuning pipeline

- **What:** `build_dataset(extended=True)` adds 46 point-in-time features: days spent in the Commons / Lords / current House / current stage so far, sittings per stage (bill-size proxy), ping-pong rounds, policy areas (renamed departments merged), congestion (Government Bills in progress), summer-recess and 5-year-limit countdowns, government majority / coalition, and title signals. New `analysis/remaining_time_v2.py`: out-of-fold target encodings (rule-of-thumb value, smoothed policy-area speed and residual), 5 expanding-window CV folds, LightGBM design experiments (feature groups, backward elimination, log vs residual target, sample weights, monotonic constraints), Optuna TPE tuning of LightGBM, XGBoost, CatBoost, hist gradient boosting, random forest and extra trees, seed averaging and a CV-weighted blend, then a single test evaluation against v1 XGBoost and the rule of thumb. Added `optuna` and `catboost`.
- **Why:** User asked to push the model much further with more features (including continuous time in the Commons), every reasonable tree-model technique, and best research practice.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest -q tests/model/test_remaining_time_v2.py tests/model/test_remaining_time_dataset.py`: 10 passed (new tests cover House-time features, recess countdown, out-of-fold encodings and the residual target).
- **Notes:** v1 outputs are unchanged: `extended=False` reproduces the v1 dataset exactly (checked column by column). A 2-trial smoke run of the pipeline scored the test set once while debugging; no design choice uses it. Government majorities are hard-coded public election results.

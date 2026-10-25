# Days to Royal Assent: model v2

**Question:** can more features, deeper tuning and ensembles beat the v1 XGBoost in
`../remaining_time_models/`?

**Short answer:**
- **Overall, only slightly:** 53.8 against 55.2 days of average error on test bills. That's a
  1.4-day gain, and its 95% interval (0.0 to 2.9 days) only just excludes zero.
- **At the earliest forecast points, by more:** 6 days better at introduction and 4 days
  better at first-House 2nd reading.
- **From first-House committee onwards, the same:** v2 and v1 are within about 2 days of
  each other.

Regenerate from `backend/` (about 15 minutes):

```bash
uv run python -u -m westminster_weather.analysis.remaining_time_v2
uv run python -m westminster_weather.analysis.stakeholder_chart v2 "The new model sharpens the earliest Royal Assent forecasts"
```

## What stayed the same as v1

- **Bills:** only the 522 that reached Royal Assent; unfinished, fallen and withdrawn bills
  are excluded.
- **Forecast points:** 11 stage points and 7 time points, each seeing only events up to that
  moment.
- **Split:** train on sessions 20–30 (2006–2019, 320 bills), test on sessions 34–39
  (2019–2026, 196 bills).
- **Comparisons:** v1 XGBoost and the rule of thumb (B1) are scored on exactly the same test
  rows.

## What was tried

Every choice below was made on **5 expanding-window cross-validation folds** inside the
training years. Each fold validates on later sessions than it trains on. The test set was
scored once at the end.

### 1. New features (46 added, all known at the forecast date)

| Group | Features |
|---|---|
| **Time in each House so far** (continuous) | days in the Commons, the Lords, the first House, the second House, the current House and the current stage |
| **Bill size** | sittings so far at 2nd reading, committee and report in each House; ping-pong rounds |
| **Policy area** | renamed departments merged into 17 stable areas (e.g. DCLG, MHCLG and DLUHC become one housing area) |
| **Encoded speed** (out-of-fold) | the rule-of-thumb value for the row; the typical speed of the bill's policy area; how much that area beats or lags the rule of thumb |
| **Congestion** | Government Bills in progress at the forecast date; bills introduced earlier in the session |
| **Calendar** | days to the summer recess; days to the Parliament's 5-year limit |
| **Politics** | government majority at the last election; coalition or minority |
| **Title signals** | "Northern Ireland", "(No. 2)", "Amendment", finance, emergency, Scotland/Wales, EU, repeal; title length |

**About the encoded features:** they are built from training targets, so they're
recomputed inside every validation fold, and training rows get out-of-fold values. A bill's
own answer never feeds its own features. Tests in `tests/model/test_remaining_time_v2.py`
check this.

### 2. Design experiments (LightGBM with fixed settings, CV error in days)

| Setting | CV error |
|---|---|
| v1 features | 50.7 |
| v1's tuned XGBoost on the same folds (reference) | 46.4 |
| + title signals | 48.2 (−2.5) |
| + policy area | 48.9 (−1.8) |
| + time in each House | 49.8 (−1.0) |
| + bill size | 49.9 (−0.8) |
| + encoded speed | 50.2 (−0.6) |
| + congestion | 50.5 (−0.3) |
| + politics | 50.8 (+0.1) |
| + calendar | 51.3 (+0.6) |
| all features | 46.9 |
| **all except calendar** (backward elimination) | **45.9** |

Other design choices, all scored on cross-validation:

| Choice tried | Result | Decision |
|---|---|---|
| **Target:** log days vs the correction to the rule of thumb | log 45.9, correction 48.4 | log kept |
| **Weights:** none, each bill once, recent sessions favoured | none 45.9, bill 48.1, recent 48.5 | no weighting |
| **Monotonic constraints** (later stage → fewer days left) | 46.6 | not used |

### 3. Tuning (Optuna Bayesian search, median pruning)

| Model | Trials | CV error (days) |
|---|---|---|
| XGBoost | 150 | 42.4 |
| LightGBM | 150 | 42.5 |
| Hist gradient boosting | 30 | 43.6 |
| CatBoost | 50 | 44.4 |
| Random forest | 25 | 45.9 |
| Extra trees | 25 | 48.0 |

The search spaces cover tree size, learning rate, regularisation, row and column sampling,
and the loss function (squared, absolute and Huber errors).

### 4. Ensembles

- **CV-weighted blend:** 51% LightGBM, 43% XGBoost and 6% hist gradient boosting. CV error
  **42.1**, the best candidate, so it became the final model.
- **Average of the top 3:** CV error 42.2.
- **Seed averaging:** every boosted model in the final fit is averaged over 5 random seeds.

## Results on test bills (2019–2026)

### Overall, stage forecast points

| Model | Avg error (days) | 95% interval |
|---|---|---|
| **v2 blend (final)** | **53.8** | 47–61 |
| v2 LightGBM | 53.9 | 48–61 |
| v2 XGBoost | 54.1 | 47–61 |
| v2 CatBoost | 54.2 | 48–61 |
| v2 hist gradient boosting | 54.2 | 48–61 |
| v2 random forest | 55.1 | 48–62 |
| v1 XGBoost (previous best) | 55.2 | 49–62 |
| Rule of thumb | 75.5 | 68–84 |

**Paired comparisons** (bootstrap over test bills):
- v2 is **0.0 to 2.9 days better than v1**.
- v2 is **17 to 26 days better than the rule of thumb**.

### By forecast point (`0_summary_for_stakeholders.png`, `1_error_by_stage.png`)

| Forecast made at | v2 | v1 | Rule of thumb | v2 vs v1 |
|---|---|---|---|---|
| Introduced (H1 1st reading) | **105** | 111 | 128 | **−6.3** |
| H1 2nd reading | **86** | 89 | 118 | **−3.6** |
| H1 committee | 79 | 80 | 114 | −0.8 |
| H1 report | 63 | 62 | 82 | +0.7 |
| H1 3rd reading | 48 | 48 | 78 | −0.2 |
| H2 1st reading | 47 | 47 | 78 | −0.2 |
| H2 2nd reading | 39 | 40 | 66 | −1.0 |
| H2 committee | 41 | 43 | 57 | −1.7 |
| H2 report | 27 | 27 | 34 | −0.3 |
| H2 3rd reading | 18 | 18 | 21 | −0.4 |
| Ping-pong | 18 | 17 | 19 | +0.2 |

**Forecasts by days since introduction** (`mae_by_days_since_introduction.csv`) show no
overall gain:

| Days since introduction | v2 vs v1 |
|---|---|
| 30 | 2 days better |
| 60 | 3 days better |
| 90 to 365 | 2 days worse |

### What the final model relies on (`4_feature_group_importance.png`)

This is measured by shuffling each feature group on the test set:

- **v1 features:** still most of the signal (+33 days of error when shuffled).
- **Encoded speed** (the rule-of-thumb value and policy-area speed): +8 days.
- **Time in each House so far:** +2.5 days.
- **Title signals:** +2.1 days.
- **Bill size:** +1.5 days.
- **Congestion, policy-area flags and politics:** under 0.5 days each.

## Why the cross-validation gain didn't carry over

On cross-validation v2 was about 4 days better than v1 (42.1 against 46.4). On test bills the
gain is 1.4 days. This is the same era shift seen in v1: bills in 2019–2026 take longer
(Brexit, COVID, two very long sessions). Patterns learned from 2006–2019 transfer only
partly. All six tuned model types land within 1 day of each other on test, which suggests the
limit is now the **information available**, not the model.

## What would most likely help next

1. **King's Speech data.** It gives a forecast point before 1st reading and a "was it in the
   Speech" signal.
2. **Bill size.** Clause and page counts, which the database is missing.
3. **Politics per bill.** Rebellion counts and Lords defeats from the Votes API, and
   amendment counts.
4. **Training on recent sessions with a rolling window.** For example, refit every session.
   Recency weighting didn't help on cross-validation, but the era shift is the main error
   source.
5. **Prediction intervals** (p10–p90), which MODEL_SPEC asks for.

## Notes

- While debugging the pipeline, a 2-trial smoke run scored the test set once. No design
  choice used that number.
- Government majorities are hard-coded public election results.
- `optuna` and `catboost` were added as dependencies.

## Files

| File | Contents |
|---|---|
| `0_summary_for_stakeholders.png` | **Stakeholder chart:** new model vs previous model vs rule of thumb, by stage |
| `1_error_by_stage.png` | Detailed error by stage, including the other v2 model families |
| `2_model_comparison.png` | Test error per model with 95% intervals |
| `3_what_helped_cv.png` | Effect of each new feature group (CV) |
| `4_feature_group_importance.png` | Feature groups the final model relies on (test permutation) |
| `mae_by_stage.csv`, `mae_by_days_since_introduction.csv` | Error tables |
| `design_experiments_cv.csv` | Every design experiment's CV error |
| `test_predictions.csv` | Every test forecast for every model |
| `metrics.json` | Tuned hyperparameters, CV scores, test metrics, intervals and the run log |

# Tree models: days left until Royal Assent

**Question:** can decision-tree-style models predict how many days a UK Government Bill has
left until Royal Assent, starting from different points in its journey? How does the error
change as the forecast starts later?

Regenerate everything in this folder from `backend/` (about 90 seconds):

```bash
uv run python -m westminster_weather.analysis.remaining_time_models
```

Code: `src/westminster_weather/analysis/remaining_time_dataset.py` (features) and
`remaining_time_models.py` (training and evaluation).

## Set-up

### Bills

- Only the **522 bills that reached Royal Assent**.
- Bills still in progress are excluded, because they have no answer yet.
- Bills that fell or were withdrawn are excluded, because they never get a Royal Assent date.

### Training and test split

The split is by time, as MODEL_SPEC §7 says:

| | Sessions | Years | Bills | Rows |
|---|---|---|---|---|
| Train | 20–30 | 2006-07 to 2017-19 | 320 | 4,365 |
| Test | 34–39 | 2019 to 2026 | 196 | 2,605 |

Session 40 is still running and only its fastest bills have finished, so it is left out.

### Forecast points

Each bill gets a forecast at every point it reached, and each point is one row:

- **11 stage points:** the start of each stage in the first House (H1) and second House (H2):
  1st reading, 2nd reading, committee, report and 3rd reading, plus ping-pong.
- **7 time points:** 30, 60, 90, 120, 180, 270 and 365 days after introduction.

### Features

There are 44 features, all built only from events known at the forecast point. A stage
point sees every sitting up to and including that stage's first sitting, in the Bills API's
own order, so stages held later the same day are not leaked. Tests in
`tests/model/test_remaining_time_dataset.py` check this.

| Group | Features |
|---|---|
| Position | stage reached; whether the bill is in its second House |
| Time and pace | days and sitting days elapsed; days since last event; **pace compared with a typical training bill at the same stage** |
| Gaps so far | days between each pair of stages already passed |
| Procedural events | Committee of the whole House, Public Bill Committee, programme motion, money resolution, ways and means, carry-over, Lords committee negatived, same-day 2nd and 3rd reading, report skipped, sittings so far |
| Timing | how far into the session and the Parliament, at introduction and now; month |
| Bill type | originating House; finance or supply title; sponsoring department (top 10 + other) |

Session end dates are **not** features, because the government rarely announces them in
advance.

### Models

- **Tuned models:**
  - decision tree, random forest, extra trees;
  - histogram gradient boosting (scikit-learn), LightGBM and XGBoost;
  - a ridge (linear) model for comparison.
- **Training target:** log(1 + days left), so the few very long bills don't dominate.
- **Hyperparameter tuning:** randomized search, 30–60 trials per model, on 3 expanding-window
  folds inside the training sessions. Each fold validates on later sessions than it trains
  on. The best model was picked on these folds, **without looking at the test set**.
- **Baselines** (MODEL_SPEC §7):
  - **B0:** typical total days by originating House, minus the days already elapsed.
  - **B1:** typical days left for training bills at the same forecast point.

## Results

### Overall, on test bills (stage forecast points)

| Model | Mean error (days) | 95% interval | Median error | Within 30 days |
|---|---|---|---|---|
| Hist gradient boosting | **54** | 48–61 | 29 | 51% |
| LightGBM | 55 | 48–62 | 29 | 51% |
| **XGBoost** (best on cross-validation) | 55 | 49–62 | 29 | 51% |
| Random forest | 56 | 48–63 | 29 | 51% |
| Extra trees | 57 | 49–64 | 29 | 51% |
| Decision tree | 59 | 52–66 | 31 | 48% |
| Linear (ridge) | 63 | 55–70 | 33 | 47% |
| B1 baseline | 75 | 68–84 | 50 | 34% |
| B0 baseline | 106 | 97–115 | 96 | 20% |

The 95% intervals come from resampling whole test bills 1,000 times.

- **The tree models beat both baselines clearly.** XGBoost is **16–24 days better than B1**
  (paired bootstrap 95% interval).
- **The boosted models and random forest are tied.** Their intervals overlap almost
  completely, so the choice between them doesn't matter on this data.
- **Trees beat the linear model by about 8 days**, so the interactions between features are
  worth capturing.

### How the error changes as the forecast starts later (`1_error_by_stage.png`)

| Forecast made at | Test rows | Avg days left | B1 error | XGBoost error | Error as % of days left |
|---|---|---|---|---|---|
| H1 1st reading | 196 | 199 | 128 | 111 | 56% |
| H1 2nd reading | 196 | 178 | 118 | 89 | 50% |
| H1 committee | 176 | 170 | 114 | 80 | 47% |
| H1 report | 120 | 158 | 82 | 62 | 40% |
| H1 3rd reading | 191 | 112 | 78 | 48 | 43% |
| H2 1st reading | 196 | 109 | 78 | 47 | 43% |
| H2 2nd reading | 195 | 84 | 66 | 40 | 48% |
| H2 committee | 147 | 84 | 57 | 43 | 51% |
| H2 report | 136 | 45 | 34 | 27 | 59% |
| H2 3rd reading | 195 | 26 | 21 | 18 | 69% |
| Ping-pong | 110 | 23 | 19 | 17 | 75% |

1. **The error falls steadily, from about 111 days at introduction to 17 days at ping-pong.**
2. **The model's biggest advantage is in the middle of the journey.** From first-House
   2nd reading to second-House 2nd reading it beats B1 by 25–34 days at most points (20 at
   first-House report). At introduction the gain is 16 days.
3. **Near the end, every method is about the same.** From second-House report onwards the
   gain over B1 is only 2–7 days, because little time is left.
4. **Relative error rises again late in the journey.** It is lowest, around 40%, at
   first-House report and 3rd reading. Exactly how long ping-pong and the wait for Royal
   Assent take is hard to predict.

Forecasts made a fixed number of days after introduction (`2_error_by_days_since_introduction.png`)
show the same trend: 89 days of error at 30 days in, 74 at 90 days and 50 at a year. B1 is
118, 107 and 75 at the same points.

### Fast-tracked and regular bills

Fast-tracked means the first House was cleared in 14 days or less; 45 test bills qualify.

| Group | XGBoost error | B1 error |
|---|---|---|
| Fast-tracked | 23 days | 70 days |
| Regular | 63 days | 77 days |

The model recognises emergency and finance bills almost immediately, and gains most there.
For regular bills it still improves on B1 by about 14 days.

### Most useful features (`4_feature_importance.png`, `5_feature_group_ablation.png`)

1. **Stage reached** and **pace compared with a typical bill at the same stage** are by far
   the strongest features. Shuffling either one adds 14–16 days of error.
2. **Bill type** matters most as a group. Removing origin, finance title and department adds
   7.5 days of cross-validated error. The HM Treasury and Northern Ireland Office indicators
   stand out, because these departments' bills are often fast-tracked.
3. **Second-House 1st → 2nd reading gap**, **same-day 2nd and 3rd reading** and **report
   skipped** are the most useful process signals.
4. **Many features overlap.** Removing any single group other than bill type changes the
   error by less than 2 days, because other features carry the same information.

## Caveats

- **The test era is harder than the training era.** Cross-validation error is about 37 days
  but test error is about 55. B1 worsens just as much (56 → 75), because bills in 2019–2026
  take longer on average: 132 days left against 113. That period includes Brexit, COVID and
  two long sessions. Some of the extra error is the model fitting the older era too closely.
- **Long bills are under-predicted.** Predictions top out at about 250 days
  (`6_predicted_vs_actual.png`). Very slow bills, such as carry-overs and paused bills, are
  forecast too early.
- **This only predicts timing for bills that pass.** The model learned only from bills that
  passed, so it says nothing about whether a bill will fail. That needs a separate model, as
  MODEL_SPEC §4 plans.
- **Cross-validation scores are slightly optimistic.** The pace feature for training rows
  uses medians from all training sessions, including each fold's validation sessions. Test
  rows only use training medians, so the test results are unaffected.
- **Point forecasts only.** There are no prediction intervals; MODEL_SPEC asks for p10–p90.
  LightGBM quantile objectives would be the natural next step.
- **Missing features:** bill size, government majority, rebellions and amendments are not in
  the database yet.

## Files

| File | Contents |
|---|---|
| `0_summary_for_stakeholders.png` | **One-slide summary:** our model against the rule of thumb, by stage (regenerate with `uv run python -m westminster_weather.analysis.stakeholder_chart`) |
| `1_error_by_stage.png` | **Detailed chart:** error by stage forecast point, all models and baselines |
| `2_error_by_days_since_introduction.png` | Error by days since introduction |
| `3_model_comparison.png` | Overall error per model with 95% bootstrap intervals |
| `4_feature_importance.png` | Top 20 features (permutation importance, XGBoost) |
| `5_feature_group_ablation.png` | Cross-validated error change when each feature group is removed |
| `6_predicted_vs_actual.png` | XGBoost predictions against actual days left |
| `mae_by_stage.csv`, `mae_by_days_since_introduction.csv` | Error tables behind charts 1 and 2 |
| `test_predictions.csv` | Every test forecast, for every model |
| `feature_importance.csv` | Permutation importance for all 44 features |
| `metrics.json` | Best hyperparameters, cross-validation scores, all metrics and intervals |

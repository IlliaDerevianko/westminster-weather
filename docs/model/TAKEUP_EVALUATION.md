# Take-up model: evaluation

Generated 2026-10-24 by `takeup/evaluate.py` (code 6607c34a28). Charts and `metrics.json` are in `backend/reports/takeup/`.

The question: when a minister commits to legislate on something, will a Government Bill on it get a first reading, and when? See `TAKEUP_PLAN.md` for the model.

## Data

- Ideas visible at 2026-10-22: **985**, of which **168** already have a bill (17%).
- Features used in the full fit: `z_src_hansard`, `z_initial_firmness`, `z_years_since_election`, `z_months_into_session`, `x_firm_commitment`, `x_repeat_mention`, `x_new_parliament`, `z_dept_treasury`, `z_dept_home_office`, `z_dept_ministry_of_housing_communities_and_local_government`, `z_dept_business_energy_and_industrial_strategy`, `z_dept_ministry_of_justice`.

## Time-split backtests

Each split trains only on what was visible at the cut (ideas triggered before it, followed up to it) and tests on ideas triggered after it, each predicted at its own trigger date. A test idea counts for a horizon only if its outcome is known (a bill came, or it was followed for the whole horizon).

### Trained before 2019-01-01, tested 2019-01-01 to 2026-10-22

Training: 453 ideas (75 with a bill by the cut). Test: 538 ideas (72 with a bill by 2026-10-22).

| Horizon | n | bills | Model AUC | Model Brier | B0 Brier | B1 AUC | B1 Brier |
|---|---|---|---|---|---|---|---|
| 26 weeks | 472 | 38 | 0.600 | 0.075 | 0.074 | 0.621 | 0.073 |
| 52 weeks | 390 | 50 | 0.571 | 0.113 | 0.112 | 0.629 | 0.109 |

B0 predicts 13% for everyone; B1 predicts 20% for ideas first seen in a written statement and 12% for Hansard (52 weeks). B0 has no ranking, so its AUC is 0.5 by definition.

Calibration, bill within 52 weeks:

| Predicted | n | mean predicted | actual |
|---|---|---|---|
| 0%-20% | 390 | 8% | 13% |
| 20%-40% | 0 | n/a | n/a |
| 40%-60% | 0 | n/a | n/a |
| 60%-80% | 0 | n/a | n/a |
| 80%-100% | 0 | n/a | n/a |

- 80% timing window (t_10 to t_90) contained the actual week for 38% of 72 bills (target 80%).
- Concordance: all ideas ranked by p52 0.578; bills ranked by predicted median week 0.500 (0.5 = chance).
- Verdicts (ideas with a bill or silent for 104+ weeks, n = 244): 'likely' 0 (n/a got a bill), 'unlikely' 18 (83% did not), 'uncertain' 93%.
- Forecasts on later 1 Januaries for still-silent ideas (n = 546): AUC 0.531, Brier 0.091 for a bill within 52 weeks.

### Trained before 2022-01-01, tested 2022-01-01 to 2026-10-22

Training: 632 ideas (118 with a bill by the cut). Test: 367 ideas (32 with a bill by 2026-10-22).

| Horizon | n | bills | Model AUC | Model Brier | B0 Brier | B1 AUC | B1 Brier |
|---|---|---|---|---|---|---|---|
| 26 weeks | 301 | 21 | 0.546 | 0.065 | 0.065 | 0.626 | 0.064 |
| 52 weeks | 219 | 28 | 0.534 | 0.114 | 0.112 | 0.632 | 0.108 |

B0 predicts 14% for everyone; B1 predicts 22% for ideas first seen in a written statement and 12% for Hansard (52 weeks). B0 has no ranking, so its AUC is 0.5 by definition.

Calibration, bill within 52 weeks:

| Predicted | n | mean predicted | actual |
|---|---|---|---|
| 0%-20% | 219 | 8% | 13% |
| 20%-40% | 0 | n/a | n/a |
| 40%-60% | 0 | n/a | n/a |
| 60%-80% | 0 | n/a | n/a |
| 80%-100% | 0 | n/a | n/a |

- 80% timing window (t_10 to t_90) contained the actual week for 22% of 32 bills (target 80%).
- Concordance: all ideas ranked by p52 0.555; bills ranked by predicted median week 0.500 (0.5 = chance).
- Verdicts (ideas with a bill or silent for 104+ weeks, n = 73): 'likely' 25 (32% got a bill), 'unlikely' 5 (100% did not), 'uncertain' 59%.
- Forecasts on later 1 Januaries for still-silent ideas (n = 113): AUC 0.601, Brier 0.107 for a bill within 52 weeks.

## Parameters (fit on all ideas)

| Parameter | Estimate | Std. error |
|---|---|---|
| `intercept` | 0.719 | 0.800 |
| `z_src_hansard` | -0.909 | 0.662 |
| `z_initial_firmness` | 0.808 | 0.272 |
| `z_years_since_election` | -0.221 | 0.151 |
| `z_months_into_session` | -0.019 | 0.028 |
| `z_dept_treasury` | -0.858 | 1.043 |
| `z_dept_home_office` | 1.882 | 2.440 |
| `z_dept_ministry_of_housing_communities_and_local_government` | 0.940 | 3.104 |
| `z_dept_business_energy_and_industrial_strategy` | 1.662 | 2.201 |
| `z_dept_ministry_of_justice` | 1.293 | 2.653 |
| `alpha_weeks_1+` | -4.310 | 0.222 |
| `alpha_weeks_5+` | -5.770 | 0.299 |
| `alpha_weeks_13+` | -6.058 | 0.259 |
| `alpha_weeks_27+` | -6.273 | 0.234 |
| `alpha_weeks_53+` | -7.124 | 0.273 |
| `alpha_weeks_105+` | -6.774 | 0.298 |
| `x_firm_commitment` | 0.988 | 0.411 |
| `x_repeat_mention` | 2.000 | 0.206 |
| `x_new_parliament` | 0.158 | 0.266 |

`z_*` move the chance an idea is ever taken up; `x_*` move the weekly chance of a first reading once the event has happened; `alpha_weeks_*` are the baseline weekly log-odds by weeks since the trigger.

## Weak spots and limitations

- Split 2019-01-01, 26 weeks: model Brier 0.075 is no better than B1 (0.073).
- Split 2019-01-01, 52 weeks: model Brier 0.113 is no better than B1 (0.109).
- Split 2019-01-01: 80% timing window covers 38% of bills.
- Split 2022-01-01, 26 weeks: model Brier 0.065 is no better than B1 (0.064).
- Split 2022-01-01, 52 weeks: model Brier 0.114 is no better than B1 (0.108).
- Split 2022-01-01: 80% timing window covers 22% of bills.
- King's Speech data is not used (decision of 2026-10-24); ideas come only from ministers' words in written statements and Hansard.
- Commitments, topics and links come from rules-only NLP (phrase lexicon, TF-IDF grouping, title similarity); wrong groupings or links show up here as label noise.
- An idea with no linked bill is treated as not (yet) taken up, so a missed link is counted as a failure of the idea.
- Merged ideas take the earliest trigger of their groups, which uses knowledge of the shared bill; test ideas that later merged may look slightly earlier than they were.
- Outcomes after a change of government are censored at the dissolution, so the test sets after 2024 are short.

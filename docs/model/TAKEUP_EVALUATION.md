# Take-up model: evaluation

Generated 2026-10-25 by `takeup/evaluate.py` (code a3b1e0e3bd). Charts and `metrics.json` are in `backend/reports/takeup/`.

The question: when a minister commits to legislate on something, will a Government Bill on it get a first reading, and when? See `TAKEUP_PLAN.md` for the model.

## Data

- Ideas visible at 2026-10-22: **1042**, of which **153** already have a bill (15%).
- Features used in the full fit: `z_src_hansard`, `z_initial_firmness`, `z_years_since_election`, `z_months_into_session`, `x_firm_commitment`, `x_repeat_mention`, `x_new_parliament`, `z_dept_treasury`, `z_dept_ministry_of_housing_communities_and_local_government`, `z_dept_home_office`, `z_dept_business_energy_and_industrial_strategy`, `z_dept_ministry_of_justice`.

## Time-split backtests

Each split trains only on what was visible at the cut (ideas triggered before it, followed up to it) and tests on ideas triggered after it, each predicted at its own trigger date. A test idea counts for a horizon only if its outcome is known (a bill came, or it was followed for the whole horizon).

### Trained before 2019-01-01, tested 2019-01-01 to 2026-10-22

Training: 468 ideas (67 with a bill by the cut). Test: 576 ideas (79 with a bill by 2026-10-22).

| Horizon | n | bills | Model AUC | Model Brier | B0 Brier | B1 AUC | B1 Brier |
|---|---|---|---|---|---|---|---|
| 26 weeks | 510 | 47 | 0.586 | 0.084 | 0.084 | 0.636 | 0.082 |
| 52 weeks | 427 | 60 | 0.548 | 0.122 | 0.121 | 0.625 | 0.118 |

B0 predicts 13% for everyone; B1 predicts 20% for ideas first seen in a written statement and 13% for Hansard (52 weeks). B0 has no ranking, so its AUC is 0.5 by definition.

Calibration, bill within 52 weeks:

| Predicted | n | mean predicted | actual |
|---|---|---|---|
| 0%-20% | 427 | 9% | 14% |
| 20%-40% | 0 | n/a | n/a |
| 40%-60% | 0 | n/a | n/a |
| 60%-80% | 0 | n/a | n/a |
| 80%-100% | 0 | n/a | n/a |

- 80% timing window (t_10 to t_90) contained the actual week for 78% of 79 bills (target 80%).
- Concordance: all ideas ranked by p52 0.543; bills ranked by predicted median week 0.483 (0.5 = chance).
- Verdicts (ideas with a bill or silent for 104+ weeks, n = 274): 'likely' 0 (n/a got a bill), 'unlikely' 267 (73% did not), 'uncertain' 3%.
- Forecasts on later 1 Januaries for still-silent ideas (n = 563): AUC 0.680, Brier 0.067 for a bill within 52 weeks.

### Trained before 2022-01-01, tested 2022-01-01 to 2026-10-22

Training: 660 ideas (105 with a bill by the cut). Test: 385 ideas (40 with a bill by 2026-10-22).

| Horizon | n | bills | Model AUC | Model Brier | B0 Brier | B1 AUC | B1 Brier |
|---|---|---|---|---|---|---|---|
| 26 weeks | 319 | 24 | 0.555 | 0.069 | 0.070 | 0.637 | 0.069 |
| 52 weeks | 236 | 32 | 0.548 | 0.119 | 0.117 | 0.613 | 0.115 |

B0 predicts 15% for everyone; B1 predicts 24% for ideas first seen in a written statement and 13% for Hansard (52 weeks). B0 has no ranking, so its AUC is 0.5 by definition.

Calibration, bill within 52 weeks:

| Predicted | n | mean predicted | actual |
|---|---|---|---|
| 0%-20% | 236 | 9% | 14% |
| 20%-40% | 0 | n/a | n/a |
| 40%-60% | 0 | n/a | n/a |
| 60%-80% | 0 | n/a | n/a |
| 80%-100% | 0 | n/a | n/a |

- 80% timing window (t_10 to t_90) contained the actual week for 45% of 40 bills (target 80%).
- Concordance: all ideas ranked by p52 0.544; bills ranked by predicted median week 0.500 (0.5 = chance).
- Verdicts (ideas with a bill or silent for 104+ weeks, n = 83): 'likely' 4 (50% got a bill), 'unlikely' 10 (60% did not), 'uncertain' 83%.
- Forecasts on later 1 Januaries for still-silent ideas (n = 118): AUC 0.645, Brier 0.123 for a bill within 52 weeks.

## Model comparison: cure model vs hazard model A vs baselines

Same splits, same test ideas, same outcome labels. The hazard model (`takeup/hazard.py`) is a discrete-time logistic hazard with one row per idea and sitting week, using recency features known at the start of each week (weeks since the last mention, mentions in the last 26 weeks, cumulative mentions, firmness so far, weeks since the trigger, source, department, election / session timing). At the trigger it knows only the trigger-day mention(s); after that its forecasts assume no new mentions.

| Split | Horizon | n | bills | Cure AUC | Hazard AUC | B1 AUC | Cure Brier | Hazard Brier | B0 Brier | B1 Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| 2019 | 26 wk | 510 | 47 | 0.586 | 0.573 | 0.636 | 0.084 | 0.084 | 0.084 | 0.082 |
| 2019 | 52 wk | 427 | 60 | 0.548 | 0.525 | 0.625 | 0.122 | 0.122 | 0.121 | 0.118 |
| 2022 | 26 wk | 319 | 24 | 0.555 | 0.637 | 0.637 | 0.069 | 0.068 | 0.070 | 0.069 |
| 2022 | 52 wk | 236 | 32 | 0.548 | 0.579 | 0.613 | 0.119 | 0.116 | 0.117 | 0.115 |

Later forecasts (each 1 January, ideas still silent; bill within 52 weeks):

| Split | n | Cure AUC | Hazard AUC | Cure Brier | Hazard Brier |
|---|---|---|---|---|---|
| 2019 | 563 | 0.680 | 0.660 | 0.067 | 0.068 |
| 2022 | 118 | 0.645 | 0.664 | 0.123 | 0.118 |

Verdicts (ideas with a bill or silent 104+ weeks; 'unlikely' accuracy should be compared with the share of these ideas that never got a bill):

| Split | Model | n | likely (correct) | unlikely (correct) | uncertain |
|---|---|---|---|---|---|
| 2019 | cure | 274 (71% no bill) | 0 (n/a) | 267 (73%) | 3% |
| 2019 | hazard | 274 (71% no bill) | 0 (n/a) | 199 (75%) | 27% |
| 2022 | cure | 83 (52% no bill) | 4 (50%) | 10 (60%) | 83% |
| 2022 | hazard | 83 (52% no bill) | 0 (n/a) | 67 (55%) | 19% |

### 80% timing windows: raw vs CQR-calibrated

Calibration (`takeup/conformal.py`): models refitted on data visible 2 years before each cut predict the ideas with a bill triggered in those 2 years; the scale-normalised conformity score max((t_10 - t)/w, (t - t_90)/w), w = t_90 - t_10, gives the finite-sample quantile q, and test windows become [t_10 - q w, t_90 + q w]. The 80% guarantee is approximate: the calibration ideas are earlier than the test ideas (drift) and only include bills that came before the cut (right truncation favours fast bills). Coverage must be read with the median width: windows of around four years cover many bills cheaply, so the timing forecasts remain vague.

| Split | Model | bills | raw coverage | CQR coverage | q | calibration bills | median width raw | median width CQR |
|---|---|---|---|---|---|---|---|---|
| 2019 | cure | 79 | 78% | 78% | 0.00 | 6 | 138 wk | 138 wk |
| 2019 | hazard | 79 | 77% | 77% | 0.00 | 6 | 102 wk | 102 wk |
| 2022 | cure | 40 | 45% | 48% | 0.01 | 18 | 236 wk | 238 wk |
| 2022 | hazard | 40 | 82% | 90% | 0.01 | 18 | 190 wk | 194 wk |

### Hazard model coefficients (fit on all ideas, C = 0.01, 101891 idea-weeks, 153 bills)

Log-odds of a first reading in a sitting week, per unit of the feature (bins: relative to weeks 1-4 since the trigger / 0-4 weeks since the last mention).

| Feature | Estimate | Std. error |
|---|---|---|
| `intercept` | -7.044 | 0.114 |
| `since_5+` | -0.236 | 0.259 |
| `since_13+` | -0.198 | 0.223 |
| `since_27+` | 0.017 | 0.196 |
| `since_53+` | 0.052 | 0.175 |
| `since_105+` | -0.324 | 0.175 |
| `log_weeks_since_trigger` | -0.347 | 0.062 |
| `gap_5+` | -0.277 | 0.239 |
| `gap_13+` | -0.063 | 0.211 |
| `gap_27+` | 0.082 | 0.205 |
| `gap_53+` | -0.540 | 0.176 |
| `recent_mentions_26w` | 0.486 | 0.139 |
| `log1p_mentions` | 1.906 | 0.319 |
| `max_firmness` | 0.133 | 0.101 |
| `any_firm` | 0.146 | 0.155 |
| `years_since_election` | -0.058 | 0.050 |
| `months_into_session` | -0.019 | 0.011 |
| `x_new_parliament` | -0.096 | 0.171 |
| `z_src_hansard` | -0.364 | 0.165 |
| `z_dept_treasury` | -0.253 | 0.367 |
| `z_dept_ministry_of_housing_communities_and_local_government` | 0.048 | 0.437 |
| `z_dept_home_office` | 0.462 | 0.522 |
| `z_dept_business_energy_and_industrial_strategy` | 0.283 | 0.437 |
| `z_dept_ministry_of_justice` | 0.022 | 0.654 |

**Hazard beats the cure model on Brier in 4/4 split-horizons and B1 in 1/4 (cure beats B1 in 0/4); mean AUC hazard 0.579 vs B1 0.628; Brier gain over the cure model +0.0002 to +0.0032 (small). Recommendation: serve the hazard model in place of the cure model (with CQR-calibrated windows), but present its probabilities as close to the base rate: it does not reliably beat the first-source baseline B1.**

## Parameters (fit on all ideas)

| Parameter | Estimate | Std. error |
|---|---|---|
| `intercept` | 0.503 | 0.936 |
| `z_src_hansard` | -1.067 | 0.717 |
| `z_initial_firmness` | 0.897 | 0.424 |
| `z_years_since_election` | -0.087 | 0.185 |
| `z_months_into_session` | -0.005 | 0.032 |
| `z_dept_treasury` | -0.554 | 1.191 |
| `z_dept_ministry_of_housing_communities_and_local_government` | 2.101 | 2.606 |
| `z_dept_home_office` | 2.242 | 2.493 |
| `z_dept_business_energy_and_industrial_strategy` | 0.823 | 2.108 |
| `z_dept_ministry_of_justice` | 0.131 | 2.300 |
| `alpha_weeks_1+` | -4.303 | 0.256 |
| `alpha_weeks_5+` | -5.669 | 0.310 |
| `alpha_weeks_13+` | -5.871 | 0.272 |
| `alpha_weeks_27+` | -6.292 | 0.271 |
| `alpha_weeks_53+` | -6.830 | 0.325 |
| `alpha_weeks_105+` | -8.093 | 0.500 |
| `x_firm_commitment` | 1.493 | 0.401 |
| `x_repeat_mention` | 1.802 | 0.222 |
| `x_new_parliament` | 0.065 | 0.331 |

`z_*` move the chance an idea is ever taken up; `x_*` move the weekly chance of a first reading once the event has happened; `alpha_weeks_*` are the baseline weekly log-odds by weeks since the trigger.

## Weak spots and limitations

- Split 2019-01-01, 26 weeks: model Brier 0.084 is no better than B1 (0.082).
- Split 2019-01-01, 52 weeks: model Brier 0.122 is no better than B1 (0.118).
- Split 2019-01-01, 26 weeks: hazard model Brier 0.084 is no better than B1 (0.082).
- Split 2019-01-01, 52 weeks: hazard model Brier 0.122 is no better than B1 (0.118).
- Split 2019-01-01: only 6 calibration bills for the cure model's window, so CQR made no adjustment.
- Split 2019-01-01: only 6 calibration bills for the hazard model's window, so CQR made no adjustment.
- Split 2022-01-01, 26 weeks: model Brier 0.069 is no better than B1 (0.069).
- Split 2022-01-01, 52 weeks: model Brier 0.119 is no better than B1 (0.115).
- Split 2022-01-01: 80% timing window covers 45% of bills.
- Split 2022-01-01, 52 weeks: hazard model Brier 0.116 is no better than B1 (0.115).
- King's Speech data is not used (decision of 2026-10-24); ideas come only from ministers' words in written statements and Hansard.
- Commitments, topics and links come from rules-only NLP (phrase lexicon, TF-IDF grouping, title similarity); wrong groupings or links show up here as label noise.
- An idea with no linked bill is treated as not (yet) taken up, so a missed link is counted as a failure of the idea.
- Merged ideas take the earliest trigger of their groups, which uses knowledge of the shared bill; test ideas that later merged may look slightly earlier than they were.
- Outcomes after a change of government are censored at the dissolution, so the test sets after 2024 are short.

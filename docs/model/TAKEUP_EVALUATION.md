# Take-up model: evaluation

Generated 2026-10-24 by `takeup/evaluate.py` (code a1d8c314a9). Charts and `metrics.json` are in `backend/reports/takeup/`.

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

## Model comparison: cure model vs hazard model A vs baselines

Same splits, same test ideas, same outcome labels. The hazard model (`takeup/hazard.py`) is a discrete-time logistic hazard with one row per idea and sitting week, using recency features known at the start of each week (weeks since the last mention, mentions in the last 26 weeks, cumulative mentions, firmness so far, weeks since the trigger, source, department, election / session timing). At the trigger it knows only the trigger-day mention(s); after that its forecasts assume no new mentions.

| Split | Horizon | n | bills | Cure AUC | Hazard AUC | B1 AUC | Cure Brier | Hazard Brier | B0 Brier | B1 Brier |
|---|---|---|---|---|---|---|---|---|---|---|
| 2019 | 26 wk | 472 | 38 | 0.600 | 0.558 | 0.621 | 0.075 | 0.074 | 0.074 | 0.073 |
| 2019 | 52 wk | 390 | 50 | 0.571 | 0.523 | 0.629 | 0.113 | 0.113 | 0.112 | 0.109 |
| 2022 | 26 wk | 301 | 21 | 0.546 | 0.626 | 0.626 | 0.065 | 0.064 | 0.065 | 0.064 |
| 2022 | 52 wk | 219 | 28 | 0.534 | 0.590 | 0.632 | 0.114 | 0.111 | 0.112 | 0.108 |

Later forecasts (each 1 January, ideas still silent; bill within 52 weeks):

| Split | n | Cure AUC | Hazard AUC | Cure Brier | Hazard Brier |
|---|---|---|---|---|---|
| 2019 | 546 | 0.531 | 0.528 | 0.091 | 0.091 |
| 2022 | 113 | 0.601 | 0.622 | 0.107 | 0.103 |

Verdicts (ideas with a bill or silent 104+ weeks; 'unlikely' accuracy should be compared with the share of these ideas that never got a bill):

| Split | Model | n | likely (correct) | unlikely (correct) | uncertain |
|---|---|---|---|---|---|
| 2019 | cure | 244 (70% no bill) | 0 (n/a) | 18 (83%) | 93% |
| 2019 | hazard | 244 (70% no bill) | 0 (n/a) | 164 (75%) | 33% |
| 2022 | cure | 73 (56% no bill) | 25 (32%) | 5 (100%) | 59% |
| 2022 | hazard | 73 (56% no bill) | 0 (n/a) | 47 (66%) | 36% |

### 80% timing windows: raw vs CQR-calibrated

Calibration (`takeup/conformal.py`): models refitted on data visible 2 years before each cut predict the ideas with a bill triggered in those 2 years; the scale-normalised conformity score max((t_10 - t)/w, (t - t_90)/w), w = t_90 - t_10, gives the finite-sample quantile q, and test windows become [t_10 - q w, t_90 + q w]. The 80% guarantee is approximate: the calibration ideas are earlier than the test ideas (drift) and only include bills that came before the cut (right truncation favours fast bills). Coverage must be read with the median width: windows of around four years cover many bills cheaply, so the timing forecasts remain vague.

| Split | Model | bills | raw coverage | CQR coverage | q | calibration bills | median width raw | median width CQR |
|---|---|---|---|---|---|---|---|---|
| 2019 | cure | 72 | 38% | 38% | 0.00 | 6 | 221 wk | 221 wk |
| 2019 | hazard | 72 | 74% | 74% | 0.00 | 6 | 206 wk | 206 wk |
| 2022 | cure | 32 | 22% | 72% | 0.16 | 15 | 221 wk | 256 wk |
| 2022 | hazard | 32 | 69% | 81% | 0.01 | 15 | 214 wk | 218 wk |

### Hazard model coefficients (fit on all ideas, C = 0.01, 95655 idea-weeks, 168 bills)

Log-odds of a first reading in a sitting week, per unit of the feature (bins: relative to weeks 1-4 since the trigger / 0-4 weeks since the last mention).

| Feature | Estimate | Std. error |
|---|---|---|
| `intercept` | -6.700 | 0.096 |
| `since_5+` | -0.257 | 0.268 |
| `since_13+` | -0.439 | 0.226 |
| `since_27+` | -0.106 | 0.193 |
| `since_53+` | -0.296 | 0.173 |
| `since_105+` | 0.044 | 0.168 |
| `log_weeks_since_trigger` | -0.266 | 0.061 |
| `gap_5+` | -0.324 | 0.240 |
| `gap_13+` | 0.106 | 0.204 |
| `gap_27+` | -0.053 | 0.201 |
| `gap_53+` | -0.543 | 0.171 |
| `recent_mentions_26w` | 0.330 | 0.114 |
| `log1p_mentions` | 1.714 | 0.252 |
| `max_firmness` | 0.110 | 0.102 |
| `any_firm` | 0.229 | 0.154 |
| `years_since_election` | -0.060 | 0.049 |
| `months_into_session` | -0.024 | 0.011 |
| `x_new_parliament` | 0.050 | 0.161 |
| `z_src_hansard` | -0.293 | 0.171 |
| `z_dept_treasury` | -0.227 | 0.340 |
| `z_dept_home_office` | 0.287 | 0.417 |
| `z_dept_ministry_of_housing_communities_and_local_government` | -0.361 | 0.417 |
| `z_dept_business_energy_and_industrial_strategy` | 0.257 | 0.366 |
| `z_dept_ministry_of_justice` | 0.338 | 0.478 |

**Hazard beats the cure model on Brier in 4/4 split-horizons and B1 in 0/4 (cure beats B1 in 0/4); mean AUC hazard 0.574 vs B1 0.627; Brier gain over the cure model +0.0004 to +0.0030 (small). Recommendation: serve the hazard model in place of the cure model (with CQR-calibrated windows), but present its probabilities as close to the base rate: it does not reliably beat the first-source baseline B1.**

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
- Split 2019-01-01, 26 weeks: hazard model Brier 0.074 is no better than B1 (0.073).
- Split 2019-01-01, 52 weeks: hazard model Brier 0.113 is no better than B1 (0.109).
- Split 2019-01-01: only 6 calibration bills for the cure model's window, so CQR made no adjustment.
- Split 2019-01-01: only 6 calibration bills for the hazard model's window, so CQR made no adjustment.
- Split 2022-01-01, 26 weeks: model Brier 0.065 is no better than B1 (0.064).
- Split 2022-01-01, 52 weeks: model Brier 0.114 is no better than B1 (0.108).
- Split 2022-01-01: 80% timing window covers 22% of bills.
- Split 2022-01-01, 26 weeks: hazard model Brier 0.064 is no better than B1 (0.064).
- Split 2022-01-01, 52 weeks: hazard model Brier 0.111 is no better than B1 (0.108).
- King's Speech data is not used (decision of 2026-10-24); ideas come only from ministers' words in written statements and Hansard.
- Commitments, topics and links come from rules-only NLP (phrase lexicon, TF-IDF grouping, title similarity); wrong groupings or links show up here as label noise.
- An idea with no linked bill is treated as not (yet) taken up, so a missed link is counted as a failure of the idea.
- Merged ideas take the earliest trigger of their groups, which uses knowledge of the shared bill; test ideas that later merged may look slightly earlier than they were.
- Outcomes after a change of government are censored at the dissolution, so the test sets after 2024 are short.
- Government Bills only (DATA_SCOPE.md): a promise delivered through a government-backed Private Member's Bill (or any other non-Government bill) counts as not taken up, so take-up is understated for such ideas.
- Served model: the API (`/api/v1/takeup/*`) serves the hazard model by default (fitted in memory on all ideas; 80% timing windows CQR-calibrated per as_of quarter on bills in a held-out 2-4 year window before it, raw with `timing_calibrated = false` when fewer than 10 calibration bills); the cure model is available with `?model=cure`.

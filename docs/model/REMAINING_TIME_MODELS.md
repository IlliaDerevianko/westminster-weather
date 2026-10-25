# Days-to-Royal-Assent tree models: process and tuning guide

This guide is for anyone (person or agent) who wants to improve the tree models that predict
**how many days a UK Government Bill has left until Royal Assent**. It records how the models
are built and evaluated, what has been tried with results, the rules that keep the evaluation
honest, and where the next gains are likely to come from.

Related documents:
- [`MODEL_SPEC.md`](./MODEL_SPEC.md): the full passage model.
- [`BUILD_PLAN.md`](./BUILD_PLAN.md).
- [`../../DATA_SCOPE.md`](../../DATA_SCOPE.md): Government Bills only.

Results write-ups:
- [`backend/reports/remaining_time_models/README.md`](../../backend/reports/remaining_time_models/README.md) (v1)
- [`backend/reports/remaining_time_models_v2/README.md`](../../backend/reports/remaining_time_models_v2/README.md) (v2)

---

## 1. Current state (October 2026)

| | Average error on test bills\* | Notes |
|---|---|---|
| Rule of thumb (B1) | 75.5 days | Median days left for past bills at the same stage and origin House |
| v1 XGBoost | 55.2 days | 44 features, random search |
| **v2 blend (current best)** | **53.8 days** | 90 features, Optuna, blend of 51% LightGBM, 43% XGBoost, 6% hist GB |

\* Mean absolute error over the 11 stage forecast points, for 196 bills from sessions 34–39
(2019–2026).

- **v2 against v1:** 0.0–2.9 days better (95% paired bootstrap interval). That's only just
  significant.
- **Where the gain is:** at the earliest forecast points. At introduction v2 errs by 105 days
  against 111 for v1. From first-House committee onwards they are within about 2 days.
- **Error falls along the pipeline:** from about 105 days at introduction to about 17 at
  ping-pong.

**Main lesson:** all six tuned model types land within 1 day of each other on test. The limit
is now the **information in the features**, not the choice of model or the tuning. Look for
gains in new data (section 9), not in more tuning.

---

## 2. Files and commands

All commands run from `backend/`.

| File | Role |
|---|---|
| `src/westminster_weather/analysis/remaining_time_dataset.py` | Builds the point-in-time dataset. `extended=False` gives the v1 features; `extended=True` adds the 46 v2 features |
| `src/westminster_weather/analysis/remaining_time_models.py` | v1: time split, baselines B0 and B1, random-search tuning, evaluation helpers (`error_metrics`, `bootstrap_mae`, `paired_diff_ci`, `per_anchor_mae`) |
| `src/westminster_weather/analysis/remaining_time_v2.py` | v2: leak-free encodings, 5-fold CV, design experiments, Optuna tuning, ensembles, one-time test scoring |
| `src/westminster_weather/analysis/stakeholder_chart.py` | Plain-language summary charts (v1 and v2) |
| `tests/model/test_remaining_time_*.py` | Leakage, split, baseline and encoding tests |
| `reports/remaining_time_models/`, `reports/remaining_time_models_v2/` | Committed outputs: charts, error tables, predictions, `metrics.json`, README |

```bash
uv run python -m westminster_weather.analysis.remaining_time_models          # v1, about 90 s
uv run python -u -m westminster_weather.analysis.remaining_time_v2           # v2, about 15 min
uv run python -m westminster_weather.analysis.stakeholder_chart v2 "<title>" # v2 summary chart
uv run ruff check . && uv run pytest -q
```

Use `python -u` so the v2 progress log is not buffered. Each line is a timestamped
`[  123s] ...` entry.

### Setup gotcha (macOS)

XGBoost and LightGBM need the OpenMP runtime. Without it they fail to import, with
`Library not loaded: @rpath/libomp.dylib`. Fix it with `brew install libomp`.

---

## 3. Problem definition

### Bills

Only **Government Bills that reached Royal Assent** are used: 522 bills.
- **Excluded:** bills still in progress (no answer yet) and bills that fell or were withdrawn
  (no Royal Assent date).
- **Consequence:** these models predict timing **given that the bill passes**. Whether it
  passes needs a separate model (MODEL_SPEC §4).

### Target

`target_days` is the number of calendar days from the forecast point to Royal Assent. Models
are trained on `log(1 + days)`.

### Forecast points

Each row is one bill at one forecast point.

- **Stage points (11):** the start of each stage.
  - First House (H1): 1st reading, 2nd reading, committee, report, 3rd reading.
  - Second House (H2): the same five stages.
  - Ping-pong.

  A stage point sees every sitting up to and including that stage's **first sitting, in the
  Bills API `sort_order`**. Stages held later the same day are therefore not visible.
- **Time points (7):** 30, 60, 90, 120, 180, 270 and 365 days after introduction. Each sees
  every sitting dated on or before that day.

Forecasts start at 1st reading because pre-Parliament data exists for only one bill (see
section 9).

### Time split (MODEL_SPEC §7)

| | Sessions | Bills | Rows |
|---|---|---|---|
| Train | 20–30 (2006-07 to 2017-19) | 320 | 4,365 |
| Test | 34–39 (2019 to 2026) | 196 | 2,605 |

Session 40 (current) is excluded: only its fastest bills have finished, which would bias the
test set.

---

## 4. Evaluation rules (keep these)

1. **Never use the test set to make a choice.** Features, targets, hyperparameters and
   ensemble weights are all picked on cross-validation, and the test set is scored once at
   the end.
   - If you look at test results and then change something, say so in the agent log and the
     README.
   - The v2 smoke run that touched the test set is recorded this way.
2. **Cross-validation is time-ordered, never random.** v2 uses 5 expanding-window folds.
   Each trains on sessions 20 up to the validation session and validates on:
   `{24}`, `{25}`, `{26, 27}`, `{28, 29}`, `{30}`.
   - Rows from the same bill are always on the same side of a fold.
3. **No look-ahead in features.** Every feature must be computable from sittings dated on or
   before the forecast date (`as_of`).
   - Session end dates are deliberately **not** features: they're rarely announced in
     advance.
   - Tests in `test_remaining_time_dataset.py` and `test_remaining_time_v2.py` enforce this.
     Add a test for every new feature.
4. **Target-derived features are out-of-fold.** The rule-of-thumb value, policy-area speed,
   policy-area residual and pace are learned from training targets. `prepare()` computes
   them:
   - from all training bills, for validation and test rows;
   - with a 5-fold split by bill, for training rows, so a bill's own target never feeds its
     own features.

   Inside cross-validation, `Folds` re-runs `prepare()` per fold.
5. **Compare like with like.** New models must be scored on the **same test rows** as v1 and
   the rule of thumb.
   - Join on `bill_id`, `anchor_type`, `anchor`; `run()` asserts full coverage.
   - Old cross-validation numbers aren't comparable after a fold change. Re-score the old
     setup on the new folds (see "v1 XGBoost setup (reference)" in v2).
6. **Report uncertainty, not just a point.**
   - 95% bootstrap intervals over test **bills**, not rows: rows of one bill aren't
     independent.
   - A paired bootstrap of the difference against the previous best.
7. **Report by forecast point.** One overall number hides where a model helps. v2's gain is
   almost all at the first two forecast points.

---

## 5. Features

### v1 (44 features)

| Group | Features |
|---|---|
| Position | `stage_index`, `in_second_house` |
| Time and pace | `elapsed_days`, `elapsed_sitting_days`, `days_since_last_event`, `pace_vs_typical` |
| Gaps between stages | `gap_h1_1r_2r` … `gap_h2_cttee_3r` |
| Procedural events | Committee of the whole House, Public Bill Committee, programme motion, money resolution, ways and means, carry-over, Lords committee negatived, same-day 2nd and 3rd reading, report skipped, sitting counts |
| Timing | days into the session and Parliament (at introduction and now), month |
| Bill type | `origin_lords`, `money_bill_title`, top-10 department one-hots |

### v2 additions (46; `extended=True`)

The table shows each group's cross-validation effect when added to v1 features (LightGBM with
fixed settings; negative is better), and whether it was kept.

| Group | CV effect | Kept |
|---|---|---|
| Title signals (Northern Ireland, (No. 2), Amendment, finance, emergency, Scotland/Wales, EU, repeal, title length) | −2.5 | yes |
| Policy area: renamed departments merged into 17 areas (`POLICY_AREAS`) | −1.8 | yes |
| Time in each House so far: Commons, Lords, first and second House, current House, current stage | −1.0 | yes |
| Bill size: sittings per stage, ping-pong rounds | −0.8 | yes |
| Encoded speed: `b1_log`, `area_speed_log`, `area_resid_log` (out-of-fold) | −0.6 | yes |
| Congestion: Government Bills in progress, bills before it in the session | −0.3 | yes |
| Politics: majority at the last election, coalition or minority (hard-coded `GOVERNMENT_MAJORITY`) | +0.1 | yes (removing it was worse) |
| Calendar: days to the summer recess, days to the 5-year limit | +0.6 | **dropped** by backward elimination |

All v2 features together: 50.7 → 46.9. Without the calendar features: 45.9.

The groups overlap, so single-group effects don't add up.

**Department lesson:** department flags help only for genuinely unusual departments. The
Northern Ireland Office has a median of 19 days to Royal Assent and HM Treasury 67. Transport
looked slow but is mid-pack (243 days). Renames split departments' bills across names, so
always map through `POLICY_AREAS`, and add new department names there when they appear.

### Final-model reliance

Permutation importance of the v2 blend on test: the extra error when each group is shuffled.

| Group | Extra error |
|---|---|
| v1 features | +33 days |
| Encoded speed | +8 |
| Time in each House | +2.5 |
| Title signals | +2.1 |
| Bill size | +1.5 |
| Congestion, policy-area flags, politics | under 0.5 each |

---

## 6. What was tried, and what happened

All cross-validation (CV) numbers are in days of mean absolute error on the v2 folds unless
noted.

| Idea | Result | Verdict |
|---|---|---|
| Log target vs predicting the correction to the rule of thumb (`target="resid"`) | 45.9 vs 48.4 | Keep the log target |
| Sample weights: each bill counts once | 48.1 vs 45.9 | Worse |
| Sample weights: each bill once, recent sessions favoured (τ = 6 sessions) | 48.5 | Worse (the teammate branch also reverted recency weighting) |
| Monotonic constraints (later stage → fewer days left) | 46.6 vs 45.9 | Not used |
| Loss function | All winning boosted models chose **absolute error** (LightGBM `regression_l1`, XGBoost `reg:absoluteerror`, CatBoost `MAE`, hist GB `absolute_error`) | Fits the MAE metric; keep absolute or Huber losses in the search space |
| Tuned model types (Optuna) | XGBoost 42.4, LightGBM 42.5, hist GB 43.6, CatBoost 44.4, random forest 45.9, extra trees 48.0 | Boosting wins |
| Ensembles | CV-weighted blend 42.1, average of top 3 42.2 | Small gain; the blend is the final model |
| Seed averaging (5 seeds per boosted model) | Used in the final fit | Cheap stability; keep it |
| v1: single decision tree, ridge (linear) | Test 59.0 and 62.6 | Trees beat linear by about 8 days |

### Cross-validation vs test

| | CV | Test |
|---|---|---|
| v1 | 37 (v1 folds) | 55 |
| v2 vs v1 gain | about 4 days (46.4 → 42.1) | 1.4 days |
| Rule of thumb (B1) | 56 | 75 |

Test bills simply take longer: 132 days left on average against 113 in training. That
period covers Brexit, COVID and two very long sessions. Expect cross-validation to overstate
gains on the newest era.

### v2 tuned hyperparameters

The full sets are in `reports/remaining_time_models_v2/metrics.json` → `tuned`.

**LightGBM:**

| Setting | Value |
|---|---|
| Trees | 964 |
| Learning rate | 0.022 |
| Leaves | 40 |
| Minimum rows per leaf (`min_child_samples`) | 103 |
| Row subsample | 0.86 |
| Column subsample | 0.49 |
| L2 / L1 regularisation | 6.7 / 0.45 |
| Loss | absolute error |

**XGBoost:**

| Setting | Value |
|---|---|
| Trees | 152 |
| Learning rate | 0.045 |
| Max depth | 6 |
| Min child weight | 38.6 |
| Row subsample | 0.56 |
| Column subsample | 0.98 |
| Loss | absolute error |

Both prefer **large leaves** (many rows per leaf, or heavy child weight). That's a sign the
data is small and noisy. Keep strong regularisation in any future search.

---

## 7. Tuning infrastructure and runtime lessons

### Optuna setup

- **Sampler:** TPE with `multivariate=True` and a fixed seed.
- **Parallel runs:** `constant_liar=True` when trials run in parallel.
- **Pruning:** `MedianPruner(n_startup_trials=15, n_warmup_steps=2)`. A trial is stopped
  early only after 3 of the 5 folds, and only once 15 trials have finished. About 4–30% of
  trials get pruned, which loses no meaningful quality.

### Trial counts

| Model | Trials |
|---|---|
| LightGBM | 150 |
| XGBoost | 150 |
| CatBoost | 50 |
| Hist gradient boosting | 30 |
| Random forest | 25 |
| Extra trees | 25 |

Search spaces are in `search_space()`.

### Speed

**Run trials in parallel, with one thread per model:** `PARALLEL_TRIALS = 8` on a 12-core
machine, and `threads=1` for each model.

The first v2 run gave each model all cores and ran trials one at a time. LightGBM alone ran
for 40+ minutes without finishing; with parallel trials it took 29 seconds. On a dataset this
small (about 4k rows), thread coordination costs more than it saves.

| Exception | Why |
|---|---|
| Hist gradient boosting runs sequentially | It uses OpenMP across all cores itself |
| Final fits use all cores | There is only one fit at a time |

**Cache fold matrices.** `Folds._arrays()` caches numpy arrays per (features, target,
weights), so trials don't rebuild them.

### Runtime breakdown (v2, 883 seconds)

| Step | Time |
|---|---|
| Design experiments | about 4 min |
| LightGBM tuning | 29 s |
| XGBoost tuning | 13 s |
| CatBoost tuning | 72 s |
| Hist GB tuning | 338 s |
| Random forest tuning | 58 s |
| Extra trees tuning | 24 s |
| Final fits and test scoring | about 2 min |

**Pausing a long run is safe; killing it is not.** Use `kill -STOP <pid>` to pause and
`kill -CONT <pid>` to resume. Progress is in memory with no checkpoints, and outputs are only
written at the end.

---

## 8. How to run an experiment

1. **Branch:** `feat/<short-name>`, per CLAUDE.md. Commit small, with an agent-log entry.
2. **New feature:**
   - Add it in `_extra_features()` (or a new `extended` level).
   - Add it to a group in `NEW_GROUPS` so the group experiments and importance pick it up.
   - Add a leakage test.
   - If it's built from targets, add it in `_encodings_from` / `_apply_encodings`, never in
     the dataset builder.
3. **New model type:** add it to `make_model()`, `search_space()` and `N_TRIALS`.
   - Give it a `threads` argument.
   - Make sure it accepts `sample_weight`.
4. **Check on cross-validation before going near the test set.** For a quick look use
   `Folds(train).score(family, params, Design(features))`. `DEFAULT_LGBM` gives fast,
   comparable feature experiments.
5. **Run the full pipeline once.** Compare to the previous best on the same test rows, and
   report:
   - the overall error with its bootstrap interval;
   - the paired difference against the previous best;
   - the error by forecast point.
6. **Write up honestly:** a README in a new `reports/<name>/` folder, including what didn't
   work. Update section 1 and section 6 of this guide.
7. **Stakeholder chart:** `stakeholder_chart.plot_v2()` works for any run folder with the same
   outputs.
   - Choose a title that states the honest takeaway.
   - Keep the three-line layout: new model, previous model, rule of thumb.

---

## 9. Where the next gains probably are

Ranked by likely impact. The models are already close to the limit of the current features.

1. **King's Speech data.** It would give a forecast point *before* 1st reading and a "was it
   announced" feature. Today only the Online Safety Bill (#3137) has pre-Parliament data. The
   `bills.kings_speech_flag` column exists but is empty.
2. **Bill size.** `size_clauses` and `size_pages` are empty. Committee sitting counts stand in
   for them today, but they're only known once committee is under way.
3. **Politics per bill**, from the Votes API: second-reading margin, rebels, Lords defeats and
   amendment counts (MODEL_SPEC Tier 1b). This should help the middle of the journey most.
4. **Era drift.** Try a rolling or recent-window refit, such as retraining every session. Also
   try a validation fold on the most recent training sessions only. Recency *weighting* didn't
   help, but drift is the main source of error.
5. **Prediction intervals.** MODEL_SPEC asks for p10–p90. Try LightGBM and XGBoost quantile
   objectives, or conformal intervals on cross-validation residuals, and check coverage on
   test.
6. **Survival framing.** XGBoost `survival:aft` would let unfinished bills be used as censored
   rows, rather than dropping them.
7. **Failure model.** Combine these timing models with a probability that the bill passes.
   That gives the full MODEL_SPEC forecast.

---

## 10. Known caveats

- **Timing only:** the models learned only from bills that passed.
- **Long bills are under-predicted:** predictions top out at about 250 days, and slow
  carry-over or paused bills come out too early.
- **Pace and cross-validation:** in v1, the pace medians for training rows use all training
  sessions, so v1's cross-validation scores were slightly optimistic. v2 uses out-of-fold
  values and fixes this.
- **Calendar days include recesses.** Sitting-day features exist, but the target is calendar
  days.
- **Hard-coded majorities:** government majorities are public election results, coded by hand.

# Westminster Weather: Passage Model Specification

**Owner:** Aleks (models) · **Version:** 0.1 · **Status:** ready to build

This document is the single source of truth for the **inside-Parliament passage model**.
It defines what the model predicts, the maths, the data it needs, and what it outputs.
Build instructions and task order are in [`BUILD_PLAN.md`](./BUILD_PLAN.md).
The database interface is in [`../../DATA_CONTRACT.md`](../../DATA_CONTRACT.md).

---

## 1. Scope

- **Population:** UK **Government Bills** only (Bills API `BillType = 1`).
- **Starting point:** the bill has already been introduced (1st reading has happened).
  Pre-Parliament stages (`LifecyclePhase.BEFORE_PARLIAMENT`) are **out of scope**.
- **End point:** Royal Assent or failure. Commencement (`AFTER_ROYAL_ASSENT`) is **out of scope** for this model.
- **Data period:** Bills API sessions with real coverage, `introducedSessionId >= 20` (about 2006–07 onwards).
  Sessions below 20 contain 0–5 bills and must be excluded.
- **Size:** about 588 government bills, 521 became Acts (checked against the live API on 2026-10-24).

### Questions the model answers, for any bill on any date `as_of`
1. **Expected time to Royal Assent**: median date and an 80% interval (p10, p90).
2. **Probability of failing at each remaining stage**, split by cause.
3. **Probability of Royal Assent at all**, and **before the current session ends**.
4. **Live update**: when new information arrives, the forecast is re-run from the bill's current state, and an alert is raised if it moved materially.

---

## 2. States

The model uses the existing enums in `westminster_weather.domain.stages`.
Do **not** create a second set of stage names. Model states are a subset of `BillStage`.

### 2.1 Modelled states and Bills API mapping

| Model state (`BillStage`) | Bills API `stageId`s | Notes |
|---|---|---|
| `COMMONS_FIRST_READING` | 6 | |
| `COMMONS_SECOND_READING` | 7 | Also absorbs 14 (programme motion), 15 (money resolution), 36 (ways and means), 25 (second reading committee) |
| `COMMONS_COMMITTEE` | 8, 49, 43, 48 | 8 = Public Bill Committee, 49 = Committee of the whole House |
| `COMMONS_REPORT_STAGE` | 9 | |
| `COMMONS_THIRD_READING` | 10 | |
| `LORDS_FIRST_AND_SECOND_READING` | 1, 2, 30 | Entered at the Lords 1st reading |
| `LORDS_COMMITTEE` | 3, 46 | |
| `LORDS_REPORT_AND_THIRD_READING` | 4, 5 | Entered at Lords report |
| `PING_PONG` | 12, 13, 42 | Optional: only happens if the second House amended the bill |
| `ROYAL_ASSENT` | 11 | Absorbing success state |

**Not modelled as separate states:**
- `PROGRAMME_AND_MONEY_MOTIONS`: usually the same day as 2nd reading. Keep it in the enum for the front end, but merge it into `COMMONS_SECOND_READING` for the model. The front end can show it as reached on the same date as 2nd reading.
- `BOTH_HOUSES_AGREE`: has no Bills API event. It is reached on the date of the last ping-pong sitting (or last 3rd reading if no ping-pong) and has zero duration.
- All other stage IDs (carry-over motions 18/19, suspension/revival 20–22, 31–34, instructions 35/37, etc.) are **events**, not states. Carry-over motions (18, 19) feed the `carry_over` feature.

### 2.2 State order depends on the originating House

`order(bill)` is the chain of states for that bill:

- **Commons-first** (`originatingHouse == "Commons"`):
  `C1R → C2R → C_COMMITTEE → C_REPORT → C3R → L1R2R → L_COMMITTEE → L_REPORT3R → [PING_PONG] → ROYAL_ASSENT`
- **Lords-first** (`originatingHouse == "Lords"`, titles end in `[HL]`):
  `L1R2R → L_COMMITTEE → L_REPORT3R → C1R → C2R → C_COMMITTEE → C_REPORT → C3R → [PING_PONG] → ROYAL_ASSENT`

`PING_PONG` is optional (square brackets). A bill can also legitimately skip a state in the data
(e.g. emergency bills where several stages happen in one day). Rule: **a state's entry date is the date of its first sitting; if a state has no sitting but a later state does, the skipped state is treated as entered and exited on the later state's entry date (zero duration).**

### 2.3 Absorbing failure states (use existing `StopReason`)

| `StopReason` | How it is labelled in historical data |
|---|---|
| `SESSION_RUNS_OUT` | Not an Act, not withdrawn, not defeated, and its last session ended by **prorogation** with no carry-over |
| `ELECTION_CALLED` | Same, but the session ended by **dissolution** |
| `PAUSED_OR_WITHDRAWN` | `billWithdrawn` is not null |
| `DEFEATED` | `isDefeated == true` (0 cases in the data, still supported) |
| `DEADLOCK` | Parliament Acts route. **Not modelled** (probability fixed at 0, documented as a limitation) |

The failure **stage** is the state the bill was in when it failed.

---

## 3. Time

- **Clock:** **sitting days**, defined as dates on which **either** House sat. Recesses don't count.
- **Sitting-day calendar** (in priority order):
  1. Parliament calendar / What's On API, if the data teammate provides it (`sitting_days` table).
  2. **Fallback (build this first):** the union of all dates that appear in any Bills API stage sitting, for any bill type, plus all Commons and Lords division dates. This is a good approximation because the Houses do business on almost every sitting day.
- **Future sitting days** (needed by the simulator): use the announced calendar if available. Otherwise, assume the **typical pattern**: weekday sitting except standard recesses, using the average share of weekdays that were sitting days in the same calendar week over the past 5 complete years. Simplest acceptable version: a sitting probability per ISO week number, estimated from history, with Monday–Thursday sitting on those weeks.
- **Output dates** are calendar dates: convert simulated sitting days back using the calendar.

---

## 4. Model components

The model has five parts. Each is estimated separately and combined in the simulator (section 5).

### 4.1 Advance hazard (core)

For bill *i* in state *k* (not absorbing), on sitting day *t*, having spent *τ* sitting days in state *k*:

```
logit P(advance on day t | still in k) =
      α_k                         # baseline speed of state k
    + f_k(τ)                      # duration dependence, piecewise constant
    + β · x_i                     # fixed bill features (known at introduction)
    + γ · z_i(t)                  # time-varying features (as of day t)
    + δ · g(d_i(t))               # end-of-session rush
```

- **`f_k(τ)`:** piecewise constant with buckets `τ ∈ {0, 1–5, 6–20, 21–60, 61+}` sitting days. Bucket `0` matters: many stages happen on the day the previous one ends.
- **`d_i(t)`:** sitting days remaining until the session end (see 4.3). For training, use the **actual** session end date. This is allowed because what matters is whether the government knew the end was coming; to limit leakage, use only the **announced** end date when an `announced_dates` row exists, else the actual end date minus a random 0–20 sitting days. Document this choice in the code.
- **`g(d)`:** indicator buckets `d ∈ {0–5, 6–15, 16–40, 41+}` (41+ is the reference).
- **Advance destination:** the next state in `order(bill)`. Special case: from the last state of the second House (`L_REPORT3R` or `C3R`), the bill goes to `PING_PONG` with probability `π_pp` (4.4), otherwise straight to `ROYAL_ASSENT`.
- **Estimation:** one pooled **discrete-time logistic regression** on bill × sitting-day rows (expect 50k–150k rows).
  - Stage intercepts `α_k` and duration buckets `f_k` are **per state**.
  - `β, γ, δ` are **shared across states** (590 bills cannot support per-state slopes).
  - L2 regularisation (`sklearn.linear_model.LogisticRegression`, `C` chosen by time-ordered cross-validation).
  - **Upgrade path, only if it clearly beats the logistic version on the validation set:** LightGBM classifier on the same rows (already in `pyproject.toml`). Keep the logistic model as the default because it is explainable.
- **Rows:** one row per (bill, sitting day) from the day after entering a state until the day it advances, fails, or is censored (`as_of` or end of data).
  Target `y = 1` on the day the next state is entered.
  Same-day transitions (several stages in one day) produce a row with `τ = 0` and `y = 1`.

### 4.2 Session-end and dissolution risk (main failure route)

About 60 of the 67 failed government bills died because the session or Parliament ended. This is handled **outside** the logistic model, by simulating the end date:

- **Session end `S`:**
  - If an announced prorogation or dissolution date exists (as of `as_of`), use it.
  - Otherwise draw from the **empirical distribution of session lengths** (in sitting days) of past sessions, **conditional on the session already having lasted as long as it has** (i.e. sample from lengths ≥ current length).
- **Dissolution `E`** (ends the Parliament, kills every bill including carried-over ones):
  - If announced, use it.
  - Otherwise: hazard per sitting day that rises with the Parliament's age. Fit a simple discrete-time logistic hazard on past Parliaments with features `years_since_last_election` (buckets: <3, 3–4, 4–4.5, 4.5+) and, if available, `government_poll_lead`. Hard cap: 5 years after the Parliament first met.
  - Elections in the data period, for labelling and fitting (verify against `parliaments` table when available): 2010-05-06, 2015-05-07, 2017-06-08, 2019-12-12, 2024-07-04.
- **Rule in simulation:** if `min(S, E)` arrives before Royal Assent, the bill fails with `SESSION_RUNS_OUT` (if `S` came first) or `ELECTION_CALLED` (if `E` came first) — **unless** the bill has a carry-over motion and `S` came first, in which case it continues into the next session (draw a new `S`).

### 4.3 Defeat in a Commons vote

Applies on the day a bill would leave `COMMONS_SECOND_READING` or `COMMONS_THIRD_READING`, and to report-stage votes.

```
P(defeat) = Φ( (R̂_i − M_t) / σ )
```

- `M_t`: government working majority on day *t* (from `political_snapshots`).
- `R̂_i`: expected number of government rebels. Tier 1 version: the rebel count at this bill's own earlier divisions if any, otherwise the median rebels in government-whipped divisions in the same Parliament. Tier 2: a member-level rebellion propensity model.
- `σ`: calibrated on all whipped Commons divisions in the data (choose `σ` so the predicted number of government defeats matches the observed number, by maximum likelihood).
- If `M_t` or rebels are unavailable, use a constant `P(defeat) = 0.002` per vote stage and log a warning.

### 4.4 Ping-pong

- **Whether ping-pong happens:** `π_pp = logistic(η_0 + η_1 · lords_defeats_cum + η_2 · amendments_agreed_cum_second_house + η_3 · size_clauses)`. Fit on completed bills (did the bill have any stage 12/13/42 sitting?). If Lords defeats are unavailable, drop that term.
- **Number of rounds** `N ≥ 1`: `N − 1 ~ Poisson(λ)`, `λ = exp(θ_0 + θ_1 · lords_defeats_cum)`. A round = one sitting of stage 12 or 13 or 42.
- **Round duration:** draw from the empirical distribution of gaps between consecutive ping-pong sittings (in sitting days).
- After the last round, the bill goes to `ROYAL_ASSENT` after a gap drawn from the empirical "last agreement → Royal Assent" distribution (often 0–3 sitting days, bunched at session end).

### 4.5 Withdrawal

- Constant per-sitting-day hazard `h_w`, fitted as (withdrawals) / (total bill sitting days at risk). Only 5 cases, so no covariates.
- Tier 2: multiply by `exp(κ · c2r_rebels_z)` if a large rebellion happened.

---

## 5. Simulator (produces every output)

For bill *i* at `as_of`, current state `k0`, time in state `τ0`:

```
repeat n = 10,000 times:
    draw S (session end) and E (dissolution) as in 4.2
    state ← k0, τ ← τ0, t ← as_of
    while state not absorbing:
        t ← next sitting day
        if t ≥ E:                       fail(ELECTION_CALLED, state);  break
        if t ≥ S:
            if carry_over:              draw new S for next session; continue
            else:                       fail(SESSION_RUNS_OUT, state); break
        with prob h_w:                  fail(PAUSED_OR_WITHDRAWN, state); break
        p_adv ← advance hazard (4.1) with features as of as_of
                (time-varying features are frozen at their as_of values, except
                 τ and days-to-session-end which evolve inside the simulation)
        if uniform() < p_adv:
            if state is a Commons vote stage and uniform() < P(defeat):
                fail(DEFEATED, state); break
            state ← next state (ping-pong branch via 4.4)
            record entry date of new state
            τ ← 0
        else:
            τ ← τ + 1
    record outcome, failure state and cause, Royal Assent date
```

- Use a fixed random seed per (bill, `as_of`) so forecasts are reproducible.
- Vectorise across simulations with NumPy (simulate all 10,000 paths together day by day). Target: under 2 seconds per bill.

### 5.1 Outputs per bill

| Output | Definition |
|---|---|
| `passage_probability` | share of paths ending in `ROYAL_ASSENT` |
| `p_ra_before_session_end` | share of paths reaching Royal Assent before the first simulated `S` |
| `time_to_royal_assent` | p10 / p50 / p90 of Royal Assent dates **among successful paths**, as days from `as_of` and as dates |
| `primary_stop_risks` | `{StopReason: share of paths failing with that cause}` |
| Per remaining state | `p_reach`, `p_fail_<cause>` (failed while in that state), `entry_date_p50` (among paths that reached it) |

These map onto the existing schema `schemas.bills.BillForecastResponse`
(`passage_probability`, `time_to_royal_assent: TimelineQuantiles`, `primary_stop_risks`).
`commencement_probability` and `time_to_commencement` stay `None` / equal to `passage_probability` until a commencement model exists; document this.
Per-state outputs need a **new** schema `StageForecastSchema` (see BUILD_PLAN task 9), matching `DATA_CONTRACT.md` §6.3.

---

## 6. Features

Features are referenced by the names in `DATA_CONTRACT.md` §5.5 (feature registry).
**Tier 1 must work end to end before any Tier 2 feature is added.**
Every feature must be computable **as of a date** without using later information.

### Tier 1 (from the Bills API alone — build first)
| Feature | Type | Definition |
|---|---|---|
| `originating_house_lords` | 0/1 | `originatingHouse == "Lords"` |
| `n_stage_sittings_cum` | int | stage sittings so far (proxy for bill complexity) |
| `committee_whole_house` | 0/1 | stage 49 occurred (fast, high-priority bills) |
| `same_day_2r_3r` | 0/1 | Commons 2nd and 3rd reading on the same day (emergency legislation) |
| `carry_over` | 0/1 | stage 18 or 19 occurred |
| `department` | category | `sponsors[].organisation.name`, top 10 departments + "other" (one-hot) |
| `introduced_months_into_session` | real | months between session start and 1st reading |
| `days_to_session_end` bucket | cat | see 4.1 |
| `parliament_age_years` | real | years since the Parliament first met |

### Tier 1b (needs Members / Votes APIs — add next)
| Feature | Source |
|---|---|
| `working_majority` | Members API, state of the parties on date (Commons) |
| `lords_government_share` | Members API, state of the parties on date (Lords) |
| `c2r_margin`, `c2r_rebels` | Commons Votes API, division at 2nd reading |
| `lords_defeats_cum` | Lords Votes API, divisions on the bill the government lost |
| `amendments_tabled_cum`, `amendments_agreed_cum` | Bills API amendments endpoints |

### Tier 2 (only if time allows)
`size_clauses`, `kings_speech_flag`, `c2r_speakers`, `c2r_word_count` (Hansard API), `critical_reports_cum`, `crisis_flag` (2020-03-01 to 2021-07-19), `government_poll_lead`.

### Alert-only signals (never model inputs)
News volume spikes (GDELT). They trigger an alert with a "news spike" reason but do not change the forecast.

---

## 7. Validation

- **Split by time, never randomly:**
  - Train: bills introduced in sessions 20–30 (about 2006–07 to 2017–19).
  - Test: sessions 34–39 (2019 onwards). Exclude the current session (in progress) from scoring.
- **Backtest procedure:** for each test bill, produce forecasts at `as_of` = the entry date of each state it reached (plus 1st reading). Use only the model trained on train sessions and features as of `as_of`.
- **Baselines to beat:**
  - **B0:** median days from 1st reading to Royal Assent, by originating House.
  - **B1:** for a bill in state *k*, the median remaining days to Royal Assent of training bills from state *k*.
- **Metrics:**
  | Metric | On |
  |---|---|
  | MAE in calendar days of `p50` Royal Assent date | bills that reached Royal Assent |
  | Coverage of the p10–p90 interval (target about 80%) | same |
  | Brier score of `passage_probability` and `p_ra_before_session_end` | all test bills |
  | Concordance index (`lifelines.utils.concordance_index`) of predicted median vs actual | all, with censoring |
- **Required charts** (save to `backend/reports/`):
  1. MAE by state the forecast was made from (should fall as the bill progresses: this proves live updating works)
  2. Model vs baselines B0 and B1
  3. Calibration plot for `passage_probability`
  4. Coefficient plot of the advance hazard (top drivers of speed)
  5. Distribution of days spent in each state

---

## 8. Live updates and alerts

1. **Triggers** (checked after each data refresh):
   `stage_change`, `lords_defeat`, `commons_rebellion` (rebels ≥ 10 or ≥ half the majority), `amendment_spike`, `carry_over`, `prorogation_announced`, `dissolution_announced`, `news_spike`.
2. On any trigger, re-run the simulator for that bill with `as_of = today`.
3. **Raise an alert** if the trigger is `stage_change`, `prorogation_announced` or `dissolution_announced`, or if the p50 Royal Assent date moved by more than **14 days**, or `passage_probability` moved by more than **0.05**.
4. Each alert stores: trigger, one-sentence plain-English summary, before and after forecast IDs, deltas, and `top_drivers` (features whose change contributed most, from the logistic coefficients × change in feature value).

Alert storage follows `DATA_CONTRACT.md` §6.4.

---

## 9. Known limitations (state these in the pitch)

- Small sample (about 590 bills, 67 failures): wide intervals are honest, not a bug.
- Defeats and Parliament Acts deadlocks are essentially absent from the data; the defeat model is structural, not learned from outcomes.
- Session end and election timing dominate failure risk and are partly political decisions.
- Pre-Parliament stages and commencement are not covered by this model.

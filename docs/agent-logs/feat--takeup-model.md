# Agent log: `feat/takeup-model`

## 2026-10-25 05:06 — refactor(takeup): bill record cache in one file instead of one per bill

- **What:** `takeup/bill_text.py` caches Bills API bill records in one file, `backend/data/cache/takeup/bills.json` (bill id -> record), instead of 588 files under `bills/`; `bill_long_titles` takes `cache_file`. The existing 588 records were merged into it and the folder removed. Also cleaned the local fetch cache (written statements, members, Hansard, GOV.UK pages, logs: ~19,800 files, 214 MB; gitignored, contents already in the database). Rebuilding the take-up source tables from scratch (`takeup/loaders.py`) would now need a fresh fetch; if labelling is rerun without the members cache, members with no stored posts count as never a minister (existing fallback).
- **Why:** Roman asked to clean up the many JSON files and combine them into one.
- **Checks:** `ruff check`: clean; `pytest tests/takeup`: 227 passed; 588 records, 583 non-empty long titles read back; `remaining_time_tails --75` results unchanged (median miss 10.2 weeks, quick bills 2.5).

## 2026-10-25 04:59 — docs(analysis): stakeholder chart, forecast miss vs days still to go by stage

- **What:** `remaining_time_tails.py` draws `3_miss_vs_time_left.png` (75/25 test bills): at each stage, typical days still to go (bars) and the new model's typical miss (line), both medians in days on one axis; listed in the stakeholder README.
- **Why:** Roman asked for the general stakeholder graph showing the miss decreasing towards Royal Assent.
- **Checks:** `ruff check`: clean; rendered and checked for overlaps. ±72 days at introduction (176 to go), ±29 at second-House 1st reading (92), ±9 at second-House 3rd reading (14).

## 2026-10-25 04:54 — docs(analysis): predicted vs real chart on test bills only

- **What:** `remaining_time_tails.py` draws `2_predicted_vs_real_test.png`: the 142 test bills (75/25 split), sorted by real time, one line for real weeks to Royal Assent and one for the new model's day-one prediction; listed in the stakeholder README.
- **Why:** Roman asked for a test-data-only graph with one predicted line and one real line.
- **Checks:** `ruff check`: clean; rendered and checked. Typical miss 10 weeks; predictions stay under ~50 weeks while real times reach 104.

## 2026-10-25 04:52 — docs(analysis): general by-stage stakeholder chart for the XGBoost tails

- **What:** `remaining_time_tails.py` saves the rule-of-thumb forecast (v2's B1, from training bills) with the test predictions and draws `1_by_stage_for_stakeholders.png` in the layout of `stakeholder_chart.plot_v2` (average error in days by stage: new model, previous model, rule of thumb, all bills), for both splits. Listed in the 75/25 stakeholder README.
- **Why:** Roman asked to repeat the general stakeholder graph.
- **Checks:** `ruff check`: clean; rendered and checked. 75/25: average ±53 days new, ±54 previous, ±73 rule of thumb; at introduction ±98 vs ±100.

## 2026-10-25 04:50 — docs(analysis): stakeholder summary for the 75/25 XGBoost results

- **What:** `backend/reports/remaining_time_tails_75_25/README.md`: plain-language summary for stakeholders (chart, how it was tested, results by kind of bill and by stage, what changed, limits, what would help next, files).
- **Why:** Roman asked for a stakeholder file.
- **Checks:** numbers taken from `metrics.json` and the chart's data (75/25 run of 6b50db0).

## 2026-10-25 04:48 — feat(analysis): what-if test, does knowing a bill's length help the XGBoost

- **What:** `remaining_time_tails.py` adds a hindsight variant with the final Act's length (`bill_texts`, `act_as_enacted`: log characters, schedules), read-only from the database. Not usable live: the Act exists only after Royal Assent and bills grow while debated. A day-one length source was looked for in the database and is missing: `bills.size_clauses` / `size_pages` are empty for all 588 bills, `bill_texts` holds only Acts (522) or latest versions (24), `api_evidence_events` only the demo bill. A Bills API publications fetcher was tried and removed (Roman: use the database only; the bill PDFs also refuse scripted downloads).
- **Why:** Roman asked whether the model can learn the length of the bill.
- **Checks:** `ruff check`: clean. 75/25, first reading: median miss 9.5 -> 7.4 weeks (long title -> + Act length), correlation 0.61 -> 0.69; over a year 30.0 -> 22.0; 30-50 weeks 9.4 -> 7.1; quick bills 6.1 -> 5.1. Upper bound only (hindsight).

## 2026-10-25 04:40 — feat(analysis): 75/25 time split for the XGBoost tails experiment

- **What:** `remaining_time_tails.py --75` trains on sessions 20-35 (bills introduced 2006-2021, 374 bills, 72%) and tests on 36-39 (2021-2026, 142 bills, 28%), the closest session boundary to 75/25; department one-hots follow the split's training bills; results and stakeholder chart in `backend/reports/remaining_time_tails_75_25/`. The default 62/38 run now also records its split in `metrics.json`, and the chart subtitle/footnote read it.
- **Why:** Roman asked for a 75/25 train/test split and the general stakeholder chart for it.
- **Checks:** `ruff check`: clean. 75/25, first reading, new model (long title + bill weights) vs v2 XGBoost: all bills 10.2 vs 10.3 weeks; quick bills 2.5 vs 6.6; typical 7.2 vs 10.4; 30-50 weeks 13.4 vs 8.4; over a year 32.5 vs 29.9. 80% range covers 77% of test bills (9% of the long ones). Hyperparameters are still the ones tuned on sessions 20-30.

## 2026-10-25 04:35 — docs(analysis): stakeholder chart covers all bills by stage

- **What:** right panel of `backend/reports/remaining_time_tails/0_summary_for_stakeholders.png` now shows the typical miss for all bills by stage, previous vs new model (was long bills only); headline "Sharper Royal Assent forecasts, most of all for quick bills".
- **Why:** Roman asked for the chart to be general, not only about long bills.
- **Checks:** `ruff check`: clean; rendered and checked. All bills: ±10 weeks on day one (previous ±11), ±1 week at the end; the two models match from report stage on.

## 2026-10-25 04:33 — docs(analysis): stakeholder chart for the XGBoost tails

- **What:** `remaining_time_tails.py --chart` draws `backend/reports/remaining_time_tails/0_summary_for_stakeholders.png` in the style of `analysis/stakeholder_chart.py` (same colours and layout): day-one miss by kind of bill (previous vs new model), and the miss on long bills by stage against all other bills.
- **Why:** Roman asked for a stakeholder graph like the existing ones.
- **Checks:** `ruff check`: clean; rendered and checked for label overlaps.

## 2026-10-25 04:30 — feat(analysis): long-title features and bill weights for the XGBoost tails

- **What:** `analysis/remaining_time_tails.py` retrains Illia's tuned v2 XGBoost (same bills, forecast points, train sessions 20-30 / test 34-39) with long-title features known at introduction (size: words, purposes, Acts named; flags: routine money, temporary, emergency, establish, regulate, criminal, NI executive), bill weights and recent-session weights, plus a p10-p90 quantile range. Results in `backend/reports/remaining_time_tails/`.
- **Why:** Roman asked to make the XGBoost more reliable for bills under 10 weeks and over 50 weeks.
- **Checks:** `ruff check`: clean. At first reading (196 test bills): under 10 weeks, median miss 10.9 -> 6.0 weeks (long title + bill weights) or 4.3 (recent weights); 10-30 weeks 9.9 -> 7.9; over 50 weeks no change (28.6 -> 30.7, always too early) and 0% inside the 80% range. Long title barely separates long bills in training (Spearman 0.03). Long bills are caught as they move: miss 31 weeks at first reading, 18 at 2nd-House first reading, 10 at 2nd-House committee. Mean error over all forecast points unchanged (59.9 -> 60.6 days).

## 2026-10-25 04:35 — feat(journey): stage 2 chart, Royal Assent predicted at first reading vs real

- **What:** `journey/stage2_ra.py`: for every Government Bill in the passage-model rolling backtest that got Royal Assent (194), the first forecast (made when the bill came out) vs the real Royal Assent. Charts `backend/reports/journey/stage2_ra_by_bill.png` (per bill: predicted p50, 80% band, real) and `stage2_ra_pred_vs_real.png` (scatter + error histogram); per-bill table `stage2_ra_pred_vs_real.csv`.
- **Why:** Roman asked for predicted vs real Royal Assent with the error and predicted time once the bill has come out (stage 3 is being built separately).
- **Checks:** `ruff check`: clean. Median |error| 12.7 weeks, mean 15.2, median signed +0.5 (no overall bias); 18% within 4 weeks; real date inside the 80% band for 55% (band too narrow). Predictions stay in 15-40 weeks: quick bills (< 10 weeks) are predicted too late and long ones (> 60 weeks) too early.

## 2026-10-25 04:15 — feat(takeup): firm-start option for stage 1 (firm commitment -> first reading)

- **What:** `TakeupData(firm_start=True)`: an idea's clock starts at its first firm commitment (firmness >= 2) visible at `as_of`; earlier mentions are ignored and ideas not yet firmly committed are not visible. `journey/stage_durations.py --firm` reruns predicted vs real with it (`backend/reports/journey/stage_durations_predicted_firm.csv`). Two tests (start date, no future rows).
- **Why:** Roman chose to narrow stage 1 to "Government commits" -> first reading (flowchart), firm commitments only.
- **Checks:** `pytest tests/takeup`: 227 passed; `ruff check`: clean. Stage 1, 27 cases: median miss 38.3 -> 31.3 weeks (mean 45.8 -> 43.0), predicted median 34 vs real 23.7. Baseline "median of earlier cases" is still better (28.8 median, 41.1 mean): firm start shortens the clock but gives no case-level timing skill.

## 2026-10-25 03:46 — Merge origin/main into feat/takeup-model

- **What:** merged PRs #20 and #22 (first-signal milestone and UI cleanup; stage-duration and originating-House analyses; tree models for days left to Royal Assent in `analysis/`). Only conflict: `backend/pyproject.toml` dependencies, kept all (pypdf, matplotlib, pyarrow, xgboost, optuna, catboost) and relocked. Unfinished files from two stopped feature-analysis agents (which used the same `analysis/` folder) were moved to a local backup instead of merged.
- **Why:** Roman asked to pull `main`.
- **Checks:** `pytest tests/takeup tests/journey tests/test_config.py`: 237 passed; `pytest tests/model tests/test_health_and_stages.py`: 131 passed, 2 skipped; `ruff check`: clean.

## 2026-10-25 01:51 — docs(journey): predicted vs real per case, current model vs improved timing rule

- **What:** `backend/reports/journey/predicted_vs_real_by_case.png` (scatter per stage) and updated `stage_durations_predicted.csv` with improved timing: stage 1 model timing corrected by its median bias on earlier resolved cases; stage 3 = median wait of earlier Acts; stage 2 unchanged.
- **Why:** Roman asked for a predicted vs real comparison after the gap remained large.
- **Checks:** median miss stage 1 38.3 -> 24.1 weeks (mean 45.8 -> 40.3); stage 2 12.7 (kept); stage 3 7.9 -> 2.4 (mean 11.5 -> 12.9). Scatter shows stage 1 and 3 predictions barely vary with the real duration (no case-level timing skill); stage 2 tracks reality.

## 2026-10-25 01:47 — docs(journey): predicted vs real time in each stage

- **What:** ran `journey/stage_durations.py` on the cleaned links (47 cases) and charted predicted vs real per stage (`backend/reports/journey/stage_durations_pred_vs_real.png`, `stage_durations_predicted.csv`).
- **Why:** Roman asked for the predicted vs actual comparison per stage.
- **Checks:** median |error|: stage 1 38 weeks (predicted median 50 vs real 24), stage 2 13 weeks (29 vs 30), stage 3 8 weeks (10 vs 3). Predicted distributions are much narrower than the real ones (models predict the typical case; the real data has long tails).

## 2026-10-25 01:43 — fix(takeup): stricter promise -> bill links; rerun checks

- **What:** `nlp.link_groups` now never links routine money bills (Supply and Appropriation, Consolidated Fund), links Finance Bills only within ~1 year, refuses procedural topics ("Madam Deputy Speaker"), and requires a strong match (≥ 0.45, or a named bill with ≥ 0.35) for links more than 2 years after the promise; 4 regression tests. Re-ran NLP on takeup-dev, the stage 1 evaluation (report and charts regenerated), the one-year-later check and the journey test; added the stage-duration charts.
- **Why:** Roman checked the real data: bills spend at most ~2 years in Parliament, yet 28 promise -> bill links had 4–5-year gaps, many to unrelated bills (e.g. "Madam Deputy Speaker" -> Financial Services Act 2012).
- **Checks:** `pytest tests/takeup/test_takeup_nlp.py`: 27 passed. Links 192 -> 164; gaps > 4 years 28 -> 5; median gap 28 weeks. Journey test, stage 1 error at the promise 89.5 -> 52.6 weeks, at +1 year 59.7 -> 37.9 (sample of 30 cases changed with the links, so stage 2/3 numbers are not like-for-like). Stage 1 hazard 80% window: 2019 split 77% coverage at 102 weeks wide (was 74% at 206). Repeated promises still ~6x more likely to get a bill (17.6% vs 3.0%).

## 2026-10-25 01:11 — feat(journey): end-to-end journey test (promise → law applied) with error chart

- **What:** `journey/journey_test.py`: for 30 sampled past cases (promise → Government Bill → Act, Royal Assent by mid-2025), predicts "when will this law apply?" at 7 checkpoints across stages 1–3 using only information known that day (stage 1 hazard model trained on earlier promises + historical medians; Aleks's rolling stage-2 backtest + median Royal Assent → applied; stage-3 `forecast_act`). Writes `backend/reports/journey/journey_test.csv` and `journey_error_by_stage.png`.
- **Why:** Roman asked for one test through all stages showing the error shrinking towards the end, with the stage-3 "weeks until the law applies".
- **Checks:** mean |error| in weeks: promise 89.5 → +1 year 59.7 → first reading 22.6 → middle stage 14.6 → last stage 10.9 → Royal Assent 7.7 (median 1.9); "+3 months" 23.0 (only 6 slow cases remain). `ruff`: clean.
- **Notes:** "law applied" = first commencement order, or start on Royal Assent when no orders are needed.

## 2026-10-25 00:59 — feat(takeup): day-one and one-year-later checks on the training data; pitch

- **What:** `takeup/insample.py`: forecasts at each promise's trigger vs reality (`stage1_insample.csv`) and, with `--updating`, forecasts one year later for promises still waiting, using the data as it stood that month (`stage1_updating.csv`); `docs/model/TAKEUP_PITCH.md`.
- **Why:** Roman asked for predicted vs real on the training data and for the "on day one nobody can tell; watch the repetition" pitch.
- **Checks:** day one: bill within 52 w AUC 0.546, Brier 0.108 vs base 0.107. One year later (708 promises): AUC 0.599, Brier 0.0249 vs base 0.0261; real rate 2.1% (no repeats, model 2.2%) vs 20.0% (3+ repeats, model 18.9%). `ruff`: clean.
- **Notes:** in-sample; the 3+ group has only 15 promises.

## 2026-10-25 00:55 — Merge Aleks's stage 2 (feat/passage-model-tasks-4-11) into feat/takeup-model

- **What:** merged Aleks's passage model (stage 2: states, hazard, failure routes, ping-pong, Monte Carlo simulator, LightGBM quantile blend, backtest, `/api/v1/bills/{bill_id}/forecast`). Only conflict: `backend/pyproject.toml` dependencies; kept all (pypdf, matplotlib, pyarrow) and relocked. Our own unfinished stage-2 attempt (`journey/passage.py` etc., never committed) was moved to a local backup instead of merged.
- **Why:** Roman: use Aleks's finished stage 2 and combine with stages 1 and 3.
- **Checks:** `pytest tests/takeup tests/journey tests/test_config.py`: 233 passed; `pytest tests/model tests/test_health_and_stages.py`: 111 passed, 2 skipped; `ruff check`: clean.

## 2026-10-25 00:50 — feat(journey): stage 3 commencement model (Royal Assent → orders finished)

- **What:** `journey/commencement.py`: P(needs commencement orders) and first-order timing (cure model, all-ones mask), and when orders finish (weekly hazard of a further order, simulated order paths), conditional on `as_of`, CQR 80% windows; backtest on Acts with Royal Assent from 2019 (`backend/reports/journey/stage3_*`). `forecast_act(bill_id, as_of)`. Does not claim "fully in force" (`is_fully_in_force` is always True in the source data).
- **Why:** Roman asked to extend the model to all stages.
- **Checks:** `pytest tests/journey/test_commencement.py`: 11 passed. Backtest: needs-orders AUC 0.88 / Brier 0.138 at Royal Assent (baselines 0.50 / 0.249, by department 0.66 / 0.244); first-order timing no better than the median; finished-by-104-weeks Brier 0.218 vs 0.256; CQR coverage 87% (first order), 94% (finish; too wide later).

## 2026-10-25 00:50 — feat(takeup): serve the hazard model with CQR windows

- **What:** take-up API defaults to the hazard model (`?model=cure` still available), CQR-calibrated timing windows per `as_of` quarter (raw with a flag if < 10 calibration bills), `/history` re-predicted week by week with only information known then; evaluation doc notes Government-Bills-only and the served model. Also: earlier `style(takeup)` lint commit on `replay.py` had no separate log entry.
- **Why:** the hazard model is more realistic than the cure model on real data (mean pi_now for silent promises 4% vs 54%).
- **Checks:** `pytest tests/takeup/test_takeup_api.py`: passes (offline, both models).

## 2026-10-25 00:19 — fix(takeup): stop extrapolation on sparse features; add stage 1 replay

- **What:** `hazard.py` switches off a static 0/1 feature seen on fewer than 15 training signals on either side (plan rule) and caps standardised values at ±5. New `takeup/replay.py`: for 9 past promises, a model trained only on earlier promises forecasts at the promise and 1/2/4 years later, next to what really happened (`backend/reports/takeup/stage1_replay.csv`).
- **Why:** Roman asked for real-vs-predicted tests on past data. The replay exposed a bug: a written-statement promise in March 2015 got a 100% forecast because written statements only start in Nov 2014, so the training data had almost none (feature spread 0.02, z = −46).
- **Checks:** `pytest tests/takeup/test_takeup_hazard.py`: 18 passed (2 new regression tests); after the fix that promise gets 15% fading to 3% (correct: never got a bill).
- **Notes:** replay findings: timing windows held the real date in 13 of 18 checkpoints for delivered promises; chance rises with repeat mentions (Post Office 20% → 95%, delivered); fails when mentions sit in other groups (Online Safety: 1 mention in its group) and for repeatedly-promised-but-dropped policies (conversion practices 25% → 83%, never introduced).

## 2026-10-24 23:58 — docs(log): checkpoint 4 hand-label results

- **What:** Roman reviewed `review_mentions.csv` (100) and `review_links.csv` (50) and confirmed every system decision correct; `score`: mentions precision 100% (85/85 kept, 0/15 wrongly rejected), links precision 100% (33/33 linked, 0/17 wrongly rejected).
- **Why:** checkpoints 4 and 8 (target ≥ 85%).
- **Checks:** `cli score`. With these sample sizes the 95% lower bounds are about 96% (mentions) and 91% (links).
- **Notes:** label quality is therefore not the main reason the models don't beat B1; the limit is the information available at the trigger.

## 2026-10-24 23:57 — feat(takeup): discrete-time hazard model with recency features and conformal windows

- **What:** `takeup/hazard.py` (one row per signal-week; recency features: weeks since last mention, mentions in last 26 weeks, cumulative mentions, max firmness; L2 logistic; forecasts assume no new mentions), `takeup/conformal.py` (CQR-style widening of the 80% timing window on a held-out period), `signals.mention_timeline(as_of)`; `evaluate.py` compares cure vs hazard vs B0/B1 on identical splits; report and charts regenerated.
- **Why:** option A chosen by Roman after the cure model failed to beat baseline B1.
- **Checks:** `pytest tests/takeup`: 200 passed; `ruff`: clean. Real backtest: hazard Brier ≤ cure in 4/4 split-horizons (tiny margins), neither beats B1 (52-week AUC: hazard 0.52/0.59, cure 0.57/0.53, B1 0.63/0.63); 80% window coverage hazard 74%/81% (CQR) vs cure 38%/72%, but windows are ~4 years wide.
- **Notes:** at the trigger the hazard model only knows the first mention, which limits ranking; API and saved runs still serve the cure model (decision pending).

## 2026-10-24 23:39 — docs(model): first real take-up evaluation report and charts

- **What:** `docs/model/TAKEUP_EVALUATION.md`, `backend/reports/takeup/` (metrics.json, calibration, pi_now curves, time-to-bill, feature effects) from the first real run on 985 ideas.
- **Why:** task 9; Roman asked for honest results.
- **Checks:** `run_evaluation(write=False)` on takeup-dev; nothing written to the database.
- **Notes:** the model does not beat baseline B1 (52-week AUC 0.57/0.53 vs 0.63; Brier 0.113 vs 0.109); 80% timing windows cover 22–38% of bills. Strong effects: repeat mention (+2.0 ± 0.2), firmness (+0.81 ± 0.27), later firm commitment (+0.99 ± 0.41).

## 2026-10-24 23:39 — feat(takeup): GOV.UK events fetch and matching

- **What:** `takeup/govuk_events.py`: fetches consultations, consultation responses, white/green papers and draft bills from the GOV.UK search API (13,952 events, cached), matches them to ideas by text (no bill information), and loads `idea_events` via the take-up upsert (refuses production).
- **Why:** task 5.
- **Checks:** `pytest tests/takeup/test_takeup_govuk.py`: 19 passed; `ruff`: clean. Dry run: 255 ideas would get 547 events.
- **Notes:** not written to the database: spot-check precision ~50–60%, mostly from weak idea topics. Re-running `nlp` renumbers ideas, so events must be re-matched afterwards.

## 2026-10-24 23:37 — feat(takeup): NLP pipeline (label, group, link) and review/score

- **What:** `takeup/nlp.py`: trigger sentences from `source_documents` (rules + an extra commitment check), Hansard minister check by date, TF-IDF topic grouping with 2-year gap splits, linking to Government Bills first read in [trigger − 7 d, trigger + 5 y] (title / long title, bill named in the text → auto), merging groups that link to the same bill with `group_id` kept for leakage control; writes `signal_mentions`, `ideas`, `idea_mentions`, `idea_bill_links` in one transaction; `review` / `score` with an overwrite guard. CLI `nlp`, `review`, `score`; `db.clear_derived_tables` (refuses production and raw tables).
- **Why:** task 7 (Tier B2–B4, B6).
- **Checks:** take-up tests 157 passed (synthetic self-test: commitment precision/recall 100%, grouping ARI ≥ 0.85 after merging, linking ≥ 85%); `ruff`: clean. Real run on takeup-dev (45 s): 25,242 documents → 12,518 trigger sentences → 1,939 commitments → 1,113 groups → 985 ideas; 192 linked (169 auto, 23 rule), 793 never taken up. Review files: `backend/data/takeup/review/review_mentions.csv` (100), `review_links.csv` (50).
- **Notes:** ~80% of raw trigger hits were not commitments (e.g. "draft Bill" mentions), hence the extra check. Estimated precision before hand-labels: commitments ~75–85%, auto links ~70–75%. Re-running `nlp` renumbers ideas, so `idea_events` must be re-matched after each run.

## 2026-10-24 23:22 — feat(frontend): promises pages with take-up forecasts

- **What:** new `/promises` page (search, verdict filter, sort, as-of picker, verdict badges, chance now / within 52 weeks, expected bill month, bill once introduced) and `/promises/[ideaId]` detail page (plain-English headline, inline-SVG chart of pi_now with the 80% window and event / mention / first-reading markers, minister quotes, events); shared types and fetch helper in `src/lib/takeup.ts`; nav links from home and `/bills`.
- **Why:** task 11.
- **Checks:** `npm run build`: passes (type check); not yet run against real take-up data (backend was down).
- **Notes:** timing in the headline is conditional on a bill coming; to be checked on real data.

## 2026-10-24 23:19 — feat(takeup): take-up API endpoints

- **What:** `takeup/api.py` (router included in `main.py`): `GET /api/v1/takeup/ideas`, `/forecast`, `/history` (all `as_of`-safe; split groups resolve to their own signal and don't reveal the later merge). Model rebuilt from the latest `takeup_runs` row, else fitted once in memory; stored `takeup_forecasts` used when present; 503 if `DATABASE_URL_TAKEUP` is not set.
- **Why:** task 10.
- **Checks:** `pytest tests/takeup/test_takeup_api.py`: 12 passed (offline, synthetic data, injected service); `ruff`: clean.
- **Notes:** saved runs keep only estimates and SEs, so CIs from a rebuilt model ignore covariances (approximate); speed against the real data not yet measured.

## 2026-10-24 23:14 — feat(takeup): evaluation, baselines and charts

- **What:** `takeup/evaluate.py`: time-split backtest (cuts 2019-01-01 and 2022-01-01; training only on signals visible at the cut, asserted), baselines B0 / B1, AUC and Brier at 26 / 52 weeks, 5-bin calibration, 80% window coverage, concordance, verdict accuracy, parameter table; 4 charts + `metrics.json` in `backend/reports/takeup/`; generator for `docs/model/TAKEUP_EVALUATION.md`; `save_run` to `takeup_runs` / `takeup_forecasts` (refuses production); `run_evaluation(write=False)`.
- **Why:** task 9.
- **Checks:** `pytest tests/takeup/test_takeup_evaluate.py`: 12 passed (synthetic data with known parameters, all recovered within 2.5 SE); `ruff`: clean.
- **Notes:** real run waits for the NLP ideas. Synthetic: 52-week AUC 0.65 vs B1 0.59 (2019 cut); B1 is competitive at the 2022 cut. At-trigger forecasts can't see later events, so early timing ranks weakly by design.

## 2026-10-24 23:02 — feat(takeup): raw-source loaders and migrate / load-raw / db-status commands

- **What:** `takeup/loaders.py` builds `source_documents` (written statements, de-duplicated Hansard contributions with `matched_phrases`, HTML stripped) and `member_government_posts` from the cache, with idempotent upserts; `cli.py` adds `migrate`, `load-raw` and `db-status` (each with an offline `--dry-run`); `db.py` upserts on a given engine and refuses production.
- **Why:** contract 10.0a/10.0b: the database is the source of truth for Tier B text.
- **Checks:** `pytest tests/takeup/test_takeup_loaders.py` (in-memory SQLite, dummy settings) and full suite: 158 passed; `ruff check`: clean. Dry run on the real cache: 25,242 documents (17,844 statements, 7,144 Hansard spoken, 254 Hansard written; 0 skipped), 2,585 posts for 1,440 members; 86 MB of text.

## 2026-10-24 23:02 — refactor(takeup): remove King's Speech data

- **What:** deleted `kings_speech.py` and its tests; removed `kings_speech_items`, `x_kings_speech`, the `kings_speech` source and event type, and the carried-over exclusion from tables, signals and tests; written statements are now the reference first source (`z_src_hansard` only). Contract section 10 and `TAKEUP_PLAN.md` updated (task 4 / checkpoint 3 removed). Local King's Speech PDFs and the review CSV deleted.
- **Why:** Roman decided not to use King's Speech data (overriding the spec's "never cut" after the trade-off was explained).
- **Checks:** full `pytest`: 158 passed; `ruff check`: clean. All non-King's-Speech leakage protections and their tests kept.
- **Notes:** `bill_text.py` kept (long titles needed for NLP linking). `bills.kings_speech_flag` (passage model) untouched. Timing estimates now rest on NLP ideas only.

## 2026-10-24 22:50 — Merge origin/main into feat/takeup-model

- **What:** merged PR #17 (legislation.gov.uk post-Royal-Assent ingestion for all 522 Acts).
- **Why:** Roman asked to bring the branch up to date with `main`.
- **Checks:** no conflicts; `pytest`: 184 passed; `ruff check`: clean.

## 2026-10-24 22:48 — feat(takeup): align data layer with contract section 10; hide DB URLs

- **What:** settings renamed to `DATABASE_URL_TAKEUP` / `DATABASE_URL_TAKEUP_UNPOOLED` (still never falling back to `DATABASE_URL`); new tables `source_documents`, `member_government_posts`; `signal_mentions.doc_id`; `ideas.origin`; `kings_speech_items.bill_id`; `takeup_forecasts` keyed by `signal_id` with the spec column names. `get_takeup_signals`: `x_kings_speech` from dated `idea_events`, carried-over King's Speech items excluded and counted, one `x_white_paper` for white or green papers. All database URL settings are now hidden from `repr()`.
- **Why:** Roman's task spec and the approved contract (checkpoint 2); a reviewer saw a failing test print the settings object, including the production URL with its password, in its terminal.
- **Checks:** full `pytest`: 183 passed (signals 55; new `tests/test_config.py` checks no password in `repr`/`str`); `ruff check`: clean.
- **Notes:** reviewer found a leak (which split group got an idea's events depended on groups dated after `as_of`); fixed with regression tests. Remaining limit: `idea_events` has no `group_id`, so for an idea whose groups are merged through a shared bill, an event matched to a later group shows on the earliest group until the merge; the Tier B producer should match events to groups (follow-up in task 7).

## 2026-10-24 22:48 — fix(takeup): conditional timing quantiles and contract output names

- **What:** `predict()` now gives timing quantiles conditional on the s silent weeks already passed, t_q(s) = min{t > s : 1 − S_u(t)/S_u(s) ≥ q} (calendar and sitting weeks); outputs renamed to the contract names (`pi`, `pi_lo`, `pi_hi`, `verdict`, `weeks_elapsed`, `pi_now`, `p26`, `p52`, `t_10`, `t_50`, `t_90`); bootstrap defaults to B = 200 and conditions on the same s; `pi_now` computed in log space.
- **Why:** the task spec names this prototype bug (quantiles were measured from week 0, so they could fall before "now").
- **Checks:** `pytest tests/takeup/test_takeup_model.py`: 36 passed (quantiles > s for s up to 258, s = 0 equals the unconditional result, exact geometric case, non-sitting weeks skipped, p13 ≤ p26 ≤ p52 ≤ pi_now); a reviewer's slow reference matched on 400 random models; `ruff`: clean.
- **Notes:** the reviewer found a divide-by-zero when pi rounds to 1 and S_u underflows; fixed by the log-space formula, with a regression test.

## 2026-10-24 22:35 — docs(contract): key take-up forecasts by signal_id

- **What:** `takeup_forecasts` gets a `signal_id` column and primary key (`run_id`, `signal_id`, `as_of`); 10.9 documents `signal_id` = `idea_id`, or `<idea_id>:<group_id>` for a group whose merge is not yet visible at `as_of`.
- **Why:** a reviewer found that one idea can produce several signals before its merge, which would collide on (`run_id`, `idea_id`, `as_of`).
- **Checks:** none (docs only; still a draft for checkpoint 2).

## 2026-10-24 22:29 — Merge origin/main into feat/takeup-model

- **What:** merged 13 commits from `main` (PRs #12–#16: `/api/v1/bills` list and dynamic tracker for any bill, bill selector page, all three lifecycle phases on the tracker, Government-Bills-only ingestion, ping-pong skipped/rounds display).
- **Why:** Roman asked to bring the branch up to date with `main`.
- **Checks:** no conflicts; `pytest`: 149 passed; `ruff check`: clean.

## 2026-10-24 22:27 — feat(takeup): take-up tables, database access and get_takeup_signals

- **What:** `takeup/tables.py` (take-up tables on their own MetaData, never created by the app's `create_all`), `takeup/db.py` (engines for the take-up Neon branch that refuse to fall back to production, `migrate()` not yet run, upserts, a strictly read-only loader for production reference tables), `takeup/signals.py` (`TakeupData.get_takeup_signals(as_of)`: visible mentions only, group split until the bill's first reading, change-of-government censoring, z/x features, sitting dates and week probabilities for the model), two optional settings in `config.py`.
- **Why:** plan tasks 2 (code) and 6/7 (`get_takeup_signals`, Tier B5).
- **Checks:** `pytest tests/takeup`: 102 passed (signals: 45, incl. a leakage test with rows on both sides of `as_of`); full suite 144 passed; `ruff check`: clean.
- **Notes:** 2 reviewers found 12 issues (critical: split groups inherited idea-level events and titles that are only known through the later bill link; major: change-of-government censoring used the dissolution before the election result was known); all fixed with tests. Found while reviewing: the existing `db.fetch_*` readers run `ensure_reference_tables_seeded_sync()`, which writes to production (idempotent `create_all` + merges), so take-up code uses its own read-only loader. Still to align with the spec / contract: env var `TAKEUP_DATABASE_URL` → `DATABASE_URL_TAKEUP`, spec column names, raw-source tables, `origin` instead of `tier`, `x_kings_speech` from `idea_events`, forecasts keyed by `signal_id`.

## 2026-10-24 22:27 — feat(takeup): port the cure model with a sitting-week mask

- **What:** `takeup/model.py`: the prototype's mixture cure model (pi, weekly hazard with 6 baseline bins, censored log-likelihood, L-BFGS + ridge, delta-method CI, verdicts, bootstrap) plus the sitting-week mask (lambda = 0 in weeks with no sitting day; expected mask from historical sitting probability for future weeks), per-row `data_end`, timing in calendar and sitting weeks, calendar coverage bounds, and a smoothing penalty between neighbouring baseline bins.
- **Why:** plan task 6 (Tier A, required change 1).
- **Checks:** `pytest tests/takeup/test_takeup_model.py`: 25 passed (incl. parameter recovery within 3 SE with a masked calendar, an all-ones mask reproducing the prototype to ~1e-3, non-sitting weeks leaving the likelihood and pi_now unchanged); `ruff`: clean.
- **Notes:** 2 reviewers found 7 issues (1 critical: a bill first read up to 7 days before its signal was treated as never taken up; it now counts as a week-1 event); all fixed with regression tests. Still to do: conditional timing quantiles t_q(s) from the task spec (the port still measures from week 0) — next commit.

## 2026-10-24 22:27 — feat(takeup): King's Speech items and links to bills

- **What:** `kings_speech.py` discovers the 13 King's/Queen's Speech briefing notes on GOV.UK (2012–2026), parses their contents pages (incl. one-word-per-line PDFs and bracketed bill tags), flags draft / carried-over / hybrid items and departments, and links items to Government Bills (exact normalised title, then TF-IDF over title + long title; `auto` / `rule` / `none`). `bill_text.py` caches bill long titles and computes first readings. Exports `backend/data/takeup/review/kings_speech_links.csv` for the checkpoint 3 hand-check.
- **Why:** plan task 4 (Tier B1 structured sources).
- **Checks:** `pytest tests/takeup/test_kings_speech.py`: 32 passed; `ruff`: clean. Real run: 372 items, 231 auto / 50 rule / 91 none, 29 carried over, 12 hybrid bills excluded.
- **Notes:** built by a workflow builder, then 2 independent reviewers (13 findings, e.g. wrong carried-over links in Oct 2019, missed bills named only in speech tags, under-flagged carry-overs) and a fixer (12 fixed, 1 threshold change rejected after checking the data). Speech date = PDF cover date (State Opening); `sessions.start_date` is one day later for 12 of 13. Reads production only through a read-only loader; nothing written to any database.

## 2026-10-24 22:26 — feat(takeup): faster, fault-tolerant Tier B fetch

- **What:** written statements now fetched at up to 20 requests/s (16 parallel); every 429/5xx is logged; Hansard runs 2 searches at a time and a search that still fails after retries is logged and skipped (not cached) so the next run retries it. `triggers.py` reformatted (whitespace only).
- **Why:** Roman approved speeding up the fetch (options A + B); Hansard returned HTTP 500 under 4 parallel searches.
- **Checks:** fetch 2015+: 17,844 statements, 0 rate-limit responses; Hansard 132/132 searches, 0 failures; 983 speakers. `ruff check` clean; `pytest tests/takeup`: 102 passed.

## 2026-10-24 22:16 — docs(model): align take-up plan and contract with the task spec

- **What:** rewrote `TAKEUP_PLAN.md` with the spec's naming (Tier A = the model, Tier B = the data pipeline B1–B6), conditional timing quantiles, `DATABASE_URL_TAKEUP`, current fetch status and a spec-reconciliation section. In `DATA_CONTRACT.md` section 10: spec names (`p26`, `p52`, `t_10/50/90`, `kings_speech_items.bill_id`), `ideas.origin` instead of `tier`, carried-over King's Speech items excluded, and a take-up feature registry (10.11).
- **Why:** Roman shared the full task spec; the plan used the opposite Tier A/B naming and missed the conditional-quantile fix and some column names.
- **Checks:** none (docs only).
- **Notes:** flagged for checkpoint 2: King's Speech before 2012 not on GOV.UK; `event_date` kept over `date` (contract convention); endpoints under `/api/v1/takeup/`; take-up registry kept separate from the passage model's 5.5. Code follow-ups after the build workflow: rename the env var, conditional quantiles + test, column names, raw-source tables, `x_kings_speech` from `idea_events`.

## 2026-10-24 22:01 — docs(contract): store raw Tier B sources in the database

- **What:** added draft tables `source_documents` (all written ministerial statements, Hansard contributions with a trigger phrase; full text) and `member_government_posts` to `DATA_CONTRACT.md` section 10; `signal_mentions` now points to its document via `doc_id`.
- **Why:** Roman wants the database to be the source of truth rather than a local cache; he chose to store everything (~40–45 MB of 1 GB) over a keyword-filtered subset.
- **Checks:** sizes measured from the cache: statements 17,844 (59 MB text), Hansard 3,895 unique contributions (16 MB text); gzip of all text 24 MB.
- **Notes:** follow-ups: add both tables to `takeup/tables.py`, a loader from cache to database, and make the NLP read from the database.

## 2026-10-24 21:58 — docs(contract): connect King's Speech items to earlier commitments

- **What:** amended the draft `DATA_CONTRACT.md` section 10: a King's Speech item is matched by text to earlier Tier B ideas and stored as their `kings_speech` event (dated on the speech), so `x_kings_speech` updates a forecast from the speech day; only unmatched items become their own Tier A ideas. Bill-based merging is kept only for de-duplication.
- **Why:** Roman pointed out that the tiers are connected (NLP finds the trigger, the King's Speech then moves the forecast). The previous draft joined them only through a shared bill, which under the `as_of` rule is invisible until first reading, so a forecast would miss the speech in exactly the window that matters.
- **Checks:** none (docs only; still a draft for checkpoint 2).
- **Notes:** follow-ups: `signals.py` must take `x_kings_speech` from `idea_events`; Tier B linking (task 7) must add the idea ↔ King's Speech item matching.

## 2026-10-24 21:32 — chore(deps): add pypdf for King's Speech briefing notes

- **What:** added `pypdf` to `backend/pyproject.toml` / `uv.lock`. Corrected the timestamps of the earlier entries in this log to the real commit times (they had been written 10–30 minutes ahead).
- **Why:** Tier A parses the King's/Queen's Speech briefing-note PDFs from GOV.UK; Roman approved the dependency at checkpoint 1.
- **Checks:** `uv run python -c "import pypdf"`: 6.21.2.

## 2026-10-24 21:30 — docs(contract): draft take-up model tables (section 10)

- **What:** added `DATA_CONTRACT.md` section 10: `kings_speech_items`, `signal_mentions`, `ideas`, `idea_mentions`, `idea_events`, `idea_bill_links`, `takeup_runs`, `takeup_forecasts`, the `get_takeup_signals(as_of)` interface and validation checks. Marked as a draft.
- **Why:** task 2 of the take-up plan (checkpoint 2). Tables are migrated only after approval, on the `takeup-dev` Neon branch first.
- **Checks:** none (docs only).
- **Notes:** `idea_mentions.group_id` keeps the pre-merge topic group, because merging ideas that link to the same bill uses future knowledge of that bill; the interface applies a merge only once the bill's first reading is ≤ `as_of`. Bill links live only in `idea_bill_links` (not also on `kings_speech_items`) so there is one source of truth.

## 2026-10-24 21:29 — feat(takeup): add Tier B fetcher with local cache

- **What:** new package `westminster_weather.takeup` with `tier_b.py` (written ministerial statements with full text, Hansard spoken / written trigger-phrase searches, Members API government posts) and `cli.py` (`fetch-tier-b`). Ported `triggers.py` from the prototype unchanged except a lint fix.
- **Why:** task 3 of the take-up plan: start the Tier B fetch immediately. It only fills `backend/data/cache/takeup/`; no database writes.
- **Checks:** `ruff check`: clean; `pytest`: 42 passed; smoke run on Sept 2024 cached full statement texts and Hansard results.
- **Notes:** the written statements *list* endpoint truncates text to ~250 characters (the prototype used it), so full text is fetched per statement; its `searchTerm` is a word search, not a phrase search, so all statements are fetched. Hansard answers in ~20 s per request, so searches run 4 at a time while request starts stay ≥ 0.35 s apart. Cache files are keyed by exact date range, so a partial run can't masquerade as a full year.

## 2026-10-24 21:28 — chore(gitignore): ignore backend/data and backend/models

- **What:** added `backend/data/` and `backend/models/` to `.gitignore`.
- **Why:** the Tier B fetch caches thousands of API responses under `backend/data/cache/`; `git check-ignore` showed they were not ignored. BUILD_PLAN already asks for both paths to be ignored.
- **Checks:** `git check-ignore -v backend/data/cache/takeup/x.json` now matches. No code changed.

## 2026-10-24 21:17 — docs(model): add Phase 1 → 2 take-up model plan

- **What:** added `docs/model/TAKEUP_PLAN.md`: design, the four changes from the prototype, Neon safety rules, features, evaluation, tasks with checkpoints, and decisions needed.
- **Why:** task 1 of Roman's take-up plan (checkpoint 1).
- **Checks:** prototype on a scratch copy: `takeup_model.py fit` / `evaluate` and `nlp_signals.py test` all ran (results in the plan, section 1). First-record keys checked against the written statements, Hansard, Members and GOV.UK APIs. No repo code changed, so `pytest` / `ruff` not rerun.
- **Notes:** written statements API starts Nov 2014; King's Speech briefing notes on GOV.UK start 2012; `db._seed_reference_tables` runs `create_all` on app start, so take-up tables need their own MetaData.

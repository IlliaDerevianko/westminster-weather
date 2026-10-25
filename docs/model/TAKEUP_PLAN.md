# Westminster Weather: Phase 1 → 2 Take-up Model Plan

> **Status (2026-10-25):** the take-up feature (promises pages, `/api/v1/takeup/*`, CLI, ingestion/NLP pipeline, evaluation and replays) has been removed. Only `takeup/model.py` (mixture cure model), `conformal.py`, `signals.py`, `tables.py` and `db.py` remain, because the stage 3 commencement model (`journey/commencement.py`) reuses them. The rest of this plan is kept as a record.

**Owner:** Roman · **Branch:** `feat/takeup-model` · **Deadline:** Sunday 25 October 2026, 16:30 (code freeze 14:30) · **Status:** checkpoint 1 approved; checkpoint 2 (contract) awaiting approval

The model answers: *the government has committed to legislate on a topic; will a bill on it get its first reading (either House), and when?*
Output per idea and `as_of`: π (chance it is ever taken up) with a 95% CI, a verdict (likely / unlikely / uncertain), π_now after the weeks already passed in silence, the chance of a bill within the next 26 / 52 weeks, and the median + 80% timing window (conditional on the weeks already passed).

This is the first model in the pipeline. Once a real bill exists, the passage model (`MODEL_SPEC.md`, BUILD_PLAN tasks 3–12, Aleks) takes over. The take-up model does not modify the passage model or its tables; the only hand-off is that ideas link to real `bill_id`s in `bills`.

**Structure** (names as in Roman's task spec):
- **Tier A: the model.** A mixture cure model that turns ideas into forecasts (section 2.2).
- **Tier B: the data pipeline.** Collects sources, finds commitments to legislate with NLP, groups them into ideas, links ideas to bills, and records dated events (section 2.3). Tier B produces everything Tier A trains and predicts on.

Both tiers are required. **King's Speech data is not used** (Roman's decision, 2026-10-24, overriding the spec's "never cut"): ideas come only from ministers' commitments found by NLP, which include both promises that became bills (teaching *when*) and promises that never did (teaching *whether*). Timing estimates rest on fewer clean, firm promises as a result.

---

## 1. Starting point (checked 2026-10-24)

**Prototype** (`~/Downloads/law-in-time`, run on a scratch copy; identical code to `law-in-time 2`):

| Check | Result |
|---|---|
| `takeup_model.py fit` on 900 synthetic signals | Parameters recovered; intercept −1.57 vs true −2.0 (2.2 SE off); the week-105+ baseline has SE 0 (no synthetic data there, so it is set by the ridge penalty alone) |
| `takeup_model.py evaluate` (cut 2021-01-01) | AUC 0.768, Brier 0.197 for "bill within 52 weeks"; 80% window coverage 87%; "likely" verdicts 82% correct, "unlikely" 73% |
| `nlp_signals.py test` | Commitment detection 100% precision and recall; firmness 100%; grouping agreement 0.90 after merging; bill linking 14 of 15 |

Known prototype bugs: `predict()` computes timing quantiles from week 0 instead of conditioning on the weeks already passed (fixed in 2.2); the written-statements list endpoint it used truncates text to ~250 characters (fixed in the fetcher, which takes the full text per statement).

**Live APIs** (first record keys confirmed; the prototype's field names all match):

| API | Keys used | Coverage / size |
|---|---|---|
| Written ministerial statements `questions-statements-api.parliament.uk/api/writtenstatements/statements` | `value.{id, dateMade, memberId, title, text, house, answeringBodyName}` | **Starts Nov 2014.** 17,844 statements since 2015 (fetched) |
| Hansard search `hansard-api.parliament.uk/search/contributions/{Spoken,Written}.json` | `{MemberId, SittingDate, DebateSection, ContributionExtId, ContributionTextFull, House}` | 2015+: 3,895 unique spoken contributions with a trigger phrase (fetched); 2006–2014 spoken + written fetching |
| Members biography `members-api.parliament.uk/api/Members/{id}/Biography` | `value.governmentPosts[].{name, startDate, endDate}` | 983 speakers checked (fetched) |

**Repo:** `bills` (588 Government Bills), `stage_sittings`, `sessions`, `sitting_days`, `parliaments` are in Neon; `data/calendar.py` has `week_sitting_probability` and `future_sitting_days`. `bills` has **no long title**, which linking needs. There is **no `political_snapshots` table**, so `working_majority` is skipped.

---

## 2. Design

### 2.1 Code layout

```
backend/src/westminster_weather/takeup/
  tables.py        # take-up tables on their OWN MetaData (see 2.4)
  db.py            # engines for DATABASE_URL_TAKEUP (Neon branch), upserts, loaders
  model.py         # Tier A: TakeUpModel, cure model + required changes (2.2)
  triggers.py      # Tier B2: phrase lexicon (ported, tuned at checkpoint 4)
  tier_b.py        # Tier B2: fetch written statements, Hansard, ministers' posts (cache)
  bill_text.py     # Tier B4: bill long titles (Bills API, cached), first readings
  govuk_events.py  # Tier B1: consultation / response / green or white paper / draft bill events
  nlp.py           # Tier B2-B4: label, group, link, merge; review + score
  signals.py       # Tier B5: get_takeup_signals(as_of), the only way the model sees data
  evaluate.py      # time-split backtest, baselines, metrics, 4 charts
  cli.py           # uv run python -m westminster_weather.takeup.cli <command>
backend/tests/takeup/        # synthetic tests + leakage, non-sitting-week, change-of-government, conditional-quantile tests
backend/data/cache/takeup/   # download cache only (gitignored); the database is the source of truth
backend/reports/takeup/      # charts + metrics.json (committed)
docs/model/TAKEUP_EVALUATION.md
```

HTTP goes through `httpx` instead of `requests`. Embeddings use the prototype's TF-IDF backend (words + character n-grams); sentence-transformers is not installed.

### 2.2 Tier A: the model

Notation and maths exactly as in the task spec: η_i = γ₀ + γ·z_i, π_i = σ(η_i); λ_i(t) = σ(α_bin(t) + β·x_i(t)) with bins 1–4, 5–12, 13–26, 27–52, 53–104, 105+; S_u(t) = ∏(1 − λ_i(s)); f_u(t) = λ_i(t)·S_u(t−1); ℓ = Σ_seen [log π + log f_u(t_i)] + Σ_not seen log(1 − π + π·S_u(c_i)); L-BFGS with a small ridge on γ, β. Outputs: π with delta-method 95% CI, verdict, S(t), π_now(s), p26, p52, bootstrap (B ≥ 200) intervals.

Required changes from the prototype:

1. **Sitting calendar.** In weeks with zero sitting days (from `get_calendar` / `sitting_days`), λ = 0: those weeks add nothing to ℓ and do not lower π_now. Implemented as a mask M_i(t) ∈ {0, 1} per idea; for future weeks M is the historical probability that the week is a sitting week (`calendar.week_sitting_probability`). Timing is reported in calendar weeks and sitting weeks.
2. **Change of government.** If the governing party changes at an election (2010, dissolution 2010-04-12; 2024, dissolution 2024-05-30) before a bill appears, follow-up ends at the dissolution and the idea is **censored**, not failed. Same-party elections (2015, 2017, 2019) switch on `x_new_parliament` at the next State Opening.
3. **Train and predict only through `get_takeup_signals(as_of)`.**
4. **First reading from the repo:** earliest sitting of Bills API stage 6 (Commons) or 1 (Lords) in `stage_sittings`, via `stage_map`; Government Bills only.
5. **Conditional timing quantiles** (prototype bug): t_q(s) = min{t > s : 1 − S_u(t)/S_u(s) ≥ q}; report t_0.5(s) and the 80% window [t_0.1(s), t_0.9(s)]. Always > s.

### 2.3 Tier B: the data pipeline

- **B1 structured sources (no NLP).** GOV.UK search API events (consultation, consultation response, green / white paper, draft bill) into `idea_events`, reusing the Phase 1 tracker's event types. (King's Speech items were removed.)
- **B2 trigger finding.** Written ministerial statements (always ministers) and Hansard spoken contributions matching `triggers.py` phrases, 2015 onwards first, then 2006–2014; Hansard speakers checked against their government posts on the date. Firmness 2 / 1 / 0; questions, negations and demands excluded; timing words upgrade medium to firm. Gemini (if `GEMINI_API_KEY` is set) confirms commitments and cleans topics; rules-only path always works.
- **B3 ideas.** Group mentions by meaning (TF-IDF), split groups with gaps over 2 years; trigger = earliest mention; mention-level rows kept so ideas can be rebuilt for any `as_of`.
- **B4 links.** Match each idea to a Government Bill first read between trigger − 7 days and trigger + 5 years, using title + long title; auto-link strong matches, Gemini or hand-check for borderline ones. Merge ideas linked to the same bill, including NLP ideas with King's Speech ideas (earliest trigger kept). Unlinked ideas = not (yet) taken up.
- **B5 `get_takeup_signals(as_of)`**: ideas from mentions dated ≤ `as_of`, links only to bills first read by `as_of`, events only ≤ `as_of`, features only from information available on `as_of`. Merges through a shared bill become visible only once that bill's first reading is ≤ `as_of`.
- **B6 quality check.** Export 100 mentions and 50 idea → bill links for hand-labelling (`cli review`), report precision (`cli score`); below 85% → fix lexicon / thresholds and rerun.

### 2.4 Neon storage and safety

- **Separate MetaData.** `db._seed_reference_tables` runs `Base.metadata.create_all` whenever the app starts, so take-up tables live on their own `MetaData` in `takeup/tables.py` and are created only by `takeup.cli migrate`.
- **Separate connection.** All take-up reads and writes use `DATABASE_URL_TAKEUP` (and optionally `DATABASE_URL_TAKEUP_UNPOOLED` for DDL) in `.env.local`, pointing at the Neon branch `takeup-dev`. Without it the take-up code refuses to run; it never falls back to `DATABASE_URL`. Applying to production is a separate, approved step at the end.
- **Database is the source of truth.** Raw sources (`source_documents`, `member_government_posts`, ~40–45 MB of the 1 GB) are stored as well as the derived tables; the local cache is only a download speed-up.
- Connection strings are never printed or committed.

Tables (full definitions in `DATA_CONTRACT.md` section 10): `source_documents`, `member_government_posts`, `signal_mentions`, `ideas`, `idea_mentions`, `idea_events`, `idea_bill_links`, `takeup_runs`, `takeup_forecasts`.

### 2.5 Features

Registry with definitions and sources: `DATA_CONTRACT.md` section 10.11. z: `first_source`, `initial_firmness`, `department` (small ones pooled), `years_since_election`, `months_into_session`. x: `firm_commitment`, `repeat_mention`, `consultation`, `consultation_response`, `white_paper`, `draft_bill`, `new_parliament`. `working_majority` skipped (no `political_snapshots` table). Any feature with fewer than ~15 positive cases is dropped and logged.

### 2.6 Evaluation

Train on triggers before 2019-01-01, test on 2019 onwards; second split at 2022-01-01. Baselines B0 (overall take-up rate within 52 weeks) and B1 (rate by `first_source`). Tier A metrics: AUC and Brier for a bill within 26 / 52 weeks, 5-bin calibration table and plot, 80% window coverage, concordance index for timing, accuracy of likely and unlikely verdicts separately, share uncertain. Tier B metrics: precision of commitments and links from the hand labels; counts of mentions, ideas, linked and unlinked ideas. Charts: calibration, π_now curves (fast, slow, never), time-to-bill distribution, feature effects with 95% CIs. Sample sizes, failures and weak spots reported honestly.

---

## 3. Tasks and checkpoints

| # | Task | Status / done when |
|---|---|---|
| 1 | Read docs + prototype; prototype tests pass; this plan | **Checkpoint 1**: approved |
| 2 | Neon branch `takeup-dev`; `DATA_CONTRACT.md` additions | **Checkpoint 2**: awaiting approval; then migrate the branch, tests pass |
| 3 | Tier B fetch in the background, first-record keys shown | 2015+ done (17,844 statements, 3,895 Hansard contributions, 983 speakers); 2006–2014 running |
| 4 | ~~King's Speech items + links; checkpoint 3~~ | **Removed** (King's Speech not used) |
| 5 | Tier B: GOV.UK events into `idea_events` | events stored with dates and URLs |
| 6 | Tier A: port the model with the required changes + conditional quantiles; synthetic tests; tests for nothing after `as_of`, zero hazard in non-sitting weeks, censoring at change of government, quantiles > s | tests pass |
| 7 | Tier B: label, group, link, merge; `get_takeup_signals(as_of)`; review CSVs | **Checkpoint 4**: mentions, ideas, % linked, never-taken-up ideas |
| 8 | (You) hand-label; `score`; if precision < 85%, fix and rerun | precision ≥ 85% |
| 9 | Tier A: fit on all ideas; baselines; `TAKEUP_EVALUATION.md`; 4 charts; save `takeup_runs` / `takeup_forecasts` | **Checkpoint 5** |
| 10 | Endpoints `GET /api/v1/takeup/ideas?as_of=`, `/api/v1/takeup/forecast?idea_id=&as_of=`, `/api/v1/takeup/history?idea_id=` | valid responses, tests |
| 11 | Frontend: π_now chart on the Phase 1 timeline with events, verdict badge and 80% window; Online Safety Bill + one dropped idea | demo works |
| 12 | Lint, full test suite, README section, PR | **Checkpoint 6** |

**Cut order if time runs out:** `/takeup/history` (static chart instead) → Tier B years before 2015 → Hansard (keep written statements) → Gemini (rules only).
**Never cut:** written statements 2015+, the `as_of` rule, the time-split evaluation, the baselines.

---

## 4. Spec reconciliation (for checkpoint 2)

Where the task spec and the repo differ, the repo docs win; these are flagged for your decision:

1. **King's Speech removed.** The spec lists it as never cut; Roman decided to drop it (2026-10-24).
2. **Date column names.** The contract's point-in-time rule names event dates `event_date`; the spec says `date`. The contract name is kept.
3. **Endpoint paths.** All existing endpoints are under `/api/v1/`, so the take-up endpoints are `/api/v1/takeup/...`.
4. **Feature registry.** The take-up features get their own registry (contract 10.11) instead of being added to the passage model's 5.5, so the passage model's documents are not changed.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Too few "never taken up" ideas | NLP commitments are required; report counts at checkpoint 4 |
| Wrong idea → bill links | long titles; hand-check of a sample of NLP links (checkpoint 4) |
| Hansard rate limit (~1,000 requests / 5 min) and HTTP 500 under parallel load | 2 searches at a time, request starts ≥ 0.35 s apart, failed searches logged and retried on the next run |
| Take-up tables leaking into production | own MetaData + separate `DATABASE_URL_TAKEUP` (2.4) |
| Clash with passage-model work | separate package, branch and tables |
| Exposed database password | ask Illia to reset it before anything is public |

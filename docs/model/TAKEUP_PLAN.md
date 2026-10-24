# Westminster Weather: Phase 1 → 2 Take-up Model Plan

**Owner:** Roman · **Branch:** `feat/takeup-model` · **Deadline:** Sunday 25 October 2026, 16:30 · **Status:** checkpoint 1 (awaiting approval)

The model answers: *the government has signalled it will legislate on a topic; will a bill on it get its first reading, and when?*
Output per idea: π (chance it is ever taken up) with a 95% CI, a verdict (likely / unlikely / uncertain), π_now after weeks of silence, P(bill within 26 / 52 weeks), and the median + 80% timing window.

Both data tiers are required. **Tier A** (King's Speech bills) gives clean, dated ideas that almost always became bills, so it teaches *when*. **Tier B** (ministers' commitments found by NLP) gives the commitments that were never taken up, so it teaches *whether*. The model is not valid without both.

This model runs in parallel with the passage model (`MODEL_SPEC.md`, Aleks). It uses separate tables and a separate package, and must not block or change that work.

---

## 1. Starting point (checked 2026-10-24)

**Prototype** (`~/Downloads/law-in-time 2`, run on a scratch copy):

| Check | Result |
|---|---|
| `takeup_model.py fit` on 900 synthetic signals | Parameters recovered; intercept −1.57 vs true −2.0 (2.2 SE off); the week-105+ baseline has SE 0 (no synthetic data there, so it is set by the ridge penalty alone) |
| `takeup_model.py evaluate` (cut 2021-01-01) | AUC 0.768, Brier 0.197 for "bill within 52 weeks"; 80% window coverage 87%; "likely" verdicts 82% correct, "unlikely" 73% |
| `nlp_signals.py test` | Commitment detection 100% precision and recall; firmness 100%; grouping agreement 0.90 after merging; bill linking 14 of 15 |

**Live APIs** (first record keys confirmed; the prototype's field names all match):

| API | Keys used | Coverage / size |
|---|---|---|
| Written ministerial statements `questions-statements-api.parliament.uk/api/writtenstatements/statements` | `value.{id, dateMade, memberId, title, text, house, answeringBodyName}` | **Starts Nov 2014.** About 1,300–1,700 per year; ~17,500 since 2015 |
| Hansard search `hansard-api.parliament.uk/search/contributions/Spoken.json` | `{MemberId, SittingDate, DebateSection, ContributionExtId, ContributionTextFull, House}` | 2015+: "we will legislate" 318, "we will bring forward legislation" 105, "when parliamentary time allows" 395, "at the earliest opportunity" 1,481, "draft bill" 1,521 |
| Members biography `members-api.parliament.uk/api/Members/{id}/Biography` | `value.governmentPosts[].{name, startDate, endDate}` | One call per distinct speaker, cached |
| GOV.UK King's/Queen's Speech briefing notes (search + content API) | PDF attachment per speech | **2012 to 2026 only** (15 speeches). 2006–2010 are not on GOV.UK. There was no speech in 2011 |

Briefing-note PDFs parse: each has a contents page listing every bill with its page number ("Draft …" marks draft bills). Tested on 2012 (19 bills found) and 2024 (contents page lists all ~40).

**Repo:** `bills` (588 Government Bills), `stage_sittings`, `sessions`, `sitting_days`, `parliaments` are in Neon; `data/calendar.py` has `week_sitting_probability` and `future_sitting_days`. `bills` has **no long title**, which linking needs.

---

## 2. Design

### 2.1 Code layout

```
backend/src/westminster_weather/takeup/
  tables.py        # takeup tables on their OWN MetaData (see 2.3)
  db.py            # engine for TAKEUP_DATABASE_URL (Neon branch), upserts, loaders
  model.py         # TakeUpModel: cure model + the four changes in 2.2
  triggers.py      # phrase lexicon (ported unchanged, then tuned at checkpoint 4)
  tier_b.py        # fetch (WMS, Hansard, minister check), label, group, link
  kings_speech.py  # Tier A: briefing-note PDFs -> kings_speech_items -> links to bills
  govuk_events.py  # Tier A: consultation / response / white paper / draft bill events
  signals.py       # get_takeup_signals(as_of): the only way the model sees data
  evaluate.py      # time-split backtest, baselines, metrics, 4 charts
  cli.py           # uv run python -m westminster_weather.takeup.cli <command>
backend/tests/takeup/   # ported synthetic tests + leakage, non-sitting-week, change-of-government tests
backend/data/cache/takeup/   # raw API responses, PDFs, Gemini answers (gitignored)
backend/reports/takeup/      # charts + metrics.json (committed)
docs/model/TAKEUP_EVALUATION.md
```

HTTP goes through `httpx` (already a dependency) instead of `requests`. Embeddings use the prototype's TF-IDF backend (no new dependency, works offline).

### 2.2 The four changes from the prototype

1. **Sitting calendar.** Bills can only be introduced on sitting days. For each idea, mask M_i(t) = 1 if either House sat in calendar week t (from `sitting_days`), else 0. The hazard becomes λ_i(t) = M_i(t)·σ(α_t + βᵀx_i(t)), so non-sitting weeks add nothing to ℓ and do not lower π_now. For weeks after `as_of` (forecasting), M is replaced by the historical probability that the week is a sitting week (`calendar.week_sitting_probability`). Timing is reported in calendar weeks and sitting weeks.
2. **Change of government.** Elections where the governing party changed: 2010 (dissolution 2010-04-12) and 2024 (dissolution 2024-05-30). An idea with no bill by that dissolution is **censored** there (δ = 0, c = weeks to dissolution), not counted as dropped. Same-party elections (2015, 2017, 2019) do not end follow-up; they switch on `x_new_parliament` at the next State Opening (first sitting of the next session, from `sessions`).
3. **No leakage.** `get_takeup_signals(as_of)` builds ideas only from mentions dated ≤ `as_of` (via `idea_mentions`), links only to bills first read ≤ `as_of`, events only ≤ `as_of`, and sets `data_end = as_of`. The backtest calls it with the cut date. A test fails if any returned date is after `as_of`.
4. **Repo bills, not CSV.** First reading = earliest sitting of Bills API stage 6 (Commons 1st reading) or 1 (Lords 1st reading) in `stage_sittings`. Government bills only (`bills`). Long titles come from `GET /Bills/{id}` (`longTitle`), cached under `backend/data/cache/takeup/`.

### 2.3 Neon storage and safety

- **Separate MetaData.** `db._seed_reference_tables` runs `Base.metadata.create_all` whenever the app starts. If the takeup tables were added to `models.py`, they would be created in the **production** database before approval. So they live on their own `MetaData` in `takeup/tables.py` and are created only by `takeup.cli migrate`.
- **Separate connection.** All takeup reads and writes use `TAKEUP_DATABASE_URL` / `TAKEUP_DATABASE_URL_UNPOOLED` in `.env.local`, pointing at a Neon branch `takeup-dev`. The existing `DATABASE_URL` is untouched. Applying to production is a separate, approved step at the end.
- Connection strings are never printed or committed.

New tables (full column definitions go into `DATA_CONTRACT.md` at checkpoint 2): `signal_mentions`, `ideas`, `idea_mentions`, `idea_events`, `idea_bill_links`, `kings_speech_items`, `takeup_runs`, `takeup_forecasts`.

### 2.4 Features

| Feature | Type | Source | Note |
|---|---|---|---|
| `first_source` | z | A, B | one-hot: King's Speech / written statement / Hansard |
| `initial_firmness` | z | B (A = 2) | 0 vague, 1 medium, 2 firm |
| `department` | z | A, B | top departments, small ones pooled |
| `years_since_election` | z | `parliaments` | |
| `months_into_session` | z | `sessions` | |
| `firm_commitment`, `repeat_mention` | x | B | dates of first firm / second mention |
| `consultation`, `consultation_response`, `white_paper`, `draft_bill` | x | GOV.UK | |
| `kings_speech` | x | A | date of the speech that announced it |
| `new_parliament` | x | `sessions` | State Opening after a same-party election |

`working_majority` and `manifesto` only if time allows. Any feature with fewer than ~15 positive cases is dropped and logged.

### 2.5 Evaluation

Train on triggers before 2019-01-01, test on 2019 onwards; second split at 2022-01-01. Baselines B0 (overall rate of a bill within 52 weeks) and B1 (rate by source type). Metrics: AUC and Brier for a bill within 26 / 52 weeks, 5-bin calibration, 80% window coverage, concordance index for timing, accuracy of likely / unlikely verdicts and share uncertain, plus label precision from the hand-check. Charts: calibration, π_now curves (fast, slow, never), time-to-bill distribution, feature effects with 95% CIs.

---

## 3. Tasks and checkpoints

| # | Task | Done when |
|---|---|---|
| 1 | Read docs + prototype; prototype tests pass; this plan | **Checkpoint 1**: plan approved |
| 2 | Neon branch `takeup-dev`; `DATA_CONTRACT.md` additions | **Checkpoint 2**: contract approved; then migrate the branch, tests pass |
| 3 | **Tier B fetch in the background** (cache only, no database needed): WMS 2015+, then Hansard trigger phrases 2015+, minister checks; then 2006–2014 | first-record keys confirmed with you (already in section 1) |
| 4 | Tier A: King's Speech items 2012–2026 + links to bills; export links CSV | **Checkpoint 3**: you hand-check the links |
| 5 | Tier A: GOV.UK events into `idea_events` | events stored with dates and URLs |
| 6 | Port the model with the four changes + `get_takeup_signals`; port synthetic tests; add leakage, non-sitting-week and change-of-government tests | tests pass |
| 7 | Tier B: label, group, link; merge with Tier A ideas linked to the same bill; export 100 mentions + 50 links for review | **Checkpoint 4**: counts of mentions, ideas, % linked, never-taken-up ideas |
| 8 | (You) hand-label; `score`; if precision < 85%, fix lexicon / thresholds and rerun 7–8 | precision ≥ 85% |
| 9 | Fit on A + B; baselines; `TAKEUP_EVALUATION.md`; 4 charts; save `takeup_runs` / `takeup_forecasts` | **Checkpoint 5** |
| 10 | Endpoints `/api/v1/takeup/ideas`, `/takeup/forecast`, `/takeup/history` | valid responses, tests |
| 11 | Frontend: π_now chart on the Phase 1 timeline; Online Safety Bill + one dropped idea | demo works |
| 12 | Lint, tests, README section, PR | **Checkpoint 6** |

**Cut order if time runs out:** `/takeup/history` (static chart instead) → Tier B before 2015 → Hansard (keep written statements) → Gemini (rules only).
**Never cut:** Tier A, Tier B written statements 2015+, the `as_of` rule, the time-split evaluation, the baselines.

---

## 4. Decisions needed at checkpoint 1

1. **King's Speech 2006–2010** are not on GOV.UK. Proposal: Tier A = the 15 speeches from 2012 to 2026 (6 in the training period, 9 in the test period); add 2006–2010 from the National Archives only if time allows.
2. **Written statements 2006–2014**: the API starts in Nov 2014. Proposal: for 2006–2014 use Hansard's written-statement search, and treat it as the first thing to cut.
3. **New dependency `pypdf`** to read the briefing-note PDFs. `matplotlib` for charts (already allowed by the build plan).
4. **Neon branch**: creating `takeup-dev` needs Neon console access, which this machine does not have. You create it (Neon console → Branches → New branch from `production`) and add its two connection strings to `.env.local` as `TAKEUP_DATABASE_URL` and `TAKEUP_DATABASE_URL_UNPOOLED`.
5. **Gemini**: no `GEMINI_API_KEY` is set, so Tier B labelling runs rules-only unless one is added to `.env.local`.
6. **Start the Tier B fetch now**, right after this checkpoint: it only writes to the local cache, so it does not need the Neon branch or the contract.

## 5. Risks

| Risk | Mitigation |
|---|---|
| Too few "never taken up" ideas | Tier B is required; report counts at checkpoint 4 |
| Wrong idea → bill links | long titles; full hand-check of Tier A, sample of Tier B |
| Hansard rate limit (~1,000 requests / 5 min) | throttle to ≤ 3 requests/s, cache every page, resumable |
| Takeup tables leaking into production | own MetaData + separate `TAKEUP_DATABASE_URL` (2.3) |
| Clash with passage-model work | separate package, branch and tables |
| Exposed database password | ask Illia to reset it before anything is public |

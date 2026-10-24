# Agent log: `feat/takeup-model`

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

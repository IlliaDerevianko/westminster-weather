# Westminster Weather — Data Contract

Defines exactly what the database stores, what the model reads, and what the model writes back.
The model talks to the database **only** through the interface in section 5. Nothing else.

Version: 0.1 · Scope: UK Government Bills (Bills API `BillType = 1`), sessions from 2006–07 onwards.

---

## 1. Conventions

- **Dates:** ISO 8601, `YYYY-MM-DD`, UK local date as published by Parliament. Timestamps: ISO 8601 UTC.
- **IDs:** use the Parliament API IDs wherever they exist (`bill_id`, `session_id`, `stage_id`, `division_id`). Never invent new IDs for API objects.
- **Houses:** exactly `'Commons'` or `'Lords'`.
- **Nulls:** `NULL` means "not happened yet / unknown". Never use `0`, `''` or `'N/A'` as a placeholder.
- **Point-in-time rule (critical):** every row that describes an event carries the date it happened (`event_date`) and the time we ingested it (`ingested_at`). The model always queries with an `as_of` date and must never see anything dated after it. This prevents data leakage in training and backtests.
- **Upserts:** ingestion is idempotent — re-running a pull updates rows by primary key, never duplicates them.
- **SQL dialect:** standard SQL, compatible with SQLite and PostgreSQL.

---

## 2. Reference tables

### 2.1 `sessions`
| Column | Type | Notes |
|---|---|---|
| `session_id` | INTEGER PK | Bills API session ID |
| `name` | TEXT | e.g. `2024-26` |
| `parliament_id` | INTEGER FK → `parliaments` | |
| `start_date` | DATE | State Opening |
| `end_date` | DATE NULL | Prorogation or dissolution; `NULL` if current |
| `end_type` | TEXT NULL | `'prorogation'` or `'dissolution'` |

### 2.2 `parliaments`
| Column | Type | Notes |
|---|---|---|
| `parliament_id` | INTEGER PK | Sequential |
| `first_meeting_date` | DATE | Used for the 5-year maximum |
| `dissolution_date` | DATE NULL | `NULL` if current |
| `election_date` | DATE NULL | Election that ended this Parliament |

### 2.3 `announced_dates`
Known future dates, recorded when announced (so the model only uses them after `announced_on`).
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `date_type` | TEXT | `'prorogation'`, `'dissolution'`, `'recess_start'`, `'recess_end'` |
| `house` | TEXT NULL | `NULL` = both Houses |
| `target_date` | DATE | The date announced |
| `announced_on` | DATE | When it became public |

### 2.4 `sitting_days`
| Column | Type | Notes |
|---|---|---|
| `date` | DATE | |
| `house` | TEXT | |
| PK | (`date`, `house`) | One row per day each House actually sat |

### 2.5 `stage_map`
Maps Bills API stages to model states. Static; filled once by hand.
| Column | Type | Notes |
|---|---|---|
| `api_stage_id` | INTEGER PK | Bills API stage ID |
| `api_stage_name` | TEXT | |
| `house` | TEXT NULL | |
| `model_state` | TEXT NULL | See table below; `NULL` = ignored by the model |

`model_state` values are the `BillStage` enum values in `backend/src/westminster_weather/domain/stages.py` (MODEL_SPEC §2.1). The Python copy of this table is `backend/src/westminster_weather/data/stage_map.py`.

| `model_state` | Commons-first order | Lords-first order | API stage IDs |
|---|---|---|---|
| `commons_1st_reading` | 1 | 4 | 6 |
| `commons_2nd_reading` | 2 | 5 | 7, 14 (programme motion), 15 (money resolution), 25 (second reading committee), 36 (ways and means) |
| `commons_committee` | 3 | 6 | 8 (Public Bill Committee), 43, 48, 49 (Committee of the whole House) |
| `commons_report_stage` | 4 | 7 | 9 |
| `commons_3rd_reading` | 5 | 8 | 10 |
| `lords_1st_and_2nd_reading` | 6 | 1 | 1, 2, 30 |
| `lords_committee` | 7 | 2 | 3, 46 |
| `lords_report_and_3rd_reading` | 8 | 3 | 4, 5 |
| `ping_pong` | 9 | 9 | 12, 13, 42 (optional: only if the second House amended the bill) |
| `royal_assent` | 10 | 10 | 11 |

For Lords-first bills the Commons states come after the Lords states. The order is derived from `bills.originating_house`. Programme and money motions are merged into `commons_2nd_reading`. All other stage IDs (e.g. carry-over motions 18, 19) map to `NULL`; carry-over motions feed `bills.carry_over_date`.

### 2.6 `political_snapshots`
Stored as validity intervals; a new row only when something changes (by-election, defection, new Parliament).
| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER PK | |
| `house` | TEXT | |
| `valid_from` | DATE | |
| `valid_to` | DATE NULL | `NULL` = still valid |
| `government_parties` | TEXT | Comma-separated party IDs (Members API) |
| `government_seats` | INTEGER | |
| `total_seats` | INTEGER | Voting members only |
| `working_majority` | INTEGER NULL | Commons only |
| `government_share` | REAL | `government_seats / total_seats` |

---

## 3. Bill tables

### 3.1 `bills`
| Column | Type | Known at | Notes |
|---|---|---|---|
| `bill_id` | INTEGER PK | intro | Bills API ID |
| `short_title` | TEXT | intro | |
| `originating_house` | TEXT | intro | |
| `introduced_session_id` | INTEGER FK | intro | |
| `introduced_date` | DATE | intro | Date of first reading |
| `department` | TEXT NULL | intro | Sponsoring department (Tier 2) |
| `kings_speech_flag` | BOOLEAN NULL | intro | Announced in King's/Queen's Speech (Tier 2) |
| `size_clauses` | INTEGER NULL | intro | From first published version |
| `size_pages` | INTEGER NULL | intro | From first published version |
| `commons_committee_type` | TEXT NULL | event | `'public_bill_committee'` or `'whole_house'` |
| `carry_over_date` | DATE NULL | event | Date carry-over motion agreed |
| `outcome` | TEXT | event | See below |
| `outcome_date` | DATE NULL | event | |
| `last_api_update` | TIMESTAMP | — | From Bills API `lastUpdate` |
| `ingested_at` | TIMESTAMP | — | |

`outcome` values: `'royal_assent'`, `'fell_session_end'`, `'fell_dissolution'`, `'withdrawn'`, `'defeated'`, `'in_progress'`.

- A session that ended within 14 days before a dissolution (the pre-election wash-up prorogations) has `sessions.end_type = 'dissolution'`, so its failed bills are `'fell_dissolution'` (`StopReason.ELECTION_CALLED`), not `'fell_session_end'` (`StopReason.SESSION_RUNS_OUT`).
- `'in_progress'` only if the bill is in the current session, or had a carry-over motion in the previous session and that session was prorogued (§8). A dissolution ends carried-over bills too.

### 3.2 `stage_sittings`
One row per sitting of a stage. The core table for durations.
| Column | Type | Notes |
|---|---|---|
| `sitting_id` | INTEGER PK | Bills API stage sitting ID |
| `bill_id` | INTEGER FK | |
| `api_stage_id` | INTEGER FK → `stage_map` | |
| `house` | TEXT | |
| `session_id` | INTEGER FK | |
| `event_date` | DATE | Date of the sitting |
| `ingested_at` | TIMESTAMP | |

### 3.2a `scheduled_sittings`
Same columns as `stage_sittings`, for sittings dated **after** the day they were ingested (future business already scheduled). We do not know when each was announced, so this table is used **only for live forecasts** (`as_of` on or after ingestion) and **never** for training or backtests. `stage_sittings` holds only sittings that have happened.

### 3.3 `divisions`
Recorded votes linked to a bill. Lords defeats are derived from here (`house = 'Lords' AND government_won = FALSE`).
| Column | Type | Notes |
|---|---|---|
| `division_id` | INTEGER | Commons/Lords Votes API ID |
| `house` | TEXT | |
| PK | (`division_id`, `house`) | IDs are per House |
| `bill_id` | INTEGER FK NULL | |
| `api_stage_id` | INTEGER NULL | |
| `event_date` | DATE | |
| `ayes` | INTEGER | |
| `noes` | INTEGER | |
| `government_position` | TEXT NULL | `'aye'`, `'no'`, or `NULL` if free vote |
| `government_won` | BOOLEAN NULL | |
| `government_rebels` | INTEGER NULL | Government-party members voting against the government |
| `ingested_at` | TIMESTAMP | |

### 3.4 `amendments`
| Column | Type | Notes |
|---|---|---|
| `amendment_id` | INTEGER PK | Bills API ID |
| `bill_id` | INTEGER FK | |
| `api_stage_id` | INTEGER | |
| `house` | TEXT | |
| `tabled_date` | DATE | |
| `is_government` | BOOLEAN | Tabled by a minister |
| `decision` | TEXT | `'agreed'`, `'negatived'`, `'withdrawn'`, `'not_moved'`, `'pending'` |
| `decision_date` | DATE NULL | |
| `ingested_at` | TIMESTAMP | |

---

## 4. Tier 2 and alert-only tables

Build only if time allows. The model must still run if these are empty.

### 4.1 `debates` (Hansard)
| Column | Type | Notes |
|---|---|---|
| `debate_id` | TEXT PK | Hansard section ID |
| `bill_id` | INTEGER FK | |
| `api_stage_id` | INTEGER NULL | |
| `house` | TEXT | |
| `event_date` | DATE | |
| `n_speakers` | INTEGER | |
| `n_contributions` | INTEGER | |
| `word_count` | INTEGER | |

### 4.2 `scrutiny_reports`
| Column | Type | Notes |
|---|---|---|
| `report_id` | INTEGER PK | |
| `bill_id` | INTEGER FK | |
| `committee` | TEXT | e.g. `'Delegated Powers'`, `'Constitution'` |
| `event_date` | DATE | |
| `critical` | BOOLEAN | Report raises significant concerns |

### 4.3 `context_flags`
| Column | Type | Notes |
|---|---|---|
| `flag` | TEXT | e.g. `'crisis'` |
| `valid_from` | DATE | |
| `valid_to` | DATE NULL | |
| `value` | REAL | Usually 1.0 |

### 4.4 `polls`
| Column | Type | Notes |
|---|---|---|
| `event_date` | DATE PK | |
| `government_lead` | REAL | Government party minus main opposition, percentage points |

### 4.5 `news_signals` (alerts only, never a model input)
| Column | Type | Notes |
|---|---|---|
| `bill_id` | INTEGER FK | |
| `event_date` | DATE | |
| `article_count` | INTEGER | GDELT |
| `zscore` | REAL | vs bill's trailing 30-day mean |

---

## 5. Model interface (the protocol)

The model calls **only these five functions** (5.1–5.4a). The database side implements them as SQL views or Python functions with identical signatures and outputs. Every function takes `as_of` and returns nothing dated after it.

### 5.1 `get_bills(as_of) -> table`
One row per bill introduced on or before `as_of`. Only columns marked "intro", plus:
- `outcome` and `outcome_date` as they stood on `as_of` (`'in_progress'` if not yet resolved by then)
- `carry_over_date` only if ≤ `as_of`

### 5.2 `get_transitions(as_of) -> table`
One row per bill per model state entered, built from `stage_sittings` + `stage_map`.
| Column | Notes |
|---|---|
| `bill_id` | |
| `model_state` | |
| `state_order` | Position in this bill's chain (depends on originating House) |
| `entered_date` | First sitting of this state |
| `exited_date` | First sitting of the next state; `NULL` if still in it on `as_of` |
| `sitting_days_in_state` | Counted from `sitting_days` for the relevant House |
| `exit_type` | `'advance'`, `'fail'`, or `NULL` (censored) |
| `fail_cause` | `'session_end'`, `'dissolution'`, `'withdrawn'`, `'defeated'`, or `NULL` |

### 5.3 `get_covariates(as_of) -> table`
Time-varying features in long format, one row per change. The model forward-fills.
| Column | Notes |
|---|---|
| `bill_id` | `NULL` = applies to all bills (e.g. political context) |
| `feature` | Name from the registry in 5.5 |
| `valid_from` | Date the value became true |
| `value` | REAL |

### 5.4 `get_calendar(as_of) -> table`
| Column | Notes |
|---|---|
| `date`, `house` | Past sitting days ≤ `as_of` |
| `announced` | Future dates from `announced_dates` with `announced_on ≤ as_of` |
| `sessions` | All sessions with `start_date ≤ as_of`, `end_date` hidden if after `as_of` |
| `parliaments` | Same rule |

### 5.4a `get_stage_sittings(as_of) -> table`
Individual stage sittings, needed where a state has several sittings that matter on their own (ping-pong rounds, MODEL_SPEC §4.4).
| Column | Notes |
|---|---|
| `bill_id`, `sitting_id`, `api_stage_id`, `house`, `session_id` | As in `stage_sittings` |
| `model_state` | From `stage_map`; `NULL` for stages the model ignores |
| `event_date` | Only sittings with `event_date ≤ as_of`, for bills introduced by `as_of` |

### 5.5 Feature registry
The model requests features by name. Adding a feature = adding a row here and a source in `get_covariates`.

| Feature | Tier | Type | Source | Known at |
|---|---|---|---|---|
| `originating_house_lords` | 1 | 0/1 | `bills` | intro |
| `size_clauses` | 1 | int | `bills` | intro |
| `committee_whole_house` | 1 | 0/1 | `bills` | event |
| `carry_over` | 1 | 0/1 | `bills.carry_over_date` | event |
| `working_majority` | 1 | int | `political_snapshots` | event |
| `lords_government_share` | 1 | real | `political_snapshots` | event |
| `c2r_margin` | 1 | int | `divisions` | event |
| `c2r_rebels` | 1 | int | `divisions` | event |
| `amendments_tabled_cum` | 1 | int | `amendments` | event |
| `amendments_agreed_cum` | 1 | int | `amendments` | event |
| `lords_defeats_cum` | 1 | int | `divisions` | event |
| `days_to_session_end` | 1 | int | `announced_dates` / model estimate | event |
| `days_to_max_parliament` | 1 | int | `parliaments` | event |
| `kings_speech_flag` | 2 | 0/1 | `bills` | intro |
| `department` | 2 | category | `bills` | intro |
| `c2r_speakers` | 2 | int | `debates` | event |
| `c2r_word_count` | 2 | int | `debates` | event |
| `critical_reports_cum` | 2 | int | `scrutiny_reports` | event |
| `crisis_flag` | 2 | 0/1 | `context_flags` | event |
| `government_poll_lead` | 2 | real | `polls` | event |

---

## 6. Model outputs (model → database)

### 6.1 `model_runs`
| Column | Type | Notes |
|---|---|---|
| `run_id` | INTEGER PK | |
| `model_version` | TEXT | e.g. `0.1.0` |
| `trained_on_until` | DATE | Last training `as_of` |
| `features_used` | TEXT | Comma-separated registry names |
| `created_at` | TIMESTAMP | |

### 6.2 `forecasts`
One row per bill per forecast.
| Column | Type | Notes |
|---|---|---|
| `forecast_id` | INTEGER PK | |
| `run_id` | INTEGER FK | |
| `bill_id` | INTEGER FK | |
| `as_of` | DATE | |
| `current_state` | TEXT | `model_state` |
| `days_in_current_state` | INTEGER | Sitting days |
| `p_royal_assent` | REAL | Eventually reaches Royal Assent |
| `p_ra_before_session_end` | REAL | |
| `ra_date_p10` | DATE | |
| `ra_date_p50` | DATE | Headline forecast |
| `ra_date_p90` | DATE | |
| `created_at` | TIMESTAMP | |

### 6.3 `forecast_stages`
Per remaining stage, for the timeline and failure chart.
| Column | Type | Notes |
|---|---|---|
| `forecast_id` | INTEGER FK | |
| `model_state` | TEXT | |
| PK | (`forecast_id`, `model_state`) | |
| `p_reach` | REAL | Probability the bill reaches this stage |
| `p_fail_defeated` | REAL | Probability it fails here, by cause |
| `p_fail_withdrawn` | REAL | |
| `p_fail_session_end` | REAL | |
| `p_fail_dissolution` | REAL | |
| `entry_date_p50` | DATE NULL | Expected date of reaching this stage |

### 6.4 `alerts`
| Column | Type | Notes |
|---|---|---|
| `alert_id` | INTEGER PK | |
| `bill_id` | INTEGER FK | |
| `created_at` | TIMESTAMP | |
| `trigger` | TEXT | `'stage_change'`, `'lords_defeat'`, `'commons_rebellion'`, `'amendment_spike'`, `'carry_over'`, `'prorogation_announced'`, `'dissolution_announced'`, `'news_spike'` |
| `summary` | TEXT | One plain-English sentence for the front end |
| `forecast_id_before` | INTEGER FK NULL | |
| `forecast_id_after` | INTEGER FK | |
| `delta_ra_days` | INTEGER NULL | Change in `ra_date_p50` |
| `delta_p_royal_assent` | REAL NULL | |
| `top_drivers` | TEXT | JSON list of `{feature, contribution}` |

**Alert rule:** write an alert if the trigger is `stage_change`, `prorogation_announced` or `dissolution_announced`, or if `|delta_ra_days| > 14`, or if `|delta_p_royal_assent| > 0.05`.

### 6.5 `backtest_forecasts`
Backtest results of any model, one row per forecast. Each model writes under its own `model_name` (e.g. `passage_v1`), so models can be compared in one table. Forecasts must use only data available on `forecast_date`.
| Column | Type | Notes |
|---|---|---|
| `model_name` | TEXT | Part of PK; e.g. `passage_v1` |
| `bill_id` | INTEGER FK | Part of PK |
| `forecast_date` | DATE | Part of PK; the `as_of` the forecast was made on |
| `state_at_forecast` | TEXT NULL | `model_state` on `forecast_date` |
| `p_royal_assent` | REAL NULL | Forecast probability of Royal Assent |
| `ra_date_p10` | DATE NULL | 10th percentile Royal Assent date (among passing paths) |
| `ra_date_p50` | DATE NULL | Median Royal Assent date |
| `ra_date_p90` | DATE NULL | 90th percentile Royal Assent date |
| `actual_outcome` | TEXT | `bills.outcome` as finally known (§3.1 values) |
| `actual_ra_date` | DATE NULL | Actual Royal Assent date; `NULL` if the bill did not pass |
| `created_at` | TIMESTAMP | When the row was written |

### 6.6 `bill_outlooks`
Presentation-ready outlook per bill in Parliament (`model/outlook.py`), computed from the passage model's simulated paths without changing its predictions: chance of law by deadlines (recesses, quarter ends, session end), weather status, weekly curve, pace and next milestone, with every sentence ready to show. Written by `uv run python -m westminster_weather.model.outlook_generate`; served by `GET /api/v1/bills/{bill_id}/outlook?target_date=YYYY-MM-DD` (computed on request). Forecasts use only data available on `forecast_date`; deadline chances are isotonic-recalibrated only when the backtest calibration check calls for it (fitted on training-era forecasts).
| Column | Type | Notes |
|---|---|---|
| `bill_id` | INTEGER FK | Part of PK |
| `forecast_date` | DATE | Part of PK; the `as_of` of the forecast |
| `target_date` | DATE | Target date of the weather status (default: next recess at least 30 days away) |
| `model_version` | TEXT | Passage model set and fit date + outlook version |
| `weather_status` | TEXT | `sunny`, `partly_cloudy`, `rain` or `storm` |
| `p_royal_assent` | REAL | Chance the bill ever becomes law |
| `p_by_target` | REAL | Chance of Royal Assent on or before `target_date` |
| `recalibrated` | BOOLEAN | Deadline chances isotonic-recalibrated |
| `outlook` | JSON | Full `BillOutlook` (`schemas/outlook.py`) |
| `generated_at` | TIMESTAMP | When the row was written |

---

## 7. Update cycle

1. **Ingestion** (database side, every 1–6 hours): pull from Bills API, Commons Votes, Lords Votes, Members API, calendar; Tier 2 sources if built. Upsert by primary key.
2. **Change detection:** compare `bills.last_api_update` and new rows in `stage_sittings`, `divisions`, `amendments`, `announced_dates`, `news_signals` since the last run.
3. **Re-forecast** (model side): for each changed bill still `in_progress`, call the interface with `as_of = today`, write to `forecasts` and `forecast_stages`.
4. **Alerts:** compare with the bill's previous forecast, apply the alert rule, write to `alerts`.
5. **Front end** reads `forecasts`, `forecast_stages` and `alerts` only.

---

## 8. Validation checks (database side, before the model runs)

- Every `stage_sittings.api_stage_id` exists in `stage_map`.
- No `event_date` is after `ingested_at`.
- `bills.outcome = 'royal_assent'` ⇔ a sitting exists with `api_stage_id = 11`.
- Every `in_progress` bill belongs to the current session or has a carry-over date.
- `political_snapshots` intervals for each House do not overlap and have no gaps.
- Sitting-day counts per session fall within a plausible range (flag sessions with fewer than 50).

---

## 9. Ownership

| Part | Owner |
|---|---|
| Sections 2–4 (tables, ingestion), section 8 (checks) | Data |
| Section 5 (interface implementation) | Data, reviewed by Models |
| Section 5.5 (feature registry), section 6 (outputs) | Models |
| Reads section 6 tables | Front end |

---

## 10. Take-up model tables (Phase 1 → 2)

Status: **draft for approval** (checkpoint 2 of `docs/model/TAKEUP_PLAN.md`). Owner: Roman.

The take-up model asks whether a government commitment to legislate becomes a bill, and when. These tables are separate from the passage model's: they are defined on their own SQLAlchemy `MetaData` (`takeup/tables.py`), are **not** created by the app's start-up `create_all`, and are migrated first on the Neon branch `takeup-dev` (`DATABASE_URL_TAKEUP`), then on `production` only after review.

Naming follows the take-up task spec: **Tier A** is the model (cure model, `takeup/model.py`), **Tier B** is the data pipeline that fills these tables (sources, NLP triggers, ideas, links, events).

An **idea** is one commitment to legislate on one topic. Ideas come only from ministers' words (written ministerial statements, Hansard), found by NLP and grouped by topic; the earliest commitment is the trigger. King's/Queen's Speech data is **not** used (decision of 2026-10-24); `bills.kings_speech_flag` in section 3 belongs to the passage model and is unaffected.

**Raw sources are stored too** (10.0a, 10.0b), so the database is the source of truth: the NLP pipeline reads from them, every commitment sentence points back to its source document, and changing the trigger lexicon never needs a re-fetch. About 25,000 documents with 86 MB of text (2006–2026; 71 MB of it from 2015), roughly 40–50 MB once Postgres compresses it. The fetcher's local cache (`backend/data/cache/takeup/`) is only a download speed-up.

### 10.0a `source_documents`
One row per written ministerial statement (all, Nov 2014 onwards) or Hansard contribution that contains a trigger phrase.
| Column | Type | Notes |
|---|---|---|
| `doc_id` | TEXT PK | `wms-<statement id>`, `han-<ContributionExtId>` (spoken) or `hanw-<ContributionExtId>` (written, pre-2015) |
| `source` | TEXT | `'wms'`, `'hansard_spoken'` or `'hansard_written'` |
| `source_id` | TEXT | API ID as published |
| `event_date` | DATE | Date made / sitting date |
| `member_id` | INTEGER NULL | Members API ID of the minister or speaker |
| `house` | TEXT NULL | `'Commons'` or `'Lords'` |
| `title` | TEXT | Statement title or debate section |
| `department` | TEXT NULL | Answering body (written statements) |
| `text` | TEXT | Full text, HTML stripped |
| `matched_phrases` | TEXT NULL | Hansard only: the search phrases that returned it, comma-separated |
| `fetched_at` | TIMESTAMP | |

### 10.0b `member_government_posts`
Government posts per member, for the "was the speaker a minister on that date" check.
| Column | Type | Notes |
|---|---|---|
| `member_id` | INTEGER | Members API ID |
| `post_name` | TEXT | e.g. `Secretary of State for Transport` |
| `start_date` | DATE | |
| `end_date` | DATE NULL | `NULL` = still in post |
| `fetched_at` | TIMESTAMP | |
| PK | (`member_id`, `post_name`, `start_date`) | |

A member with no rows has held no government post. Members whose posts were fetched are listed in the fetch log, so "no rows" is distinguishable from "not fetched".

### 10.2 `signal_mentions`
One row per sentence in which the government commits (or seems to commit) to legislate.
| Column | Type | Notes |
|---|---|---|
| `mention_id` | TEXT PK | `<doc_id>-<sentence index>` |
| `source` | TEXT | `'wms'`, `'hansard_spoken'` or `'hansard_written'` |
| `doc_id` | TEXT FK → `source_documents` | Document the sentence comes from |
| `event_date` | DATE | Date made / sitting date |
| `member_id` | INTEGER NULL | Members API ID of the speaker |
| `is_minister` | BOOLEAN NULL | Speaker held a government post on `event_date`; `NULL` = unknown |
| `department` | TEXT NULL | Answering body (written statements) |
| `title` | TEXT | Statement title or debate section |
| `sentence` | TEXT | The matched sentence |
| `snippet` | TEXT | Sentence plus one either side |
| `phrase` | TEXT | Trigger phrase matched |
| `firmness` | INTEGER | 0 vague, 1 medium, 2 firm |
| `is_commitment` | BOOLEAN | Kept as a government commitment |
| `topic` | TEXT | Short topic phrase |
| `labelled_by` | TEXT | `'rules'`, `'gemini'` or `'hand'` |
| `ingested_at` | TIMESTAMP | |

### 10.3 `ideas`
| Column | Type | Notes |
|---|---|---|
| `idea_id` | TEXT PK | `nlp-<nnnnn>` |
| `origin` | TEXT | Always `'nlp'` (kept for future sources) |
| `topic` | TEXT | |
| `trigger_date` | DATE | Earliest mention (all data); `get_takeup_signals` recomputes it as of a date |
| `department` | TEXT NULL | |
| `first_source` | TEXT | `'wms'`, `'hansard_spoken'`, `'hansard_written'` |
| `initial_firmness` | INTEGER | Firmness of the first mention |
| `n_mentions` | INTEGER | |
| `ingested_at` | TIMESTAMP | |

### 10.4 `idea_mentions`
| Column | Type | Notes |
|---|---|---|
| `idea_id` | TEXT FK → `ideas` | |
| `mention_id` | TEXT | `signal_mentions.mention_id` |
| `group_id` | TEXT | Topic group before ideas linked to the same bill were merged |
| PK | (`idea_id`, `mention_id`) | |

`group_id` exists for leakage control: merging two groups because they link to the same bill uses knowledge of that bill, so `get_takeup_signals(as_of)` applies the merge only if the bill's first reading is ≤ `as_of`.

### 10.5 `idea_events`
Time-varying events for an idea (the model's `x` features).
| Column | Type | Notes |
|---|---|---|
| `idea_id` | TEXT FK → `ideas` | |
| `event_type` | TEXT | `'consultation'`, `'consultation_response'`, `'green_paper'`, `'white_paper'`, `'draft_bill'` |
| `event_date` | DATE | |
| `title` | TEXT | |
| `source_url` | TEXT | |
| `ingested_at` | TIMESTAMP | |
| PK | (`idea_id`, `event_type`, `event_date`) | |

### 10.6 `idea_bill_links`
| Column | Type | Notes |
|---|---|---|
| `idea_id` | TEXT PK, FK → `ideas` | One link per idea |
| `bill_id` | INTEGER NULL FK → `bills` | `NULL` = checked, no matching bill |
| `similarity` | REAL NULL | Text similarity of idea vs bill title + long title |
| `decided_by` | TEXT | `'auto'`, `'gemini'`, `'rule'` or `'hand'` |
| `checked` | BOOLEAN | Hand-checked |
| `ingested_at` | TIMESTAMP | |

The bill's first reading date is not copied here; it is read from `stage_sittings` (stage 6 or 1), so it can never disagree with the bills data.

### 10.7 `takeup_runs`
| Column | Type | Notes |
|---|---|---|
| `run_id` | INTEGER PK | |
| `created_at` | TIMESTAMP | |
| `code_version` | TEXT | Git commit |
| `train_cut` | DATE | Ideas with trigger before this date were used for fitting |
| `features` | TEXT | Comma-separated `z_*` and `x_*` names |
| `parameters` | TEXT | JSON: estimates and standard errors |
| `metrics` | TEXT | JSON: evaluation results |

### 10.8 `takeup_forecasts`
| Column | Type | Notes |
|---|---|---|
| `run_id` | INTEGER FK → `takeup_runs` | |
| `signal_id` | TEXT | Row of `get_takeup_signals(as_of)`: the `idea_id`, or `<idea_id>:<group_id>` for a group not yet merged at `as_of` (10.4) |
| `idea_id` | TEXT FK → `ideas` | |
| `as_of` | DATE | |
| PK | (`run_id`, `signal_id`, `as_of`) | |
| `pi`, `pi_lo`, `pi_hi` | REAL | Chance ever taken up and its 95% CI |
| `verdict` | TEXT | `'likely'`, `'unlikely'` or `'uncertain'` |
| `weeks_elapsed` | INTEGER | Weeks since trigger with no bill |
| `pi_now` | REAL | Updated chance after `weeks_elapsed` silent weeks |
| `p26`, `p52` | REAL | Chance of a first reading within the next 26 / 52 weeks from `as_of` |
| `t_10`, `t_50`, `t_90` | INTEGER NULL | Timing quantiles in weeks from trigger, conditional on `weeks_elapsed` silent weeks (always > `weeks_elapsed`); `NULL` beyond the 5-year window |
| `created_at` | TIMESTAMP | |

### 10.9 Interface: `get_takeup_signals(as_of) -> table`
The take-up model reads data only through this function. One row per idea with at least one mention dated ≤ `as_of`.
| Column | Notes |
|---|---|
| `signal_id`, `title` | `idea_id`, or `<idea_id>:<group_id>` for a group whose merge is not yet visible at `as_of` (10.4); topic |
| `signal_date` | Earliest mention ≤ `as_of` |
| `bill_date` | First reading of the linked bill if ≤ `as_of`, else `NULL` |
| `data_end` | `as_of`, or the dissolution date if the governing party later changed before a bill appeared (censored, not failed) |
| `z_*` | Features known at `signal_date` |
| `x_*` | Dates time-varying events happened, only if ≤ `as_of` |

Rules: mentions, events and links dated after `as_of` are invisible; merges of groups follow `idea_mentions.group_id` (10.4); first reading = earliest sitting of Bills API stage 6 or 1. Merging through a shared bill de-duplicates ideas and stays invisible until that bill's first reading.

### 10.10 Validation checks
- Every `idea_mentions.mention_id` exists in `signal_mentions`.
- Every `signal_mentions.doc_id` exists in `source_documents`; every `source_documents.member_id` of a Hansard row has been checked against `member_government_posts`.
- Every `idea_bill_links.bill_id` (non-null) exists in `bills` and has a first-reading sitting on or after the idea's `trigger_date` minus 7 days.
- No `signal_mentions.event_date` or `idea_events.event_date` after `ingested_at`.

### 10.11 Take-up feature registry
Features the take-up model requests by name (separate from the passage model's registry in 5.5). All are computed by `get_takeup_signals(as_of)` from information dated ≤ `as_of`. Any feature with fewer than ~15 positive cases in the training data is dropped and logged.

| Feature | Type | Definition | Source |
|---|---|---|---|
| `first_source` | z, one-hot | Where the idea first appeared: `written_statement` (reference), `hansard` | `ideas.first_source` |
| `initial_firmness` | z, 0/1/2 | Firmness of the first mention | `signal_mentions.firmness` |
| `department` | z, one-hot | Lead department; small departments pooled into `other` | `source_documents.department` |
| `years_since_election` | z, real | Years from the last general election to the trigger | `parliaments` |
| `months_into_session` | z, real | Months from the session's start to the trigger | `sessions` |
| `working_majority` | z, int | **Skipped:** needs `political_snapshots`, which does not exist yet | — |
| `firm_commitment` | x, date | First firm (2) mention after the trigger | `signal_mentions` |
| `repeat_mention` | x, date | Second mention of the idea | `signal_mentions` |
| `consultation` | x, date | Consultation launched | `idea_events` (GOV.UK) |
| `consultation_response` | x, date | Government response published | `idea_events` (GOV.UK) |
| `white_paper` | x, date | White or green paper published | `idea_events` (GOV.UK) |
| `draft_bill` | x, date | Draft bill published | `idea_events` (GOV.UK) |
| `new_parliament` | x, date | State Opening after a same-party election (2015, 2017, 2019) | `sessions`, `parliaments` |

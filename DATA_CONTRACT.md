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

The model calls **only these four functions**. The database side implements them as SQL views or Python functions with identical signatures and outputs. Every function takes `as_of` and returns nothing dated after it.

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

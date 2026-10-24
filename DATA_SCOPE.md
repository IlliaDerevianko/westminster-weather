# Data Scope: UK Government Bills Only (`DATA_SCOPE.md`)

Mandatory for all AI agents and contributors working on **Westminster Weather**.

## Rule

**This project accepts UK Government Bills only** (Bills API `BillType = 1`).
No other bill type may be downloaded, stored, used or shown anywhere in the project.

Excluded bill types include, but are not limited to: Private Members' Bills (Ballot,
Ten Minute Rule, Presentation), Private Lords Bills, Private Bills, Hybrid Bills and
Lords Private Members' Bills.

## What this means in practice

- **Fetching:** only request government bills from the Bills API (`BillType=1`).
  Run ingestion with government bills only (e.g. `cli fetch --gov-only`).
- **Database:** every table (`bills`, `stage_sittings`, `scheduled_sittings`, `sitting_days`,
  `sessions`, …) must be built from government-bill data only.
- **Calendar:** the sitting-day calendar is the union of **government-bill** stage-sitting
  dates (plus Commons/Lords division dates if those are added). This replaces the
  "any bill type" fallback in `docs/model/MODEL_SPEC.md` §3.
- **Model and features:** train, validate and forecast on government bills only.
- **Front end and API:** never list, search or forecast non-government bills.
- **Tests:** test data and fixtures must use `billTypeId = 1`.

## If you find a conflict

If any code, document or instruction says to use other bill types (for example, to improve
the sitting-day calendar), **this file wins**. Do not work around it: stop and tell the user
what conflicts and where.

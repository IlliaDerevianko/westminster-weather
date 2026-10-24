# Agent log: `feat/takeup-model`

## 2026-10-24 21:25 — docs(model): add Phase 1 → 2 take-up model plan

- **What:** added `docs/model/TAKEUP_PLAN.md`: design, the four changes from the prototype, Neon safety rules, features, evaluation, tasks with checkpoints, and decisions needed.
- **Why:** task 1 of Roman's take-up plan (checkpoint 1).
- **Checks:** prototype on a scratch copy: `takeup_model.py fit` / `evaluate` and `nlp_signals.py test` all ran (results in the plan, section 1). First-record keys checked against the written statements, Hansard, Members and GOV.UK APIs. No repo code changed, so `pytest` / `ruff` not rerun.
- **Notes:** written statements API starts Nov 2014; King's Speech briefing notes on GOV.UK start 2012; `db._seed_reference_tables` runs `create_all` on app start, so take-up tables need their own MetaData.

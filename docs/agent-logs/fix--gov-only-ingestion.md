# Agent log: `fix/gov-only-ingestion`

## 2026-10-24 21:28 — fix(data): ingest UK Government Bills only

- **What:** `data/bills_api.py` now only lists and downloads `BillType = 1` (removed `include_all_types`), and `build_tables_from_payloads` drops any other bill type before building tables, so `sitting_days` and `sessions` come from government bills only; the `all_stage_sittings` table is gone. `cli fetch` downloads government bills only; `--gov-only` is still accepted (hidden, no effect) so existing commands keep working. Added two tests.
- **Why:** Aleks: the project accepts government bills only (`DATA_SCOPE.md`, branch `docs/government-bills-only`). Changing the teammate's ingestion code was approved by Aleks.
- **Checks:** `ruff`: clean; `pytest`: 46 passed.
- **Notes:** the Neon DB was already loaded government-only, so no re-ingestion is needed.

# Agent log: `docs/government-bills-only`

## 2026-10-24 21:24 — docs: restrict project data to government bills only

- **What:** added `DATA_SCOPE.md` (government bills only, `BillType = 1`, everywhere), referenced it from AGENTS.md §5 and CLAUDE.md §5, and changed MODEL_SPEC §3 and BUILD_PLAN task 1 so the sitting-day calendar uses government-bill sittings only.
- **Why:** Aleks: the project does not accept other bill types.
- **Checks:** none (docs only).
- **Notes:** `data/bills_api.py` (teammate's code) still defaults to `include_all_types=True` and the `cli fetch` default downloads all types; not changed without approval. The Neon DB is already government-only (loaded with `--gov-only`).

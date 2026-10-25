# Agent log: `feat/stage3-frontend`

## 2026-10-25 06:36 — feat(stage3): real in-force status on the bill page; stage 3 reads act_texts

- **What:** `domain/bills.py` uses `act_in_force_status`: "Law fully in force" is completed only when no provision is prospective (date = latest provision start date), otherwise current with "N of M sections still prospective"; "Amended or repealed" with no instrument is skipped ("none recorded") instead of upcoming. The stage 3 model (`journey/stage3_xgb.py`, `analysis/commencement_xgb.py`) reads `act_texts` instead of the dropped `bill_texts`. Updated `tests/test_health_and_stages.py`: the Online Safety Act is not fully in force (3 of 404 sections prospective).
- **Why:** Roman asked to see stage 3 on the frontend from the database, shown as done where the law already exists.
- **Checks:** `ruff check .`: clean; `pytest`: 379 passed. API: Tobacco and Vapes Act -> law in force current (173 of 372 prospective), commencement orders upcoming, stage 3 forecast median 34 days; Medicines and Medical Devices Act -> current (1 of 56), amended skipped. Page screenshot of bill 3879 checked.

## 2026-10-25 06:35 — feat(data): Act texts and in-force status from legislation.gov.uk

- **What:** `clients/legislation_texts.py` fetches, for every Act in `act_commencements`, the enacted XML (plain text -> new table `act_texts`) and the current revised XML (`Status="Prospective"` provisions -> new table `act_in_force_status`: section groups, prospective count, fully in force, latest start date). ORM rows in `models.py`, upsert helper `save_act_texts_to_db` in `db.py`, downloads cached in gitignored `backend/data/cache/legislation/`. Ran it against production: 522 Acts written to both tables; 356 fully in force, 166 with provisions not yet in force. Tests in `tests/test_legislation_texts.py`.
- **Why:** Roman asked to fetch what the database lacks for stage 3: `bill_texts` (Act text, created outside the repo) had been dropped, and `act_commencements.is_fully_in_force` was true for all 522 Acts (the in-force date was just the last commencement order or Royal Assent).
- **Checks:** `ruff check .`: clean; parser tests pass; commencement-clause feature shares identical to the old `bill_texts` (0.51 / 0.23 / 0.24 / 0.14).

## 2026-10-25 05:50 — fix(stage3): no stage 3 forecast instead of an error when a source table is missing

- **What:** `journey/stage3_xgb.py` loads its data without caching failures: if a source table is missing or the database is unreachable it logs a warning and returns no stage 3 forecast, and retries on the next call. Test in `tests/journey/test_stage3_xgb.py`.
- **Why:** the production table `bill_texts` (Act texts; created outside this repo, no ingest here builds it) was dropped after PR #21 merged. The running API only worked from data cached before the drop; after a restart every Act's `/forecast` would have failed.
- **Checks:** `ruff check .`: clean; `pytest tests/journey`: 15 passed; restarted API without `bill_texts`: `/api/v1/bills/3137/forecast` and `/2700/forecast` return 200 with null stage 3 fields, `/api/v1/bills/2700` 200.

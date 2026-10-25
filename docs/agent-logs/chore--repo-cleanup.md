# Agent log: `chore/repo-cleanup`

## 2026-10-25 08:43 — chore: remove the unused bill_outlooks generator and table definition

- **What:** Deleted `model/outlook_generate.py`, `db.save_bill_outlooks_to_db`, the `BillOutlookRow` ORM model and DATA_CONTRACT §6.6.
- **Why:** Category D of the user's cleanup. `GET /api/v1/bills/{id}/outlook` computes outlooks on request; nothing reads the `bill_outlooks` table.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 393 passed.
- **Notes:** The `bill_outlooks` table itself was not dropped from the database (if it exists, it is simply no longer written or created).

## 2026-10-25 08:41 — chore: remove unused API clients

- **What:** Deleted `clients/govuk_content.py`, `clients/parliament_bills.py` and `clients/legislation.py` (nothing imports them) and the two config settings only they used (`legislation_gov_uk_base_url`, `gov_uk_content_api_base_url`). `parliament_bills_api_base_url` stays (used by `data/bills_api.py`).
- **Why:** User asked to clean up unused code; chose categories A–D (dead code, research experiments, take-up backend, outlook generator). This is A.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 393 passed.

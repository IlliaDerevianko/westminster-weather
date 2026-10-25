# Agent log: `chore/repo-cleanup`

## 2026-10-25 08:41 — chore: remove unused API clients

- **What:** Deleted `clients/govuk_content.py`, `clients/parliament_bills.py` and `clients/legislation.py` (nothing imports them) and the two config settings only they used (`legislation_gov_uk_base_url`, `gov_uk_content_api_base_url`). `parliament_bills_api_base_url` stays (used by `data/bills_api.py`).
- **Why:** User asked to clean up unused code; chose categories A–D (dead code, research experiments, take-up backend, outlook generator). This is A.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 393 passed.

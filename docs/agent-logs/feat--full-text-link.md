# Agent log: `feat/full-text-link`

## 2026-10-25 07:25 — feat(api): full_text_url and full_text_label on every bill

- **What:** `GET /api/v1/bills/{id}` returns `full_text_url` and `full_text_label`: the Act on legislation.gov.uk (`ukpga/{year}/{number}`, current revised text) once the bill has Royal Assent as of the selected date, otherwise the bill's published texts on bills.parliament.uk (`/bills/{id}/publications`). Three tests: an Act, the same Act before its Royal Assent, a bill in Parliament.
- **Why:** User asked for a link to the full bill text on every bill page.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 392 passed. legislation.gov.uk/ukpga/2025/5 returns 200; bills.parliament.uk answers scripts with a Cloudflare challenge (403), but the Bills API lists 10 publications for bill 4305.
- **Notes:** The Online Safety Bill live-API path does not set the fields; the front end falls back to the bills.parliament.uk link. Branched from `chore/remove-promises-page`.

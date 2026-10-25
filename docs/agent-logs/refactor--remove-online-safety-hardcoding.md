# Agent log: `refactor/remove-online-safety-hardcoding`

## 2026-10-25 07:38 — refactor: remove Online Safety Bill special-casing

- **What:** Removed the `bill_id == 3137` live-API load in `evaluate_parliament_bill_from_db`; the hand-written Online Safety Bill fetch/sync/load functions in `domain/bills.py` (~350 lines); the `GET /api/v1/bills/online-safety-bill` path and `POST /api/v1/bills/online-safety-bill/sync` endpoint; the front end's default to bill 3137 and its `online-safety-bill` alias (with no bill in the URL it now opens the most recently introduced bill). The sync-endpoint test is dropped and the pre-Parliament history test now uses the generic `/api/v1/bills/3137`.
- **Why:** User: "don't load online safety act separately there shouldn't be any hardcoding".
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 393 passed (was 394; one sync test removed); `npx tsc --noEmit`: clean. Live: old alias and sync endpoint 404, `/api/v1/bills/3137` 200, `/` 200.
- **Notes:** Bill 3137's stored pre-Parliament records in `tracked_bills` (6 stages) are kept and still shown through the generic path, with the same timeline at every test date. Nothing can rebuild those records any more; they were only ever written by the removed sync. Left as is: the search box example hint mentions "Online Safety, … 3137", and `model/cli.py` lists 3137 among its sample bills for the sanity report. `clients/govuk_content.py` and `clients/parliament_bills.py` are now unused. The user's own uncommitted edits to `page.tsx` were left out. Branched from `fix/list-latest-stage`.

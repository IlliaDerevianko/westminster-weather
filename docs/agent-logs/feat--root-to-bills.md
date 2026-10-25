# Agent log: `feat/root-to-bills`

## 2026-10-25 10:16 — feat(ui): make the bill directory the home page

- **What:** `next.config.mjs` redirects `/` without a `bill_id` to `/bills` (307, temporary); `/?bill_id=…` still opens a bill. The tracker's fallback that fetched the whole bill list to open the newest bill is replaced by a client-side `router.replace("/bills")` for the same case; the unused `BillSummary` type is removed.
- **Why:** User: the empty URL `/` should point to the bills page.
- **Checks:** `npx tsc --noEmit`: clean. Dev server: `/` -> 307 to `/bills`; `/?bill_id=4305` 200; `/bills` 200.

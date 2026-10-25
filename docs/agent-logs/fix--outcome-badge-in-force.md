# Agent log: `fix/outcome-badge-in-force`

## 2026-10-25 07:05 — fix(frontend): show "Law fully in force" as the overall outcome once it is reached

- **What:** The header badge shows "Law fully in force" when that stage is completed for the selected date; otherwise a readable label for `bills.outcome` ("Royal Assent", "Fell at session end", …) instead of the raw value.
- **Why:** User: bill 4207 (Taxation (Energy and Vehicles) Act 2026) said "Overall Outcome: royal_assent" though it is fully in force; fix it for all bills. `bills.outcome` (DATA_CONTRACT §3.1) never goes past Royal Assent, so the badge now also looks at the stages.
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=4207` loads (200). API for 4207: `law_fully_in_force` completed. Badge not checked visually in a browser.
- **Notes:** Frontend only; the API's `outcome` field is unchanged. The bill selector dropdown still shows raw `outcome` values (the list endpoint has no stage data). The user's own uncommitted edits to `page.tsx` were left out of this commit. Branched from `fix/time-travel-caption`.

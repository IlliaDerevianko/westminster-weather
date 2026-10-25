# Agent log: `fix/time-travel-caption`

## 2026-10-25 07:01 — fix(frontend): drop the phase count and date range from the time-travel caption

- **What:** The date-picker caption now reads "Time-travel across all phases of <bill>." without "3" and without the "(first signal → outcome)" date range.
- **Why:** User asked to remove the time frame in brackets and the number 3.
- **Checks:** `npx tsc --noEmit`: clean. Frontend only, so no backend checks.
- **Notes:** Branched from `fix/zero-regulations-label` so the dev server keeps showing that fix; the branch can be rebased onto `main` once that one is merged.

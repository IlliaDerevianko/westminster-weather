# Agent log: `chore/ui-copy-tweaks`

## 2026-10-25 08:03 — chore(frontend): trim UI copy on the tracker, bill list and outlook panel

- **What:** The user's own edits, committed at their request: date picker label drops "(`as_of_date`)"; the legend's "red stop branch" sentence is removed; stage cards no longer show the entered date or "≤ as_of_date" next to the sittings count; the bill list's row button reads "Details →"; the outlook panel's "From N simulated paths…" footnote is removed.
- **Why:** User asked to commit, push and merge their changes.
- **Checks:** `npx tsc --noEmit`: clean. Frontend only.
- **Notes:** With the entered date removed, a stage card with a different completed date now reads "→ <completed date>" with no start date. The edits leave two whitespace-only lines in `page.tsx`. Branched from `refactor/remove-online-safety-hardcoding`.

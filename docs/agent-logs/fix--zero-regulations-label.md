# Agent log: `fix/zero-regulations-label`

## 2026-10-25 06:59 — refactor(stages): take the "none yet" badge from the backend summary; test the rule generally

- **What:** The frontend reads the badge from the backend summary's text before " — " instead of mapping stage names to labels. The bill-3751 test is replaced by one rule checked on Acts 33, 40, 45, 57 and 3751: a zero-SI stage after Royal Assent is `skipped` if the Act is fully in force, otherwise `upcoming` with a "None … yet — …" summary and a legislation.gov.uk link.
- **Why:** User: "don't hardcode it for specific one make it general!" The backend rule was already general; the frontend label map and the single-bill test were not.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 384 passed; `npx tsc --noEmit`: clean. Scan of all 418 Acts with a zero-SI stage: 639 stages now "None … yet", 331 skipped, none left as a bare "Upcoming".

## 2026-10-25 06:55 — fix(frontend): show "None drafted/laid yet" instead of a faded "Upcoming"

- **What:** Post-Royal-Assent stages that are upcoming with a backend summary get a "None drafted yet" / "None laid yet" / "None made yet" badge and a dashed white card instead of the faded grey "Upcoming" card. The summary already shows in the card's detail box.
- **Why:** User chose option 2: make zero-SI regulation stages on bills like 3751 readable as "none yet", not "not reached".
- **Checks:** `npx tsc --noEmit`: clean; `/?bill_id=3751` loads (200) on the dev server. ESLint is not configured in the frontend, so not run. Badge not checked visually in a browser.

## 2026-10-25 06:55 — fix(stages): say "none yet" for post-Royal-Assent stages with zero SIs

- **What:** When an Act has no Commencement Orders, Draft SIs or laid SIs and is not yet fully in force, those stages stay `upcoming` but now carry a summary ("None drafted yet — …", "None laid yet — …") and a legislation.gov.uk link. New test for bill 3751 (Water (Special Measures) Act 2025).
- **Why:** User asked why bill 3751's "Regulations drafted" and "Regulations laid" boxes were plain grey "Upcoming". Since PR #23 the live in-force check finds 2 of 33 sections still prospective, so zero-SI stages are `upcoming` rather than `skipped`, which looked the same as a bill before Royal Assent.
- **Checks:** `uv run ruff check .`: clean; `uv run pytest`: 380 passed.
- **Notes:** No API schema change; status values are unchanged. Not checked whether legislation.gov.uk has regulations under this Act that ingestion missed.

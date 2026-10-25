## 2026-10-25 — feat(ui): paginate the government bill directory

- **What:** Added client-side pagination with 20 bills per page, visible result ranges, numbered desktop navigation and compact mobile Previous/Next controls. Search and filters run over the complete in-memory list and reset to page one. Page changes focus and scroll to the results heading.
- **Why:** User requested pagination while retaining all bills in memory.
- **Checks:** Production build and TypeScript passed. Component logic exercised in a Node harness with mocked React hooks and 45 synthetic government-bill summaries: first and final pages, disabled next, filter reset, empty results, clearing filters and a single API fetch all passed. Backend ruff passed; required pytest invocation retains the previously diagnosed readline exit 139. Diff whitespace check passed.
- **Notes:** Browser layout and real DOM focus behavior were not verified in this sandbox.

## 2026-10-25 — feat(ui): increase the base font size by 1.5 pixels

- **What:** Increased the root font size from 16px to 17.5px, scaling rem-based interface typography proportionally.
- **Why:** User requested a 1.5px font increase.
- **Checks:** Frontend production build and type checks passed; backend ruff passed; pytest still exits 139 due to the previously diagnosed local Python readline crash. Diff whitespace check passed.

## 2026-10-25 — fix(frontend): isolate development assets from production builds

- **What:** Configured Next.js development to use .next-dev while production continues to use .next; updated generated-type includes, ignore rules and restart instructions.
- **Why:** Production build checks can overwrite assets expected by a running development server, leaving localhost unstyled. Existing generated output contained production CSS but no development layout CSS.
- **Checks:** Production build and type checks passed; both built pages reference existing stylesheets containing the redesigned selectors and font. Config phase checks confirm distinct dev/build directories and matching build/start directories. Backend ruff passed.
- **Notes:** Diagnosed pytest exit 139 as Anaconda Python readline import segmentation fault. A temporary environment with the existing standard Python 3.12 and installed OpenMP runtime ran the full suite: 166 passed, 60 failed, 31 setup errors. All 91 unsuccessful cases contain database DNS resolution errors. Existing backend environment was preserved. Runtime CSS confirmation awaits restarting the user's frontend; this sandbox cannot bind or access localhost.

## 2026-10-25 — feat(ui): reduce repeated bill context

- **What:** Moved the Parliament bill number into About this bill, placed phase descriptions in a collapsed About this phase disclosure, and removed the Current phase label from phase navigation. Kept current-position indicators in the overview and stage timeline.
- **Why:** Apply the three information-density changes approved by the user.
- **Checks:** Frontend production build and type checks passed; backend ruff passed; pytest exited 139 before producing results; diff whitespace check passed.
- **Notes:** Phase help resets to collapsed when switching phases. Stage counts and forecast percentages remain unchanged. Browser interaction checks remain unverified under the previously observed local-server restriction.

## 2026-10-25 — feat(ui): refresh typography and enlarge the header logo

- **What:** Unified headings and body text under Avenir Next with local system fallbacks. Increased logo width from 240 to 330 pixels on desktop and 180 to 230 on mobile, preserving proportions and allowing narrow headers to wrap.
- **Why:** User requested a different font family and larger top-left logo. No information was removed.
- **Checks:** Clean frontend production build and type checks passed; backend ruff passed; pytest exited 139 before reporting results; diff whitespace check passed.
- **Notes:** Initial incremental build failed on a missing Next.js page entry; moving the generated cache aside and rebuilding succeeded. Browser layout remains unverified under the previously observed server-binding restriction.

## 2026-10-25 — feat(ui): simplify the weather forecast presentation

- **What:** Added restrained SVG weather icons in the specified status colours, a clear passage probability, teal charts and expandable forecast details. Kept pace and the next milestone visible. Made wide charts scroll on small screens and handled empty forecast curves.
- **Why:** Keep the main forecast easy to understand while retaining supporting information.
- **Checks:** Final `npm run build` passed compilation and type checks; `uv run ruff check .` passed; `uv run pytest` again exited 139 before reporting results. `git diff --check` passed.
- **Notes:** Browser visual and interaction checks remain unverified because the environment rejects local server binding with EPERM. Next.js reports an existing multiple-lockfile workspace warning.

## 2026-10-25 — feat(ui): simplify the bill directory and journey

- **What:** Replaced dense cards with searchable bill rows, outcome filters, an understated header and a single-phase timeline. Moved bill context, history and evidence into expandable controls. Added shared responsive styling using the requested palette.
- **Why:** Make the interface easier to scan and reduce competing visual elements.
- **Checks:** Frontend production build and TypeScript passed; backend `uv run ruff check .` passed. `uv run pytest` and direct `.venv/bin/python -m pytest -q` both exited 139 before producing results. `git diff --check` passed.
- **Notes:** Remote main could not be fetched (DNS unavailable), so the branch uses local main. Browser verification blocked by local server bind EPERM. Forecast-specific presentation follows separately.

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

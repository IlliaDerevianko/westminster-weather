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

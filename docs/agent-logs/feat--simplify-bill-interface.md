## 2026-10-25 — feat(ui): simplify the bill directory and journey

- **What:** Replaced dense cards with searchable bill rows, outcome filters, an understated header and a single-phase timeline. Moved bill context, history and evidence into expandable controls. Added shared responsive styling using the requested palette.
- **Why:** Make the interface easier to scan and reduce competing visual elements.
- **Checks:** Frontend production build and TypeScript passed; backend `uv run ruff check .` passed. `uv run pytest` and direct `.venv/bin/python -m pytest -q` both exited 139 before producing results. `git diff --check` passed.
- **Notes:** Remote main could not be fetched (DNS unavailable), so the branch uses local main. Browser verification blocked by local server bind EPERM. Forecast-specific presentation follows separately.

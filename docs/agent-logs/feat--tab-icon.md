# Agent log: `feat/tab-icon`

## 2026-10-25 10:18 — feat(ui): new browser tab icon (tower with sun clock face)

- **What:** `frontend/src/app/icon.svg` is the user's `icon-full-colour (1).svg` (charcoal tower, amber sun clock face), drawing unchanged, framed in a centred square viewBox (`-8 0 116 116`, was `22 0 56 116`) so the tall icon is not stretched in the square tab slot. `favicon.ico` is regenerated from it (32x32, transparent) for browsers without SVG icons.
- **Why:** User asked to set this as the logo shown in the browser tab.
- **Checks:** File inspected before use: plain shapes, no scripts or external links. Rendered preview checked. Dev server serves `/favicon.ico` (200) and `/icon.svg`; pages link both.
- **Notes:** The header logo (`public/brand/logo-horizontal-full-colour.svg`) is unchanged. Browsers cache tab icons, so a hard refresh may be needed to see it.

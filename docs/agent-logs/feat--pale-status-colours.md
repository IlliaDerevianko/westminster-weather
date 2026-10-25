# Agent log: `feat/pale-status-colours`

## 2026-10-25 10:11 — feat(ui): in progress in yellow, Acts in green

- **What:** In progress status text is now amber/yellow (#8A6510) and Acts / current law green (#2E7D3A); failed stays red.
- **Why:** User: in progress should be yellow and the third stage green, not the other way round.
- **Checks:** Contrast on white 5.3:1 and 5.1:1. Screenshot of `/bills` checked.
- **Notes:** The green is outside the site palette, added at the user's request.

## 2026-10-25 10:11 — feat(ui): status as coloured, slightly bolder text instead of tinted boxes

- **What:** Removed the pale background boxes; the status text keeps its colour (in progress teal #0B6A64, current law amber #8A6510, failed red #A93636) and is now weight 500.
- **Why:** User: "don't put boxes around, just change the font colour slightly, maybe make it a bit thicker".
- **Checks:** Contrast on white 6.4:1, 5.3:1, 6.4:1. Screenshot of `/bills` checked.

## 2026-10-25 10:10 — feat(ui): pale status tints on the bill directory

- **What:** Each bill's status on `/bills` gets a pale tint: in progress = pale teal (text #0B6A64), Acts / current law = pale amber, the "sunny" colour (text #8A6510), failed (fell, withdrawn, defeated) = pale red, the "storm" colour (text #A93636).
- **Why:** User asked to slightly change colours to pale ones to indicate in progress, failed and current on the bills page.
- **Checks:** `npx tsc --noEmit`: clean. Text contrast on each tint: 5.7:1, 4.8:1, 5.5:1 (all above 4.5:1). Screenshot of `/bills` checked: in progress and Act tints render as intended; the failed tint was checked by contrast only, not on screen.
- **Notes:** Interpreted "current" as Acts that are now law (their status shows the latest stage, e.g. "Law fully in force").

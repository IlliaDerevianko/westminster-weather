# Agent log: `feat/larger-text-and-logo`

## 2026-10-25 09:29 — feat(frontend): prediction chart a bit shorter

- **What:** `BillOutlookPanel.tsx`: chart canvas height 380 -> 300 and axis labels 16 -> 15; still full width.
- **Why:** Roman asked for the chart a bit smaller.
- **Checks:** `npx tsc --noEmit`: clean; screenshot of bill 4035's outlook checked.

## 2026-10-25 09:27 — feat(frontend): bigger prediction chart

- **What:** `BillOutlookPanel.tsx`: the chance-of-law chart takes the full panel width (was half) and a taller canvas (1000x380, was 640x200), with larger axis labels (16), a thicker line and a bolder target marker; the deadlines list moves below it in two columns.
- **Why:** Roman asked for a bigger prediction graph.
- **Checks:** `npx tsc --noEmit`: clean; screenshot of bill 4035's outlook checked.

## 2026-10-25 09:26 — feat(frontend): bigger logo; remove the Tracker link and the black bill-selector bar

- **What:** `SiteHeader.tsx`: logo height 64 -> 96 px (header 80 -> 112 px), "Tracker" link removed ("All bills" stays). `page.tsx`: removed the black top bar (bill selector and the yellow "Browse All Bills" button) and the state and imports only it used; bills are picked from the directory.
- **Why:** Roman asked to remove the Tracker button and the black navbar with the yellow button, and make the logo bigger.
- **Checks:** `npx tsc --noEmit`: clean; screenshot of the top of bill 4035 checked.

## 2026-10-25 09:24 — feat(frontend): larger text across the site and the Westminster Weather logo

- **What:** base font size 18px (was the browser default 16px) in `globals.css`, so every rem-based Tailwind size grows by 12.5%; the 10 fixed 10-11px sizes became `text-xs` so they scale too; the chart's "target" label 10 -> 12. New `src/app/SiteHeader.tsx` on every page (via `layout.tsx`): the horizontal full-colour logo linking home, plus Tracker / All bills links. Logo in `public/brand/`, favicon `src/app/icon.svg` and `favicon.ico` from the logo pack.
- **Why:** Roman asked for larger text throughout and the Westminster Weather logo.
- **Checks:** `npx tsc --noEmit`: clean; screenshots of the bill page (4035) and the directory checked.

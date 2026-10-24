## 2026-10-25 — Antigravity — `feat(ui): display red stop box only on stage where bill actually stopped`

- **Branch**: `feat/fast-tracked-bypassed-stages`
- **Summary**: Updated the 3-phase pipeline view in `frontend/src/app/page.tsx` so the red stop-risk box (`stopMeta`) on the right of a stage card is rendered **only** when that stage actually stopped (`stage.status === "stopped"` and `stage.active_stop_reason` is set). Bills that turn into law or are still in progress now render full-width (`col-span-12`) stage cards with no unused red stop boxes, while stopped bills (e.g. Bill `#3703` *Tobacco and Vapes Bill* at `commons_report_stage` due to `election_called`) display only the taken stop box on the exact stage where they stopped.
- **Files touched**:
  - `frontend/src/app/page.tsx`
  - `docs/agent-logs/feat--fast-tracked-bypassed-stages.md`
- **Verification**: `npx tsc --noEmit` and `uv run pytest -q` passed.

## 2026-10-25 — Antigravity — `feat(stages): mark bypassed pre-parliamentary and 0-sitting parliamentary stages as fast_tracked`

- **Branch**: `feat/fast-tracked-bypassed-stages`
- **Summary**: Added `StageStatus.FAST_TRACKED` (`"fast_tracked"`) so that when a bill bypasses formal pre-parliamentary stages (`signal_appears` .. `slot_in_programme` with no separate pre-legislative records prior to First Reading) or dispenses with a Parliamentary stage without formal sittings (e.g. `commons_report_stage` with 0 sittings on Bill `#4296` *Church Appointments (Repeal of Discriminatory Provision) Bill* when proceeding directly from Committee of the whole House to 3rd Reading), those stages are explicitly marked `"fast_tracked"` with a `Fast-Tracked` / `Fast-Tracked (0 sittings)` badge, `»` icon, and explanatory summary in the UI.
- **Files touched**:
  - `backend/src/westminster_weather/domain/bills.py`
  - `backend/tests/test_health_and_stages.py`
  - `frontend/src/app/page.tsx`
  - `docs/agent-logs/feat--fast-tracked-bypassed-stages.md`
- **Verification**: `uv run ruff check .`, `uv run pytest -q`, and `npx tsc --noEmit` all passed.

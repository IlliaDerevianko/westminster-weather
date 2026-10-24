## 2026-10-25 — Antigravity — `feat(stages): mark bypassed pre-parliamentary and 0-sitting parliamentary stages as fast_tracked`

- **Branch**: `feat/fast-tracked-bypassed-stages`
- **Summary**: Added `StageStatus.FAST_TRACKED` (`"fast_tracked"`) so that when a bill bypasses formal pre-parliamentary stages (`signal_appears` .. `slot_in_programme` with no separate pre-legislative records prior to First Reading) or dispenses with a Parliamentary stage without formal sittings (e.g. `commons_report_stage` with 0 sittings on Bill `#4296` *Church Appointments (Repeal of Discriminatory Provision) Bill* when proceeding directly from Committee of the whole House to 3rd Reading), those stages are explicitly marked `"fast_tracked"` with a `Fast-Tracked` / `Fast-Tracked (0 sittings)` badge, `»` icon, and explanatory summary in the UI.
- **Files touched**:
  - `backend/src/westminster_weather/domain/bills.py`
  - `backend/tests/test_health_and_stages.py`
  - `frontend/src/app/page.tsx`
  - `docs/agent-logs/feat--fast-tracked-bypassed-stages.md`
- **Verification**: `uv run ruff check .`, `uv run pytest -q`, and `npx tsc --noEmit` all passed.

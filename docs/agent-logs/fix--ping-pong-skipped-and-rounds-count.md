## 2026-10-24 — Antigravity — `feat(ui): display skipped (0 rounds) and ping-pong round count on stage cards`

- **Branch**: `fix/ping-pong-skipped-and-rounds-count`
- **Summary**: Updated `frontend/src/app/page.tsx` to render `StageStatus === "skipped"` with a `Skipped (0 rounds)` badge and `0` icon when ping-pong did not occur, display `Done (N rounds)` / `Current (N rounds)` and the round summary when ping-pong sittings occurred, and count `"skipped"` alongside `"completed"` in the Phase 2 progress counter (`9/9 done`).
- **Files touched**:
  - `frontend/src/app/page.tsx`
  - `docs/agent-logs/fix--ping-pong-skipped-and-rounds-count.md`
- **Verification**: `npx tsc --noEmit` passed with 0 errors.

## 2026-10-24 — Antigravity — `fix(domain): mark unused ping_pong as skipped with 0 rounds and expose sitting_count`

- **Branch**: `fix/ping-pong-skipped-and-rounds-count`
- **Summary**: Added `StageStatus.SKIPPED` (`"skipped"`) and `sitting_count: int` to `TrackedStageNode` and `TrackedStageNodeSchema` so that when a UK Government Bill passes both Houses and reaches Royal Assent without any ping-pong sittings, the `ping_pong` stage is marked `"skipped"` with `sitting_count=0` (`"0 ping-pong rounds — Second House agreed to the Bill without exchanging amendments."`), and when ping-pong sittings occurred, the exact number of ping-pong rounds is exposed.
- **Files touched**:
  - `backend/src/westminster_weather/domain/bills.py`
  - `backend/src/westminster_weather/schemas/bills.py`
  - `backend/src/westminster_weather/main.py`
  - `backend/tests/test_health_and_stages.py`
  - `docs/agent-logs/fix--ping-pong-skipped-and-rounds-count.md`
- **Verification**: `uv run ruff check .` and `uv run pytest -q` passed.

## 2026-10-24 — Antigravity — `feat(phase3): ingest legislation.gov.uk post-Royal-Assent data for all 522 Acts into Neon`

- **Branch**: `feat/post-royal-assent-ingestion`
- **Summary**: Added `ActCommencementRow` (`act_commencements`) and `PostRoyalAssentEventRow` (`post_royal_assent_events`) tables in Neon Lakebase Postgres, implemented `clients/legislation.py` to ingest all 522 UK Government Acts (`ukpga`) and 4,673 post-Royal-Assent events (1,522 Commencement Order SIs, 347 Draft SIs, 807 substantive Laid SIs, and 431 Amending/Repealing instruments) from `legislation.gov.uk`, exposed `get_act_commencements(as_of)` and `get_post_royal_assent_events(as_of)` on `DatabaseInterface` for the forecasting model, and wired Phase 3 (`After Royal Assent`) stages into `evaluate_parliament_bill_from_db` and the Next.js UI.
- **Files touched**:
  - `backend/src/westminster_weather/models.py`
  - `backend/src/westminster_weather/db.py`
  - `backend/src/westminster_weather/clients/legislation.py`
  - `backend/src/westminster_weather/data/interface.py`
  - `backend/src/westminster_weather/domain/bills.py`
  - `backend/tests/test_health_and_stages.py`
  - `frontend/src/app/page.tsx`
  - `docs/agent-logs/feat--post-royal-assent-ingestion.md`
- **Verification**: `uv run ruff check .`, `uv run pytest -q`, and `npx tsc --noEmit` all passed.

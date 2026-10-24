# Westminster Weather

**Westminster Weather** is a tool for predicting whether UK Government Bills will pass and estimating the time until their commencement.

## Purpose

The primary purpose of Westminster Weather is to enable businesses to make informed decisions based on upcoming UK legislative changes and provide clear timelines for when those changes will take effect.

## Core Capabilities

- **Passage Prediction**: Predict whether UK Government Bills will pass into law.
- **Time-to-Commencement Estimation**: Estimate the timeline until a bill's commencement.
- **Business Decision Timelines**: Provide actionable timelines around legislative changes so businesses can plan and make informed decisions.

## Running Locally

The project has two parts that run side by side: a FastAPI backend in [`backend/`](./backend) and a Next.js frontend in [`frontend/`](./frontend). All commands below start from the repository root (the `westminster-weather` folder).

### Prerequisites

- **Python 3.12** and [**uv**](https://docs.astral.sh/uv/) (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Node.js 18.18+** and **npm**

### 1. Start the backend (port 8000)

```bash
cd backend
uv sync
uv run uvicorn westminster_weather.main:app --reload --port 8000
```

- Health check: <http://127.0.0.1:8000/health>
- Interactive API docs: <http://127.0.0.1:8000/docs>

Bill data is currently held in memory, so no database is needed to run the app. Settings in [`config.py`](./backend/src/westminster_weather/config.py) can be overridden with a `backend/.env` file.

### 2. Start the frontend (port 3000)

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The frontend proxies `/api/v1/*` requests to the backend at `127.0.0.1:8000`, so the backend must be running.

### Running tests and linting

```bash
cd backend
uv run pytest
uv run ruff check .
```

## Building the Bill Dataset (Stage 1)

Fetches every UK Government Bill (about 600, from 2006 onwards) and every stage sitting from the [Bills API](https://bills-api.parliament.uk), and writes them into the `stage_map`, `bills` and `stage_sittings` tables defined in `DATA_CONTRACT.md`. No API key is needed; a full run takes about a minute.

```bash
cd backend
uv run python -m westminster_weather.dataset.fetch_bills                   # -> backend/data/westminster_weather.db (SQLite)
uv run python -m westminster_weather.dataset.fetch_bills --limit 20        # quick test run
uv run python -m westminster_weather.dataset.fetch_bills --db "postgresql+psycopg://user:pass@localhost:5432/westminster_weather"
```

Re-running is safe: rows are updated by primary key, never duplicated. What is filled in:

- **`bills`**: title, originating House, session, first reading date, sponsoring department, committee type, carry-over date and `outcome` (`royal_assent`, `fell_dissolution`, `fell_session_end`, `withdrawn`, `defeated`, `in_progress`).
- **`stage_sittings`**: one row per sitting that has already happened. Scheduled future sittings are left out, so the data never includes anything dated after it was fetched.
- **`stage_map`**: every API stage, with the model state from the contract (or `NULL` if the model ignores it).

Not yet filled: `kings_speech_flag`, `size_clauses` and `size_pages` (not in the Bills API), the `outcome_date` of `fell_session_end` bills (prorogation dates are not in the Bills API), and the `sessions`/`parliaments` tables (the Bills API has no sessions endpoint).

## Development Workflow

All contributors and AI agents must follow the repository conventions defined in [`AGENTS.md`](./AGENTS.md) and [`CLAUDE.md`](./CLAUDE.md):

- **Branch Naming**: Use `<type>/<short-kebab-case-description>` (e.g., `feat/...`, `fix/...`, `docs/...`).
- **Pull Requests Only**: Never push directly to `main`. All changes must be merged into `main` via Pull Requests.

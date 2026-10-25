# Westminster Weather

**Westminster Weather** is a tool for predicting whether UK Government Bills will pass and estimating the time until their commencement.

## Purpose

The primary purpose of Westminster Weather is to enable businesses to make informed decisions based on upcoming UK legislative changes and provide clear timelines for when those changes will take effect.

## Core Capabilities

- **Passage Prediction**: Predict whether UK Government Bills will pass into law.
- **Time-to-Commencement Estimation**: Estimate the timeline until a bill's commencement.
- **Business Decision Timelines**: Provide actionable timelines around legislative changes so businesses can plan and make informed decisions.

## Running Locally

The project has two parts that run side by side: a FastAPI backend in [`backend/`](./backend) backed by **Neon Lakebase Postgres** and a Next.js frontend in [`frontend/`](./frontend). All commands below start from the repository root (`westminster-weather`).

### Prerequisites

- **Python 3.12** and [**uv**](https://docs.astral.sh/uv/) (`brew install uv` or `curl -LsSf https://astral.sh/uv/install.sh | sh`)
- **Node.js 18.18+** and **npm**
- **Neon CLI** (`npm i -g neon@latest`) with `.env.local` populated via `neon deploy` or `neon env pull` (`DATABASE_URL`, `DATABASE_URL_UNPOOLED`, `NEON_BRANCH`)

### 1. Start the backend (port 8000)

```bash
cd backend
uv sync
uv run python -m westminster_weather.model.cli train   # once: builds the passage model files (~10 s)
uv run uvicorn westminster_weather.main:app --reload --port 8000
```

The `train` step writes the fitted passage models to `backend/models/` (gitignored). Without it, `/api/v1/bills/{id}/forecast` returns an error ("model set 'live' not found"). Rerun it after new data is ingested.

- Health check (including Neon Postgres connectivity): <http://127.0.0.1:8000/health>
- Interactive API docs: <http://127.0.0.1:8000/docs>

### 2. Start the frontend (port 3000)

In a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Development assets are written to `frontend/.next-dev`; production builds use
`frontend/.next`. This keeps `npm run build` from replacing the CSS and JavaScript
used by a running development server. After changing this configuration, restart
`npm run dev` and refresh the browser.

Open <http://localhost:3000>. The frontend proxies `/api/v1/*` requests to the backend at `127.0.0.1:8000`, so the backend must be running.

### Running tests and linting

```bash
cd backend
uv run pytest
uv run ruff check .
```

## Development Workflow

All contributors and AI agents must follow the repository conventions defined in [`AGENTS.md`](./AGENTS.md) and [`CLAUDE.md`](./CLAUDE.md):

- **Branch Naming**: Use `<type>/<short-kebab-case-description>` (e.g., `feat/...`, `fix/...`, `docs/...`).
- **Pull Requests Only**: Never push directly to `main`. All changes must be merged into `main` via Pull Requests.

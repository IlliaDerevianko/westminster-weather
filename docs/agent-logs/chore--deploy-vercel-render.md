# Agent log: `chore/deploy-vercel-render`

## 2026-10-25 10:24 — chore(deploy): free demo deploy on Render (API) and Vercel (site)

- **What:** `render.yaml` blueprint for the backend (free web service, rootDir `backend`; build: uv sync from the lockfile, then `cli train` because `backend/models/` is gitignored; start: uvicorn on `$PORT`; health check `/health`; `DATABASE_URL`/`DATABASE_URL_UNPOOLED` entered in the dashboard). `frontend/next.config.mjs` forwards `/api/v1/*` to `BACKEND_URL` (default `http://127.0.0.1:8000`). README "Deploying a free demo" section; `.vercel/` gitignored.
- **Why:** User asked to deploy quickly and for free to Vercel and Render for a demo.
- **Checks:** Rehearsed from a clean worktree: Render's build commands (lockfile install, models trained) and start command (port 8011: `/health`, `/api/v1/bills`, `/api/v1/bills/4305`, outlook all 200); `BACKEND_URL=http://127.0.0.1:8011 next build` succeeded; `next start` on 3011: `/` -> 307 `/bills`, pages 200, icons 200, 588 bills through the proxy, bill page with outlook rendered. Memory in a fresh process: bill list, bill page and outlook peak ~260 MB (fits the 512 MB free plan).
- **Notes:** `/api/v1/bills/{id}/forecast` peaks ~757 MB on its first call (stage 3 XGBoost fitted in memory) and would exceed the free plan; the site does not call it. Free Render sleeps after 15 idle minutes (~1 minute cold start). Branched from `feat/tab-icon`.

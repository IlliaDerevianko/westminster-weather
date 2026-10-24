# Claude Code Guidelines (`CLAUDE.md`)

This file provides guidance to Claude Code and other AI coding assistants when working in the **Westminster Weather** repository.

See also [`AGENTS.md`](./AGENTS.md) for the complete repository agent instructions.

## Git Workflow & Branching Rules (Mandatory)

### 1. Consistent Branch Naming
Never work directly on `main`. Always create and switch to a branch named according to this convention before making changes:

```text
<type>/<short-kebab-case-description>
```

**Supported `<type>` prefixes:**
- `feat/` — New features (e.g., `feat/hourly-weather-view`)
- `fix/` — Bug fixes (e.g., `fix/null-wind-speed`)
- `docs/` — Documentation changes (e.g., `docs/add-agent-guidelines`)
- `refactor/` — Code refactoring (e.g., `refactor/weather-models`)
- `test/` — Adding or updating tests (e.g., `test/api-parser`)
- `chore/` — Maintenance, dependencies, configuration (e.g., `chore/update-deps`)
- `ci/` — CI/CD pipeline configuration (e.g., `ci/pr-checks`)

**Formatting rules:**
- Use lowercase kebab-case (`a-z`, `0-9`, `-`).
- Keep descriptions brief and meaningful.

### 2. Merge to `main` Only via Pull Requests
- **Do NOT push or merge directly into `main`.**
- Every change to `main` must go through a Pull Request:
  1. Branch off `main` (`git checkout -b <type>/<short-description>`).
  2. Stage and commit changes on your branch.
  3. Push the branch to `origin` (`git push -u origin <type>/<short-description>`).
  4. Open a Pull Request targeting `main` (`gh pr create --base main`).
  5. Merge to `main` exclusively via the Pull Request (`gh pr merge`).

### 3. Small Commits, Made Without Being Asked
- Commit after every logical unit of work (one function, fix, test file or doc section), **even if the user has not asked you to commit**. Do not batch a whole task into one commit.
- One concern per commit; run `uv run ruff check .` and `uv run pytest` (from `backend/`) first when code changed.
- Commit automatically, but **push or open a PR only when asked**. Never commit to `main`.

### 4. Agent Log on Every Commit
- Every commit must add an entry to `docs/agent-logs/<branch-name>.md` (`/` → `--`, e.g. `docs/agent-logs/feat--passage-model.md`), staged in the **same commit**, newest entry first.
- Entry: `## YYYY-MM-DD HH:MM — <commit summary>` followed by **What**, **Why**, **Checks** (commands and results) and optional **Notes**.
- Full rules and the entry template: [`AGENTS.md`](./AGENTS.md) sections 3–4.


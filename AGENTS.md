# Agent Guidelines (`AGENTS.md`)

This document defines the repository conventions and Git workflow rules that all AI agents and contributors must follow when working on **Westminster Weather**.

## 1. Branch Naming Convention

All work must be done on a dedicated branch with a consistent, descriptive name using the following format:

```text
<type>/<short-kebab-case-description>
```

### Allowed Branch Prefixes (`<type>`)
- `feat/` — New features or capabilities (e.g., `feat/weather-forecast-api`)
- `fix/` — Bug fixes (e.g., `fix/temperature-unit-conversion`)
- `docs/` — Documentation creation or updates (e.g., `docs/branch-conventions`)
- `refactor/` — Code restructuring without changing external behavior (e.g., `refactor/api-client`)
- `test/` — Adding or updating tests (e.g., `test/forecast-service-unit-tests`)
- `chore/` — Build tasks, tooling, dependencies, or configuration (e.g., `chore/setup-linter`)
- `ci/` — Continuous integration and workflow changes (e.g., `ci/github-actions-tests`)

### Naming Rules
- Use **lowercase alphanumeric characters and hyphens (`-`) only** in the description (kebab-case).
- Keep branch names concise, clear, and scoped to a single task.
- Do not use underscores, spaces, or uppercase characters.

---

## 2. Pull Request & Merge Policy

### No Direct Commits to `main`
- **`main` is a protected trunk branch.** Never commit or push directly to `main`.
- All changes—regardless of size—must be developed on a feature/topic branch following the naming convention above.

### Merge Only via Pull Requests
1. Create a new branch from the latest `main`:
   ```bash
   git checkout main
   git pull origin main
   git checkout -b <type>/<short-description>
   ```
2. Commit your changes with clear, descriptive commit messages.
3. Push the branch to the remote repository:
   ```bash
   git push -u origin <type>/<short-description>
   ```
4. Open a Pull Request (PR) targeting `main` (e.g., via `gh pr create`).
5. Merge into `main` **only** through the Pull Request after review and verification.
6. Delete the feature branch after the PR is merged.

---

## 3. Commit Frequency (Small Commits by Default)

Agents must commit **small, frequent commits on their own initiative**, even when the user has not asked for a commit.

- Commit as soon as one logical unit of work is complete (one function, one fix, one test file, one doc section). Do not wait until the end of a task or for the user to say "commit".
- One concern per commit: never bundle unrelated changes. If a diff needs "and" to describe it, split it.
- Before each commit, run the checks that apply (from `backend/`: `uv run ruff check .` and `uv run pytest`). If a check fails and the commit is still worth keeping, say so in the commit message and in the log entry.
- Use the commit message format `<type>(<scope>): <summary>`, with the same `<type>` prefixes as branch names, and a short body explaining *why*.
- Committing is automatic; **pushing and opening Pull Requests are not**. Push or open a PR only when the user asks (or a task plan says so). Never commit to `main` (section 2).

---

## 4. Agent Log (One Entry per Commit)

Every commit made by an agent must include an entry in the branch's log file, so a human can see what was done and why without reading diffs.

- **File:** `docs/agent-logs/<branch-name>.md`, with `/` in the branch name replaced by `--` (e.g. branch `feat/passage-model` → `docs/agent-logs/feat--passage-model.md`). Create it with the first commit on the branch. One file per branch avoids merge conflicts between parallel branches.
- **When:** add the entry in the **same commit** as the change it describes (stage the log file together with the code).
- **Order:** newest entry at the top.
- **Entry format:**

  ```markdown
  ## YYYY-MM-DD HH:MM — <commit summary, same as the commit message's first line>

  - **What:** what changed, in plain English (1–3 bullets).
  - **Why:** the reason or the instruction that led to it.
  - **Checks:** commands run and their results (e.g. `pytest`: 36 passed; `ruff`: clean), or "none (docs only)".
  - **Notes:** anything surprising, assumptions made, follow-ups or known problems. Omit if none.
  ```

- Keep entries short and factual. Report failures and skipped checks honestly.

---

## 5. Data Scope: Government Bills Only

**This project accepts UK Government Bills only (Bills API `BillType = 1`).** Never download, store, use or display any other bill type (Private Members' Bills, Private Bills, Hybrid Bills, etc.), including for helper data such as the sitting-day calendar. Full rules: [`DATA_SCOPE.md`](./DATA_SCOPE.md). If anything conflicts with this rule, `DATA_SCOPE.md` wins: stop and tell the user.

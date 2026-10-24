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

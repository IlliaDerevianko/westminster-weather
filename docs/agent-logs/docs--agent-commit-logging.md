# Agent log: `docs/agent-commit-logging`

## 2026-10-24 20:17 — docs: require small commits and a per-commit agent log

- **What:** added AGENTS.md sections 3–4 and CLAUDE.md sections 3–4: agents commit small, frequent commits without being asked, and add an entry to `docs/agent-logs/<branch>.md` in every commit. Added `docs/agent-logs/README.md` and this log.
- **Why:** requested by Aleks, so agent work is easy to follow commit by commit.
- **Checks:** none (docs only).
- **Notes:** one log file per branch was chosen to avoid merge conflicts between parallel branches. Pushing and PRs still happen only when asked.

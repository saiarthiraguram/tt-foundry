---
name: create-pr
description: Create a properly-formatted GitHub pull request for tt-xla. Handles pre-flight branch validation, area detection from changed paths, PR title in "[Area] imperative" form, PR body from template, CODEOWNERS-derived reviewers, label mapping, branch push, and initial CI status report. Use when the user says "create a PR", "open a PR for these changes", or invokes /create-pr.
---

# create-pr — tt-xla PR Creation

Guides Claude Code through creating a structured PR for the tt-xla repo. Use when the user has staged commits on a feature branch and is ready to open a PR.

Usage: `/create-pr [area]` — optional `area` skips auto-detection (e.g. `vLLM`, `CI`, `pjrt`, `Test Infra`).

## Phases

Run sequentially. Stop and report to the user if any phase fails — do not silently continue.

### 1. Pre-flight validation
- `git branch --show-current` — must NOT be `main`. If on main, abort and tell the user to create a feature branch.
- `git status --porcelain` — if uncommitted/untracked changes exist, ask the user whether to stash, commit, or abort.
- `git log main..HEAD --oneline` — must have ≥1 commit ahead of main. If empty, abort.
- `git fetch origin main` then `git log origin/main..HEAD --oneline` — confirm branch isn't already merged.

### 2. Linked issue
- Ask the user: "Existing GitHub issue number, or shall I draft one from the diff?"
- If existing: validate via `gh issue view <N> --repo tenstorrent/tt-xla` and capture the URL.
- If drafting: summarize from commit messages + `git diff main..HEAD --stat` and `gh issue create --repo tenstorrent/tt-xla` (confirm title/body with user before creating).

### 3. Area detection & PR title
- Read `references/commit-template.md` for the prefix table.
- If user passed `[area]` argument, use it verbatim.
- Otherwise, run `git diff main..HEAD --name-only` and apply the Path-to-Prefix Lookup. If a single area dominates, pick that prefix; if multi-area, use bare-verb style.
- Generate title: `[Area] <imperative verb phrase>` — ≤72 chars, no trailing period, no `feat:`/`fix:` conventional-commit prefixes.
- **Confirm the title with the user before proceeding.**

### 4. PR body
- Load `references/pr-body-template.md`.
- Auto-fill:
  - **Ticket** — link from phase 2
  - **Problem description** — derived from issue / commit messages
  - **What's changed** — summarize commits + `git diff --stat`
  - **Test commands run** — suggest tests based on changed paths (e.g. `tests/torch/` → `pytest -v tests/torch -m single_device`)
  - **Checklist** — pre-tick boxes you can verify (pre-commit, SPDX); leave human-judgment items unticked

### 5. Reviewers
- Read `.github/CODEOWNERS` (use the table in `references/pr-body-template.md` as a quick reference if CODEOWNERS isn't checked out).
- For each changed file, find the most-specific matching owner block.
- Union the owners, then always include at least one global (`*`) owner.
- Cap at 15 reviewers. Strip the requesting user from the list.

### 6. Labels
- Map detected area → labels: `vLLM` → `vllm`, `CI` → `ci`, `pjrt` → `pjrt`, `Test Infra` → `testing`, `FX fusing` → `fx-fusing`, `build` → `build`, `tools` → `tooling`.
- `gh label list --repo tenstorrent/tt-xla` to check existence. Create missing labels only if the user confirms.

### 7. Push & create
- `git push -u origin <branch>` (use `-u` only if upstream not set).
- `gh pr create --repo tenstorrent/tt-xla --base main --title "<title>" --body-file <tmpfile> --reviewer <list> --label <list>`.
- Pass the PR body via `--body-file` (a HEREDOC tmp file), never via `--body` inline, to preserve formatting.

### 8. Post-create
- Echo the PR URL.
- `gh pr checks <N>` — report which workflows started.
- Remind the user they can run the ci-review skill (or `gh pr checks --watch`) to track CI later.

## Safety rules

- **Never** push to `main` or force-push.
- **Never** create the PR before the user confirms the title.
- **Never** request review from the PR author themselves.
- If `gh` is not authenticated, stop and tell the user to run `gh auth login` — do not try to work around it.
- Match the scope of what the user asked: if they say "just push and open the PR, skip the issue", skip phase 2.

## References

- `references/commit-template.md` — area prefixes, title rules, examples
- `references/pr-body-template.md` — PR body template + CODEOWNERS quick reference

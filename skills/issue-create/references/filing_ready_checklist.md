# Filing-ready checklist (before `gh-command.sh`)

## `draft.md` must NOT contain

- `_Ran gh issue list_` or search query strings
- `gh unavailable` / `unauthenticated` narration
- Paths under `.claude/issues/` or "this draft is v2"
- `manifest.json` / skill / process meta

## `draft.md` Related issues must contain

- Real issues: `#NNNN` or `org/repo#NNNN` with one-line context
- **Or** a single clean negative line:
  `No similar issues found in tenstorrent/tt-xla at time of investigation.`
- Parent tracker / upstream owner when known (#4705, tt-mlir#3370)
- Duplicate guidance: "expand #4784 in place" not "this draft is investigated form"

## `manifest.json` dev-only fields

- `related_issue_search` — queries + outcome for the developer
- `dev_adds_manually` — Type: Bug, milestone, assignee, parent link
- `recommended_next_steps` — echo of escalation/diagnosis (optional)

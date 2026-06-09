---
name: issue-create
description: Draft a GitHub issue package from a bringup escalation, pytest log, or manual investigation. Writes .claude/issues/<YYYYMMDD>_<slug>/ with title.txt, draft.md, manifest.json, and a ready-to-paste gh-command.sh. Never files issues automatically — developer reviews and runs gh manually. Use when bringup ESCALATES, when filing op/runtime bugs from model tests, or when the user says "draft an issue" or invokes /issue-create.
allowed-tools: Bash Read Write Edit Grep Glob
---

# issue-create — GitHub issue draft package

Produce a **reviewable issue draft** under `.claude/issues/` in the consuming repo
(e.g. `tt-xla`). The skill **never** runs `gh issue create` — the developer
reviews `draft.md`, edits if needed, then runs `gh-command.sh` manually.

Usage:

```
/issue-create [--from-bringup <safe_key>] [--from-log <path>] [--slug <name>] [--repo tt-xla|tt-mlir|tt-metal] [--category oom|pcc|compile|runtime|missing_op]
```

- `--from-bringup` — read `.claude/bringup/<safe_key>/escalation.json`,
  `bringup_steps.txt`, and referenced `logs/*.log`.
- `--from-log` — primary failure log when not using bringup artifacts.
- `--slug` — short snake_case label for the folder (default: infer from model/op).
- `--repo` — target GitHub repo (default: infer from `references/repo_routing.md`).
- `--category` — override manifest category when inference is ambiguous.

## Output layout

```
.claude/issues/<YYYYMMDD>_<slug>/
├── title.txt          # one line — becomes gh --title
├── draft.md           # full issue body (GitHub-flavored Markdown)
├── manifest.json      # machine metadata + dev_adds_manually checklist
└── gh-command.sh      # copy-paste gh issue create (NOT auto-run)
```

Folder date is **UTC** `YYYYMMDD` at draft time. Slug must be filesystem-safe
(lowercase, underscores, ≤ 48 chars).

## Steps

### 1. Gather inputs

**From bringup (`--from-bringup`):**
- `escalation.json` — `result`, `reason`, `arch`, `blocker`, `what_works`,
  `recommended_next_steps`, log paths (when present)
- `diagnosis_<component>.json` — when no `escalation.json`; use `verdict`,
  `experiments`, `root_cause`, `recommendations` / `recommended_next_steps`
- `bringup_steps.txt` — timeline, component, iteration table
- `state.json` — `model_key`, HF id, component list, `failure_reasons`
- Latest failing `logs/*_run.log` or `logs/*_verify.log` (grep `TT_FATAL`, `AssertionError`, `PCC`, `Out of Memory`)

**From log (`--from-log`):**
- Read the full log; extract error class, stack top, buffer sizes, op names, PCC values
- If log references a `model_key` or pytest node id, capture it

**Always search for related issues** before drafting (use the target `repo` from
step 2 — not always `tt-xla`):

```bash
# Adjust query from failure signature (op name, model family, error class)
gh issue list --repo tenstorrent/tt-xla --search "grid_sample OOM" --limit 10 2>/dev/null || true
```

**When `gh` works:** record hits in `manifest.json` → `related_issues` /
`upstream_issues` and cite them in `draft.md` § Related issues as `#NNNN` or
`tenstorrent/tt-xla#NNNN` with one-line context (phase, component, what differs).

**When search returns no hits:** in `draft.md` write only:
`No similar issues found in <repo> at time of investigation.` — **do not** paste
the queries or `_Ran gh issue list …_` into the body. Store queries in
`manifest.json` → `related_issue_search`.

**When `gh` is unavailable:** `draft.md` still uses filing-ready wording
(`No similar issues identified in <repo> (GitHub search pending before filing).`).
Put queries in `related_issue_search`; add `dev_adds_manually` reminder to
confirm on GitHub before filing. Never narrate the tooling failure in `draft.md`.

### 2. Classify and route

Use `references/repo_routing.md`:

| Symptom | Typical repo | manifest `category` |
|---------|--------------|---------------------|
| Model pytest PCC / atol on tt-xla test | `tenstorrent/tt-xla` | `pcc` |
| DRAM/L1 OOM during model run on tt-xla | `tenstorrent/tt-xla` | `oom` |
| TTMLIR / StableHLO lowering, missing op | `tenstorrent/tt-mlir` | `compile` |
| tt-metal kernel / program factory (after isolated repro) | `tenstorrent/tt-metal` | `runtime` |

Set `manifest.json` → `"type": "investigated"` when you ran experiments;
`"type": "repro_only"` when only a pytest repro exists.

### 3. Write title.txt

One line, imperative, specific. Include model or op and failure mode.

Examples (from validated drafts):
- `grid_sample: ttir.floor decomposition fails to lower — "BinaryOpType cannot be mapped to BcastOpMath"`
- `Playground v2.5 text_encoder_2: whole-model PCC ~0.971 from cumulative error across 32 CLIPTextModelWithProjection encoder blocks`

Rules: ≤ 120 chars; no trailing period; prefix with model/op name when known.

### 4. Write draft.md

Follow `references/draft_body_template.md`. Required sections:

1. **Describe the bug** (or **Summary** for upstream-only bugs) — model key, arch, observed vs expected metric
2. **Call chain** — ASCII tree from model → submodule → failing op
3. **Key observations** — bullet facts; cite buffer sizes, PCC per layer, what changed vs related issues
4. **Experiments / sanities** — table when multiple repros exist
5. **Steps to reproduce** — fenced `bash` with exact pytest / checkout commands
6. **Logs** — absolute paths to log files on the host; paste 5–15 line excerpt of the decisive error
7. **Expected behavior**
8. **Suggested next steps** — **required** when `--from-bringup` and
   `escalation.json` → `recommended_next_steps` **or**
   `diagnosis_<component>.json` → `recommendations` is non-empty. Numbered list;
   preserve order. If absent, infer from `blocker` / `failure_reasons` /
   `diagnosis` root cause.
9. **Related issues** — filing-ready bullets only (see
   `references/draft_body_template.md` §9). Hits, upstream owners, parent
   trackers, then optionally `No similar issues found in <repo>…`. **Never**
   include search-query narration or internal draft paths.
10. **Notes** (optional) — arch, classification, branch. No draft-version meta.

For **historical / non-reproducible** bugs, add an upfront status note and list
`verify still-reproducible before filing` in `dev_adds_manually`.

### 5. Write manifest.json

Schema: `references/manifest_schema.md`. Minimum fields:

```json
{
  "repo": "tenstorrent/tt-xla",
  "type": "investigated",
  "category": "oom",
  "arch": "n150",
  "model_key": "oft/pytorch-single_device-inference",
  "suggested_labels": ["bug"],
  "upstream_issues": [],
  "related_issues": ["tenstorrent/tt-xla#3419"],
  "log_paths": ["/tmp/oft4244_model.log"],
  "draft_body": "draft.md",
  "title_file": "title.txt",
  "dev_adds_manually": [
    "milestone",
    "assignee",
    "parent_tracker",
    "Type: Bug (tt-xla UI field — gh CLI cannot set)"
  ]
}
```

Add optional fields when known: `branch`, `test_path`, `pcc_observed`, `pcc_required`,
`atol_observed`, `notes`, `related_issue_search` (when `gh` search could not run),
`recommended_next_steps` (echo of `escalation.json` when present).

### 6. Write gh-command.sh

Executable shell script. **Must include** header comment:
`# Do NOT run automatically — issue-create skill never files issues on behalf of the developer.`

Template:

```bash
#!/usr/bin/env bash
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

gh issue create \
  --repo tenstorrent/tt-xla \
  --title "$(cat "$DIR/title.txt")" \
  --label bug \
  --body-file "$DIR/draft.md"

# Optional — add manually:
#   --assignee @me
#   --milestone "<name>"
#   link parent tracker in GitHub UI after creation
```

Use `$DIR`-relative paths so the script works from any cwd.
`chmod +x` the script.

### 7. Report to user

Print a one-screen summary:

```
[issue-create] drafted: .claude/issues/<folder>/
  repo:     tenstorrent/tt-xla
  category: oom
  title:    <title.txt first line>
  logs:     <paths>
  related:  #3419 | PENDING (gh unavailable — see manifest related_issue_search)
  next_steps: 3 items from escalation.json

Next: review draft.md → run gh-command.sh manually (or edit flags first).
dev_adds_manually: milestone, assignee, parent_tracker, Type field
If related_issues empty: run gh searches from manifest.related_issue_search first.
```

## Hard rules

- **Never** run `gh issue create` (or open a browser) automatically.
- **Never** file duplicate issues without checking `related_issues` search (or
  completing the search deferred via `related_issue_search` when `gh` was down).
- **Always** emit `### Suggested next steps` when `--from-bringup` and
  `escalation.json` or `diagnosis_*.json` supplies recommendations.
- **`draft.md` is filing-ready** — no `Ran gh issue list`, no search strings in
  Related issues, no `.claude/issues/` paths, no "v2 draft" meta.
- **Never** omit log paths — always cite at least one on-disk log the developer can attach.
- **Read-only on YAML** — this skill does not mutate `test_config_inference_single_device.yaml`.
- Prefer **investigated** drafts (sanities, allocation sizes, call chain) over bare repro dumps.
- tt-xla issues often need **Type: Bug** set in the GitHub UI — always list in `dev_adds_manually`.

## Integration with bringup FSM

Invoke after:
- `/model-bringup` or `/model-bringup-multichip` writes `escalation.json` with `ESCALATED`
- `/model-bringup-diagnose` sets `escalation_skill` but human filing is still needed for upstream bugs
- `/model_issue_pick` confirms an XFAIL still reproduces and user wants a fresh issue draft

Pair with `/failure_summary` to pick which XFAIL entry to draft next.

## References

- `references/draft_body_template.md` — section-by-section body guide + examples
- `references/manifest_schema.md` — full `manifest.json` field list
- `references/repo_routing.md` — which repo owns which failure class
- `references/filing_ready_checklist.md` — pre-flight: no process meta in `draft.md`

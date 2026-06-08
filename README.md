# tt-foundry

Claude Code skills that drive the end-to-end model bringup pipeline for
Tenstorrent hardware (PJRT / tt-xla / tt-forge-models).

## What's here

```
skills/
├── model-bringup/                    # E2E orchestrator (FSM)
├── model-bringup-scaffold/           # VALIDATE stage — loader scaffold + state.json (HuggingFace default)
├── model-bringup-scaffold-github/    # VALIDATE variant — model source lives on GitHub (vendor or port)
├── model-bringup-scaffold-pipeline/  # VALIDATE variant — multi-component HF DiffusionPipeline (per-component loaders, shard specs)
├── model-bringup-overview/           # OVERVIEW stage — CPU sanity + golden.pt capture, model_overview.md
├── model-bringup-run/                # FIRST_RUN / VERIFY stages — pytest under 5-min budget
├── model-bringup-diagnose/           # DIAGNOSE stage — pattern-match failure log → JSON
├── model-bringup-repair/             # REPAIR stage — apply strategy (monkey_patch, runtime_debug, …)
├── model-bringup-config-update/      # CONFIG_UPDATE stage — write final YAML status
├── model-bringup-finalize/           # FINALIZE stage — multi-arch reverify, pre-commit, PR draft
├── runtime-failure-debugger/         # Op-level bisect invoked by runtime_debug repair strategy
├── graph-break-analysis/             # Auxiliary: torch.compile graph-break investigation
├── issue-create/                     # Draft GitHub issue packages (.claude/issues/) — never auto-files
├── model_issue_pick/                 # XFAIL re-triage (single entry)
├── failure_summary/                  # YAML digest of all KNOWN_FAILURE_XFAIL entries
├── potential_new_models/             # SOTA bringup-candidate suggester
├── triage-unpack-forward-output/     # Triage FAILED_FE_COMPILATION "no unpack_forward_output" cases
├── code-reviewer/                    # C++/Python code review checklist + pre-PR self-review (lint/SPDX/tests/commits)
└── create-pr/                        # Open a tt-xla PR (area-prefixed title, body template, CODEOWNERS)
```

## High-level flow

```
                            ┌──────────────────────┐
     /model-bringup <key> ──▶│      model-bringup   │
                            │     (orchestrator)   │
                            └──────────────────────┘
                                       │
   ┌─────────┬─────────┬─────────┬─────┴────┬─────────┬───────────────┬──────────┐
   ▼         ▼         ▼         ▼          ▼         ▼               ▼          ▼
scaffold  overview   run    diagnose     repair    verify     config-update   finalize
   │                                        │
   ├─ scaffold-github  (GitHub-hosted)      ▼
   └─ scaffold-pipeline (DiffusionPipeline) runtime-failure-debugger
                                            │
                                            ▼
                                  graph-break-analysis
```

Auxiliary skills hang off the same FSM but enter from different states:

- `issue-create` — draft `title.txt` + `draft.md` + `manifest.json` + `gh-command.sh`
  under `.claude/issues/` from bringup escalations or pytest logs (developer runs `gh` manually)
- `model_issue_pick` — re-triage a single existing `KNOWN_FAILURE_XFAIL` entry
- `failure_summary` — digest the XFAIL list for triage / sharing
- `potential_new_models` — suggest the next wave of bringup candidates
- `triage-unpack-forward-output` — fix one training-test failure pattern
  (`FAILED_FE_COMPILATION` with "no unpack_forward_output handler")
- `code-reviewer` — review a diff (C++/Python checklist, standards, antipatterns)
  or run the mechanical pre-PR self-review (lint / SPDX / test coverage / commit messages)
- `create-pr` — open the PR after `finalize` produces the branch + body draft

## Consuming tt-foundry from another repo

The skills are plain `SKILL.md` files under `skills/<skill-name>/`. Two ways
to use them from a sibling project (e.g. `tt-xla`):

1. **Submodule.** Add `tt-foundry` as a git submodule (e.g. under
   `third_party/tt-foundry`), then point your `.claude/settings.json` plugin
   list (or symlinks in `.claude/skills/`) at the skills directory.
2. **Direct copy.** Copy individual `skills/<name>/SKILL.md` into the
   consuming repo's `.claude/skills/`. Faster, but creates drift — prefer
   the submodule path for anything beyond a one-off experiment.

## Repo invariants

- **No code, only prompts.** This repo contains `SKILL.md` files and docs.
  All actual code (pytest invocations, YAML parsing, etc.) lives in the
  consuming repo (`tt-xla` today).
- **Read-only on the consuming repo's YAML.** The skills can recommend
  changes, but they only mutate YAML when explicitly invoked with `--apply`
  or equivalent.
- **One source of truth per skill.** If you find yourself editing the same
  skill in both `tt-foundry` and a consuming repo's `.claude/skills/`,
  upstream the change to `tt-foundry` and drop the local copy.

## License

(Add license file when this repo gets a home / remote.) #To-DO

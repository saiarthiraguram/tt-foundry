# tt-foundry

Claude Code skills that drive the end-to-end model bringup pipeline for
Tenstorrent hardware (PJRT / tt-xla / tt-forge-models).

## What's here

```
skills/
├── model-bringup/                    # E2E orchestrator (FSM)
├── model-bringup-scaffold/           # VALIDATE stage  — loader scaffold + state.json
├── model-bringup-run/                # FIRST_RUN / VERIFY stages — pytest under 5-min budget
├── model-bringup-diagnose/           # DIAGNOSE stage — pattern-match failure log → JSON
├── model-bringup-repair/             # REPAIR  stage  — apply strategy (monkey_patch, …)
├── model-bringup-config-update/      # CONFIG_UPDATE  — write final YAML status
├── runtime-failure-debugger/         # Op-level bisect for runtime_debug repairs
├── graph-break-analysis/             # Auxiliary: torch.compile graph-break investigation
├── model_issue_pick/                 # XFAIL re-triage (single entry)
├── failure_summary/                  # YAML digest of all KNOWN_FAILURE_XFAIL entries
└── potential_new_models/             # SOTA bringup-candidate suggester
```

## High-level flow

```
                          ┌──────────────────────┐
   /model-bringup <key> ──▶│       model-bringup  │
                          │      (orchestrator)  │
                          └──────────────────────┘
                                     │
   ┌──────────┬──────────┬───────────┼────────────┬────────────────┐
   ▼          ▼          ▼           ▼            ▼                ▼
scaffold    run     diagnose      repair       verify       config-update
                                     │
                                     ▼
                           runtime-failure-debugger
                                     │
                                     ▼
                            graph-break-analysis
```

The three newer skills (`model_issue_pick`, `failure_summary`,
`potential_new_models`) hang off the same FSM but enter from different
states: re-triage existing XFAIL entries, digest the YAML for triage, and
suggest the next bringup wave respectively.

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

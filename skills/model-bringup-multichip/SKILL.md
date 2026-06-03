---
name: model-bringup-multichip
description: Promotion-only multichip tensor-parallel bringup for PyTorch models. Runs only after single-chip /model-bringup exhausts n150 and p150 with weight-bound failure. Implements Megatron TP (get_mesh_config + load_shard_spec). Use when promotion.json exists or user explicitly continues from weight_predicted scaffold.
allowed-tools: Bash Read Write Edit Grep Glob Task Agent
---

# Model Bringup — Multichip TP (promotion-only)

See `references/` for DRAM budgets, OOM classification, arch eligibility, dtype ladder, and shard templates.

## Entry gate

Do **not** start cold. Require one of:
- `.claude/bringup/<safe_key>/promotion.json` from `/model-bringup`, or
- `--force-multichip` **and** `weight_fit.json` shows `weight_predicted` on all eligible arches.

## Invocation

`/model-bringup-multichip <model_key> [--component <name>] [--arch n300-llmbox] [--from-promotion]`

## FSM

```
VALIDATE_TP  →  FIRST_RUN_TP  →  (fail) DIAGNOSE + classify-oom  →  REPAIR_SHARD  →  VERIFY_TP
                      ↓ pass
              CONFIG_UPDATE_TP (--apply)  →  FINALIZE
```

## Delegate skills

| Stage | Skill |
|-------|--------|
| VALIDATE_TP | `model-bringup-scaffold-torch-tp` |
| FIRST_RUN_TP / VERIFY_TP | `model-bringup-run-torch-tp` |
| DIAGNOSE | `model-bringup-diagnose` + `model-bringup-classify-oom` |
| REPAIR_SHARD | `model-bringup-repair-shard-spec` |
| CONFIG_UPDATE_TP | `model-bringup-config-update-torch-tp` |
| FINALIZE | `model-bringup-finalize` |

Read `references/arch_eligibility.md` — single-chip on n150 **and** p150 must be exhausted (weight-bound) unless `--force-multichip`.

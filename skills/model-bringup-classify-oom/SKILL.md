---
name: model-bringup-classify-oom
description: Classify HW failure logs into activation vs weight-bound vs arch_insufficient for single-chip model bringup. Drives whether to REPAIR on same arch, retry p150, or write promotion.json for multichip. Invoke after FIRST_RUN failure or from model-bringup-diagnose when OOM suspected.
allowed-tools: Bash Read Write Grep
---

# Model Bringup — Classify OOM

## Invocation

`/model-bringup-classify-oom --log <path> [--arch n150|p150] [--model-key <key>]`

Reads:
- `.claude/bringup/<safe_key>/weight_fit.json`
- pytest log at `<path>`
- optional `logs/iter_*_result.json`

## Output

Write `.claude/bringup/<safe_key>/last_oom_classification.json` and return:

```json
{
  "class": "activation | weight_runtime | weight_predicted | arch_insufficient | dtype_only | shardy_fe | fe_pcc | other",
  "arch": "n150",
  "evidence": "short quote from log",
  "promote_multichip": false,
  "retry_arch": null,
  "recommended_repair": "reduce_resolution | dtype_bf16_activations | fix_mesh | escalate | none"
}
```

Full rules: `model-bringup-multichip/references/oom_classification.md`.

## Decision order

1. If log has FE compile / Shardy / PCC keywords → `fe_pcc` or `shardy_fe` (not promotion).
2. If `arch` is `n150`, log has DRAM OOM, and `weight_fit` shows `eligible_archs` contains `p150` with `fits_bf16` or `fits_fp32` for p150 but n150 failed → `arch_insufficient`, `retry_arch: "p150"`, `promote_multichip: false`.
3. If OOM mentions intermediate / activation / buffer >> weight estimate → `activation`, `recommended_repair: reduce_resolution` (or tiling/flags per model).
4. If OOM on weight load / param allocation and weight exceeds budget on this arch → `weight_runtime`, `promote_multichip: false` until orchestrator confirms all arches tried.
5. If scaffold had `weight_predicted` for this arch only → `weight_predicted`.

Set `promote_multichip: true` only when orchestrator sets flag `--check-promotion` after all `eligible_archs` exhausted with weight_runtime/predicted.

## Bringup steps log

Append one line to `bringup_steps.txt`:

```
CLASSIFY_OOM: arch=<arch> class=<class> promote=<bool> repair=<recommended_repair>
```

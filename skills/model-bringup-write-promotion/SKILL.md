---
name: model-bringup-write-promotion
description: Writes promotion.json when single-chip bringup exhausts all eligible arches with weight-bound failure. Invoked by model-bringup orchestrator before handing off to model-bringup-multichip.
allowed-tools: Bash Read Write
---

# Model Bringup — Write Promotion

## Invocation

`/model-bringup-write-promotion <model_key> [--component <name>]`

## Preconditions

- `state.arch_results` records **weight_bound** (or equivalent) for every arch in
  `state.arch_queue` / `weight_fit.json` → `eligible_archs`.
- **No** arch has `passed`.
- Dtype ladder completed per arch (fp32 → activation repair → bf16).

## Steps

1. Read `.claude/bringup/<safe_key>/state.json` and `weight_fit.json`.
2. Run the helper script (preferred over hand-written JSON):
   ```bash
   python model-bringup-multichip/scripts/write_promotion.py \
     --bringup-dir .claude/bringup/<safe_key> \
     --model-key "<model_key>" \
     --component "<component>"
   ```
   Default `--component` = first component in `weight_fit.json` or `model`.
3. Verify output matches `model-bringup-multichip/references/promotion_schema.md`.
4. Update `state.json`: `stage: "promotion_pending"`, append history:
   ```json
   { "stage": "write_promotion", "result": "promotion_pending", "details": { "path": "promotion.json" } }
   ```
5. Append to `bringup_steps.txt`:
   ```
   WRITE_PROMOTION: eligible=<archs> weight_bytes_bf16=<N> suggested_chips=<N>
   ```

## Output

```
[write-promotion] promotion.json written
  eligible_archs_tried: [n150, p150]
  suggested_multichip: n300-llmbox x4
  Next: /model-bringup-multichip <model_key> --from-promotion
```

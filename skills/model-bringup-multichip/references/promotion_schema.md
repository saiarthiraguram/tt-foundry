# promotion.json schema

Written by `/model-bringup` when all eligible single-chip arches are weight-bound.

Path: `.claude/bringup/<safe_key>/promotion.json`

```json
{
  "model_key": "...",
  "component": "transformer",
  "reason": "weight_bound_all_eligible_arches",
  "eligible_archs_tried": ["n150", "p150"],
  "arch_results": {
    "n150": { "result": "weight_bound", "dtype_last": "bf16", "log": "logs/iter_n150_run.log" },
    "p150": { "result": "weight_bound", "dtype_last": "bf16", "log": "logs/iter_p150_run.log" }
  },
  "weight_bytes_bf16": 20000000000,
  "suggested_multichip_arch": "n300-llmbox",
  "suggested_chip_count": 4,
  "tt_xla_sha": "abc1234",
  "created_at": 1710000000
}
```

`/model-bringup-multichip` refuses to start without this file unless `--force-multichip` + scaffold `weight_predicted`.

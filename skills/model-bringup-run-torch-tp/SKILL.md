---
name: model-bringup-run-torch-tp
description: FIRST_RUN_TP and VERIFY_TP for multichip PyTorch models. Runs component tests or test_all_models_torch tensor_parallel nodes with SPMD env.
allowed-tools: Bash Read Write
---

# Run — Torch TP

## Invocation

`/model-bringup-run-torch-tp <model_key> [--arch n300-llmbox] [--iteration N] [--timeout SECS]`

## Environment

```bash
export TT_XLA_ARCH=<arch>
export TT_VISIBLE_DEVICES=<from scaffold_multichip.json>
export CONVERT_SHLO_TO_SHARDY=1   # DiT / video transformers when needed
# xr.use_spmd() in test or via test body
```

## Test discovery

1. Prefer `state.details.test_path` from weight_fit / component scaffold.
2. Else collect `tests/runner/test_models.py` for `tensor_parallel-inference`.
3. Mochi MVP: `tests/torch/models/mochi/test_transformer.py::test_transformer_sharded`

## Logging

Same as `model-bringup-run`: `logs/iter_<N>_run.log`, json-report, `last_stage` from `._bringup_stage.txt`.

Append history with `stage: first_run_tp`, `arch`, `details.chip_count`.

---
name: model-bringup-run-torch-tp
description: FIRST_RUN_TP and VERIFY_TP for multichip PyTorch models. Runs component tests or test_all_models_torch tensor_parallel nodes with SPMD env.
allowed-tools: Bash Read Write
---

# Run — Torch TP

## Invocation

`/model-bringup-run-torch-tp <model_key> [--arch n300-llmbox] [--iteration N] [--timeout SECS]`

## Host gate (before pytest)

Read active component from `weight_fit.json` → `parallelism_mode` and
`scaffold_multichip.json` → `chip_count` (expected mesh).

```bash
MODE=$(jq -r '.components[] | select(.name=="<component>") | .parallelism_mode' weight_fit.json)
MESH=$(jq -r '.chip_count // 0' scaffold_multichip.json)
python .../probe_host.py --parallelism-mode "$MODE" --expected-mesh-chips "$MESH" -o host_probe.json

VIS=$(jq -r '.recommended_tt_visible_devices.multichip_bringup' host_probe.json)
if [ "$(jq -r '.can_run_component // .can_run_multichip_bringup' host_probe.json)" != "true" ]; then
  jq -r '.component_skip_reason // .multichip_skip_reason // .tt_visible_devices_env_skip_reason' host_probe.json
  # STOP — tell user to fix TT_VISIBLE_DEVICES, install tt-smi, or change host
fi
export TT_XLA_ARCH=<arch>
export TT_VISIBLE_DEVICES="$VIS"   # from tt-smi boards — e.g. 0,1,2,3 when 8 runtime chips
export TT_XLA_SPMD=1
export CONVERT_SHLO_TO_SHARDY=1
```

`TT_VISIBLE_DEVICES` from **tt-smi** (`visible_board_count`). **`runtime_chip_count`**
drives mesh only — do not set `0..7` unless `visible_board_count==8`.

Valid TP on n300 llmbox: **`valid_tp_degrees`** in probe (typically **2, 4, 8** for 8 chips).
Skip if scaffold `chip_count` is not in that list.

**Head divisibility (DiT / LLM):** if the component has `num_attention_heads`, intersect
probe degrees with `{d | num_heads % d == 0}`. Document the chosen degree in
`scaffold_multichip.json` (`head_constraint` field). Do not pick 8-way TP when
heads=28 (HunyuanImage) — use 4 or 2.

**Large LM text encoders on multichip:** if the test OOMs on replicated logits,
set `logits_to_keep=1` in loader inputs so lm_head runs on the last token only
(FLUX.2 Mistral3 TE pattern); hidden states for the pipeline remain full-seq.

## Test discovery

1. Prefer `state.details.test_path` from weight_fit / component scaffold.
2. Else collect `tests/runner/test_models.py` for `tensor_parallel-inference`.
3. Mochi MVP: `tests/torch/models/mochi/test_transformer.py::test_transformer_sharded`

## Logging

Same as `model-bringup-run`: `logs/iter_<N>_run.log`, json-report, `last_stage` from `._bringup_stage.txt`.

Append history with `stage: first_run_tp`, `arch`, `details.chip_count`.

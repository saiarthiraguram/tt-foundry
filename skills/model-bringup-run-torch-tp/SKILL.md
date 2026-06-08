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

**Large LM text encoders on multichip (`logits_to_keep`):** optional memory fix when
OOM is from **replicated full-sequence logits** (`lm_head` output), not from backbone
activations. `logits_to_keep=1` skips vocab projection on all but the last token;
the transformer stack still runs on the **full prompt length**.

**Apply only when ALL of:**

1. Component is a **causal LM** (`*ForCausalLM` or equivalent) that accepts
   `logits_to_keep` in `forward`.
2. Log / classify-oom evidence points to **logits / lm_head** buffer
   (`[batch, seq, vocab]` replicated on multichip) — not attn/MLP activations.
3. Loader wrapper / test **`run_graph_test` compares hidden states** (or another
   non-logits tensor the pipeline consumes). Confirm from loader + capture spec:
   output is e.g. `hidden_states[-1]` with shape `(batch, seq, hidden)` — **not**
   logits and **not** a last-token-only hidden slice unless the pipeline truly
   uses one position.
4. **CPU golden uses the same** `logits_to_keep` (and same wrapper output) so PCC
   stays apples-to-apples.

**Do not apply when:**

- Encoder is **encoder-only** (T5, CLIP, UMT5, ByT5, …) — no `logits_to_keep`.
- Test or pipeline consumes **full-sequence logits** or **per-token logits**.
- Wrapper returns **last-token hidden states only** but downstream needs **every
  token** — fix the wrapper/output contract first; do not use `logits_to_keep` as
  a substitute for full-seq conditioning.
- OOM is **weight-bound** or **backbone activation** bound — `logits_to_keep`
  will not help; use shard repair / resolution / promotion instead.

**Does not change weights or pipeline semantics** when guards pass: full-seq hidden
states for cross-attention remain; only unused `lm_head` work is dropped
(FLUX.2 Mistral3 TE pattern).

## Test discovery

1. Prefer `state.details.test_path` from weight_fit / component scaffold.
2. Else collect `tests/runner/test_models.py` for `tensor_parallel-inference`.
3. Mochi MVP: `tests/torch/models/mochi/test_transformer.py::test_transformer_sharded`

## Logging

Same as `model-bringup-run`: `logs/iter_<N>_run.log`, json-report, `last_stage` from `._bringup_stage.txt`.

Append history with `stage: first_run_tp`, `arch`, `details.chip_count`.

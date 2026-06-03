# Dtype ladder — single-chip before multichip

## Order (mandatory)

1. **CPU OVERVIEW** — `torch.float32` on CPU (`model-bringup-overview` skill).
2. **FIRST_RUN on HW** — fp32 weights and activations unless loader forces bf16.
3. **Activation repair** on **same arch** — resolution, tiling, compile flags (no dtype change yet).
4. **bf16 activations** on **same arch** — inputs + forward in bf16; weights bf16 only if still OOM or loader requires.
5. **Promotion** — only if **weight_bound** on **every** eligible single-chip arch after steps 2–4.

## Rules

- Do **not** default HW runs to bf16 to “save time.”
- Do **not** promote to multichip because fp32 OOM’d once without classifying activation vs weight.
- After promotion, multichip FIRST_RUN uses the **last single-chip dtype** that was attempted (usually bf16).

## CONFIG_UPDATE markers

Record in `state.json` → `details.dtype_ladder`:
```json
{
  "overview": "fp32",
  "first_run_by_arch": { "n150": "fp32", "p150": "bf16" },
  "promotion_dtype": "bf16"
}
```

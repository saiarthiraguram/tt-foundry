# Dtype ladder — single-chip before multichip

## Order (default)

1. **CPU OVERVIEW** — `torch.float32` on CPU (`model-bringup-overview`), unless the
   loader/doc explicitly uses another dtype for CPU sanity.
2. **FIRST_RUN on HW** — see **Source dtype** below (fp32 default, bf16 when justified).
3. **Activation repair** on **same arch** — resolution, tiling, compile flags (no dtype change yet).
4. **bf16 activations** on **same arch** — when still activation-bound after step 3.
5. **Promotion** — only if **weight_bound** on **every** eligible single-chip arch after steps 2–4.

## Source dtype (skip fp32 when justified)

Before FIRST_RUN, inspect the **upstream inference path**:

| Signal | Action |
|--------|--------|
| Loader module constant `DTYPE = torch.bfloat16` (e.g. `glm_image/pytorch/src/model_utils.py`, many pipeline loaders) | FIRST_RUN **bf16** on HW; record `"dtype": "bf16"` |
| HF `config.torch_dtype` / `from_pretrained(..., torch_dtype=bfloat16)` in loader | FIRST_RUN **bf16** |
| Reference inference script / HF README uses bf16 end-to-end | FIRST_RUN **bf16** |
| Generative video/image model with no fp32 checkpoint published | FIRST_RUN **bf16**; do not force fp32 weights |
| LLM / classic fp32-safe checkpoint, no bf16 signal | FIRST_RUN **fp32**, then bf16 only after activation repair or weight pressure |

**Rule:** match the **dominant dtype of the source artifact**, not an artificial
fp32 pass. CPU OVERVIEW may still use fp32 for numerical sanity unless the loader
requires bf16 inputs.

Scaffold should record in `weight_fit.json` / `state.json`:

```json
{
  "details": {
    "source_dtype": "bf16",
    "source_dtype_reason": "loader.DTYPE | hf_config | inference_script"
  }
}
```

## Rules

- Do **not** promote to multichip because bf16 OOM'd once without classifying activation vs weight.
- Do **not** downgrade bf16 source models to fp32 “for completeness.”
- After promotion, multichip FIRST_RUN uses the **last single-chip dtype** attempted (usually bf16).

## CONFIG_UPDATE / state markers

Record in `state.json` → `details.dtype_ladder`:

```json
{
  "overview": "fp32",
  "source_dtype": "bf16",
  "first_run_by_arch": { "n150": "bf16", "p150": "bf16" },
  "promotion_dtype": "bf16"
}
```

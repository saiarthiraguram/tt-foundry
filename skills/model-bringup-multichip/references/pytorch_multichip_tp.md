# PyTorch multichip tensor parallel bringup

PyTorch **tensor parallel (TP)** on multichip hosts (n300-llmbox, Galaxy, lb-blackhole).
Covers **Megatron 1D**, **2D FSDP-style**, and **MoE** patterns — see
`architecture_shard_templates.md` for per-family maps.

## When to use multichip TP

Only after `/model-bringup` writes `promotion.json` (all eligible single-chip
arches weight-bound). Do **not** start here for activation OOM or n150-only
failures that pass on p150.

## Loader contract

```python
def get_mesh_config(self, num_devices: int):
    """Return (mesh_shape, mesh_names)."""
    ...

def load_shard_spec(self, model):
    """Return dict[param → partition_spec] or None to replicate."""
    ...
```

## TP pattern summary

| Pattern | Mesh axes | Typical models | Reference |
|---------|-----------|----------------|-----------|
| **Megatron 1D** | `(None, "model")` | DiT, VAE3D, T5 in pipelines | `mochi/pytorch/src/utils.py` |
| **FSDP-style 2D** | `("batch", "model")` | Llama, Qwen, Gemma, Mistral | `llama/causal_lm/pytorch/loader.py` |
| **MoE TP** | `(batch, model)` + expert rules | DeepSeek V3.2, GPT-OSS, Kimi K2 | `deepseek/deepseek_v3_2_exp/pytorch/loader.py` |
| **Replicate** | any | small text encoders | `load_shard_spec` → `None` |

## Environment and test invocation

**Runner TP** — registered in
`tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml` (and
`torch_llm/` variants):

```bash
TT_VISIBLE_DEVICES=0,1,2,3 TT_XLA_SPMD=1 TT_XLA_ARCH=n300-llmbox \
  python -m pytest tests/runner/test_models.py -k "<model_key>" ...
```

**Component TP** — under `tests/torch/models/<family>/`:

```bash
TT_VISIBLE_DEVICES=... TT_XLA_SPMD=1 CONVERT_SHLO_TO_SHARDY=1 \
  python -m pytest tests/torch/models/mochi/test_transformer.py::test_transformer_sharded ...
```

Typical test body:

```python
mesh = loader.get_mesh(num_devices)  # or get_mesh(mesh_shape, mesh_names)
shard_spec = loader.load_shard_spec(model)
xs.mark_sharding(..., mesh, shard_spec)
```

Use **`dtype_override=torch.bfloat16`** when loader/source dtype is bf16.

## Small components on TP mesh

Return **`None`** from `load_shard_spec` (full replicate). Attach to the same
mesh as the heavy component — do **not** add input data-parallel sharding in
this skill set unless the loader already supports `batch_axis="data"`.

## Chip-count heuristic

| Weight (bf16) | Typical mesh |
|---------------|--------------|
| ≤ 12 GiB/device budget on target | 2–4 devices |
| 12–24 GiB/device | 4 devices |
| > 24 GiB/device | 8–32 devices (Galaxy) |

Exact count from `promotion.json` → `suggested_chip_count` and
`dram_budget_torch_tp.md`.

## Out of scope (v1)

| Not supported | Reason |
|---------------|--------|
| JAX / Flax loaders | PyTorch-only path |
| Runner data-parallel YAML | TP promotion path only |
| Multi-host DP | Not in promotion FSM |
| Auto mesh without `load_shard_spec` | Explicit specs required |

## Repair loop

TP failures → `model-bringup-repair-shard-spec` (mesh divisibility, wrong
column/row map, MoE expert layout). OOM → `classify-oom` in multichip context.

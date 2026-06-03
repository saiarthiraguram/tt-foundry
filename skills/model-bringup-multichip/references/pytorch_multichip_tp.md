# PyTorch multichip tensor parallel bringup

Megatron-style **tensor parallel (TP) only** for PyTorch loaders on multichip
hosts (e.g. n300-llmbox). **JAX, data-parallel, and multi-host DP are out of
scope** for this skill set.

Use with:
- `model-bringup-scaffold-torch-tp`
- `model-bringup-run-torch-tp`
- `architecture_shard_templates.md`

## When to use multichip TP

Only after `/model-bringup` writes `promotion.json` (all eligible single-chip
arches weight-bound). Do **not** start here for activation OOM or n150-only
failures that pass on p150.

## Loader contract (PyTorch)

Every multichip loader must implement:

```python
def get_mesh_config(self, num_devices: int):
    """Return (mesh_shape, mesh_names). Megatron: shard on 'model' axis."""
    if num_devices not in MESH_SHAPES:
        raise ValueError(f"Unsupported device count {num_devices}")
    return MESH_SHAPES[num_devices], MESH_NAMES

def load_shard_spec(self, model):
    """Return dict[param_name -> (shard_dim, mesh_axis)] or None to replicate."""
    ...
```

Reference implementations:
- Mochi DiT: `third_party/tt_forge_models/mochi/pytorch/src/utils.py`
- Mochi VAE: `shard_vae_decoder_specs` in same file

## Mesh shapes (Megatron, 1D model axis)

```python
MESH_SHAPES = {1: (1, 1), 2: (1, 2), 4: (1, 4), 8: (1, 8), 32: (8, 4)}
MESH_NAMES = (None, "model")
```

- Column-parallel weight: `("model", None)`
- Row-parallel weight: `(None, "model")`
- Bias after row-parallel: `(None,)`

Head count must divide `mesh_shape[1]` when sharding attention on the model axis.

## Environment and test invocation (TP)

Register tests in
`tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`.

Run with visible devices and SPMD enabled:

```bash
TT_VISIBLE_DEVICES=0,1,2,3 TT_XLA_SPMD=1 TT_XLA_ARCH=n300-llmbox \
  python -m pytest <test_node> -svv --tb=long
```

Component tests under `tests/torch/models/<family>/` typically:

```python
mesh = loader.get_mesh(num_devices)
shard_spec = loader.load_shard_spec(model)
xs.mark_sharding(..., mesh, shard_spec)
```

Log files for TP runs: `logs/iter_tp_<N>_run.log` (see `model-bringup-run-torch-tp`).

## Small components on TP mesh

Text encoders and other small modules: return **`None`** from
`load_shard_spec` (full replicate). Still attach to the same mesh as the
DiT/transformer for fabric consistency — do **not** use input data-parallel
sharding in this skill set.

## Chip-count heuristic

| Weight (bf16) | Typical mesh |
|---------------|--------------|
| ≤ 12 GiB/device budget on target | 2–4 devices |
| 12–24 GiB/device | 4 devices |
| > 24 GiB/device | 8+ devices |

Exact count comes from `promotion.json` → `suggested_chip_count` and
`dram_budget_torch_tp.md`.

## Out of scope (v1)

| Not supported | Reason |
|---------------|--------|
| JAX / Flax loaders | PyTorch-only bringup path |
| Data parallel (batch sharding) | Megatron TP only |
| Multi-host DP | Not in promotion path |
| Auto mesh from XLA SPMD without loader specs | Requires explicit `load_shard_spec` |

## Repair loop (multichip)

On TP failure, route to `model-bringup-repair-shard-spec` (mesh divisibility,
wrong column/row spec) — not single-chip `model-bringup-repair`.

On OOM during TP, invoke `model-bringup-classify-oom` with multichip context;
activation-bound failures may need resolution or tiling changes on the TP mesh,
not a return to single-chip unless promotion was wrong.

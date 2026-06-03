# PyTorch Megatron TP shard templates (Torch only)

Megatron: shard on **`model`** axis only; other dim `None` (replicate along batch axis).

```python
# Column-parallel
("model", None)
# Row-parallel
(None, "model")
# 1D bias — replicate after row-parallel
(None,)
```

## Transformer LLM (Llama/Qwen/Mistral)

Layers: `model.model.layers[*].self_attn.{q,k,v,o}_proj`, `mlp.{gate,up,down}_proj`.

## DiT dual-stream (Mochi-style)

Reference: `third_party/tt_forge_models/mochi/pytorch/src/utils.py` → `shard_transformer_specs`.

- Q/K/V image + text streams: `("model", None)`
- `to_out`, `to_add_out`: `(None, "model")`
- FF `net[0].proj`: `("model", None)`; `net[2]`: `(None, "model")`

`get_mesh_config`: `MESH_SHAPES = {1:(1,1), 2:(1,2), 4:(1,4), 8:(1,8), 32:(8,4)}`, `MESH_NAMES = (None, "model")`.

## VAE 3D decoder (Mochi-style)

`shard_vae_decoder_specs`: conv column on out channels, row on in; GroupNorm channel-sharded.

## Text encoder (T5-XXL in pipeline)

`load_shard_spec` → **`None`** (replicate on multichip mesh); still use same mesh as heavy components for uniform fabric.

## mesh_config

```python
def get_mesh_config(self, num_devices: int):
    if num_devices not in MESH_SHAPES:
        raise ValueError(...)
    return MESH_SHAPES[num_devices], MESH_NAMES
```

Head divisibility: attention heads must divide `mesh_shape[1]` when sharding heads on model axis.

## Small component on multichip mesh

Return `None` from `load_shard_spec` — weights replicate; do **not** use data-parallel input sharding in this skill set.

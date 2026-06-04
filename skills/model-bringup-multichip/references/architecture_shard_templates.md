# PyTorch tensor-parallel shard templates

Catalog of **TP patterns already used** in tt-forge-models loaders and multichip
tests. Pick the closest family, then adapt parameter names via
`model.named_parameters()`.

Shared API (all patterns):

```python
def get_mesh_config(self, num_devices: int) -> tuple[tuple[int, ...], tuple[str, ...]]:
    ...

def load_shard_spec(self, model) -> dict | None:
    """Map param/buffer → partition spec, or None to replicate on mesh."""
```

Reference base: `third_party/tt_forge_models/base.py`.

---

## Pattern A — Megatron 1D (`model` axis only)

**Mesh:** `MESH_NAMES = (None, "model")`  
**Shapes:** `{1:(1,1), 2:(1,2), 4:(1,4), 8:(1,8), 32:(8,4)}` (family-specific)

| Spec tuple | Meaning |
|------------|---------|
| `("model", None)` | Column-parallel (shard output / in_features dim) |
| `(None, "model")` | Row-parallel (shard input / out_features dim) |
| `(None,)` | Replicate 1D (bias after row-parallel) |

**Used by:** video/image DiT pipelines — shard on hidden/head dims, replicate batch.

| Family | Loader | Shard helpers |
|--------|--------|---------------|
| Mochi | `mochi/pytorch/loader.py` | `src/utils.py` → `shard_transformer_specs`, `shard_vae_decoder_specs` |
| HiDream I1 | `hidream_i1/pytorch/loader.py` | `shard_hidream_transformer_specs`, `shard_t5_encoder_specs`, `shard_llama_specs` |
| GLM-Image | `glm_image/pytorch/loader.py` | `shard_transformer_specs`, `shard_vae_specs`, `shard_text_encoder_specs`, `shard_vision_language_encoder_specs` |
| HunyuanVideo | `hunyuan_video/pytorch/loader.py` | `shard_transformer_specs`, `shard_vae_specs`, `shard_text_encoder_specs`, `shard_text_encoder_2_specs` |
| HunyuanImage 2.1 / 1.5 | `hunyuan_image_2_1/`, `hunyuan_1_5/` | `shard_transformer_specs`, `shard_text_encoder_specs`, … |
| Lumina-Image | `lumina_image/pytorch/` | `shard_transformer_specs`, `shard_vae_specs`, `shard_text_encoder_specs` |
| OmniGen | `omnigen/pytorch/` | `shard_transformer_specs`, `shard_vae_specs`, `shard_vae_encoder_specs` |
| Krea realtime | `krea_realtime_video/pytorch/` | `shard_transformer_specs`, `shard_text_encoder_specs` |

**DiT attention (dual-stream, Mochi-style):** Q/K/V → `("model", None)`; output
proj → `(None, "model")`; FF up/gate → `("model", None)`; FF down → `(None, "model")`.

**VAE 3D decoder:** conv — column on out channels, row on in; GroupNorm
channel-sharded (see `shard_vae_decoder_specs`).

**Small / encoder-only components:** `load_shard_spec` → **`None`** (replicate);
still use the same mesh as the heavy component for fabric uniformity
(e.g. Mochi `text_encoder` returns `(1,1)` mesh + `None` spec).

---

## Pattern B — 2D FSDP-style (`batch` + `model` axes)

**Mesh:** `("batch", "model")` — common for **causal LMs** on llmbox / Galaxy.

| Spec tuple | Typical weight |
|------------|----------------|
| `(None, "batch")` | embed_tokens |
| `("model", "batch")` | lm_head, q/k/v/up/gate |
| `("batch", "model")` | o_proj, down_proj |
| `("batch",)` | layer norms |

**Used by:** runner `tensor_parallel-inference` LLMs.

| Family | Loader |
|--------|--------|
| Llama | `llama/causal_lm/pytorch/loader.py` |
| Qwen 2 / 2.5 / 3 | `qwen_*/causal_lm/pytorch/loader.py` |
| Gemma | `gemma/pytorch/loader.py` |
| Mistral / Pixtral | `mistral/pytorch/`, `mistral/pixtral/pytorch/` |
| Falcon | `falcon/pytorch/loader.py` |
| GPT-OSS (MoE) | `gpt_oss/pytorch/loader.py` |
| Solar, Command, Arcee, OLMo3, Phi4, … | respective `*/pytorch/loader.py` |

**Large 70B+ / Galaxy:** mesh `(4, 8)` or `(2, num_devices//2)` when
`num_devices == 32` (see Llama 405B / 70B branches in `get_mesh_config`).

**Prefill / strategy parameterization:** `ModelLoaderPrefill.load_shard_spec(model, strategy="fsdp", batch_axis="batch")` — also supports `batch_axis="data"` when inputs are sharded (advanced; not default bringup path).

**Replicate small variants:** some loaders return `None` for tiny models
(e.g. Llama 3.2 1B) — run replicated on mesh.

---

## Pattern C — MoE / sparse expert TP

Expert weights shard on **both** `(batch, model)` or expert-specific axes; dense
layers follow Pattern B.

| Family | Loader | Notes |
|--------|--------|-------|
| DeepSeek V3.2 | `deepseek/deepseek_v3_2_exp/pytorch/loader.py` | Galaxy `(4,8)` only; MLA + `A2aSparse` MoE; `enable_sparse_mlp()` at load |
| GPT-OSS | `gpt_oss/pytorch/loader.py` | MoE layers + custom inject; mesh `(2,4)` @ 8 chips, `(4,8)` @ 32 |
| Kimi K2 / K2.5 | `kimi_k2/pytorch/loader.py`, `kimi_k2/k2_5/` | MoE + large mesh |
| DeepSeek V4 | `tests/torch/models/deepseek_v4/test_deepseek_v4_tp.py` | see loader + `utils.py` in test tree |
| GLM4 MoE | `tests/torch/models/glm4_moe/` | component tests for MoE blocks |

**Repair hints:** expert count must divide mesh axis; router/gate often replicated;
see existing `load_shard_spec` in each loader before inventing new maps.

---

## Pattern D — Vision-language / multimodal

| Family | Components | Pattern |
|--------|------------|---------|
| GLM-Image | TextEncoder, VisionLanguageEncoder, Transformer, Vae | B-style mesh; VLM uses `shard_vision_language_encoder_specs` |
| Mistral Pixtral | vision + language | `mistral/pixtral/pytorch/loader.py` |
| Gemma3 multimodal | `gemma3/multimodal/pytorch/loader.py` | combined mesh + shard spec |

---

## Pattern E — Runner-only / replicate

Some entries in `test_config_inference_tensor_parallel.yaml` use TP collect paths
but **`load_shard_spec` → None`** for small sub-models. Validate in loader before
assuming sharding exists.

---

## Choosing a template (scaffold-torch-tp)

```
1. Is it a DiffusionPipeline / video DiT component?
   → Pattern A (Megatron 1D), copy nearest family in table above.
2. Is it a causal LM in runner TP YAML?
   → Pattern B (batch+model FSDP-style).
3. Does load_model() call enable_sparse_mlp / MoE modules?
   → Pattern C — copy DeepSeek V3.2 or GPT-OSS loader.
4. Multimodal encoder + DiT?
   → Pattern D — GLM-Image or Pixtral reference.
5. Unknown param names?
   → Walk named_parameters(); apply column/row rules from Pattern A or B.
```

## Head / channel divisibility

- Attention heads must divide the sharded axis size (Pattern A: `mesh_shape[1]`;
  Pattern B: check head_dim × num_heads vs shard dim).
- Conv out_channels must divide column-parallel axis for VAE blocks.

## Tests to copy

| Pattern | Example test |
|---------|--------------|
| A sharded DiT | `tests/torch/models/mochi/test_transformer.py::test_transformer_sharded` |
| A + run_graph_test | `tests/torch/models/hunyuan_video/test_transformer.py::test_transformer_sharded` |
| B runner TP | collect `tests/runner/test_models.py` → `tensor_parallel-inference` |
| C MoE TP | `tests/torch/models/deepseek_v4/test_deepseek_v4_tp.py` |

## Out of scope (v1 bringup skills)

- JAX / Flax (`falcon/bounty_jax/…`) — PyTorch path only
- Runner **data_parallel** YAML (`test_config_inference_data_parallel.yaml`) — not promotion default
- Input activation data-parallel sharding unless loader already implements `batch_axis="data"`

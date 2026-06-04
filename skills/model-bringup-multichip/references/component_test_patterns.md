# Component test patterns

Applies to **`tests/torch/models/<family>/`** — one compile target per file, not
full pipeline `generate()`.

## Principles

1. **One compile target per test file** — e.g. DiT/transformer step, VAE decode,
   text-encoder encode, not end-to-end multi-minute pipeline loops.
2. **Loaders in tt-forge-models** — tests import `ModelLoader(variant=...)` or
   `ModelLoader(subfolder=...)`; avoid reaching into `pipe.unet` in the test body.
3. **No default arch markers** — do **not** use `@pytest.mark.n150` /
   `@pytest.mark.p150`. Most components run on both Wormhole and Blackhole when
   DRAM allows. See `arch_eligibility.md` for skip-on-device guards.
4. **Fair CPU/Forge compare** — clone mutable state (KV cache, buffers) before
   CPU golden if `run_graph_test` mutates in-place.

## Standard component roles

Map pipeline submodules to these **roles** when scaffolding `weight_fit.json`
and choosing reference loaders:

| Role | Typical loader variant / subfolder | Example test file |
|------|-----------------------------------|---------------------|
| **transformer** / **dit** / **unet** | `Transformer`, `UNet`, `subfolder="transformer"` | `test_transformer.py`, `test_unet.py`, `test_wan_dit.py` |
| **vae_decoder** | `Vae`, `VAE`, `subfolder="vae"` | `test_vae_decoder.py` |
| **vae_encoder** | encoder-only wrapper | `test_vae_encoder.py` |
| **text_encoder** | T5/CLIP/UMT5 first encoder | `test_text_encoder.py` |
| **text_encoder_2** … **text_encoder_N** | second+ text stacks | `test_text_encoder_2.py`, … |
| **vision_language_encoder** | VLM conditioning stack | (glm_image loader — add test when present) |
| **image_token_step** / **mmgpt** | autoregressive image-token path | Janus `ImageTokenStep` variants |
| **gen_img_embed** / **gen_vision_decode** | decode-side helpers | Janus embed/decode components |
| **llm_backbone** | causal LM inside a pipeline | HiDream `shard_llama_specs` path |
| **moe / sparse_mlp** | MoE FFN blocks | `test_deepseek_v4_tp.py`, `test_glm4_moe.py` |

## Families under `tests/torch/models/` (representative)

| Family directory | Components (test files) | Loader / shard reference |
|------------------|-------------------------|--------------------------|
| `mochi/` | transformer, vae_decoder, text_encoder | `third_party/tt_forge_models/mochi/pytorch/` |
| `hunyuan_video/` | transformer, vae_decoder, text_encoder, text_encoder_2 | `.../hunyuan_video/pytorch/` |
| `HiDream_I1/` | transformer, vae_decoder, text_encoder … text_encoder_4 | `.../hidream_i1/pytorch/` |
| `HunyuanImage_2_1/`, `Hunyuan_1_5/` | transformer, vae_decoder, text_encoder(s) | `.../hunyuan_image_2_1/`, `.../hunyuan_1_5/` |
| `lumina_image/` | transformer, vae_decoder, text_encoder | `.../lumina_image/pytorch/` |
| `glm_image/` | (loader: TextEncoder, VLM, Transformer, Vae) | `.../glm_image/pytorch/` |
| `cog_videox/` | transformer, vae_decoder, text_encoder | `.../cog_videox/pytorch/` (when loader present) |
| `omnigen/` | transformer, vae_decoder, vae_encoder | `.../omnigen/pytorch/` |
| `krea_realtime/` | transformer, vae_decoder, text_encoder | `.../krea_realtime_video/pytorch/` |
| `wan2_2/` | wan_dit, vae_decoder, vae_encoder, umt5_text_encoder | `.../wan2_2/pytorch/` (when present) |
| `playground_v2_5/`, `sdxl_lightning/` | unet, vae_decoder, text_encoder(s) | diffusion UNet-style |
| `janus_pro/` | image_token_step, gen_img_embed, gen_vision_decode | `.../janus_pro/text_to_image/pytorch/loader.py` |
| `deepseek_v3_2_exp/`, `deepseek_v4/` | full model / MoE / TP | `.../deepseek/*/pytorch/loader.py` |
| `kimi_k2/`, `glm4_moe/` | large MoE LLM | respective loaders |

When adding a new pipeline family, mirror the **directory name** under
`tests/torch/models/<family>/` and list each `test_*.py` in `weight_fit.json`
→ `components[].test_path`.

## Per-component arch eligibility

Store in **`weight_fit.json`** per component (not pytest markers):

```json
{
  "name": "transformer",
  "eligible_archs": ["n150", "p150"],
  "p150_only": false
}
```

Example: a 7B MMGPT/image-token block may set `"p150_only": true` and
`eligible_archs: ["p150"]` while sibling components stay dual-arch.

In the component test, skip only when runtime device class disagrees:

```python
from tests.runner.test_utils import get_xla_device_arch

if get_xla_device_arch() == "wormhole" and weight_fit["p150_only"]:
    pytest.skip("component requires p150 (Blackhole)")
```

## model-bringup-run mapping

Store in `state.json` → `details.test_path` when not using runner collect:

```
tests/torch/models/mochi/test_transformer.py::test_transformer_sharded
tests/torch/models/hunyuan_video/test_transformer.py::test_transformer_sharded
tests/torch/models/HiDream_I1/test_transformer.py::test_transformer
```

Set `TT_XLA_ARCH` to the current arch in the per-arch loop (`n150` / `p150`).

## Multichip component tests

Sharded runs (see `mochi/test_transformer.py`, `hunyuan_video/test_transformer.py`):

- `xr.use_spmd()`, `CONVERT_SHLO_TO_SHARDY=1` for DiT / video transformers when needed
- `get_mesh` + `loader.load_shard_spec` + `xs.mark_sharding`
- `dtype_override=torch.bfloat16` when loader/HF defaults to bf16

Record mesh + chip count in `scaffold_multichip.json` for `model-bringup-run-torch-tp`.

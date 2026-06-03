# Component test patterns

Reference: [PR #4810 Janus-Pro](https://github.com/tenstorrent/tt-xla/pull/4810).

## Principles

1. **One compile target per test file** — ImageTokenStep, GenImgEmbed, GenVisionDecode, not full pipeline `generate()`.
2. **Loaders in tt-forge-models** — tests import `ModelLoader(subfolder=...)`; no `pipe.unet` in test body.
3. **Arch markers** — `@pytest.mark.n150`, `@pytest.mark.p150` per component eligibility.
4. **Fair CPU/Forge compare** — clone mutable state (e.g. KV cache) before CPU golden if `run_graph_test` mutates in-place.

## Arch split example (Janus)

| Component | Pro-1B | Pro-7B ImageToken |
|-----------|--------|-------------------|
| ImageToken prefill/decode | n150 + p150 | **p150 only** |
| GenImgEmbed | n150 + p150 | n150 + p150 |
| GenVisionDecode | n150 + p150 | n150 + p150 |

## model-bringup-run mapping

Store in `state.json` → `details.test_path` when not using `test_all_models_torch`:

```
tests/torch/models/mochi/test_transformer.py::test_transformer_sharded
tests/torch/models/janus_pro/test_image_token_step.py::...
```

Set `TT_XLA_ARCH` to current arch in the per-arch loop.

## Multichip component tests

May use direct SPMD setup (see mochi `test_transformer_sharded`):

- `xr.use_spmd()`, `CONVERT_SHLO_TO_SHARDY=1` for DiT
- `get_mesh` + `loader.load_shard_spec` + `xs.mark_sharding`

Record in `scaffold_multichip.json` for `model-bringup-run-torch-tp`.

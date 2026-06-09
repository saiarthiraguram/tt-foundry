---
name: model-bringup-scaffold-torch-tp
description: VALIDATE_TP stage for promotion-only multichip bringup. Adds or validates get_mesh_config and load_shard_spec (Megatron 1D, FSDP-style 2D, or MoE TP) on PyTorch loaders. Registers tensor_parallel YAML entries on --apply.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Scaffold — Torch tensor parallel

## Gate

Require `.claude/bringup/<safe_key>/promotion.json` unless `--force-multichip`.

## Steps

1. Read `promotion.json` + `weight_fit.json` for component and `suggested_chip_count`.
2. **Classify TP pattern** (see `architecture_shard_templates.md`):
   - Pipeline DiT / VAE → **Pattern A** (Megatron 1D, `(None, "model")` mesh)
   - Runner causal LM → **Pattern B** (FSDP-style, `("batch", "model")` mesh)
   - MoE / sparse expert → **Pattern C** (DeepSeek V3.2, GPT-OSS, Kimi K2 refs)
   - Multimodal encoder + DiT → **Pattern D** (GLM-Image, Pixtral)
3. Copy shard helpers from the nearest family loader under
   `third_party/tt_forge_models/<family>/pytorch/` (`src/model_utils.py` or
   `src/utils.py`).
4. Ensure `get_mesh_config(num_devices)` and `load_shard_spec(model)` on the
   target loader (or component wrapper).
5. Small components: `load_shard_spec` → `None` (replicate on mesh).
6. Dry-run import:
   `python -c "from third_party.tt_forge_models.<family>.pytorch import ModelLoader; ..."`
7. Write `scaffold_multichip.json` with `tp_pattern`, mesh, chip_count,
   `TT_VISIBLE_DEVICES` hint.
8. Propose YAML diff for
   `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`
   (or `torch_llm/` variant) with `supported_archs` for multichip hosts — dry-run
   unless `--apply`.

Record in `state.json`: `details.multichip_scaffold_path`, `details.tp_pattern`,
`stage: validate_tp`.

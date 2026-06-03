---
name: model-bringup-scaffold-torch-tp
description: VALIDATE_TP stage for promotion-only multichip bringup. Adds or validates get_mesh_config and Megatron load_shard_spec on PyTorch loaders. Registers tensor_parallel YAML entries on --apply.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Scaffold — Torch tensor parallel

## Gate

Require `.claude/bringup/<safe_key>/promotion.json` unless `--force-multichip`.

## Steps

1. Read `promotion.json` + `weight_fit.json` for component and `suggested_chip_count`.
2. Pick reference loader by architecture (see `model-bringup-multichip/references/architecture_shard_templates.md`).
3. Ensure `get_mesh_config(num_devices)` and `load_shard_spec(model)` in loader or `src/model_utils.py`.
4. Small components: `load_shard_spec` → `None` (replicate on mesh).
5. Dry-run: `python -c "from third_party.tt_forge_models.<family>.pytorch import ModelLoader; ..."`.
6. Write `scaffold_multichip.json` with mesh, chip_count, `TT_VISIBLE_DEVICES` hint.
7. Propose YAML diff for `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml` (dry-run unless `--apply`).

Record in `state.json`: `details.multichip_scaffold_path`, `stage: validate_tp`.

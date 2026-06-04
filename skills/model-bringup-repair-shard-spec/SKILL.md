---
name: model-bringup-repair-shard-spec
description: REPAIR stage for multichip TP. Fixes mesh shape, shard specs (Megatron, FSDP-style, MoE), and activation footprint on the TP mesh. No single-chip arch changes.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Repair — shard spec (TP only)

## Strategies

### fix_mesh_shape

Adjust `MESH_SHAPES` / `get_mesh_config` when head divisibility fails or device
count mismatch. Check Pattern A vs B in `architecture_shard_templates.md`.

### fix_shard_spec

Edit `shard_*_specs()` or inline `load_shard_spec` using `named_parameters()` +
templates. Match column/row rules for the family's TP pattern.

### fix_moe_layout

Expert / router shard map wrong — copy from DeepSeek V3.2, GPT-OSS, or Kimi K2
loaders; verify expert count divides mesh axis.

### reduce_activation_footprint

Lower resolution on **same** multichip mesh before adding chips.

### runtime_debug

Delegate when TP OOM persists after shard fix — same as `model-bringup-repair`.

## Blocked

- Do not route back to n150/p150 single-chip from here.
- Do not use runner data-parallel input sharding unless loader already supports it.

---
name: model-bringup-repair-shard-spec
description: REPAIR stage for multichip TP. Fixes mesh shape, Megatron shard specs, and activation footprint on the TP mesh. No single-chip arch changes.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Repair — shard spec (TP only)

## Strategies

### fix_mesh_shape

Adjust `MESH_SHAPES` / `get_mesh_config` when head divisibility fails or device count mismatch.

### fix_shard_spec

Edit `shard_*_specs()` using `named_parameters()` + templates in
`model-bringup-multichip/references/architecture_shard_templates.md`.

### reduce_activation_footprint

Lower resolution on **same** multichip mesh before adding chips.

### runtime_debug

Delegate when TP OOM persists after shard fix — same as `model-bringup-repair`.

## Blocked

- Do not route back to n150/p150 single-chip from here.
- Do not use data-parallel input sharding in this skill set.

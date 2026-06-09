---
name: model-bringup-multichip
description: Promotion-only multichip tensor-parallel bringup for PyTorch models. Runs only after single-chip /model-bringup exhausts n150 and p150 with weight-bound failure. Supports Megatron 1D, FSDP-style 2D, and MoE TP patterns (get_mesh_config + load_shard_spec). Use when promotion.json exists or user explicitly continues from weight_predicted scaffold.
allowed-tools: Bash Read Write Edit Grep Glob Task Agent
---

# Model Bringup — Multichip TP (promotion-only)

See `references/` for DRAM budgets, OOM classification, arch eligibility, dtype ladder,
shard templates, `host_device_probe.md`, and `pytorch_multichip_tp.md`.

Scripts: `scripts/compute_weight_fit.py`, `scripts/write_promotion.py`, `scripts/probe_host.py`.

**Before any run:** `probe_host.py` → `host_probe.json` (see `host_device_probe.md`).

- **Single-chip (n150/p150):** only when `runtime_chip_count==1` on a **dedicated** host — never on
  llmbox/qb/galaxy/lb fabric (no `TT_VISIBLE_DEVICES=0` workaround).
- **Multichip TP:** only when `runtime_chip_count>=2`; `TT_VISIBLE_DEVICES` from **tt-smi**
  (`visible_board_count`, e.g. `0,1,2,3`); mesh from **runtime** (`runtime_chip_count`, e.g. 8).
- **Per component:** pass `--parallelism-mode` from `weight_fit.json`; skip with
  `component_skip_reason` if wrong host (e.g. `single_device` VAE on n300 llmbox).
- **Valid TP degrees:** 2/4/8 on 8-chip llmbox; **2/4/8/32** on Galaxy — see `valid_tp_degrees` in probe JSON and `host_device_probe.md`.

## Entry gate

Do **not** start cold. Require one of:
- `.claude/bringup/<safe_key>/promotion.json` from `/model-bringup`, or
- `--force-multichip` **and** `weight_fit.json` shows `weight_predicted` on all eligible arches.

## Invocation

`/model-bringup-multichip <model_key> [--component <name>] [--arch n300-llmbox] [--from-promotion]`

For pipeline families, pass `--component` only where
`weight_fit.json` has `parallelism_mode: tensor_parallel` (or promotion.json
lists that component). Skip `single_device_only: true` parts — they keep
`tests/torch/models/...` nodes with `@pytest.mark.single_device` and
**replicate** on the TP mesh (`load_shard_spec` → `None`). Do not register
component tests in runner YAML.

## FSM

```
VALIDATE_TP  →  FIRST_RUN_TP  →  (fail) DIAGNOSE + classify-oom  →  REPAIR_SHARD  →  VERIFY_TP
                      ↓ pass
              CONFIG_UPDATE_TP (--apply)  →  FINALIZE
```

## Delegate skills

| Stage | Skill |
|-------|--------|
| VALIDATE_TP | `model-bringup-scaffold-torch-tp` |
| FIRST_RUN_TP / VERIFY_TP | `model-bringup-run-torch-tp` |
| DIAGNOSE | `model-bringup-diagnose` + `model-bringup-classify-oom` |
| REPAIR_SHARD | `model-bringup-repair-shard-spec` |
| CONFIG_UPDATE_TP | `model-bringup-config-update-torch-tp` |
| FINALIZE | `model-bringup-finalize` |

Read `references/arch_eligibility.md` — single-chip on n150 **and** p150 must be exhausted (weight-bound) unless `--force-multichip`.

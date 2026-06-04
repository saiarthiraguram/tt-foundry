# OOM and failure classification

Output JSON schema for `model-bringup-classify-oom`:

```json
{
  "class": "activation | weight_runtime | weight_predicted | arch_insufficient | dtype_only | shardy_fe | fe_pcc | other",
  "arch": "n150 | p150 | ...",
  "evidence": "one-line",
  "promote_multichip": false
}
```

Set `promote_multichip: true` only when `class` is `weight_runtime` or `weight_predicted` **and** all eligible single-chip arches are exhausted.

## activation

- Log mentions intermediate / activation / buffer size >> weight footprint.
- OOM scales when resolution, `num_frames`, or latent H×W×T changes.
- `weight_bytes_bf16` fits in `0.7 * DRAM[arch]` for one chip.
- **Action:** REPAIR on same arch (reduce_resolution, enable_vae_tiling, enable_compile_flags), then dtype_bf16_activations.

## weight_runtime

- OOM during weight load, constant fold, or param DRAM allocation.
- Message like `allocate ... B DRAM buffer` for weights, not activations.
- **Action:** if all arches weight-bound after ladder → promotion.json.

## arch_insufficient

- Run on `n150` failed DRAM OOM but `weight_fit.json` has `eligible_archs` containing `p150` with fits_bf16/fits_fp32.
- **Not** multichip; continue single-chip on `p150`.
- Example: component with `"p150_only": true` (large MMGPT / image-token block) — skip n150 in orchestrator; component test may `pytest.skip` when `get_xla_device_arch() == "wormhole"`.

## weight_predicted

- Scaffold: `weight_bytes_fp32 > 0.85 * DRAM[arch]`.
- Optional single probe run unless `--skip-single-run`.

## dtype_only

- fp32 failed; bf16 on same arch passed.
- Stay single-chip; update dtype in state.

## shardy_fe / fe_pcc

- Shardy divisibility, compare on non-1×1 mesh, FE compile errors, PCC — use standard DIAGNOSE/REPAIR, not multichip promotion.

## Log snippets (examples)

| Snippet | Likely class |
|---------|----------------|
| `allocate 32463388672 B DRAM` on video DiT full res | activation (mochi transformer) |
| `Not enough space to allocate ... across 12 banks` with small weight estimate | activation or arch_insufficient |
| Pro-7B-scale MMGPT on n150 only | arch_insufficient → p150 only |

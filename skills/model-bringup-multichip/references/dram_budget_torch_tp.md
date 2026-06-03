# DRAM budget — single-chip and multichip planning

## Single-chip canonical DRAM (bytes)

| Arch | Hardware | DRAM per device | Use in weight_fit |
|------|----------|-----------------|-------------------|
| `n150` | Wormhole | **12 GiB** (12 × 2^30) | `12949672960` |
| `p150` | Blackhole | **32 GiB** (32 × 2^30) | `34359738368` |

Usable planning budget per arch: **85%** of DRAM (`0.85 * dram_bytes`) for **weight** gate.

## Weight bytes

```
weight_bytes_fp32 = num_params * 4
weight_bytes_bf16 = num_params * 2
```

Prefer live `sum(p.numel())` from loader; else config estimate; else name heuristic.

## Per-arch eligibility (scaffold)

For each component and each arch in `[n150, p150]`:

```
if weight_bytes_fp32 > 0.85 * DRAM[arch]:
    weight_predicted[arch] = true
    eligible_single_chip[arch] = false   # may still run one probe unless --skip-single-run
else:
    eligible_single_chip[arch] = true
```

## Activation derate (classification only)

Multiply estimated activation footprint by derate when **classifying** OOM — does **not** auto-trigger multichip.

| activation_class | derate on n150 | derate on p150 |
|------------------|----------------|----------------|
| text_encoder | 1.0 | 1.0 |
| image | 1.2 | 1.2 |
| video | 2.0 | 1.5 |
| vae | 1.5 | 1.5 |
| t2i_submodule | 1.2 | 1.2 |

If OOM log shows large intermediate buffer but `weight_bytes_bf16 / 1 chip < 0.7 * DRAM[arch]` → classify **activation**, not weight_bound.

## Multichip hosts (promotion tier)

| Arch | Typical devices | Mesh examples |
|------|-----------------|---------------|
| `n300-llmbox` | 8 | `(1,8)`, `(2,4)` |
| `lb-blackhole` | 4 | `(1,4)`, `(2,2)` |
| `galaxy-wh-6u` | 32 | `(8,4)`, `(4,8)` |

Per-device DRAM for multichip planning: use same order of magnitude as single-chip family (WH ~12 GiB, BH ~32 GiB per chip) unless lab notes say otherwise.

## Chip count heuristic (promotion only)

```
required_bytes_per_chip = (weight_bytes_active_dtype + activation_reserve) / chip_count
chip_count = min { n in {1,2,4,8,32} : required_bytes_per_chip <= DRAM[host_per_device] }
```

Uniform `chip_count` across pipeline components on one host demo.

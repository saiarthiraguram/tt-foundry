# Arch eligibility — n150 and p150 single-chip

## Defaults

- **`--archs n150,p150`** when `weight_fit.json` → `eligible_archs` contains both.
- Run **full single-chip ladder on each** eligible arch in order: **n150**, then **p150**.
- **`--arch p150`** when scaffold marks **p150-only** (e.g. Janus Pro-7B ImageTokenStep).

## weight_fit.json fields

```json
{
  "component": "transformer",
  "weight_bytes_fp32": 40000000000,
  "weight_bytes_bf16": 20000000000,
  "eligible_archs": ["n150", "p150"],
  "per_arch": {
    "n150": { "dram_gib": 12, "fits_fp32": false, "fits_bf16": true, "weight_predicted": true },
    "p150": { "dram_gib": 32, "fits_fp32": true, "fits_bf16": true, "weight_predicted": false }
  },
  "supported_archs": [],
  "p150_only": false
}
```

After HW bringup, `supported_archs` = arches that **PASSED** (subset of eligible).

## Promotion gate

Promote to multichip only when:

1. Every arch in `eligible_archs` completed the ladder (fp32 → activation repair → bf16), and  
2. Each ended **weight_bound**, and  
3. **No** arch PASSED.

If **any** arch PASSED → CONFIG_UPDATE `supported_archs: [n150]` and/or `[p150]` — **no multichip**.

## pytest / YAML alignment

Match [PR #4810](https://github.com/tenstorrent/tt-xla/pull/4810):

- Mark tests `@pytest.mark.n150` / `@pytest.mark.p150` per component.
- YAML `supported_archs` must match passing arches only.

## CONFIG_UPDATE

```yaml
<model_key>:
  status: EXPECTED_PASSING
  supported_archs: [n150, p150]   # only arches that passed HW
```

Use `arch_overrides` when one logical model_key needs different status per arch.

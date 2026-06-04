# Arch eligibility — n150 and p150 single-chip

## Defaults

- **`--archs n150,p150`** when `weight_fit.json` → `eligible_archs` contains both.
- Run **full single-chip ladder on each** eligible arch in order: **n150**, then **p150**.
- **`--arch p150`** when scaffold marks **p150-only** (e.g. a component that only fits
  Blackhole DRAM — see `weight_fit.json` → `p150_only`).

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

1. Every arch in `eligible_archs` completed the ladder (dtype → activation repair → bf16 if needed), and  
2. Each ended **weight_bound**, and  
3. **No** arch PASSED.

If **any** arch PASSED → CONFIG_UPDATE `supported_archs: [n150]` and/or `[p150]` — **no multichip**.

## Runner tests vs component tests

### `tests/runner/` (collect via `test_all_models_torch`)

Arch selection is driven by **YAML config**, not per-test arch markers:

- Single-device:
  `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
- Tensor parallel:
  `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`
- LLM variants:
  `tests/runner/test_config/torch_llm/test_config_inference_*.yaml`

Set **`supported_archs`** on each model entry to the machines that passed bringup
(e.g. `[n150, p150]` or `[p150]` only). The runner resolves `TT_XLA_ARCH` and
filters entries against the current host.

Use **`arch_overrides`** when one logical `model_key` needs different status or
arch lists per machine (see `tests/runner/validate_test_config.py`).

**Do not** add `@pytest.mark.n150` / `@pytest.mark.p150` on runner-collected tests.

### `tests/torch/models/` (component tests)

Component tests are **not** registered in runner YAML. Default: **no arch marker**
— the test runs on whatever device is present (`TT_XLA_ARCH` / current silicon).

Add a **skip guard** only when a component genuinely cannot run on the current
device class. Use `get_xla_device_arch()` from `tests/runner/test_utils.py`:

```python
from tests.runner.test_utils import get_xla_device_arch

def test_transformer():
    arch = get_xla_device_arch()
    if arch == "wormhole" and _component_is_p150_only():
        pytest.skip("transformer requires Blackhole (p150) — see weight_fit.json")
    ...
```

Map device class → bringup arch when documenting skips:

| `get_xla_device_arch()` | Single-chip bringup arch |
|-------------------------|--------------------------|
| `wormhole`              | `n150`                   |
| `blackhole`             | `p150`                   |

Record per-component eligibility in **`weight_fit.json`** (`eligible_archs`,
`p150_only`); mirror passing arches into runner YAML when the component is also
exposed as a `model_key`.

## CONFIG_UPDATE

```yaml
<model_key>:
  status: EXPECTED_PASSING
  supported_archs: [n150, p150]   # only arches that passed HW
```

Use `arch_overrides` when one logical model_key needs different status per arch.

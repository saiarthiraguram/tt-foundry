# weight_fit.json schema

Written at VALIDATE: `.claude/bringup/<safe_key>/weight_fit.json`

Pipeline models: one file with `components[]` array; monolithic models: single top-level object.

```json
{
  "model_key": "family/pytorch-Variant-single_device-inference",
  "hf_repo": "org/model",
  "param_estimate_source": "loader | config | name_heuristic",
  "components": [
    {
      "name": "transformer",
      "num_params": 10000000000,
      "weight_bytes_fp32": 40000000000,
      "weight_bytes_bf16": 20000000000,
      "activation_class": "video",
      "eligible_archs": ["p150"],
      "p150_only": false,
      "parallelism_mode": "single_device",
      "single_device_only": true,
      "test_path": "tests/torch/models/<family>/test_transformer.py::test_transformer",
      "per_arch": {
        "n150": {
          "dram_gib": 12,
          "dram_bytes": 12949672960,
          "budget_bytes": 11007202022,
          "fits_fp32": false,
          "fits_bf16": false,
          "weight_predicted": true
        },
        "p150": {
          "dram_gib": 32,
          "dram_bytes": 34359738368,
          "budget_bytes": 29205737613,
          "fits_fp32": true,
          "fits_bf16": true,
          "weight_predicted": false
        }
      },
      "supported_archs": [],
      "test_path": "tests/torch/models/.../test_....py::..."
    }
  ]
}
```

`budget_bytes = floor(0.85 * dram_bytes)`.

`fits_fp32` := `weight_bytes_fp32 <= budget_bytes` (activation derate not applied to fits_*; used only in classify-oom).

Scaffold sets `eligible_archs` from `fits_fp32 || fits_bf16` OR post-probe update.

`parallelism_mode`: **`single_device`** (default when weights fit one chip) or
**`tensor_parallel`** when weight-bound on all eligible single-chip arches.
Drives pytest marker (`single_device` vs `tensor_parallel`) and which
`test_path` node to run.

`test_path`: pytest node under **`tests/torch/models/<family>/`** — **not**
`test_all_models_torch[...]` for pipeline components.

`single_device_only`: **true** for encoders/VAE that fit one chip; never TP-promote;
may replicate on mesh when a sibling DiT shards.

**PCC:** enforced in the test file (`required_pcc=0.99`), not runner YAML.

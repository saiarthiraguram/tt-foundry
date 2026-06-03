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

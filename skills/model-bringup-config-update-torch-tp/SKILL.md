---
name: model-bringup-config-update-torch-tp
description: CONFIG_UPDATE for tensor_parallel YAML after multichip PASSED. Dry-run by default; --apply writes test_config_inference_tensor_parallel.yaml.
allowed-tools: Read Write Edit Bash Grep
---

# Config update — torch TP

## Invocation

`/model-bringup-config-update-torch-tp --result PASSED|ESCALATED [--arch n300-llmbox] [--apply]`

## Target (pipeline vs monolithic)

| Surface | On PASSED (--apply) |
|---------|---------------------|
| **Pipeline component** (`test_path` under `tests/torch/models/`) | Update sharded test: `@pytest.mark.tensor_parallel`, `bringup_status=EXPECTED_PASSING`, keep `required_pcc=0.99` in `ComparisonConfig`. **No** runner YAML. |
| **Monolithic** `test_models.py` key | `tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml` |

## PASSED (--apply) — monolithic YAML only

```yaml
<model_key>:
  status: EXPECTED_PASSING
  supported_archs: [<multichip arch>]
  assert_pcc: false
```

## PASSED (--apply) — pipeline component test

Edit `tests/torch/models/<family>/test_<role>.py`:
- Promote sharded node (`test_*_sharded`) to `BringupStatus.EXPECTED_PASSING`
- Ensure `@pytest.mark.nightly`, `@pytest.mark.model_test`, `@pytest.mark.tensor_parallel`
- `ComparisonConfig(pcc=PccConfig(required_pcc=0.99))` must already pass from run-torch-tp

## Dry-run

Write `.claude/bringup/<safe_key>/config_update_tp_proposed.md` with diff + provenance.

Do not add pipeline component entries to runner YAML when promoting TP.

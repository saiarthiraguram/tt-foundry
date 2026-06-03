---
name: model-bringup-config-update-torch-tp
description: CONFIG_UPDATE for tensor_parallel YAML after multichip PASSED. Dry-run by default; --apply writes test_config_inference_tensor_parallel.yaml.
allowed-tools: Read Write Edit Bash Grep
---

# Config update — torch TP

## Invocation

`/model-bringup-config-update-torch-tp --result PASSED|ESCALATED [--arch n300-llmbox] [--apply]`

## Target file

`tests/runner/test_config/torch/test_config_inference_tensor_parallel.yaml`

## PASSED (--apply)

```yaml
<model_key>:
  status: EXPECTED_PASSING
  supported_archs: [<multichip arch>]
  assert_pcc: false   # generative default; set true when stable
```

Mirror `bringup_status` in component test file if present.

## Dry-run

Write `.claude/bringup/<safe_key>/config_update_tp_proposed.md` with diff + provenance (tt-xla SHA, promotion.json arch_results).

Do not modify single_device YAML entries when promoting — they reflect single-chip attempts.

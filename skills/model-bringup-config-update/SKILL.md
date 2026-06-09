---
name: model-bringup-config-update
description: Config update stage of the model bringup pipeline. Updates the test YAML config and bringup_status marker in the test fixture to reflect the final outcome (PASSED, TIMEOUT, or ESCALATED). Invoked by the model-bringup orchestrator as the final stage.
allowed-tools: Read Write Edit Bash Grep
---

# Model Bringup — Config Update

You are the **config update** stage of the model bringup pipeline.

## Invocation
`/model-bringup-config-update <model_key> --result <PASSED|TIMEOUT|ESCALATED> [--arch <arch>] [--apply]`

When the orchestrator finishes the per-arch loop with **at least one passing
arch**, pass `--result PASSED` once (no per-arch invoke). Read
`state.arch_results` for the full `supported_archs` list.

For **promotion** (all arches weight-bound), the orchestrator invokes
`model-bringup-write-promotion` instead — this skill is not used for that path.

## Responsibility
Update the test YAML configuration and the test file's `bringup_status`
marker to reflect the final bringup outcome.

## Default mode: dry-run

Unless `--apply` is passed, this skill **does not mutate any file**. It
writes a proposed-change report to
`.claude/bringup/<safe_key>/config_update_proposed.md` with:
- A unified-diff snippet for the YAML entry.
- A unified-diff snippet for the test-fixture `bringup_status` line.
- The reason the change is being proposed (PASSED / TIMEOUT / ESCALATED).
- The provenance block (tt-xla SHA, tt-foundry SHA, timestamp, source skill).

The orchestrator should re-invoke with `--apply` once the user (or a
non-interactive policy) approves. This matches the opt-in mutate convention
that `model_issue_pick` already uses, so the pipeline never silently writes
to the YAML.

## Provenance block (write into both the proposed-change report and the
final escalation_report when in ESCALATED state)

```
Provenance:
  tt-xla SHA       : <short sha>
  tt-foundry SHA   : <short sha or 'not a submodule'>
  Generated        : <YYYY-MM-DD HH:MM>
  Source skill     : model-bringup-config-update
  Result classified from : json_report | stdout_fallback
```

## Steps

### 1. Determine test surface (pipeline vs monolithic)

Read `state.json` / `weight_fit.json`:

| `details.scaffold_variant` or `test_path` | Config target |
|-------------------------------------------|---------------|
| `"pipeline"` or `test_path` under `tests/torch/models/` | **Component test file only** — do **not** edit runner YAML |
| Monolithic / `test_all_models_torch[...]` | `test_config_inference_single_device.yaml` |

### 2. Locate bringup_status

**Pipeline components:** `bringup_status=BringupStatus.<value>` in
`tests/torch/models/<family>/test_*.py` (`record_test_properties` or inline).

**Monolithic:** fixture in `third_party/tt_forge_models/<family>/pytorch/tests/test_*.py`
or runner-driven test under `tests/runner/`.

### 3. Locate YAML (monolithic only)

If **not** a pipeline component test, check:
1. `tests/runner/test_config/torch/test_config_inference_single_device.yaml`
2. Other YAML under `tests/runner/test_config/` for run_mode / parallelism.

**Skip YAML entirely** when `test_path` matches `tests/torch/models/<family>/`.

### 4. Apply the update (only if `--apply` was passed)

If `--apply` was NOT passed: write the proposed-change diff to
`config_update_proposed.md` per the **Default mode: dry-run** section above
and stop. The orchestrator records the dry-run path in state.json under
`history[].details.proposed_path` and exits this stage.

If `--apply` WAS passed: continue with the per-result branches below.

**If result == PASSED:**

1. **Resolve `supported_archs` from `state.arch_results`** (preferred over
   `--arch` alone):
   ```bash
   PASSING=$(jq -r '.arch_results | to_entries[] | select(.value == "passed" or .value.result == "passed") | .key' state.json | sort -u | tr '\n' ',' | sed 's/,$//')
   ```
   - If `arch_results` is empty, fall back to `--arch` (single-arch debug).
   - Example: both n150 and p150 passed → `supported_archs: [n150, p150]`.
   - Example: only p150 passed → `supported_archs: [p150]`.
2. Set `bringup_status=BringupStatus.EXPECTED_PASSING` in the **component test**
   (pipeline) or monolithic fixture.
3. Ensure the test is **not** marked with `pytest.mark.skip` (unless documented OOM).
4. **Pipeline components:** confirm the test already enforces PCC at bringup time:
   ```python
   comparison_config=ComparisonConfig(pcc=PccConfig(required_pcc=0.99))
   ```
   If missing or below `0.99`, add/fix in the test file — runner YAML `required_pcc`
   does **not** apply to `tests/torch/models/` nodes. Bringup only counts as PASSED
   if `model-bringup-run` executed that test and PCC passed.
5. **Monolithic only:** update YAML:
   ```yaml
   <model_key>:
     status: EXPECTED_PASSING
     supported_archs: [<passing arches>]
     required_pcc: 0.99
     assert_pcc: false
   ```
6. Update **`weight_fit.json`**: set top-level or per-component
   `supported_archs` to the same passing list (mirror YAML).
7. Update `state.json`: set `stage: "passed"`, persist
   `details.supported_archs: [<list>]`.
8. Append history entry:
   `{ "stage": "config_update", "result": "passed", "details": { "supported_archs": [...] } }`.

See `model-bringup-multichip/references/arch_eligibility.md`.

**If result == TIMEOUT:**
- Set `bringup_status=BringupStatus.UNKNOWN` in the test fixture.
- Add `pytest.mark.skip(reason="<timeout_reason>")` to the test variant.
- In the YAML config, add entry with `status: NOT_SUPPORTED_SKIP`.
- Update `state.json`: set `stage: "not_supported_skip"`.
- Append history entry: `{ "stage": "config_update", "result": "not_supported_skip" }`.
- Write `.claude/bringup/<safe_key>/escalation_report.md` with timeout details.

**If result == ESCALATED:**
- Set `bringup_status=BringupStatus.KNOWN_FAILURE_XFAIL` in the test fixture (or
  `BringupStatus.UNSPECIFIED` if cause is truly unknown).
- Add `@pytest.mark.xfail(reason="<last failure_reason from state.json>")` to the test.
- In the YAML config, add entry with `status: KNOWN_FAILURE_XFAIL` and `reason:`.
- Update `state.json`: set `stage: "escalated"`.
- Append history entry: `{ "stage": "config_update", "result": "escalated" }`.
- Write `.claude/bringup/<safe_key>/escalation_report.md` with:
  - model_key, arch
  - All history entries (stage, result, details)
  - Final diagnosis and repair attempts
  - Recommended next human action

### 4. Write to bringup_steps.txt

Timing capture: `_step_start_ts`/`_step_start_iso` at the very start of this
skill's execution; `_step_end_ts`/`_step_end_iso` after the YAML and fixture
edits are flushed. After that, also capture pipeline-level totals:
```bash
_pipeline_end_ts=$(date +%s)
_pipeline_end_iso=$(date -Iseconds)
_pipeline_start_ts=$(jq -r '.pipeline_start_ts' .claude/bringup/<safe_key>/state.json)
_total_elapsed_s=$((_pipeline_end_ts - _pipeline_start_ts))
# Human-readable form: Xh Ym Zs (drop leading 0h / 0m components)
```

Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP <N> — Config Update (model-bringup-config-update)
--------------------------------------------------------------------------------
Start    : <_step_start_iso>
End      : <_step_end_iso>
Elapsed  : <_step_end_ts - _step_start_ts>s

Result        : PASSED | TIMEOUT | ESCALATED

bringup_status updated:
  File : <test file path>
  From : BringupStatus.<old>
  To   : BringupStatus.<new>

YAML config updated:
  File  : tests/runner/test_config/torch/test_config_inference_single_device.yaml
  Entry :
    <yaml block>

[If ESCALATED or TIMEOUT:]
  Escalation report: .claude/bringup/<safe_key>/escalation_report.md

CONFIG UPDATE RESULT: PASSED | TIMEOUT | ESCALATED
```

Then close the file with the final summary block. Use the pipeline-level
totals captured above:
```
================================================================================
FINAL RESULT
================================================================================
<✓|✗> <model_key> — <PASSED|ESCALATED|TIMEOUT> after <N> repair iteration(s)
  Loader created  : yes | no
  Applied patches : <list or 'none'>
  Start time      : <pipeline_start_iso from state.json>
  End time        : <_pipeline_end_iso>
  Total elapsed   : <Xh Ym Zs>  (<_total_elapsed_s>s)
  YAML entry      : <key added to YAML or 'none'>
================================================================================
```

Also persist `pipeline_end_ts`, `pipeline_end_iso`, and `total_elapsed_seconds`
in `state.json` so a downstream aggregator (e.g. a multi-model timing study)
can read structured timing data without re-parsing the .txt log.

### 5. Output

On PASSED:
```
[config-update] PASSED
  bringup_status → EXPECTED_PASSING
  test file:  <path>
  yaml entry: <path>
```

On TIMEOUT:
```
[config-update] UNKNOWN
  bringup_status → UNKNOWN
  reason:     exceeded 300s execution limit
  report:     .claude/bringup/<safe_key>/escalation_report.md
```

On ESCALATED:
```
[config-update] ESCALATED
  bringup_status → KNOWN_FAILURE_XFAIL
  report:     .claude/bringup/<safe_key>/escalation_report.md
```

---
name: model-bringup-scaffold
description: Scaffold and validation stage of the model bringup pipeline. Validates that a model_key has a loader and is importable. If the loader is missing, creates loader.py and package __init__.py files following the pattern of existing models. Initializes state.json. Invoked by the model-bringup orchestrator at the VALIDATE stage.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Model Bringup — Scaffold & Validation

You are the **scaffold and validation** stage of the model bringup pipeline.

> **Pipeline models — delegate.** If the HF target resolves to a
> `DiffusionPipeline` (i.e. its `model_index.json` has `_class_name` ending in
> `Pipeline` and lists multiple component sub-folders: `unet`/`transformer`,
> `vae`, `text_encoder`, …), do **not** scaffold here. A single loader cannot
> serve a multi-component pipeline cleanly. Hand off to
> `model-bringup-scaffold-pipeline`, which scaffolds one variant per
> component, captures real per-component I/O shapes, and emits
> shard specs / `TT_VISIBLE_DEVICES` planning.

## Invocation
`/model-bringup-scaffold <model_key> [--arch <arch>] [--custom-test]`

- `--custom-test` — opt-in flag. When set (or auto-detected via the
  heuristic in Step 4c), scaffold writes a per-model test file in addition
  to relying on the generic `tests/runner/test_models.py` discovery. Use
  for models where the generic runner cannot drive the loader cleanly
  (multi-modal inputs, non-tensor outputs, or any case where the model
  needs assertions on one specific output field).

## Responsibility
Validate that the model_key is ready to run before any test execution begins.
If any file is missing, **create it** following the conventions of existing models.
Initialize the bringup state directory and state.json.

---

## Step 1 — Parse model_key

The model_key may be in one of two formats:

**Format A — structured key** (preferred):
```
<family>/<framework>-<variant>-<parallelism>-<run_mode>
```
Example: `ltx2/pytorch-Fast-single_device-inference`
- `family` = `ltx2`
- `framework` = `pytorch`
- `variant` = `Fast`
- `parallelism` = `single_device`
- `run_mode` = `inference`

**Format B — HuggingFace model ID** (e.g. `google/bert_uncased_L-2_H-128_A-2`):
- Detect: Part 1 does not match `<framework>-<variant>-<parallelism>-<run_mode>`
- Normalize: derive `family` from the model name (e.g. `bert_tiny`), set `framework=pytorch`, `parallelism=single_device`, `run_mode=inference`, derive `variant` from the repo name
- Synthesize the structured key and continue

---

## Step 2 — Locate or create the loader

Check for `third_party/tt_forge_models/<family>/pytorch/loader.py`.

**If loader exists:** proceed to Step 3.

**If loader is absent:** create the full model directory structure (see "Creating a new loader" below), then proceed to Step 3.

---

### Creating a new loader

#### 2a. Find a reference model
Find the most similar existing model to use as a template:
- For text/NLP models: look at `bert/`, `gpt_neo/`, `llama/`
- For image models: look at `resnet/`, `clip/`
- For video/diffusion: look at `flux/`, `ltx2/`
- Use `find third_party/tt_forge_models -name loader.py | head -5` to browse

Read the reference `loader.py` in full before writing anything.

#### 2b. Inspect the HuggingFace model
Use `python -c` to inspect the model:
```python
from transformers import AutoModel, AutoConfig
config = AutoConfig.from_pretrained("<hf_model_id>")
print(config)
model = AutoModel.from_pretrained("<hf_model_id>")
print(type(model))
# Print forward signature
import inspect
print(inspect.signature(model.forward))
```
Record: model class, input names, output structure, hidden dim, num layers.

#### 2c. Create directory structure
```
third_party/tt_forge_models/<family>/
├── __init__.py
└── pytorch/
    ├── __init__.py          ← re-exports ModelLoader, ModelVariant
    └── loader.py            ← model loader
```

No `tests/` subdirectory is needed. `tests/runner/test_models.py` discovers models
automatically via the dynamic loader from `loader.py` alone.

#### 2d. Write `loader.py`
Follow the reference model's structure exactly:
- `ModelVariant(StrEnum)` — one entry per checkpoint; name the variant after the HF repo slug (e.g. `L2_H128_A2`)
- `ModelConfig` or `LLMModelConfig` — set `pretrained_model_name`
- `ModelLoader(ForgeModel)`:
  - `_get_model_info()` — use appropriate `ModelTask` (e.g. `NLP_MASKED_LM`, `MM_VIDEO_TTT`)
  - `load_model()` — load from HuggingFace with `torch_dtype=torch.bfloat16`
  - `load_inputs()` — generate synthetic inputs matching the model's `forward()` signature
  - `unpack_forward_output()` — extract tensor from dataclass/tuple/dict output
- SPDX header at top

#### 2e. Write `pytorch/__init__.py`
```python
# SPDX-FileCopyrightText: (c) 2025 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
from .loader import ModelLoader, ModelVariant
```

---

## Step 3 — Validate imports

Run:
```bash
python -c "from third_party.tt_forge_models.<family>.pytorch import ModelLoader, ModelVariant; print('OK')"
```

If this fails, fix the import error before proceeding.

---

## Step 4 — Validate discoverability via collect

Confirm the model is visible to `tests/runner/test_models.py`:
```bash
pytest -q --collect-only tests/runner/test_models.py 2>&1 \
  | grep "test_all_models_torch\[<family>/pytorch-"
```

If no lines appear, the loader failed to import during collection — check the import error in the collect output and fix it before proceeding.

---

## Step 4b — Pre-flight size gate

Before any pytest run, estimate the model's parameter count and reject
anything that obviously cannot fit on a single device. This prevents the
pipeline from burning an hour on a 70B model that will OOM during compile.

Order of preference:

1. **Live count from the loader** (most accurate):
   ```python
   from third_party.tt_forge_models.<family>.pytorch import ModelLoader
   m = ModelLoader().load_model()
   n = sum(p.numel() for p in m.parameters())
   ```
   Skip if the loader requires HF download and offline-mode is set, or if
   `load_model()` would itself cost minutes (large checkpoints).

2. **Config-only count** (fast, weights not loaded):
   ```python
   from transformers import AutoConfig
   c = AutoConfig.from_pretrained("<hf_id>")
   # most HF configs expose num_parameters or hidden_size+num_layers+vocab_size
   ```

3. **Name-based heuristic** (always available — same regex table as
   `failure_summary`):
   - Explicit `\d+(?:\.\d+)?B` tokens → that many billion params
   - `\d+M` tokens → that many million
   - Fallback: 100M

**Gate logic** (single-device-inference target on n150):

| Params estimate | Action |
|---|---|
| < 14 B  | proceed |
| 14–30 B | warn — likely OOM, **emit shard plan** (see below), then ask the user to confirm before continuing |
| > 30 B  | **reject** with `result=blocked`, `block_reason="exceeds single-device capacity; route to multi_chip pipeline"` (still emit shard plan so escalation has something to act on) |

Record the estimate (and its source: `loader | config | name_heuristic`) in
state.json under `details.param_estimate` so escalation reports can show
the gate decision.

### Shard plan (warn-band or reject)

When the estimate is ≥ 14 B, compute and persist a shard plan into
`state.json` under `details.shard_plan`:

```json
{
  "params_billions": <X>,
  "fits_single_device_n150": false,
  "suggested_mesh": "1x2" | "1x4" | "1x8" | "2x4",
  "tt_visible_devices": "0,1" | "0,1,2,3" | ...,
  "rationale": "<one line: 'fp16 param bytes / device L1 budget' or 'matches mesh used by <peer model>'>"
}
```

Mesh selection (rules of thumb — pick the smallest mesh that fits):
- ~14–20 B fp16 → `1x2` (`TT_VISIBLE_DEVICES=0,1`)
- ~20–30 B fp16 → `1x4` (`TT_VISIBLE_DEVICES=0,1,2,3`)
- > 30 B fp16   → `1x8` or `2x4` (still reject for single-device pipeline
  but record the plan so the user can re-invoke against the multi-chip
  pipeline with the right environment).

Also print the plan to the steps log under
`STEP 1 — Parse & Scaffold` so the orchestrator surfaces it on warn/reject.

---

## Step 4c — Optional: custom per-model test file

The generic `tests/runner/test_models.py` discovery handles the vast
majority of models. **Skip this step** unless one of the following is true:

1. The user passed `--custom-test`.
2. **Auto-detect trigger** — any of:
   - `load_inputs()` returns a dict whose keys do not match the
     `model.forward` signature exactly (post-`unpack` keys, kwargs-only
     args, etc.).
   - The loader exposes `load_model(component=...)` (pipeline-style
     multi-component model — the generic runner only sees the default
     component).
   - `unpack_forward_output` returns a non-tensor that the runner cannot
     coerce (e.g. a list of tensors with different shapes).

If triggered, scaffold writes
`third_party/tt_forge_models/<family>/pytorch/tests/test_<family>.py`
following the template below (and creates `tests/__init__.py` if absent).

Template:
```python
# SPDX-FileCopyrightText: (c) 2026 Tenstorrent AI ULC
#
# SPDX-License-Identifier: Apache-2.0
import pytest
import torch
import torch_xla.core.xla_model as xm

from third_party.tt_forge_models.<family>.pytorch import ModelLoader, ModelVariant


@pytest.mark.single_device
@pytest.mark.record_test_properties(
    category="model_test",
    model_name="<family>",
    run_mode="inference",
    parallelism="single_device",
    bringup_status="<inferred BringupStatus, default INCOMPLETE>",
)
def test_<family>_inference_single_device():
    loader = ModelLoader()
    model = loader.load_model().eval()
    inputs = loader.load_inputs()

    device = xm.xla_device()
    model = model.to(device)
    inputs = {k: v.to(device) if torch.is_tensor(v) else v for k, v in inputs.items()}

    out = model(**inputs) if isinstance(inputs, dict) else model(*inputs)
    tensor = loader.unpack_forward_output(out) if hasattr(loader, "unpack_forward_output") else out
    xm.mark_step()

    assert torch.is_tensor(tensor)
    assert torch.isfinite(tensor.cpu()).all(), "non-finite output on device"
```

Notes:
- Leave the assertion body intentionally light — the runner's PCC and
  bringup_status reporting do the heavy lifting elsewhere.
- Mark `bringup_status=INCOMPLETE` initially; `model-bringup-config-update`
  flips it at the end of the pipeline.
- If `--custom-test` was passed but no auto-trigger fired, still write the
  file. Do not overwrite an existing test_<family>.py — if one is already
  present, log "custom test already exists" and continue.

Record in state.json:
```json
"details": { "custom_test_path": "third_party/tt_forge_models/<family>/pytorch/tests/test_<family>.py" | null }
```

## Step 5 — Initialize bringup state

Create `.claude/bringup/<safe_model_key>/` with subdirectories `logs/` and `patches/`.

Write `state.json`:
```json
{
  "model_key": "<model_key>",
  "arch": "<arch>",
  "stage": "validate",
  "iteration": 0,
  "history": [],
  "applied_patches": [],
  "failure_reasons": [],
  "created_at": <unix_timestamp>,
  "updated_at": <unix_timestamp>
}
```

Append history entry: `{ "stage": "validate", "result": "passed", "details": { "loader_path": "third_party/tt_forge_models/<family>/pytorch/loader.py", "loader_created": true|false } }`.

---

## Step 6 — Write to bringup_steps.txt

Append a section to `.claude/bringup/<safe_key>/bringup_steps.txt`.
If the file does not exist yet (first stage), write the header block first:
```
================================================================================
MODEL BRINGUP LOG
================================================================================
Model Key  : <model_key>
Arch       : <arch>
Date       : <YYYY-MM-DD>
================================================================================
```

Then append:
```
--------------------------------------------------------------------------------
STEP 1 — Parse & Scaffold (model-bringup-scaffold)
--------------------------------------------------------------------------------
Input model_key : <original model_key>
Format detected : A (structured) | B (HuggingFace model ID)
Normalized to   : family=<family>  variant=<variant>  parallelism=<p>  run_mode=<r>

Loader path     : third_party/tt_forge_models/<family>/pytorch/loader.py
Loader created  : yes | no

[If created:]
  Reference model : <reference loader path>
  Model class     : <class name>
  HF model ID     : <id>
  Input signature : <key input fields>
  Files written   :
    - third_party/tt_forge_models/<family>/__init__.py
    - third_party/tt_forge_models/<family>/pytorch/__init__.py
    - third_party/tt_forge_models/<family>/pytorch/loader.py

Import validation  : python -c "from ... import ModelLoader, ModelVariant" → OK | FAILED
Collect validation : pytest --collect-only tests/runner/test_models.py | grep '<family>/pytorch-' → <N> test(s) found | NONE FOUND

Size gate          : <X> B params (source: loader | config | name_heuristic) → proceed | warn | reject
Shard plan         : <mesh> / TT_VISIBLE_DEVICES=<list> | n/a (single-device fits)

Custom test file   : <path or 'none — generic runner suffices'>

SCAFFOLD RESULT: PASSED | FAILED
```

## Step 7 — Output

On success:
```
[scaffold] PASSED
  loader:         third_party/tt_forge_models/<family>/pytorch/loader.py
  collect check:  <N> test(s) visible in tests/runner/test_models.py
  loader_created: yes | no
```

Only exit with failure (and let the orchestrator escalate) if:
- The HF model ID is unreachable / does not exist
- The model inspection (`AutoModel.from_pretrained`) fails with an unrecoverable error
- The import validation still fails after creating the loader (syntax/logic error)
- The size gate (Step 4b) rejects the model as too large for single-device

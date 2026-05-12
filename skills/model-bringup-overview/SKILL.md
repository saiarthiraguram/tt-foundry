---
name: model-bringup-overview
description: Overview, CPU sanity, and golden-reference capture stage of the model bringup pipeline. Writes a one-page model_overview.md, runs the loader on CPU with random inputs to confirm it produces a finite forward output, and persists a golden.pt reference tensor for downstream PCC comparison. Invoked by the model-bringup orchestrator at the OVERVIEW stage, between VALIDATE and FIRST_RUN.
allowed-tools: Bash Read Write Edit Grep
---

# Model Bringup — Overview, CPU Sanity & Golden Capture

You are the **overview + CPU sanity + golden capture** stage of the model
bringup pipeline. Runs **after** scaffold succeeds and **before** any
hardware execution. The goal is to surface loader/forward bugs in seconds on
CPU, write a human-readable model card, and persist a reference output
tensor so the repair stage can compute real PCC rather than blindly
lowering the threshold.

## Invocation
`/model-bringup-overview <model_key> [--skip-golden]`

`--skip-golden` is honoured when the forward output is huge (>1 GiB) or
non-tensor in a way that does not fit a single `.pt` file.

## State location
All artefacts under `.claude/bringup/<safe_key>/`:
- `model_overview.md` — the one-pager (G1).
- `cpu_sanity.log` — stdout/stderr of the CPU forward (G2).
- `golden.pt` — captured output (G3); `golden_meta.json` records its shape/dtype.

`safe_key` = `model_key` with `/` replaced by `__`.

---

## Step 1 — Load loader metadata

Import the loader and pull a metadata snapshot. Random weights are fine —
do **not** download HF checkpoints here (see `user_bringup_prefs`).

```python
from third_party.tt_forge_models.<family>.pytorch import ModelLoader, ModelVariant
loader = ModelLoader()
info = loader._get_model_info()      # ModelTask, model_name, etc.
model = loader.load_model()          # random init OK
inputs = loader.load_inputs()
import inspect, torch
sig = inspect.signature(model.forward)
n_params = sum(p.numel() for p in model.parameters())
```

Capture (best effort — wrap in `try/except` and record `null` on failure):
- `info.model_name`, `info.task`, `info.framework`
- `type(model).__name__`, fully-qualified module path
- `n_params` (live), HF id from the loader's `pretrained_model_name`
- Input field names + per-field shape + dtype
- `forward` signature string

---

## Step 2 — Write `model_overview.md`

Layout:

```
# <family> / <variant>

| Field            | Value |
|------------------|-------|
| HF model ID      | <id> |
| Model class      | <fully qualified> |
| Task             | <ModelTask enum> |
| Modality         | <text | vision | audio | video | multimodal> |
| Parameters       | <N> (≈ <X> B) |
| Forward signature| `<inspect.signature>` |

## Inputs
- `<field>`: shape=<tuple>, dtype=<dtype>

## Expected output (CPU random-init forward)
<filled by Step 4>

## References
- HF page : https://huggingface.co/<id>
- Paper   : <arXiv link if known, else 'none'>

## Bringup notes
- Source skill : model-bringup-overview
- Generated    : <YYYY-MM-DD HH:MM>
- tt-xla SHA   : <short sha>
```

Modality is derived from `ModelTask` (e.g. `NLP_*` → text, `CV_*` → vision,
`MM_*` → multimodal). If the enum is unfamiliar, leave it as `unknown` —
do not invent.

Paper link: only fill if the loader file or HF README references a known
arXiv id. Otherwise `none` — never fabricate a URL.

---

## Step 3 — CPU forward sanity gate

Run the loader's forward on CPU with the random-init model and synthesized
inputs, capturing stdout+stderr to `cpu_sanity.log`:

```python
import torch
model = model.to(torch.float32).eval()    # use fp32 on CPU; bf16 is HW-only
with torch.inference_mode():
    out = model(**inputs) if isinstance(inputs, dict) else model(*inputs)

# Unpack via the loader if available — same path tt-forge-models uses on HW.
if hasattr(loader, "unpack_forward_output"):
    tensor = loader.unpack_forward_output(out)
else:
    tensor = out

assert torch.is_tensor(tensor), f"unpack_forward_output returned {type(tensor)}"
assert torch.isfinite(tensor).all(), "output contains NaN/Inf on CPU"
print("CPU_SANITY_OK", tuple(tensor.shape), tensor.dtype)
```

Run the whole block inside a single `python -c` (or a temp script). Tee
the output to `cpu_sanity.log`.

**Pass criteria** (all must hold):
- Script exits zero.
- Last log line is `CPU_SANITY_OK <shape> <dtype>`.

**Fail handling** — do **not** proceed to golden capture. Return:
```json
{"result": "failed", "stage": "cpu_sanity", "log": "<path>", "reason": "<short>"}
```
The orchestrator will surface this as a loader bug (not a HW bug) and
escalate without ever invoking `model-bringup-run`.

Common failure modes worth recording in `reason`:
- `forward_signature_mismatch` — `load_inputs()` keys do not match
  `forward()` params.
- `unpack_missing` — `unpack_forward_output` not implemented and forward
  returns a dataclass/tuple.
- `dtype_mismatch` — model has bf16 params on CPU; loader did not cast.
- `nonfinite_output` — random init produced NaN/Inf (rare; flag for
  investigation, do not suppress).

---

## Step 4 — Capture golden reference

Unless `--skip-golden` is set and the previous step recorded
`reason: too_large`, persist the unpacked tensor:

```python
torch.save(tensor.detach().cpu(), ".claude/bringup/<safe_key>/golden.pt")
```

Write `golden_meta.json`:
```json
{
  "shape": [<dim>, ...],
  "dtype": "<str>",
  "n_elements": <int>,
  "rng_seed": <if loader exposes one>,
  "input_hash": "<sha256 of concatenated input bytes, 16 chars>",
  "size_bytes": <int>
}
```

Append the captured shape/dtype to the `## Expected output` section of
`model_overview.md` (the line was a placeholder until now).

**Size guard.** Refuse to write if the tensor would exceed 1 GiB. In that
case set `golden: skipped_too_large` in the result and let downstream PCC
work proceed without a stored reference.

---

## Step 5 — Update state.json

Append a `history` entry:
```json
{
  "stage": "overview",
  "timestamp": <now>,
  "result": "passed" | "failed",
  "details": {
    "overview_md":  "model_overview.md",
    "cpu_sanity":   "cpu_sanity.log",
    "golden":       "golden.pt" | "skipped_too_large" | null,
    "golden_meta":  "golden_meta.json" | null,
    "n_params":     <int>,
    "modality":     "<text|vision|multimodal|...>",
    "output_shape": [<...>],
    "output_dtype": "<str>"
  }
}
```

Persist `details.n_params` and `details.output_shape` at the top level of
state.json under `details.overview` too — the repair stage reads from there
when computing PCC against `golden.pt`.

---

## Step 6 — Bringup steps log

Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP <N> — Overview / CPU Sanity / Golden (model-bringup-overview)
--------------------------------------------------------------------------------
Overview MD   : .claude/bringup/<safe_key>/model_overview.md
CPU sanity    : .claude/bringup/<safe_key>/cpu_sanity.log → <PASS|FAIL: reason>
Golden tensor : .claude/bringup/<safe_key>/golden.pt (<shape>, <dtype>, <bytes>)
                | skipped_too_large | skipped_failed_cpu

Model snapshot:
  class       : <type>
  n_params    : <int> (≈ <X> B)
  modality    : <text|vision|...>
  input keys  : <list>
  out shape   : <tuple>

OVERVIEW RESULT: PASSED | FAILED
```

---

## Step 7 — Output

On success:
```
[overview] PASSED
  card    : .claude/bringup/<safe_key>/model_overview.md
  golden  : .claude/bringup/<safe_key>/golden.pt  (<shape>, <dtype>)
```

On CPU-sanity failure:
```
[overview] FAILED at cpu_sanity — <reason>
  log     : .claude/bringup/<safe_key>/cpu_sanity.log
```

Only exit with failure (and let the orchestrator escalate) if Step 3 fails.
Step 4 falling back to `skipped_too_large` is a soft skip, not a failure.

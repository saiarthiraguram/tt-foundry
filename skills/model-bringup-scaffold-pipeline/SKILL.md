---
name: model-bringup-scaffold-pipeline
description: Scaffold per-component loaders for pipeline models (Stable Diffusion family, video-generation DiTs, any multi-stage HF DiffusionPipeline). Where model-bringup-scaffold writes one loader that returns one model, this skill introspects the pipeline, locates each component's source file, instruments it to capture real I/O shapes/dtypes, then writes a loader where load_model() returns the requested *component* and load_inputs() builds inputs matched to that component's forward signature. Also picks TT_VISIBLE_DEVICES and emits shard specs (uniform across components) based on per-component param counts and the user's target device. Use this skill whenever the HF target is a DiffusionPipeline or any multi-component pipeline — not the default model-bringup-scaffold.
allowed-tools: Bash Read Write Edit Grep Glob AskUserQuestion
---

# Model Bringup — Pipeline Scaffold

You are the **pipeline-specialized** scaffold stage of the model bringup pipeline.
Use this skill *instead of* `model-bringup-scaffold` when the HuggingFace
target is a multi-component pipeline (anything that loads via
`DiffusionPipeline.from_pretrained(...)` and exposes `.unet` / `.transformer` /
`.vae` / `.text_encoder` etc.).

## Invocation
`/model-bringup-scaffold-pipeline <hf_repo_or_model_key> [--arch <arch>] [--device <n150|p150|auto>]`

## Why this exists

The default scaffold skill assumes one model → one loader → one test. Pipeline
models break that assumption in three places:

1. **`load_model()` returning the whole pipeline** forces test scripts to dig
   `pipe.unet`, `pipe.vae`, etc. out themselves. All model-shaping code must
   live in `tt-forge-models`, not in the runner.
2. **`load_inputs()` for a multi-component pipeline is ambiguous** — inputs to
   the UNet, VAE encoder, VAE decoder, and text encoder are different shapes
   and dtypes. A single `load_inputs()` cannot serve all of them.
3. **Device sizing is per-component, not per-pipeline.** A 7 B aggregate
   pipeline can still OOM the UNet if its activations are huge (common in
   video-gen). Shard specs must be chosen so every component runs on the
   *same* chip count (uniform fabric across a pipeline demo).

This skill scaffolds a loader where each component is a *variant*, with its
own `load_model()` slice and its own `load_inputs()` builder generated from
real captured shapes/dtypes — not guessed.

---

## Step 0 — Confirm target is a pipeline

Before doing any work, verify the HF id resolves to a pipeline class:

```python
from diffusers import DiffusionPipeline
import inspect
cls = DiffusionPipeline.from_pretrained.__func__
# Cheap check: list config.json keys — pipelines have _class_name == "<...>Pipeline"
import json, huggingface_hub
p = huggingface_hub.hf_hub_download("<hf_repo>", "model_index.json")
idx = json.load(open(p))
print(idx.get("_class_name"))
print({k: v for k, v in idx.items() if not k.startswith("_")})
```

- If `_class_name` ends in `Pipeline` and `model_index.json` lists multiple
  sub-folders (`text_encoder`, `unet`/`transformer`, `vae`, `scheduler`, …)
  → proceed.
- Otherwise → **abort** with `result=wrong_skill, hint="use model-bringup-scaffold"`.

Do not fall through to single-component scaffolding here — wrong-target
detection should be loud, not silent.

---

## Step 1 — Parse model_key / hf_repo

Accept either:
- A bare HF repo id (`playgroundai/playground-v2.5-1024px-aesthetic`)
- A structured `model_key` (`playground_v2_5/pytorch-Aesthetic_1024px-single_device-inference`)

Derive `family` from the repo slug (lowercased, non-alnum → `_`), e.g.
`playground-v2.5-1024px-aesthetic` → `playground_v2_5_1024px_aesthetic`.
Truncate to a reasonable length (≤ 40 chars).

Persist the chosen `family`, `variant_prefix` (the human-readable variant
stem before the per-component suffix), and `hf_repo` in
`.claude/bringup/<safe_key>/scaffold_pipeline.json` so later steps don't
re-derive.

---

## Step 2 — Pull architecture + parameter counts (NO weights yet)

Goal: enumerate the pipeline's components and their parameter counts *before*
downloading multi-GB weights. Use HF `config.json` per subfolder where
possible.

```python
from huggingface_hub import hf_hub_download
import json, importlib, diffusers
idx = json.load(open(hf_hub_download("<hf_repo>", "model_index.json")))

components = {}              # name -> {class_name, params_estimate, source_file}
for name, spec in idx.items():
    if name.startswith("_") or not isinstance(spec, list): continue
    module_name, class_name = spec   # e.g. ["diffusers", "UNet2DConditionModel"]
    try:
        cfg = json.load(open(hf_hub_download("<hf_repo>", f"{name}/config.json")))
    except Exception:
        cfg = {}
    components[name] = {"module": module_name, "class": class_name, "config": cfg}
```

For each component, compute an approximate parameter count:

- **UNet / transformer (DiT)**: `hidden_size² × num_layers × C` heuristic
  using `cfg`'s `block_out_channels`, `num_attention_heads`, `cross_attention_dim`,
  `transformer_layers_per_block`. If a `num_parameters` field is present in
  the config (newer DiT releases include it), prefer that.
- **VAE**: usually ~80 M; if `cfg.block_out_channels` and `layers_per_block`
  present, use a coarse scaling.
- **Text encoder(s)**: `AutoConfig.from_pretrained("<repo>", subfolder="text_encoder")`
  → use `hidden_size² × num_hidden_layers × 12` as the standard transformer
  estimate.
- **Scheduler / safety_checker / feature_extractor**: skip — no parameters
  to compile.

Fallback: name-based heuristic from `failure_summary` (`\d+B`/`\d+M` regex,
size words like `tiny`/`small`/`base`/`large`/`xl`). The fallback applies
*per component*, not to the aggregate pipeline.

Save the enumeration as a table in `scaffold_pipeline.json`:
```json
{
  "components": [
    {"name": "text_encoder",   "class": "CLIPTextModel",                 "params": 123_000_000, "source": "transformers/models/clip/modeling_clip.py"},
    {"name": "text_encoder_2", "class": "CLIPTextModelWithProjection",   "params": 695_000_000, "source": "transformers/models/clip/modeling_clip.py"},
    {"name": "unet",           "class": "UNet2DConditionModel",          "params": 2_600_000_000, "source": "diffusers/models/unets/unet_2d_condition.py"},
    {"name": "vae",            "class": "AutoencoderKL",                 "params": 84_000_000, "source": "diffusers/models/autoencoders/autoencoder_kl.py"}
  ],
  "total_params": 3_500_000_000,
  "denoising_steps_default": 50
}
```

---

## Step 3 — Locate each component's source file via class name

For every entry in the components table, resolve `<module>.<class>` to a real
file path on disk so step 4 can patch in logging:

```python
import importlib, inspect
mod = importlib.import_module(module_name)
cls = getattr(mod, class_name)
src = inspect.getsourcefile(cls)
```

Record `source_file` in `scaffold_pipeline.json` (absolute path, plus a
repo-relative form). If `inspect.getsourcefile` returns `None` (rare:
class is defined in a `.pyc`-only or zipimported module), fall back to
`importlib.import_module(module_name).__file__` and grep within that file
or its package for `class <class_name>(`.

Schedulers and feature extractors are intentionally skipped here.

---

## Step 4 — Insert logging instrumentation (offline capture only)

Goal: capture the **real** input/output tensor shapes, dtypes, and (where
small) tensor contents for every component on one forward pass of the
pipeline. This is what step 5's `load_inputs()` builders consume.

For every component's source file:

1. Locate the `forward(self, ...)` method via `ast.parse` (do not regex —
   `forward` can be deep inside the class). Record the original line range.
2. Inject at the **first body line**:
   ```python
   import logging; _BR_LOG = logging.getLogger("tt_bringup.<component>")
   _BR_LOG.info(
       "FWD_IN class=%s shapes=%s dtypes=%s",
       type(self).__name__,
       {k: tuple(v.shape) for k, v in locals().items() if hasattr(v, "shape")},
       {k: str(v.dtype) for k, v in locals().items() if hasattr(v, "dtype")},
   )
   ```
3. Inject just before every `return` in `forward`:
   ```python
   _BR_LOG.info("FWD_OUT class=%s out=%r", type(self).__name__,
       <expr summarising the return value's shapes/dtypes>)
   ```
4. For the **denoising loop** specifically (pipelines expose this on the
   pipeline class itself, usually inside `__call__` — look for the
   `for i, t in enumerate(timesteps)` or `for t in self.progress_bar(...)`
   loop), inject `_BR_LOG.info("DENOISE_STEP %d/%d", i, len(timesteps))`.

**Critical**: these edits go into a **temporary venv overlay**, not the
installed `diffusers` / `transformers` packages. Either:

- Copy the source files into `.claude/bringup/<safe_key>/_overlay/<pkg>/<...>.py`
  and prepend the overlay path to `PYTHONPATH` for the capture run only, **or**
- Use `inspect.getsource` + `exec` with monkey-patched `forward` methods on
  the loaded class objects.

Do **not** mutate the installed-package source files — that contaminates
other tests and other models on the same machine. Always restore on exit
(write a cleanup hook in the capture script).

Persist the chosen instrumentation strategy in `scaffold_pipeline.json`
under `capture.strategy`.

---

## Step 5 — Run one tiny pipeline pass and harvest

Run the pipeline **once** with minimal inputs (lowest resolution, batch=1,
num_inference_steps=2) under the overlay from step 4, with logging routed
to `.claude/bringup/<safe_key>/capture.log`:

```bash
DIFFUSERS_VERBOSITY=info \
PYTHONPATH=.claude/bringup/<safe_key>/_overlay:$PYTHONPATH \
python - <<'PY'
import logging, torch
from diffusers import DiffusionPipeline
logging.basicConfig(level=logging.INFO)
pipe = DiffusionPipeline.from_pretrained("<hf_repo>", torch_dtype=torch.bfloat16)
pipe.to("cpu")  # we only need shapes; do not waste a TT card on this
out = pipe("a small red cube on a wooden table",
           num_inference_steps=2, height=64, width=64)
PY
```

Parse `capture.log` and produce a per-component **I/O spec** dict:

```python
spec = {
  "unet": {
    "inputs": [
      {"name": "sample",            "shape": (1, 4, 8, 8), "dtype": "torch.bfloat16"},
      {"name": "timestep",          "shape": (1,),         "dtype": "torch.int64"},
      {"name": "encoder_hidden_states", "shape": (1, 77, 2048), "dtype": "torch.bfloat16"},
      {"name": "added_cond_kwargs", "kind": "dict", "fields": {
         "text_embeds": {"shape": (1, 1280), "dtype": "torch.bfloat16"},
         "time_ids":    {"shape": (1, 6),    "dtype": "torch.bfloat16"}}}
    ],
    "outputs": [{"shape": (1, 4, 8, 8), "dtype": "torch.bfloat16"}],
    "called_per_step": True
  },
  ...
}
```

Also record:
- `denoise_steps_observed`: the iteration count printed by the loop probe.
- Whether each component is called **once per pipeline call** or **once per
  denoise step** (callees-per-step distinction matters for perf budgeting).

Save `spec` to `scaffold_pipeline.json` under `io_spec`.

---

## Step 6 — Device target & shard-spec planning

Ask the user (via `AskUserQuestion`) which target device:

| Option | Aggregate fit (rule of thumb) | TT_VISIBLE_DEVICES default |
|---|---|---|
| `n150`           | ≤ 7 B  | `0`                  |
| `p150`           | ≤ 12 B | `0`                  |
| `auto`           | based on max(component params)  | derived |

Then compute the **chip count** needed:

```python
def needed_chips(params, device, is_video_gen):
    # Video-gen activations balloon — derate by 2× on n150, 1.5× on p150.
    derate = 2.0 if (is_video_gen and device == "n150") else (1.5 if is_video_gen else 1.0)
    effective = params * derate
    cap = {"n150": 7_000_000_000, "p150": 12_000_000_000}[device]
    import math
    return max(1, math.ceil(effective / cap))
```

`is_video_gen` is True if the pipeline class name contains
`Video|I2V|T2V|HunyuanVideo|LTX|Wan|CogVideo`.

Then apply the **uniform-chip-count rule**: if *any* component returns
`needed_chips > 1`, set the pipeline-wide `chip_count` to that max value,
and emit shard specs for **every** component (even ones that would fit on
1). Pipeline demos run all components on the same fabric.

Record in `scaffold_pipeline.json`:
```json
{
  "device": "n150",
  "chip_count": 2,
  "tt_visible_devices": "0,1",
  "shard_specs": {
    "text_encoder":   {"strategy": "data_parallel"},
    "text_encoder_2": {"strategy": "data_parallel"},
    "unet":           {"strategy": "tensor_parallel", "mesh": [1, 2]},
    "vae":            {"strategy": "data_parallel"}
  }
}
```

Shard-spec selection rules:
- `params < per_device_cap` → `data_parallel` (replicate across chips).
- `params ≥ per_device_cap` → `tensor_parallel` with mesh shape
  `[1, chip_count]` (1-D shard, prefer last dim).
- If user passes `--device auto`, choose the smallest device that fits the
  largest component without exceeding `chip_count = 1`; only escalate to
  multi-chip if no single device fits.

If the largest component still does not fit at `chip_count = 8` (the
fabric max we currently support per host), abort with
`result=blocked, block_reason="exceeds host fabric capacity"`.

---

## Step 7 — Write the per-component loader

Directory layout (same as `model-bringup-scaffold`, with one variant
*per component*):

```
third_party/tt_forge_models/<family>/
├── __init__.py
└── pytorch/
    ├── __init__.py          # re-exports ModelLoader, ModelVariant
    └── loader.py
```

`loader.py` must:

1. Define `ModelVariant(StrEnum)` with one entry per component variant.
   Variant names follow `<VariantPrefix>_<Component>`, e.g.
   `AESTHETIC_1024PX_UNET`, `AESTHETIC_1024PX_VAE_DECODER`,
   `AESTHETIC_1024PX_TEXT_ENCODER`. Keep them stable — the test runner
   keys off them in the YAML.
2. `_VARIANTS` maps each variant to a `ModelConfig(pretrained_model_name=<repo>)`
   plus a `subfolder` (the HF model_index.json key — `"unet"`, `"vae"`,
   `"text_encoder"`, etc.).
3. `ModelLoader.__init__` stores `self.component_name = variant_config.subfolder`
   so `load_model()` and `load_inputs()` can branch on it.
4. `load_model()` returns **only the requested component**:
   ```python
   from diffusers import DiffusionPipeline
   pipe = DiffusionPipeline.from_pretrained(repo, torch_dtype=dtype)
   self._component = getattr(pipe, self.component_name)
   return self._component
   ```
   Free `pipe` after extraction (`del pipe`) so we do not hold every
   component in RAM.
5. `load_inputs()` reads the per-component spec from step 5 and synthesizes
   matching tensors with `torch.randn(shape, dtype=...)` (or
   `torch.randint` for integer dtypes like timestep). For dict-valued
   inputs (`added_cond_kwargs`), build the nested dict explicitly.
   **Do not** call the pipeline to derive inputs — that defeats the
   purpose of synthetic inputs.
6. `unpack_forward_output()` matches the output spec from step 5.

Embed the captured I/O spec as a Python literal at the top of the loader
under `_COMPONENT_IO_SPEC = { ... }` so the loader is self-contained and
reproducible without re-running capture.

The same shard-spec dict from step 6 is also embedded near the top, so
the runner can read it without re-deriving:
```python
SHARD_SPECS = { "unet": {...}, "vae": {...}, ... }
TT_VISIBLE_DEVICES = "0,1"
```

---

## Step 8 — Wire up the test runner

The runner-side updates needed so component tests actually execute:

1. **YAML registration**: append one block per component variant to
   `tests/runner/test_config/torch/test_config_inference_single_device.yaml`,
   each at `EXPECTED_PASSING` (or `KNOWN_FAILURE_XFAIL` if the diagnose
   stage hasn't been run yet — caller's choice). Example:
   ```yaml
   playground_v2_5_1024px_aesthetic/pytorch-Aesthetic_1024px_unet-single_device-inference:
     status: UNSPECIFIED
     required_pcc: 0.95
   playground_v2_5_1024px_aesthetic/pytorch-Aesthetic_1024px_vae_decoder-single_device-inference:
     status: UNSPECIFIED
     required_pcc: 0.95
   ...
   ```
2. **Multi-chip variants**: if `chip_count > 1`, the variants live in
   `test_config_inference_multi_chip.yaml` instead, and `parallelism` in
   the variant key changes from `single_device` to e.g. `n300` /
   `tg_8chip` / matching the fabric.
3. **`TT_VISIBLE_DEVICES` plumbing**: emit a one-line note in the bringup
   log telling the operator (or downstream model-bringup-run skill) that
   the run must set `TT_VISIBLE_DEVICES=<value>` and reference
   `SHARD_SPECS` from the loader.

This skill does *not* execute the tests — that is `model-bringup-run`'s
job. It only registers them. Caller delegates the actual run.

---

## Step 9 — State, log, and output

`state.json` gets a `details.scaffold_variant: "pipeline"` marker, plus:

```json
{
  "components_scaffolded": ["unet", "vae", "text_encoder", ...],
  "denoise_steps_observed": 2,
  "device": "n150",
  "chip_count": 2,
  "tt_visible_devices": "0,1",
  "io_spec_path": ".claude/bringup/<safe_key>/scaffold_pipeline.json"
}
```

Append to `bringup_steps.txt`:

```
--------------------------------------------------------------------------------
STEP 1P — Pipeline Scaffold (model-bringup-scaffold-pipeline)
--------------------------------------------------------------------------------
HF repo            : <repo>
Family             : <family>
Pipeline class     : <e.g. StableDiffusionXLPipeline>
Total params       : ~<X.Y B>
Components         : <n>
  - <name>  class=<...>  params=~<...>  source=<...>
  - ...
Capture pass       : <PASS|FAIL>  steps=<N>  log=.claude/bringup/<safe_key>/capture.log
Device target      : <n150|p150|auto-resolved-to-...>
Chip count         : <k>
TT_VISIBLE_DEVICES : <value>
Shard specs        :
  - <component> : <data_parallel|tensor_parallel mesh=...>
  - ...

Loader path        : third_party/tt_forge_models/<family>/pytorch/loader.py
Variants written   :
  - <FAMILY>_<COMPONENT>
  - ...

YAML entries added :
  - <model_key 1>
  - ...

SCAFFOLD RESULT    : PASSED
```

Terminal output (one screen):

```
[scaffold-pipeline] PASSED
  hf repo         : <repo>
  components      : <n>  (unet, vae, text_encoder, ...)
  total params    : ~<X.Y B>
  device          : <n150|p150>
  chip_count      : <k>     (uniform across components)
  TT_VISIBLE_DEVICES=<value>
  loader          : third_party/tt_forge_models/<family>/pytorch/loader.py
  yaml entries    : <n> appended
  next stage      : model-bringup-run (per variant)
```

---

## Hard rules

- **No source-package mutation.** Instrumentation in step 4 must be in a
  temp overlay or runtime monkey-patch. Never write into the installed
  `diffusers` / `transformers` tree — the next test on this machine
  inherits the contamination.
- **One forward capture pass.** Step 5 runs the real pipeline exactly
  once, on CPU, with minimal resolution and 2 denoise steps. We need
  shapes, not images.
- **All model code stays in tt-forge-models.** The test runner must not
  reach into `pipe.unet` itself; the loader returns the component. If you
  catch yourself writing `pipeline.<component>` in a test file, the
  loader is wrong.
- **Synthetic inputs only.** `load_inputs()` builds tensors with the
  captured shape/dtype using `torch.randn`/`torch.randint`. It does
  **not** re-run the pipeline to derive inputs — that breaks the
  component-isolation contract.
- **Uniform chip count across components.** If one component needs N
  chips, all components get shard specs for N chips. Pipeline demos run
  on a single fabric.
- **Bail early on wrong target.** If `model_index.json` does not declare
  a `Pipeline`, return `result=wrong_skill` and let
  `model-bringup-scaffold` handle the single-component case.
- **Size gate still applies, but per-component.** If the largest
  component cannot fit on the chosen device at `chip_count = 8`, block
  the pipeline — do not silently downgrade to a smaller variant.

---

## Cross-reference

- `model-bringup-scaffold` — sibling skill for single-component models;
  this skill replaces it for pipelines.
- `model-bringup-run` — executes per-component variants after this
  skill registers them.
- `failure_summary` — name-based param heuristic table (also used here as
  the step-2 fallback).

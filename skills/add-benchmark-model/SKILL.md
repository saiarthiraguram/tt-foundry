---
name: add-benchmark-model
description: Add a model to the tt-xla benchmark suite and wire it into the nightly perf pipeline. Handles two shapes — single-forward models (vision / llm / encoder / multimodal) via a config-driven test_<model> entry through a shared harness, and multi-component diffusion pipelines (SD / SDXL / Playground / MMDiT) via the four-artifact template (benchmark pipeline + test entry, PCC-gated nightly test, and demo script). Applies the loader-first workaround discipline (InternVL3) and the CPU→TT→CPU eviction / per-component fp32-twin PCC discipline (SDXL-Lightning, Playground v2.5), registers the model in perf-bench-matrix.json, and verifies locally on n150 before landing. Use when the user says "add <model> to benchmarks", "wire <model> into nightly", "add a PCC-gated nightly / demo for <model>", or invokes /add-benchmark-model.
---

# add-benchmark-model — tt-xla Benchmark Onboarding

Add a model to `tests/benchmark/` and register it in the nightly perf matrix.

There are **two shapes** of benchmark-add, and the first thing to decide is which one:

1. **Single-forward model** (vision / encoder / LLM / one-tensor VLM) → one `test_<model>`
   entry driving a shared harness. Reference: the InternVL3 bringup (commit `807201eac`,
   `[Benchmark] Add InternVL3 multimodal benchmark and wire into nightly`); lessons in
   `references/internvl3-case-study.md`. **Read it before touching a VLM or any model that
   uses HF sdpa / dynamic RoPE / boolean-mask indexing.** This is the default path (phases below).
2. **Multi-component diffusion pipeline** (SD / SDXL / Playground / MMDiT / video-DiT — any
   multi-stage `DiffusionPipeline`) → a **four-artifact** bringup: a benchmark pipeline +
   test entry, a **PCC-gated nightly** test, and a **demo script**. References are
   SDXL-Lightning and Playground v2.5; the full template lives in
   `references/imagegen-pipeline-template.md`. **Read it before touching any pipeline model**
   and follow the pipeline phases (§ *Pipeline models* below) instead of the single-forward phases.

Usage: `/add-benchmark-model <model-key-or-loader-path>`

## Architecture of the benchmark suite

Each modality has a **harness** in `tests/benchmark/benchmarks/` and a
**config-driven test file** in `tests/benchmark/`:

| Modality | Harness | Test file | Input shape |
|----------|---------|-----------|-------------|
| Vision (image classifier/segmenter) | `vision_benchmark.py` | `test_vision.py` | single image tensor |
| LLM (decode/prefill) | `llm_benchmark.py` | `test_llms.py` | token ids, generation |
| Encoder (embeddings) | `encoder_benchmark.py` | `test_encoders.py` | token ids, single forward |
| Multimodal / VLM | `multimodal_benchmark.py` | `test_multimodal.py` | **dict** of kwargs (`input_ids`/`attention_mask`/`pixel_values`) |
| Image-gen / diffusion pipeline | `imagegen_benchmark.py` (+ per-model `benchmarks/<model>_pipeline.py`) | `test_imagegen.py` | text prompt → multi-step denoise → image. **See § Pipeline models** — this is a four-artifact bringup, not a single `test_<model>`. |

The split is deliberate: **model-specific config lives in `test_<model>`, reusable
measurement logic lives in the harness.** A new model almost always means a new
`test_<model>` function in an existing test file — not a new harness. Only write a
new harness for a genuinely new modality (new input contract). If you do, model it
on the closest existing harness.

## Phases (single-forward models)

Run sequentially. Stop and report to the user if a phase fails — do not silently continue.
**If the model is a diffusion pipeline, skip these and use § Pipeline models instead.**

### 1. Classify the model & pick the harness
- Determine modality from the loader / model card. Map to the table above.
- **If it's a multi-component `DiffusionPipeline`, stop here and switch to § Pipeline models.**
- Confirm a `tt-forge-models` loader exists for the model
  (`third_party/tt_forge_models/<family>/pytorch/loader.py` exposing `ModelLoader`,
  `ModelVariant`, `load_model`, `load_inputs`, `get_model_info`). If it does **not**,
  this is a bringup, not a benchmark-add — stop and point the user at `/model-bringup`.
- Pick the target test file. If the modality has no harness yet, flag it to the user
  and propose authoring one (rare).

### 2. Establish the I/O contract
- `load_model(dtype_override=...)` returns the model; `load_inputs(dtype_override=...)`
  returns the inputs. Inspect their actual return types — the harness depends on it:
  - Vision harness feeds **one tensor**; multimodal harness feeds a **dict of kwargs**
    and keeps integer tensors integer (only weights + `pixel_values` are bf16 — see
    `multimodal_benchmark.py:_move_inputs_to_device`).
  - Determine what `extract_output_tensor_fn` must return: the harness computes PCC on
    a single tensor, so extract `.logits` / the wrapper's tensor output.
- Run **one CPU forward** with the loader's inputs to confirm it produces a finite
  output before involving hardware. This is your golden reference.

### 3. Write `test_<model>`
- Add a `def test_<model>(output_file, request):` to the chosen test file, following
  the existing entries (see `test_internvl3` in `test_multimodal.py` for the canonical
  shape). It must:
  1. Import `ModelLoader` / `ModelVariant` from the loader **inside the function**
     (keeps collection cheap and import-light).
  2. Build the model and inputs via the loader.
  3. Define `load_inputs_fn(dtype)` and `extract_output_tensor_fn(output)`.
  4. Call the shared `test_<modality>(...)` entry point with the config
     (`data_format`, `required_pcc`, etc.).
- **Loader-first discipline (critical — see case study):** any model-specific
  workaround (attention impl, monkeypatch, config override) **belongs in the loader**,
  not the test. Put it in the test only as an interim measure when the loader is
  unwritable, and leave a comment saying so + why. The runner path benefits when it's
  in the loader.

### 4. Apply compiler-gap workarounds (only if needed)
Don't pre-emptively add these — only when a real compile/run failure points at them.
The InternVL3 bringup hit all three; full root-cause writeups are in the case study:
- **sdpa → illegal StableHLO `select` region** → pass `attn_implementation="eager"`.
- **boolean-mask indexing → `nonzero` dynamic-shape abort** → replace the
  data-dependent path with a static-shape expanded mask (monkeypatch).
- **dynamic RoPE rank-0 scalar → `ArrayRef::back()` assert** → force
  `rope_type="default"` (no-op for fixed-length inputs).

When a workaround papers over a genuine tt-mlir lowering gap (not a model quirk),
**recommend the user file a tracking issue** (`/issue-create`) — these are general
VLM problems, not one-offs.

### 5. Wire into the nightly matrix
- Edit `.github/workflows/perf-bench-matrix.json`. Append an entry to the `tests`
  array next to its modality siblings:
  ```json
  { "name": "<short_name>", "pytest": "tests/benchmark/test_<file>.py::test_<model>" }
  ```
- Add optional keys **only when required** (the entry inherits `test-defaults` otherwise):
  - `"pyreq": "<pip spec>"` — extra Python deps (e.g. a pinned `transformers`).
  - `"libreq"` — extra system libs.
  - `"runs-on": "<runner>"` — override the default `["n150","p150-perf"]` (e.g.
    `n300-llmbox`, `galaxy-wh-6u`, `qb2-blackhole`) for multi-chip / big models.
  - `"accuracy-testing": true` — register a parallel accuracy variant (LLM pattern).
  - `vllm` / `skip-*-dump` — only for vLLM or dump-heavy entries.
- Keep the JSON valid (trailing-comma-free); the file is parsed by the workflow.

### 6. Verify locally on n150
- Run the test on hardware and confirm it passes with PCC ≥ threshold:
  ```bash
  pytest -svv tests/benchmark/test_<file>.py::test_<model>
  ```
- Capture wall-clock. VLMs/large models compile slowly (InternVL3 ≈ 19 min). If the
  run exceeds the nightly per-entry budget, tell the user the timeout needs raising.
- Sanity-check the PCC: a suspiciously perfect `1.000000` may mean a degenerate/cached
  comparison — confirm it's a real graph. The bringup runner path measured 0.9849 for
  InternVL3, so a benchmark PCC of 1.0 warranted a second look.

### 7. Decide the PCC threshold
- Default is `DEFAULT_REQUIRED_PCC = 0.97`. Use the modality default unless the model's
  observed PCC justifies otherwise.
- If a prior `/model-bringup` set a YAML `required_pcc` for the runner path, align with
  it (or explain the divergence). InternVL3's benchmark used `0.90` while the runner
  YAML used `0.97` — note discrepancies for the reviewer.

### 8. Pre-commit, commit, PR
- Run `pre-commit run --files <changed files>` (black on the test, JSON formatting).
- Commit message: `[Benchmark] Add <model> benchmark and wire into nightly`, ending with
  the required `Co-Authored-By:` trailer.
- Hand off to `/create-pr` (area `Benchmark` / `Test Infra`).

## Pipeline models (diffusion / multi-component)

For a multi-stage `DiffusionPipeline`, a benchmark-add is a **four-artifact** bringup, not a
single `test_<model>`. **Read `references/imagegen-pipeline-template.md` first** — it has the
exact code shape, the frozen `_perf` schema, the CPU→TT→CPU eviction discipline, and the
SDXL-Lightning / Playground v2.5 knob tables. Run these phases sequentially; stop and report
on any failure.

### P1. Classify & confirm the pipeline
- Confirm it's a genuine multi-component pipeline (text encoder(s) + heavy net + VAE) with a
  `tt-forge-models` loader exposing per-component `ModelVariant`s. If the loader is missing
  or single-component, this is a bringup/scaffold — point the user at `/model-bringup`
  (`model-bringup-scaffold-pipeline`), not here.
- Pull the model's own reference `DiffusionPipeline` to read the **scheduler class, step
  count, guidance/CFG, and VAE scaling** — these are per-model and must NOT be copied from
  the reference models.

### P2. Establish the generate() flow on CPU
- Run one CPU forward of the full pipeline (all components fp32) to confirm a finite image
  and to fix the exact tensor flow (tokenize → TE1 → TE2 → denoise loop → VAE decode),
  including CFG batch handling if the model uses guidance. This flow is shared across all
  four artifacts.

### P3. Artifact 1 — benchmark pipeline (`benchmarks/<model>_pipeline.py`)
- Write `<Model>Config` (per-component `_on_tt` toggles + `compile_options`) and
  `<Model>Pipeline`. Load all components on CPU; register `.compile(backend="tt")` in
  `load_models()`; do the **CPU→TT→CPU eviction** inline in `generate()` (one component
  resident at a time). UNet in **bf16**, text encoders + VAE in **fp32**.
- Populate the frozen `_perf` dict every `generate()` call.
- Apply model opt-level rules (SDXL/Playground: TE+UNet at `optimization_level=0`, VAE
  switched to `1` inline and restored). Confirm the model's real requirement — don't assume.

### P4. Artifact 2 — benchmark test entry (`test_imagegen.py::test_<model>`)
- Add a thin `test_<model>(output_file, request)` that imports the pipeline **inside the
  function**, defines `build_pipeline_fn(compile_options)` → `(pipeline, generate_fn)`, and
  calls the shared `test_imagegen(...)`. Thread `compile_options` into the Config.

### P5. Artifact 3 — PCC-gated nightly (`tests/torch/models/<model>/test_<model>_pipeline.py`)
- Same flow but **all components on TT**, each forward checked against a lazy **fp32 CPU
  twin** fed the *same input* (clone inputs before the CPU→TT move); assert
  `pcc >= PCC_THRESHOLD` (0.99) and **fail fast**. UNet checked every step. Copy the nightly
  markers + `record_test_properties` block verbatim (change names).

### P6. Artifact 4 — demo script (`examples/pytorch/<model>.py`)
- Standalone runnable (`python examples/pytorch/<model>.py`), no pytest/PCC/perf — same flow,
  saves a PNG for human eyeballing. Mirror `examples/pytorch/sdxl_lightning.py`.

### P7. Wire into the matrix
- Append `{ "name": "<model>", "pytest": "tests/benchmark/test_imagegen.py::test_<model>" }`
  to the `tests` array (single top-level list entry) in `perf-bench-matrix.json`. `runs-on`
  override only for multichip/big models.

### P8. Verify on n150
- Benchmark entry passes (two-pass warmup + steady-state, image saved), nightly PCC ≥ 0.99
  on every component, demo produces a sane image. Record wall-clock (pipelines compile slowly)
  and surface any nightly-timeout implication to the user.

### P9. Pre-commit, commit, PR
- `pre-commit run --files <changed files>`. Split commits along the landing-PR grain if the
  user wants (benchmark wire-in, then PCC-gated nightly + demo) or land together. Titles like
  `[<Model>] Wire e2e pipeline into nightly and benchmark CI` and
  `<Model>: add PCC-gated nightly + demo scripts`. Hand off to `/create-pr`.

## Definition of done

For **pipeline models**, use the pipeline checklist in
`references/imagegen-pipeline-template.md` instead. For **single-forward models**:

- [ ] `test_<model>` added to the correct test file, model-specific config self-contained.
- [ ] Workarounds in the loader (or test, with a comment justifying the interim placement).
- [ ] Entry registered in `perf-bench-matrix.json` with valid JSON and minimal keys.
- [ ] Passes locally on n150 with PCC ≥ threshold; wall-clock recorded.
- [ ] Nightly timeout / dependency-pin implications surfaced to the user.
- [ ] Tracking issues filed for any compiler gaps the workarounds paper over.

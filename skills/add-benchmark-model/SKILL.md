---
name: add-benchmark-model
description: Add a model to the tt-xla benchmark suite and wire it into the nightly perf pipeline. Picks the right harness (vision / llm / encoder / multimodal), writes a config-driven test_<model> entry that drives a single forward pass through the shared benchmark, applies the loader-first workaround discipline learned from InternVL3 (eager attention, static-shape masks, default RoPE for compiler gaps), registers the model in perf-bench-matrix.json, and verifies locally on n150 before landing. Use when the user says "add <model> to benchmarks", "wire <model> into nightly", or invokes /add-benchmark-model.
---

# add-benchmark-model — tt-xla Benchmark Onboarding

Add a model to `tests/benchmark/` and register it in the nightly perf matrix.
The reference implementation is the InternVL3 bringup (commit `807201eac`,
`[Benchmark] Add InternVL3 multimodal benchmark and wire into nightly`). The hard
lessons from that bringup — what to put in the loader vs the test, and which
compiler gaps need source-level workarounds — live in
`references/internvl3-case-study.md`. **Read it before touching a VLM or any model
that uses HF sdpa / dynamic RoPE / boolean-mask indexing.**

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

The split is deliberate: **model-specific config lives in `test_<model>`, reusable
measurement logic lives in the harness.** A new model almost always means a new
`test_<model>` function in an existing test file — not a new harness. Only write a
new harness for a genuinely new modality (new input contract). If you do, model it
on the closest existing harness.

## Phases

Run sequentially. Stop and report to the user if a phase fails — do not silently continue.

### 1. Classify the model & pick the harness
- Determine modality from the loader / model card. Map to the table above.
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

## Definition of done
- [ ] `test_<model>` added to the correct test file, model-specific config self-contained.
- [ ] Workarounds in the loader (or test, with a comment justifying the interim placement).
- [ ] Entry registered in `perf-bench-matrix.json` with valid JSON and minimal keys.
- [ ] Passes locally on n150 with PCC ≥ threshold; wall-clock recorded.
- [ ] Nightly timeout / dependency-pin implications surfaced to the user.
- [ ] Tracking issues filed for any compiler gaps the workarounds paper over.

# Image-Gen Pipeline Template — PCC-gated nightly, benchmark, and demo

The reference bringups are **SDXL-Lightning** and **Playground v2.5** — multi-component
text-to-image diffusion pipelines (text_encoder + text_encoder_2 + unet + vae) where
every component runs on Tenstorrent. Use this template whenever the model is a
**multi-stage `DiffusionPipeline`** (SD / SDXL / MMDiT / video-DiT), not a single-forward
model. Single-forward models (vision / encoder / one-tensor VLM) stay on the harness path
in the main `SKILL.md`.

Landing PRs (read these diffs when the file paths below drift):
- `843fe0549` — `[Playground v2.5] Wire e2e pipeline into nightly and benchmark CI` (#5044): first
  image-gen benchmark pipeline + the `imagegen_benchmark.py` harness + `test_imagegen.py`.
- `04c41c5e2` — `[SDXL Lightning] Add e2e pipeline in nightly and benchmark CI` (#5244).
- `ed2a19012` — `[Benchmark] Wire image-gen e2e pipelines into nightly + benchmark CI (SD1.5, SD3)` (#5085).
- `c46c1e56b` — `Enable VAE on TT for Playground v2.5 and SDXL Lightning` (#5402).
- `1b6b2a300` — `Playground v2.5 + SDXL Lightning: add PCC-gated nightly + demo scripts` (#5480):
  added `examples/pytorch/{playground_v2_5,sdxl_lightning}.py`, rewrote the per-component
  unit tests into one PCC-gated e2e nightly test, deleted the old
  `test_text_encoder.py` / `test_unet.py` / `test_vae_decoder.py` stubs.
- `5b47da74b` — `[SDXL Lightning] Migrate _perf to model-agnostic benchmark schema` (#5582):
  froze the `_perf` dict schema the harness reads (below).

## The four artifacts

Bringing up one pipeline model produces **four files** plus one matrix entry:

| # | Artifact | Path | Purpose | PCC? | Perf? |
|---|----------|------|---------|------|-------|
| 1 | Benchmark pipeline | `tests/benchmark/benchmarks/<model>_pipeline.py` | Per-component TT toggles + `_perf` timing | no | yes |
| 2 | Benchmark test entry | `tests/benchmark/test_imagegen.py::test_<model>` | Thin config; builds pipeline, calls shared harness | no | yes |
| 3 | PCC-gated nightly | `tests/torch/models/<model>/test_<model>_pipeline.py` | All components on TT + fp32 CPU twins, fail-fast per-component PCC | yes | no |
| 4 | Demo script | `examples/pytorch/<model>.py` | Standalone `python examples/pytorch/<model>.py`, saves a PNG | no | no |

Matrix: append one entry to the `tests` array in `.github/workflows/perf-bench-matrix.json`
(details below). The benchmark test entry (#2) is what the matrix runs.

All four share the **same generate() control flow** (tokenize → TE1 → TE2 → denoise loop →
VAE decode). Keep them in lock-step: a change to component placement or dtype in one must be
mirrored in the others. They differ only in what they measure (perf vs PCC vs nothing) and
whether they carry the per-component `_on_tt` toggles (benchmark + demo do; nightly runs
everything on TT).

## Artifact 1 — benchmark pipeline (`benchmarks/<model>_pipeline.py`)

A `<Model>Config` + `<Model>Pipeline` pair.

- **`Config`** carries `model_id`, image dims, derived latent dims, and one bool per
  component (`text_encoder_on_tt`, `text_encoder_2_on_tt`, `unet_on_tt`, `vae_on_tt`,
  all default `True`) plus `compile_options: Optional[dict]` (harness-supplied — see VAE
  opt-level note). Toggles let you bisect a bad component onto CPU without editing code.
- **`setup()`** → `load_models()` + `load_scheduler()` + `load_tokenizers()`.
- **`load_models()`** loads every component on **CPU** and, for TT-bound components, only
  registers the dynamo backend: `model.compile(backend="tt")`. It does **not** move
  anything to the device here.
- **`generate()`** does the CPU→TT→CPU eviction dance (below) and records `_perf`.

### CPU→TT→CPU eviction discipline (critical)

Multi-component pipelines OOM TT DRAM if all components are resident at once. The rule:
**at most one heavy component on the device at a time.** Per component, inside `generate()`:

```python
if self.config.<comp>_on_tt:
    self.<comp> = self.<comp>.to(device)        # CPU → TT, right before the forward
    <inputs> = <inputs>.to(device=device)
t0 = time.perf_counter()
out = self.<comp>(<inputs>)
if self.config.<comp>_on_tt:
    out = out.to("cpu")                          # TT → CPU; the cpu cast forces the sync
self._perf["components"]["<comp>"] = time.perf_counter() - t0
if self.config.<comp>_on_tt:
    self.<comp> = self.<comp>.to("cpu")          # evict before the next component loads
```

The `.to("cpu")` on the output is what **forces the XLA sync**, so the timer must close
*after* it. For the UNet, move it to the device **once before** the denoising loop and evict
**once after** — only the per-step `sample`/`timestep` tensors cross the boundary inside the
loop; hoist the loop-invariant embeds/time_ids to the device before the loop.

### dtype placement

- Text encoders + VAE: **fp32** on TT.
- UNet: **bf16** on TT (fp32 UNet weights ≈ 10 GB — won't fit). `unet_dtype = bf16 if unet_on_tt else fp32`.

### opt-level (model-specific — do not assume)

SDXL-Lightning and Playground both run text encoders + UNet at
`optimization_level=0` (text_encoder_1 hits *"Unsupported buffer type"* at level 1), but the
**VAE needs `optimization_level=1`** (composite `ttnn.group_norm`; GroupNorm decomposition at
level 0 OOMs the VAE). Switch inline around the VAE forward and restore, merging rather than
clobbering the harness options:

```python
if self.config.vae_on_tt:
    torch_xla.set_custom_compile_options({**self.config.compile_options, "optimization_level": 1})
    ...
    torch_xla.set_custom_compile_options(self.config.compile_options)   # restore after
```

### `_perf` schema (frozen — the harness reads it, #5582)

`generate()` must populate `self._perf` every call:

```python
self._perf = {
    "components": {},                 # {name: seconds}   scalar per-stage (te1, te2, vae)
    "steps": [],                      # [seconds, ...]    one per heavy-net step
    "step_metric_name": "unet_step",  # or "transformer_step" for MMDiT
    "total": None,                    # full generate() wall time
}
```

The harness turns this into `images_per_second`, `e2e_latency`, `<step_metric>_mean_s`,
`cpu_overhead_s`, and one `<component>_s` measurement per scalar component. Keep the schema
model-agnostic so `imagegen_benchmark.py` never needs per-model edits.

## Artifact 2 — benchmark test entry (`test_imagegen.py::test_<model>`)

Thin. Imports the pipeline **inside the function**, defines a `build_pipeline_fn` and a
matching `generate_fn`, and delegates to the shared `test_imagegen(...)`:

```python
def test_<model>(output_file, request):
    from benchmarks.<model>_pipeline import <Model>Config, <Model>Pipeline

    prompt = "..."
    num_inference_steps = <N>
    height = width = 1024

    def build_pipeline_fn(compile_options):
        pipeline = <Model>Pipeline(config=<Model>Config(compile_options=compile_options))
        pipeline.setup()

        def generate_fn(prompt, steps):
            return pipeline.generate(prompt=prompt, num_inference_steps=steps, seed=DEFAULT_SEED)

        return pipeline, generate_fn

    test_imagegen(
        build_pipeline_fn=build_pipeline_fn,
        model_info_name="<model-dashed-name>",
        output_file=output_file, request=request,
        prompt=prompt, num_inference_steps=num_inference_steps,
        height=height, width=width,
        optimization_level=0,                        # only if the model needs it
        output_image_path="test_<model>_output.png",
    )
```

`compile_options` is threaded into the `Config` so the VAE opt-level switch can **merge**
into the harness options instead of overwriting them. The harness (`imagegen_benchmark.py`)
runs **two passes**: a **warmup** (`generate(prompt, 1)` — 1 step, triggers first-forward
compile of every component) and a **steady-state** (`generate(prompt, num_inference_steps)`
— all cache hits; this pass's image is saved and its latency drives throughput). One image
per run → `samples_per_sec = 1 / steady_state_time`.

## Artifact 3 — PCC-gated nightly (`tests/torch/models/<model>/test_<model>_pipeline.py`)

Structurally the same pipeline, but **every component runs on TT** (no `_on_tt` toggles) and
each TT forward is immediately checked against a **lazily-loaded fp32 CPU twin** fed the
*same input the TT component saw*. Fail-fast the moment any PCC drops below threshold.

```python
_PCC_EVALUATOR = TorchComparisonEvaluator(ComparisonConfig(assert_on_failure=False))
_PCC_CONFIG = PccConfig()
def _pcc(device_out, golden_out) -> float:
    return float(_PCC_EVALUATOR._compare_pcc(device_out, golden_out, _PCC_CONFIG))

def _cpu_twin(self, variant):                     # lazy fp32 twin, one per component
    if variant not in self._cpu_twins:
        self._cpu_twins[variant] = ModelLoader(variant).load_model(dtype_override=torch.float32)
    return self._cpu_twins[variant]
```

Per component: **clone the input before the CPU→TT move** (`tokens_1_cpu = tokens_1.clone()`,
`latents_cpu = latents.clone()`), run TT, run the fp32 twin on the clone, then:

```python
pcc = _pcc(tt_out, golden_out)
logger.info(f"[PCC] <comp>: pcc={pcc:.6f}")
assert pcc >= PCC_THRESHOLD, f"<comp> PCC {pcc:.6f} below threshold {PCC_THRESHOLD}"
```

The pipeline **continues on the TT outputs** (real deployment behavior); the twin only
provides a clean fp32 reference. The UNet is checked **every denoising step** (bf16 TT vs
fp32 twin fed the identical fp32 inputs). `PCC_THRESHOLD = 0.99` for both reference models —
use that unless the model justifies otherwise.

Markers + properties (copy verbatim, change names):

```python
@pytest.mark.nightly
@pytest.mark.model_test
@pytest.mark.single_device
@pytest.mark.large
@pytest.mark.record_test_properties(
    category=Category.MODEL_TEST,
    model_name="<Model>_Pipeline",
    model_group=ModelGroup.RED,
    run_mode=RunMode.INFERENCE,
    bringup_status=BringupStatus.PASSED,
)
def test_<model>_pipeline():
    xr.set_device_type("TT")
    ...
```

## Artifact 4 — demo script (`examples/pytorch/<model>.py`)

Standalone, runnable (`python examples/pytorch/<model>.py`), no pytest, no PCC, no perf
timing. Same generate() flow and per-component `_on_tt` toggles as the benchmark pipeline,
but it **saves a real PNG** so a human can eyeball quality. Copyright header + a `Run:`
docstring. Mirror `examples/pytorch/sdxl_lightning.py` / `playground_v2_5.py`; the family
also has `sd_v1_4_pipeline.py`, `sd_v1_5_pipeline.py`, `sd_v3_pipeline.py`, `sdxl-pipeline.py`.

## Matrix wiring

`.github/workflows/perf-bench-matrix.json` is a **single-element top-level list**; the entry
has `test-defaults` (`runs-on: ["n150-perf","p150-perf"]`, `libreq: "libgl1 libglib2.0-0"`,
etc.) and a `tests` array. Append next to the other image-gen siblings:

```json
{ "name": "<model>", "pytest": "tests/benchmark/test_imagegen.py::test_<model>" }
```

Add `runs-on` override only for multichip / big models (e.g. FLUX pins to 4-chip blackhole
`qb2`); `pyreq` / `libreq` only when the base env lacks a dep. Keep the JSON valid.

## Model-specific knobs that are NOT shared

Do not copy these blindly — they differ per model:

| Knob | SDXL-Lightning | Playground v2.5 |
|------|----------------|-----------------|
| Scheduler | `EulerDiscreteScheduler(timestep_spacing="trailing")` | `EDMDPMSolverMultistepScheduler` |
| Steps | 4 (distilled) | 50 |
| CFG | none (`guidance_scale=0`, batch stays 1) | yes (`cfg_scale=3.0`; uncond+cond concat → batch 2, chunk after UNet) |
| Prompt | `"A girl smiling"` | `"Astronaut in a jungle, cold color palette, muted colors, detailed, 8k"` |

CFG doubles the UNet batch (concat uncond+cond, then `chunk(2)` and combine
`uncond + scale*(cond-uncond)`); no-CFG models feed `noise_pred` straight to the scheduler.
Pull the scheduler class, step count, guidance, and VAE scaling from the model's own
reference `DiffusionPipeline`, not from these two.

## Definition of done (pipeline models)

- [ ] `benchmarks/<model>_pipeline.py` — Config with `_on_tt` toggles + `compile_options`; `generate()` populates the frozen `_perf` schema.
- [ ] `test_imagegen.py::test_<model>` — thin `build_pipeline_fn`/`generate_fn`, in-function import, correct `optimization_level`.
- [ ] `tests/torch/models/<model>/test_<model>_pipeline.py` — all components on TT, per-component fp32-twin PCC ≥ threshold, fail-fast, nightly markers.
- [ ] `examples/pytorch/<model>.py` — standalone demo saves a PNG.
- [ ] Matrix entry appended; JSON valid.
- [ ] CPU→TT→CPU eviction verified (one component resident at a time); UNet bf16, VAE opt_level=1 inline+restore.
- [ ] Verified on n150: benchmark passes, nightly PCC ≥ threshold, demo produces a sane image; wall-clock recorded.

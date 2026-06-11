# InternVL3 Benchmark Bringup — Case Study & Workaround Playbook

The reference run for `add-benchmark-model`. Commit `807201eac`
(`[Benchmark] Add InternVL3 multimodal benchmark and wire into nightly`) added:

- `tests/benchmark/benchmarks/multimodal_benchmark.py` — the VLM harness (dict-of-kwargs
  inputs, integer tensors kept integer, single forward pass, PCC vs CPU golden).
- `tests/benchmark/test_multimodal.py::test_internvl3` — config-driven entry
  (`INTERNVL3_1B_HF`, bf16, `required_pcc=0.90`).
- `.github/workflows/perf-bench-matrix.json` — `internvl3_1b` entry.

Result on n150: `1 passed in 1142.74s (~19 min)`, PCC=1.000000, 0.48 samples/sec.

## Why a new harness was justified here

Vision harness feeds one image tensor and force-casts everything to the model dtype.
VLMs need a **dict** of `input_ids`/`attention_mask`/`pixel_values`, and the integer
inputs (`input_ids`, `attention_mask`) **must stay integer** for embedding lookup —
only weights and `pixel_values` are bf16. That's a different input contract, so
`multimodal_benchmark.py` was modeled on `vision_benchmark.py` but with
`_move_inputs_to_device` that does not coerce dtype. This is the bar for "new harness":
a genuinely different I/O contract, not just a different model.

## The five fixes (cascade — each unblocked the next)

| # | Blocker | Root cause | Fix | Where it belongs |
|---|---------|-----------|-----|------------------|
| 1 | `InternalTorchDynamoError: eval() arg 1 must be a string` | GuardBuilder passed the whole `guard` object where torch's `get()` expects `guard.name` | `self.get(guard)` → `self.get(guard.name)` in `python_package/tt_torch/utils.py:99` | tt-xla core (standalone bug — affects any model hitting an NN_MODULE ID_MATCH guard) |
| 2 | `stablehlo.select op using value defined outside the region` (~24 ops) | HF `AttentionMaskConverter._unmask_unattended` (sdpa-only) emits a bool `torch.all` reduce whose region constants get CSE-hoisted out → illegal in StableHLO | `attn_implementation="eager"` | **loader** (passed as `load_model` kwarg; merges into `from_pretrained` model_kwargs) |
| 3 | `setman(...)` trisc1 layernorm build failure | tt-metal `recip.h` uses an sfpi API absent in the pinned sfpi 7.45.0 | recip.h one-liner (tt-metal PR #43621) + point `runtime/sfpi` at 7.46.0 | **dependency pin** — needs a tt-mlir/tt-metal SHA bump; NOT committable to tt-xla |
| 4 | SIGABRT compiling `get_placeholder_mask` | `inputs_embeds[special_image_mask].numel()` → boolean-index → `nonzero` (dynamic shape) hard-aborts XLA | monkeypatch `InternVLModel.get_placeholder_mask` → static expanded `[B,S,H]` mask (`masked_scatter` in forward still does the real placement) | **loader** (interim: test) |
| 5 | `ArrayRef::back(): Assertion !empty()` in tt-mlir | Qwen2 text config `rope_type="dynamic"` → `dynamic_frequency_update` computes `torch.max(position_ids)+1` = rank-0 scalar; lowering calls `ArrayRef::back()` on empty shape | force `rope_type="default"` on all rotary modules (no-op for fixed 1025-token seq) | **loader** (interim: test) |

## Reusable workaround patterns (the generalizable part)

These three recur across HF VLMs / transformer models. Apply only when a real failure
points at them — verify the error signature first.

### A. sdpa → illegal StableHLO region
**Signature:** `stablehlo.select op using value defined outside the region` during
MHLO→StableHLO. **Cause:** sdpa-only `_unmask_unattended` "fully-masked row" trick.
**Fix:** `attn_implementation="eager"` — eager adds the mask directly, no region capture.
```python
model = loader.load_model(dtype_override=data_format, attn_implementation="eager")
```

### B. boolean-mask indexing → `nonzero` dynamic shape
**Signature:** SIGABRT / hard compiler abort around a `*_mask` selection that calls
`tensor[bool_mask]`. **Cause:** advanced boolean indexing lowers to `nonzero`, whose
output shape is data-dependent. **Fix:** replace with a static-shape expanded mask; if
the real placement is done elsewhere (e.g. `masked_scatter`), the check is only a
validation and can be dropped.
```python
def _placeholder_mask_static(self, input_ids, inputs_embeds, image_features):
    special_image_mask = input_ids == self.config.image_token_id
    return special_image_mask.unsqueeze(-1).expand_as(inputs_embeds).to(inputs_embeds.device)
InternVLModel.get_placeholder_mask = _placeholder_mask_static
```

### C. dynamic RoPE rank-0 scalar → `ArrayRef::back()` assert
**Signature:** `ArrayRef::back(): Assertion !empty()` in tt-mlir. **Cause:** NTK/dynamic
rope's `dynamic_frequency_update` produces a rank-0 scalar from `torch.max(position_ids)+1`.
**Fix:** force `rope_type="default"` — safe when seq length << original context (dynamic
scaling is a no-op there) and `inv_freq` is already initialized.
```python
for _m in model.modules():
    rt = getattr(_m, "rope_type", None)
    if isinstance(rt, str) and "dynamic" in rt:
        _m.rope_type = "default"
```

## Loader-vs-test placement — the discipline

Fixes #2/#4/#5 were placed in `test_internvl3` only because
`third_party/tt_forge_models/internvl3/pytorch/loader.py` was **root-owned / unwritable**.
The correct home is the loader: it benefits the runner path too and keeps the benchmark
test thin. When you must put a workaround in the test, leave a comment stating it's
interim and why (see the verbatim comments in `test_internvl3`).

## Nightly / landing gotchas

1. **Dependency pin (fix #3)** is the real nightly blocker: until the tt-mlir/tt-metal
   pin includes tt-metal PR #43621 (`d998791a686`), nightly compiles layernorm against
   the system sfpi and fails on `setman`. Working-tree symlinks do NOT survive a clean
   checkout. Surface this to the user — a benchmark entry that passes locally can still
   redfail nightly on a stale pin.
2. **Timeout:** compile + run ≈ 19 min. Default matrix entries have no extended budget —
   recommend ≥2400s for slow VLMs.
3. **PCC sanity:** benchmark measured 1.000000 but the bringup runner path measured
   0.9849. A perfect 1.0 can indicate a degenerate/cached comparison — confirm it's a
   real graph before trusting it.
4. **File tracking issues** for #4 and #5 — they are general tt-mlir VLM lowering gaps
   (boolean-index `nonzero` abort; dynamic-RoPE rank-0 assert), not InternVL-specific.

## Source pointers
- Resume notes: `INTERNVL3_RESUME.md` (repo root)
- Memory: `project_internvl3_benchmark.md`, `feedback_tt_xla_guard_bug.md`,
  `feedback_sdpa_stablehlo.md`
- sfpi resolution order: `tt_metal/jit_build/build.cpp:113`
  (`{root_ + "runtime/sfpi", "/opt/tenstorrent/sfpi"}` — local first, then system)

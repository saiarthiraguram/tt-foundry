# Repo routing — where to file

Default target is the repo whose CI/test surfaced the failure. File upstream
when the root cause is clearly in lowering or tt-metal and you have (or will
have) an isolated repro.

## tenstorrent/tt-xla

**File here when:**
- Failure is from `tests/runner/test_models.py` or `tests/torch/models/**`
- Model-level PCC / atol assertion in tt-xla evaluators
- Model-level OOM during pytest on tt device (even if op-level root cause)
- Bringup `escalation.json` references a tt-xla `model_key`

**Labels:** usually `bug`. Set **Type: Bug** in GitHub UI after creation.

**manifest `category`:** `pcc`, `oom`, or `runtime` depending on failure class.

## tenstorrent/tt-mlir

**File here when:**
- TTMLIR / StableHLO compile failure before runtime execution
- Missing lowering for `torch.*` / `aten.*` op
- Investigation originated in tt-forge-fe / mlir test (note upstream context)
- Historical grid_sample / floor decomposition class bugs

**manifest `category`:** `compile` or `missing_op`.

## tenstorrent/tt-metal

**File here when:**
- Isolated tt-metal / ttnn repro exists (repeat, bcast, binary_ng factory, etc.)
- tt-xla issue already tracks model surface; this is the kernel-level follow-up
- PJRT stack shows `tt_metal/...` or `ttnn::operations` as root abort

**manifest `category`:** `runtime`.

Link from tt-xla issue → tt-metal issue in `upstream_issues` / body cross-links.

## Decision flow

```
pytest fails on tt-xla test?
  yes → default tenstorrent/tt-xla
        → if isolated op sanity also fails with same ttnn op → note tt-metal follow-up
  no → compile-only (no device execution)?
        yes → tenstorrent/tt-mlir
        no  → ask user or use escalation.json blocker text
```

## Search before filing

```bash
gh issue list --repo tenstorrent/tt-xla --search "<op> OOM" --limit 10
gh issue list --repo tenstorrent/tt-mlir --search "<op> lowering" --limit 10
```

If a related issue exists with the same lowering path, reference it in
`related_issues` and focus the new draft on **what changed** (model, alloc size,
arch) rather than duplicating the full investigation.

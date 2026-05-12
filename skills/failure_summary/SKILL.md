---
name: failure_summary
description: Builds a Markdown (or CSV) table of every KNOWN_FAILURE_XFAIL entry in tests/runner/test_config/torch/test_config_inference_single_device.yaml with columns (test name, scope, failure reason). Includes both top-level XFAIL entries and per-arch arch_overrides XFAIL. Use when the user wants a digest of currently-xfailed models for triage, sharing, or filing GitHub issues.
allowed-tools: Bash Read Write
---

# Failure Summary — KNOWN_FAILURE_XFAIL digest

Produce a Markdown (or CSV) table of every `KNOWN_FAILURE_XFAIL` entry in the
inference single-device YAML so the user can scan failure reasons in one view.

## Invocation

`/failure_summary [--arch n150] [--out <path>] [--format md|csv]`

- `--arch` (default `n150`): used to resolve `arch_overrides` XFAILs.
- `--out` (default `.claude/failure_summary/summary.md` for md,
  `.claude/failure_summary/summary.csv` for csv): output path.
- `--format` (default `md`): `md` for GitHub-flavored Markdown, `csv` for CSV.

## Steps

### 1. Enumerate candidates

Use Python — do not hand-roll a grep — and capture both top-level and
`arch_overrides` XFAILs. Preserve YAML order.

```python
import yaml
from pathlib import Path
p = Path("tests/runner/test_config/torch/test_config_inference_single_device.yaml")
data = yaml.safe_load(p.read_text())["test_config"] or {}
ARCH = "<arch>"
rows = []
for key, cfg in data.items():
    if cfg is None: continue
    if cfg.get("status") in ("KNOWN_FAILURE_XFAIL", "known_failure_xfail"):
        rows.append((key, "top", (cfg.get("reason") or "").strip()))
        continue
    ov = (cfg.get("arch_overrides") or {}).get(ARCH)
    if ov and ov.get("status") in ("KNOWN_FAILURE_XFAIL", "known_failure_xfail"):
        rows.append((key, f"arch:{ARCH}", (ov.get("reason") or "").strip()))
```

If `len(rows) == 0`, print `[failure_summary] no KNOWN_FAILURE_XFAIL entries
for arch=<arch>` and exit without writing a file.

### 2. Categorize each row

The flat 77-row table is too wide to scan. Bucket each row by inferring a
failure category from its reason. Match in this order; first hit wins:

| Category label | Regex (case-insensitive) |
|---|---|
| `OOM — DRAM`            | `Out of Memory.*DRAM buffer` |
| `OOM — L1`              | `Out of Memory.*L1 buffer` |
| `Circular buffer (L1)`  | `circular buffers.*beyond max L1 size` |
| `Shape rank`            | `Can't convert shape rank` |
| `Missing op`            | `failed to legalize operation 'ttir\.([a-z_]+)'` (capture op) |
| `Shardy propagation`    | `Shardy propagation` |
| `PCC drop`              | `PCC comparison failed.*pcc=([0-9.]+).*Required: pcc=([0-9.]+)` |
| `Dtype mismatch`        | `mat1 and mat2 must have the same dtype` |
| `Dynamo / Sparse`       | `torch\._dynamo\|SparseSequential` |
| `Module init`           | `assert isinstance\(self\._model, torch\.nn\.Module\)` |
| `Async XLA tensor`      | `Check failed: handle->HasValue.*async operation` |
| `Arange invalid range`  | `ttir\.arange.*Invalid range` |
| `Tensor not allocated`  | `tensor has a non-zero number of elements, but its data is not allocated` |
| `Scatter legalize`      | `stablehlo\.scatter failed to legalize` |
| `Dynamic shapes`        | `Dynamic shapes` |
| `Other`                 | (fallback) |

Also extract:
- **Issue number**: `https://github.com/tenstorrent/tt-xla/issues/(\d+)` → `#<n>`,
  or `—` if absent.
- **Op name** (only for `Missing op`): captured group from the regex.
- **PCC values** (only for `PCC drop`): captured `(observed, required)`.
- **Short model name**: drop the trailing `-single_device-inference` suffix
  (it's the same for every row of this YAML).

### 3. Estimate approximate model size

For each row, infer an approximate parameter count from the model name
(regex match against `<full_key>`, case-insensitive, first hit wins).
This is a coarse heuristic — the user is using it to pick small models to
iterate on quickly, not to plan capacity.

| Pattern (regex on model key) | Approx params |
|---|---|
| `(\d+(?:\.\d+)?)\s*B(?:[_\-]\|$\|\b)` (captured N) | N × 10⁹ |
| `(\d+(?:\.\d+)?)\s*M(?:[_\-]\|$\|\b)`             | N × 10⁶ |
| `bi_lstm_crf`                                    | 5 M |
| `hippynn`                                        | 1 M |
| `mobilenetv3`                                    | 5 M |
| `\bnano\b`                                       | 10 M |
| `\btiny\b`                                       | 30 M |
| `\bsmall\b`                                      | 60 M |
| `\bbase\b`                                       | 100 M |
| `\blarge\b`                                      | 300 M |
| `resnet101\|r101`                                | 45 M |
| `resnet50\|r50`                                  | 25 M |
| `resnet34\|r34`                                  | 22 M |
| `resnet18\|r18`                                  | 12 M |
| `vgg16`                                          | 138 M |
| `perceiver`                                      | 30 M |
| `vit\|vilt`                                      | 90 M |
| (fallback)                                       | 100 M |

Watch out for false positives on the `\d+B`/`\d+M` rules — only match when
followed by `_`, `-`, end-of-string, or a word boundary, so `R50dcn`,
`P4`, etc. don't get parsed as parameter counts.

Format the rendered value as `~1.5 B`, `~30 M`, etc.

### 4. Render

For `--format md`, structure:

```
# KNOWN_FAILURE_XFAIL summary (arch: <arch>)

Source: tests/runner/test_config/torch/test_config_inference_single_device.yaml
Total entries: <N>

## Quick-pick: 3 smallest models

For fastest iteration. Sizes are approximate, derived from the model name.

| Rank | Model | Approx params | Category | Issue |
|------|-------|---------------|----------|-------|
| 1 | <smallest> | ~1 M | <cat> | #NNNN |
| 2 | <2nd smallest> | ~5 M | <cat> | #NNNN |
| 3 | <3rd smallest> | ~5 M | <cat> | #NNNN |

## By category

| Category | Count |
|----------|-------|
| <Cat 1>  | <c1>  |
| <Cat 2>  | <c2>  |
...
```

For the Quick-pick table:
- Sort all entries by ascending estimated params.
- Tie-break by YAML order (i.e. stable sort).
- Take the first 3.
- Skip the section entirely if `len(rows) < 3`.

Order categories in the `By category` table by descending count (largest
buckets first — the things worth attacking in bulk).

Then one section per non-empty category, also in descending-count order:

```
## <Category label> (<count>)

| Model (short name) | Scope | Issue | Extra |
|--------------------|-------|-------|-------|
| <short_name>       | top   | #1497 |       |
...
```

Per-section column rules:
- **Extra** column is **only** included for `Missing op` (op name) and
  `PCC drop` (`pcc=0.978 → req=0.99`). For all other categories, drop the
  Extra column.
- Drop the **Scope** column if every row in that section has the same scope
  (which is the common case — keeps the table narrow). Note the scope under
  the heading instead, e.g. `_All entries: top._`
- Issue cell renders `#NNNN` (no full URL — too noisy in a scan view). Keep
  the URL only in the `Other` section's reason snippet, so unmatched rows
  still have a clickable link.

For `Other`, keep a narrower fallback table with a truncated reason:

```
## Other (<count>)

| Model | Scope | Reason (first 90 chars) | Issue |
|-------|-------|-------------------------|-------|
| ... | ... | ... | #NNNN |
```

Cell-content rules (apply everywhere):
- Replace `|` inside any cell with `\|`.
- Collapse newlines/tabs to a single space.
- If a reason is empty, render `_(none)_`.
- Keep `<` and `>` literal — GitHub Markdown handles them in table cells.

For `--format csv`, **skip categorization** and emit the flat 4-column form
(`name,scope,category,reason`) with `csv.writer` default quoting. CSV
consumers want one row per entry; categories are a column there, not a
section break.

### 3. Write and report

Create parent dirs for `--out` if missing. Write the rendered output. Print a
one-line summary to the terminal:

```
[failure_summary] arch=<arch>  entries=<N>  format=<md|csv>  out=<path>
```

If `<N> > 0`, also print the first 3 rows of the table inline so the user can
sanity-check without opening the file.

## Hard rules

- **Read-only on the YAML.** This skill never edits the YAML — it only reports.
- **Single source.** Only the inference single-device YAML; do not pull from
  training or multi-chip configs unless extended explicitly.
- **Stable ordering.** Preserve YAML order (do not sort) so the output diffs
  cleanly against prior runs.
- **No live test runs.** This skill is YAML-only. For re-verifying whether an
  entry's recorded reason still reproduces, use `model_issue_pick` instead.

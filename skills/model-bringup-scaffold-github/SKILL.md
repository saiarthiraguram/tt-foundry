---
name: model-bringup-scaffold-github
description: Scaffold a tt-forge-models loader for a model whose source lives on GitHub (no HuggingFace mirror). Asks the user whether to vendor the repo as a submodule under third_party/tt_forge_models/third_party/<repo>/ or to port the model code into <family>/pytorch/src/ (the maptr / transfuser / bevformer pattern). Writes loader.py + __init__.py, validates import + collect, and hands off to the standard OVERVIEW / FIRST_RUN / ... FSM. Used by the model-bringup orchestrator when the model_key is a github short form, a git+ URL, or a full github.com URL.
allowed-tools: Bash Read Write Edit Grep Glob
---

# Model Bringup — GitHub Scaffold

You are the **GitHub variant** of the scaffold stage. Use when the model
lives on GitHub with no HuggingFace mirror. Same role as
`model-bringup-scaffold`, different inputs and a different way of laying
the source on disk.

After this skill exits PASSED, downstream stages (`model-bringup-overview`,
`-run`, `-diagnose`, `-repair`, `-finalize`) run unchanged — they only
require that `ModelLoader().load_model()` returns an `nn.Module`.

## Invocation
`/model-bringup-scaffold-github <model_key> [--arch <arch>] [--mode submodule|port] [--entry <module.path:ClassName>] [--ref <git_ref>] [--custom-test]`

- `model_key` (required) — one of:
  - `github:<org>/<repo>` — short form. Optional `@<ref>` selects a SHA,
    tag, or branch, e.g. `github:open-mmlab/mmdetection3d@v1.4.0`.
  - `https://github.com/<org>/<repo>[.git]` — full URL.
  - `git+https://...` — also accepted (pip-style).

  All three normalize to `<repo_url>` + optional `<ref>`. `family` is
  derived from the repo name (lower-snake-case, strip `pytorch_`,
  `_pytorch`, `-model`, etc.).

- `--mode submodule | port` — see **Vendor mode** below. If omitted, the
  skill **asks** via `AskUserQuestion`.
- `--entry "<module.path:ClassName>"` — entry point (e.g.
  `mmdet3d.models.detectors:FCOS3D`). Required because we cannot
  `AutoModel.from_pretrained` a class out of GitHub. If omitted, the
  skill asks.
- `--ref <git_ref>` — overrides the ref parsed from `@<ref>`.
- `--custom-test` — same meaning as in `model-bringup-scaffold`.

---

## Step 0 — Normalize the model_key

Parse one of the three forms. Set:
- `repo_url` (e.g. `https://github.com/open-mmlab/mmdetection3d`)
- `ref` (SHA / tag / branch; default `HEAD` resolved to a specific SHA)
- `family` (snake-case, derived from repo name; user-overridable via the
  ask below)

Construct the structured model_key the orchestrator already understands:
```
<family>/pytorch-<variant>-single_device-inference
```
where `variant` defaults to the repo's `<ref>` short SHA (8 chars) or the
ref name if it was a tag. The user can override this via the variant
question below.

---

## Step 1 — Ask the user (only when args missing)

Use `AskUserQuestion` for genuine forks. Skip the question if the user
already passed the flag.

**Q1 — Vendor mode** (skip if `--mode` was passed):

> The model code can either be vendored as a submodule (keeps the full
> GitHub repo intact under `third_party/tt_forge_models/third_party/`) or
> ported (just the model-code subset copied into
> `third_party/tt_forge_models/<family>/pytorch/src/`). Which mode?

Options:
- `Submodule` — full repo as a git submodule. **Recommended for repos
  with non-trivial install scripts, native extensions, or many internal
  imports.** Cost: pulls everything (data scripts, docs, tests). The
  loader imports from `third_party.tt_forge_models.third_party.<repo>.…`.
- `Port` — copy only the model-code subset. **Recommended for small
  research repos (<5 .py files of model code).** Cost: you maintain the
  copy; need to re-port on upstream updates. The loader imports from
  `.src.<module>` and you can add a `SRC_VENDORED_FROM.txt` provenance
  file.

**Q2 — Entry class** (skip if `--entry` was passed):

> What is the Python entry point of the model? Format:
> `module.path:ClassName` — e.g. `mmdet3d.models.detectors:FCOS3D`.

**Q3 — Variant slug** (only if the orchestrator did not pass one):

> Variant slug for `ModelVariant`? (Defaults to the short SHA of the
> selected ref.)

Record the answers in state.json under `details.scaffold_github`:
```json
{
  "repo_url":   "<url>",
  "ref":        "<sha or tag>",
  "mode":       "submodule | port",
  "entry":      "<module:Class>",
  "family":     "<family>",
  "variant":    "<variant_slug>"
}
```

---

## Step 2 — Vendor the source

### Mode A — Submodule

Target path: `third_party/tt_forge_models/third_party/<repo>/`

```bash
git -C third_party/tt_forge_models submodule add --depth 1 "<repo_url>" third_party/<repo>
git -C third_party/tt_forge_models/third_party/<repo> fetch --depth 1 origin "<ref>"
git -C third_party/tt_forge_models/third_party/<repo> checkout "<ref>"
git -C third_party/tt_forge_models submodule update --init --recursive
```

If `third_party/tt_forge_models/third_party/` does not exist yet, create
it and add a `__init__.py` so it is importable as a Python package.

If the repo is huge (> 1 GiB checked out), fall back to a shallow clone
**without** submoduling — write a note in the steps log and proceed.

Pin the ref. Write `third_party/tt_forge_models/third_party/<repo>/.bringup_ref`:
```
ref:    <full sha>
ref_kind: <sha|tag|branch>
fetched: <YYYY-MM-DD>
url:    <repo_url>
```

The import path the loader will use:
```
from third_party.tt_forge_models.third_party.<repo>.<module.path> import <ClassName>
```

If the repo has its own `setup.py` / `pyproject.toml` and its model code
is not directly importable from the source tree (e.g. it expects to be
installed), inject a `sys.path.insert(0, …)` line at the top of the
loader. Do **not** `pip install -e` automatically — that mutates the
venv state of the user's whole environment, which is out of scope for
scaffold.

### Mode B — Port

Target path: `third_party/tt_forge_models/<family>/pytorch/src/`

1. Clone the repo to a temp dir:
   ```bash
   tmp=$(mktemp -d)
   git -C "$tmp" clone --depth 1 "<repo_url>" repo
   git -C "$tmp/repo" fetch --depth 1 origin "<ref>"
   git -C "$tmp/repo" checkout "<ref>"
   ```
2. Identify the model-code subset to copy. Heuristic:
   - Resolve the entry `<module.path:ClassName>` to the file containing
     `class <ClassName>`. Start from that file.
   - Walk relative imports from the entry file (BFS over `from .…` and
     `from <repo_root>…`) to collect a transitively-closed file set.
   - Copy each collected `.py` into
     `third_party/tt_forge_models/<family>/pytorch/src/`, preserving the
     subpackage layout (`<a>/<b>/<file>.py`).
   - Add `__init__.py` to every intermediate directory.
3. Write a provenance file at
   `third_party/tt_forge_models/<family>/pytorch/src/SRC_VENDORED_FROM.txt`:
   ```
   Source repo : <repo_url>
   Ref         : <full sha>
   Ref kind    : <sha|tag|branch>
   Fetched     : <YYYY-MM-DD>
   Entry       : <module:Class>
   Files       :
     - <rel path 1>
     - <rel path 2>
     ...
   ```
   This is the breadcrumb the next person needs to re-port if the model
   changes upstream.

The import path the loader will use:
```
from .src.<module> import <ClassName>
```
(where `<module>` is the entry file's path relative to `src/`, with
slashes turned into dots and `.py` stripped).

**Refuse to port** if the collected `.py` set exceeds 30 files or 5 MB
total — that's a strong signal that submodule is the right mode. Fail
with `result=blocked`, `block_reason="port set too large; rerun with --mode submodule"`.

---

## Step 3 — Write loader.py

Same template skeleton as `model-bringup-scaffold` step 2d, with three
differences:

1. `ModelVariant` — populated with the user's `<variant_slug>` (default
   short-SHA).
2. `ModelConfig.pretrained_model_name` is **not** set (there is no HF
   id). Instead set a new field `source_repo` = `<repo_url>@<ref>`.
3. `load_model()` body imports the entry class from the path computed in
   Step 2 and instantiates it from a config-only path (random weights —
   per `user-bringup-prefs`):

```python
def load_model(self, dtype_override=None):
    # Import path produced by model-bringup-scaffold-github
    {imports}

    # No HF download. Construct from default config / random weights.
    {ctor_invocation}     # filled by the skill: usually MyModel(**default_cfg)
    if dtype_override is not None:
        model = model.to(dtype_override)
    return model.eval()
```

The skill fills `{ctor_invocation}` by inspecting the class signature:
- If `MyClass.__init__` has no required positional args → `model = MyClass()`.
- If it has required args, prompt the user once for a minimal config
  dict (`AskUserQuestion`, free-form), wrap it in `model =
  MyClass(**ctor_kwargs)`, and persist `ctor_kwargs` in state.

`load_inputs()` body: synthesize tensors matching the class's `forward`
signature, same heuristic as `model-bringup-scaffold` (use
`inspect.signature(model.forward)`, default shapes from annotations
where present, else `(1, 3, 224, 224)` for image-like, `(1, 128)` for
text-like).

`unpack_forward_output()` body: by default return the input unchanged
(model returns a tensor). If the user provided forward output shape
during the ask, generate the unpack from there.

Also write `pytorch/__init__.py` re-exporting `ModelLoader, ModelVariant`
exactly as `model-bringup-scaffold` does.

---

## Step 4 — Validate imports + discovery

Same as `model-bringup-scaffold` Step 3 + Step 4:

```bash
python -c "from third_party.tt_forge_models.<family>.pytorch import ModelLoader, ModelVariant; print('OK')"
pytest -q --collect-only tests/runner/test_models.py 2>&1 \
  | grep "test_all_models_torch\[<family>/pytorch-"
```

If the loader import fails because the repo expects `sys.path` munging or
an installed package, edit loader.py to inject the right path and retry
**once**. If it still fails, exit with `result=failed`,
`failure_reason=<import error one-liner>` so the orchestrator escalates
to a human rather than thrashing.

---

## Step 4b — Pre-flight size gate + shard plan

Inherit the same gate logic as `model-bringup-scaffold` Step 4b. Param
estimate sources, in order:

1. Live count from the loader (load random-weights model, sum
   `p.numel()`).
2. Name-based heuristic on `<repo_url>` and `<entry_class>`.
   (`AutoConfig` is unavailable for GitHub-only models.)

Same gate thresholds (14 B warn, 30 B reject) and same shard plan emit.

---

## Step 4c — Optional custom test file

Same heuristics as `model-bringup-scaffold` Step 4c. GitHub-vendored
models more often need a custom test (non-standard input ctors are
common) — when `--mode port` is selected, default `--custom-test` to
true unless the user explicitly disabled it.

---

## Step 5 — Initialize bringup state

Same shape as `model-bringup-scaffold` Step 5, plus persist the GitHub
provenance fields under `details.scaffold_github`:

```json
{
  "stage": "validate",
  "result": "passed",
  "details": {
    "loader_path":      "third_party/tt_forge_models/<family>/pytorch/loader.py",
    "loader_created":   true,
    "scaffold_variant": "github",
    "scaffold_github": {
      "repo_url": "<url>",
      "ref":      "<sha>",
      "mode":     "submodule | port",
      "entry":    "<module:Class>",
      "src_path": "third_party/tt_forge_models/<family>/pytorch/src/" | "third_party/tt_forge_models/third_party/<repo>/"
    }
  }
}
```

---

## Step 6 — Bringup steps log

Append to `.claude/bringup/<safe_key>/bringup_steps.txt`:
```
--------------------------------------------------------------------------------
STEP 1 — Parse & Scaffold (model-bringup-scaffold-github)
--------------------------------------------------------------------------------
Input model_key : <original>
Repo URL        : <url>
Ref             : <sha> (<sha|tag|branch>)
Family          : <family>
Variant         : <variant_slug>
Entry           : <module:Class>

Vendor mode     : submodule | port
Vendor path     : <relative path>
Provenance file : <.bringup_ref or SRC_VENDORED_FROM.txt>

Loader created  : yes
Files written   :
  - third_party/tt_forge_models/<family>/__init__.py
  - third_party/tt_forge_models/<family>/pytorch/__init__.py
  - third_party/tt_forge_models/<family>/pytorch/loader.py
  [+ submodule files or src/ files]

Import validation  : python -c "from ... import ModelLoader, ModelVariant" → OK | FAILED
Collect validation : pytest --collect-only ... → <N> test(s) found | NONE FOUND

Size gate          : <X> B params (source: loader | name_heuristic) → proceed | warn | reject
Shard plan         : <mesh> / TT_VISIBLE_DEVICES=<list> | n/a

Custom test file   : <path or 'none — generic runner suffices'>

SCAFFOLD RESULT: PASSED | FAILED
```

---

## Step 7 — Output

On success:
```
[scaffold-github] PASSED
  repo            : <repo_url>@<short_sha>
  vendor          : submodule | port
  vendor path     : <relative path>
  loader          : third_party/tt_forge_models/<family>/pytorch/loader.py
  collect check   : <N> test(s) visible in tests/runner/test_models.py
```

On failure: same escalation rules as `model-bringup-scaffold` (failed
import, blocked size gate, blocked port size).

---

## Notes for the orchestrator

- Same exit codes as `model-bringup-scaffold` — orchestrator does not
  need to special-case the variant.
- The OVERVIEW skill runs unchanged: it imports `ModelLoader`, runs
  `load_model()` + `load_inputs()` on CPU, and captures golden. Because
  the loader returns a random-init model, CPU sanity should be fast
  (single forward) — the same as the HF path.
- If the user later updates the source (new SHA, new `--ref`), they
  should re-invoke this skill with `--ref` to refresh provenance. The
  pipeline does not auto-detect upstream drift.

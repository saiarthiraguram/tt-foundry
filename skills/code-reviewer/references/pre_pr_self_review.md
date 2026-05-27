# Pre-PR Self-Review — tt-xla

Mechanical "is my branch ready to PR?" checks. Run these against `git diff main..HEAD` (or `--staged` if not yet committed) before opening a PR.

This complements `code_review_checklist.md` (which is judgment-heavy): the steps here are scriptable Yes/No checks.

## Phase 1 — Lint gate

Run and capture exit codes:

```bash
pre-commit run --files $(git diff main..HEAD --name-only)
black --check $(git diff main..HEAD --name-only -- '*.py')
clang-format --dry-run --Werror $(git diff main..HEAD --name-only -- '*.cpp' '*.h' '*.hpp')
```

Report PASS/FAIL per tool. Any FAIL → block the PR until fixed.

## Phase 2 — SPDX header on new files

For each newly-added source file (`git diff main..HEAD --diff-filter=A --name-only`), check that:

- `*.cpp`/`*.h`/`*.hpp` start with `// SPDX-License-Identifier: Apache-2.0`
- `*.py` starts with `# SPDX-License-Identifier: Apache-2.0`
- `*.cmake`/`CMakeLists.txt` start with `# SPDX-License-Identifier: Apache-2.0`

Skip generated files, vendored third-party code, and anything under `third_party/`.

## Phase 3 — Diff risk flags

Examine the diff and flag (don't auto-block — surface to the user):

- `TODO` / `FIXME` / `XXX` / `HACK` markers introduced in the diff
- `print(` / `std::cerr` / `std::cout` / `printf(` added (use loguru / logging instead)
- New public functions / classes ≥40 lines with no test coverage (see Phase 4)
- Single PR touches files owned by ≥3 distinct CODEOWNERS blocks — consider splitting
- New `// removed` / `# removed` comments — dead-code crumbs, delete instead
- Submodule pointer changes in `third_party/` without a matching `Uplift` commit message

## Phase 4 — Test coverage

Map changed source files to expected test locations and flag any source change with no test in the diff:

| Source path changed | Expected test path |
|---|---|
| `pjrt_implementation/src/` | `tests/jax/` or `tests/torch/` |
| `python_package/tt_torch/` | `tests/torch/` |
| `python_package/tt_jax/` | `tests/jax/` |
| `integrations/vllm_plugin/` | `tests/integrations/vllm_plugin/` |
| `scripts/` | (test optional — flag only if logic-bearing) |

If source changed but no test file in `git diff main..HEAD --name-only` matches the expected directory, surface it as: "Source change in X, no test added under Y."

## Phase 5 — Commit message check

For each commit on the branch (`git log main..HEAD --format='%s'`):

1. Title length ≤72 chars
2. Either matches `^\[(vLLM plugin|vLLM|CI|Test Infra|pjrt|FX fusing|test|build|python-package|tools)\] ` *or* starts with one of `Add `, `Fix `, `Update `, `Enable `, `Remove `, `Disable `, `Uplift `
3. No trailing period
4. No conventional-commit prefixes (`feat:`, `fix:`, `chore:`, etc.)
5. Imperative mood (heuristic: first word doesn't end in `ed` or `ing`)

Report per-commit PASS/FAIL.

## Output format

End with a single block the user can paste into a PR review or scratch note:

```
PR-readiness for <branch>:
- Lint:           PASS / FAIL (<tool>: <details>)
- SPDX:           PASS / FAIL (<N> new files missing header)
- Diff risks:     <N> flags — <one-line summary>
- Test coverage:  PASS / GAP (<source> → no test in <dir>)
- Commit msgs:    PASS / FAIL (<commit>: <reason>)

Verdict: READY / NEEDS-FIXES
```

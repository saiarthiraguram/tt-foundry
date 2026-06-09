# draft.md body template

GitHub-flavored Markdown. Match the tone of validated drafts from
`tenstorrent/tt-xla` branch `akannan/issue_creation_skill`.

## Section order

### 1. Opening (`### Describe the bug` or `### Summary`)

- First bullet: **model key** or upstream context + failure metric
- Second bullet: surface error vs underlying TT_FATAL / PCC assertion
- Third bullet (optional): arch, dtype, what partially improved vs related issues

For upstream-only bugs (tt-mlir / forge-fe), use `### Upstream context` + `### Summary`
and note historical status if not reproducible on current `main`.

### 2. Call chain

ASCII tree, 3–6 levels:

```
ModelWrapper
  → Submodule
      → failing_op(...)
          → StableHLO / TTIR lowering
              → ttnn.<op>   # exact op from log
```

### 3. Key observations

Bullets with **concrete numbers**:
- Buffer sizes (bytes), shapes, per-bank free DRAM
- PCC per layer / cumulative sweep breakpoints
- Whether isolated op reproduces vs whole-model only
- Comparison to related issue (#NNNN) — same path, smaller/larger alloc

### 4. Experiments / sanities (when applicable)

| Test | Result | Notes |
|------|--------|-------|
| Whole model | OOM / PCC 0.97 | log path |
| Isolated op sanity | OOM / pass | shapes |

Skip this table for single-repro cases.

### 5. Steps to reproduce

Fenced bash block with **exact** commands:

```bash
pytest -svv "tests/runner/test_models.py::test_all_models_torch[<model_key>]"
```

Include `git checkout <branch>` when repro is branch-specific.

Paste a **short** failing traceback excerpt (5–15 lines) after the commands when helpful.

### 6. Logs

- List absolute paths (`/tmp/...`, `.claude/bringup/.../logs/...`)
- Name the decisive line (e.g. "line 29: TT_FATAL Out of Memory")
- Paste one excerpt block for the primary error

### 7. Expected behavior

One paragraph — what should happen instead.

### 8. Suggested next steps

**Required** when drafting with `--from-bringup` and `escalation.json` has a
non-empty `recommended_next_steps` array. Emit as `### Suggested next steps` with
a **numbered list** — one item per array entry, preserving order and technical
content (e.g. `runtime_debug: …`, `reduce_activation_footprint: …`).

If `recommended_next_steps` is missing but `blocker.recommended_next_steps` or
`state.json` → `failure_reasons` imply clear levers, infer 1–3 items and still
include this section for bringup escalations.

Skip this section only for `--from-log` / bare repro drafts with no escalation
handoff and no investigation notes.

### 9. Related issues

**`draft.md` is filing-ready GitHub body text.** Write only what a reviewer sees
on the issue — never process narration (`Ran gh issue list`, search query strings,
`gh unavailable`, internal `.claude/issues/` paths, or "this draft supersedes …").

**When search found hits** — bullet each with context:

- `#3419` — earlier grid_sample OOM; same lowering path, larger alloc
- `tenstorrent/tt-mlir#3370` — owning compile fix; this issue tracks model surface
- Distinguish **phase** (compile vs runtime) and **component** (transformer vs text_encoder)

**When search returned no hits** — one clean line (no italics, no query strings):

```markdown
### Related issues

- No similar issues found in `tenstorrent/tt-xla` at time of investigation.
```

If companion issues exist (parent tracker, sibling component, upstream owner), list
those **above** the "no similar" line. Record the queries you ran in
`manifest.json` → `related_issue_search` only (not in `draft.md`).

**When `gh` was unavailable before filing** — still use filing-ready wording:

```markdown
- No similar issues identified in `tenstorrent/tt-xla` (GitHub search pending before filing).
```

Put queries in `manifest.related_issue_search` and `dev_adds_manually`:
`related_issues: confirm no duplicate on GitHub before filing`.

**Duplicate / update-in-place:** if an open issue already tracks the same
failure (e.g. bare PCC tracker with empty body), say:
`#4784 — same component and failure mode; expand that issue with this investigation
rather than filing a duplicate.`

**Comment-on-upstream vs new issue:** if the root fix lives in an open tt-mlir
issue, cite it and state this repro belongs as a **comment on #3370**, not a
duplicate compiler issue.

### 10. Notes (optional)

Arch, classification (model vs op-level), historical status, branch name.
Do **not** duplicate `recommended_next_steps` here — they belong in §8.
Do **not** mention draft versions, skill invocation, or manifest fields.

## Style rules

- Use backticks for op names, shapes, file paths
- Prefer tables for sweep / sanity results
- Do not paste entire 500-line logs — excerpt + path
- For OOM: always include allocation size, bank count, free per-bank if in log
- For PCC: include observed, required, atol; note cumulative vs single-layer
- **Never** in `draft.md`: `_Ran gh issue list_`, `(searched … — no hits)`,
  `.claude/issues/<folder>`, `this draft is v2`, `gh CLI cannot set` in Related
  issues (Type: Bug reminder belongs in Notes or `dev_adds_manually` only once)

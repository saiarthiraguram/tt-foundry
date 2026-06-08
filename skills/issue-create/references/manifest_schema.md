# manifest.json schema

Each issue draft folder includes `manifest.json` for automation and review checklists.

## Required fields

| Field | Type | Description |
|-------|------|-------------|
| `repo` | string | Target repo, e.g. `tenstorrent/tt-xla`, `tenstorrent/tt-mlir` |
| `type` | string | `investigated` \| `repro_only` |
| `category` | string | `oom` \| `pcc` \| `compile` \| `runtime` \| `missing_op` |
| `arch` | string | e.g. `n150`, `p150`, `n300-llmbox` |
| `suggested_labels` | string[] | Usually `["bug"]`; add `enhancement` only when appropriate |
| `upstream_issues` | string[] | Issues in **other** repos that own the root fix |
| `related_issues` | string[] | Same-repo or cross-repo duplicates / prior reports |
| `log_paths` | string[] | Absolute paths to logs cited in `draft.md` |
| `draft_body` | string | Always `draft.md` |
| `title_file` | string | Always `title.txt` |
| `dev_adds_manually` | string[] | Fields gh CLI cannot set or that need human judgment |

## Optional fields

| Field | Type | When to set |
|-------|------|-------------|
| `model_key` | string | tt-xla runner key, e.g. `oft/pytorch-single_device-inference` |
| `branch` | string | Feature branch with repro or bisect artifacts |
| `test_path` | string | Pytest node id for component-level repro |
| `pcc_observed` | number | Observed PCC when category is `pcc` |
| `pcc_required` | number | Threshold from evaluator (usually 0.99) |
| `atol_observed` | number | Observed atol when relevant |
| `notes` | string | Historical bugs, filing caveats, "do not file yet" |
| `related_issue_search` | string | Audit trail only — queries run and outcome (`no hits` / `gh pending`). **Never** copy into `draft.md` |
| `recommended_next_steps` | string[] | Echo of `escalation.json` → `recommended_next_steps` when `--from-bringup` |

## Example — OOM (tt-xla model)

```json
{
  "repo": "tenstorrent/tt-xla",
  "type": "investigated",
  "category": "oom",
  "arch": "n150",
  "model_key": "oft/pytorch-single_device-inference",
  "suggested_labels": ["bug"],
  "upstream_issues": [],
  "related_issues": ["tenstorrent/tt-xla#3419"],
  "log_paths": ["/tmp/oft4244_model.log"],
  "draft_body": "draft.md",
  "title_file": "title.txt",
  "dev_adds_manually": [
    "milestone",
    "assignee",
    "parent_tracker",
    "Type field (Bug) in tt-xla UI"
  ]
}
```

## Example — compile (tt-mlir, historical)

```json
{
  "repo": "tenstorrent/tt-mlir",
  "type": "investigated",
  "category": "compile",
  "arch": "n150",
  "suggested_labels": ["bug"],
  "upstream_issues": [],
  "related_issues": [],
  "log_paths": [],
  "draft_body": "draft.md",
  "title_file": "title.txt",
  "dev_adds_manually": [
    "milestone",
    "assignee",
    "parent_tracker",
    "verify still-reproducible before filing (historical bug)"
  ],
  "notes": "Historical bug, not reproducible on current main. Draft retained for template validation."
}
```

## Example — OOM escalation (`gh` unavailable)

```json
{
  "repo": "tenstorrent/tt-xla",
  "type": "investigated",
  "category": "oom",
  "arch": "n300-llmbox",
  "model_key": "flux_2_dev/pytorch-Dev-transformer",
  "suggested_labels": ["bug"],
  "upstream_issues": [],
  "related_issues": [],
  "related_issue_search": "gh unavailable; dev must search 'transformer OOM tensor parallel activation' / 'replicated activation Shardy' before filing",
  "recommended_next_steps": [
    "runtime_debug: identify replicated intermediate and force-shard",
    "reduce_activation_footprint: seq_len<=128 MVP pass"
  ],
  "log_paths": [".claude/bringup/flux_2_dev/logs/iter_3_run.log"],
  "draft_body": "draft.md",
  "title_file": "title.txt",
  "dev_adds_manually": [
    "milestone",
    "assignee",
    "parent_tracker",
    "Type: Bug (tt-xla UI field)",
    "related_issues: run gh issue list search and cross-link before filing"
  ]
}
```

## Example — PCC (component test)

```json
{
  "repo": "tenstorrent/tt-xla",
  "type": "investigated",
  "category": "pcc",
  "arch": "n150",
  "model_key": "playground_v2_5/text_encoder_2",
  "branch": "kkannan/may21_playground_v2_5_encoder_2_pcc_drop",
  "test_path": "tests/torch/models/playground_v2_5/test_text_encoder_2.py::test_text_encoder_2",
  "pcc_observed": 0.9711294738181017,
  "pcc_required": 0.99,
  "atol_observed": 2.083254814147949,
  "suggested_labels": ["bug"],
  "upstream_issues": [],
  "related_issues": [],
  "log_paths": ["/tmp/playground_te2_pcc4709.log"],
  "draft_body": "draft.md",
  "title_file": "title.txt",
  "dev_adds_manually": ["milestone", "assignee", "parent_tracker", "Type: Bug (UI field)"]
}
```

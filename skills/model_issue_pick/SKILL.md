---
name: model_issue_pick
description: Picks a KNOWN_FAILURE_XFAIL entry from tests/runner/test_config/torch/test_config_inference_single_device.yaml, runs the model's pytest in the background, then analyzes the log to verify whether the recorded failure still reproduces, the test is now silently passing (recommend RM_XFAIL → EXPECTED_PASSING), or the failure has changed (recommend updating the reason). Use when the user asks to triage/refresh XFAIL entries in the inference YAML.
allowed-tools: Bash Read Write Edit Grep
---

# XFAIL Entry Re-verification

You re-verify a single `KNOWN_FAILURE_XFAIL` entry in
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` by
running its pytest in the background and classifying the new log.

## Invocation

`/model_issue_pick [model_key] [--pick first|random|index=N] [--arch n150] [--timeout 600] [--apply]`

- `model_key` (optional positional): full key like
  `perceiverio_vision/pytorch-Vision_Perceiver_Conv-single_device-inference`.
  If omitted, the skill selects an XFAIL entry per `--pick` (default `random`).
- `--pick` (default `random`): selection strategy when no `model_key` is given.
  - `first` — earliest XFAIL entry by line number
  - `random` — uniform random over candidates
  - `index=N` — Nth candidate (0-based) in line-order
- `--arch` (default `n150`): used to (a) resolve `arch_overrides` and (b) set
  `TT_XLA_ARCH` for the pytest subprocess.
- `--timeout` (default `600`): wall-clock seconds for pytest.
- `--apply` (default off): write the recommended status change back to the
  YAML. Without `--apply`, the skill **only reports**.

## Workspace

`./.claude/model_issue_pick/<safe_key>/` where `<safe_key>` is `model_key`
with `/` replaced by `__`. Files:

- `run.log` — full pytest stdout+stderr for the iteration.
- `report.md` — verdict, evidence, recommended YAML change.
- `pid` — background pytest pid (cleaned up at completion).

## Steps

### 1. Enumerate KNOWN_FAILURE_XFAIL candidates

Use Python (one-shot via `python -c`) on the YAML — do not hand-roll a grep:

```python
import yaml, sys
from pathlib import Path
p = Path("tests/runner/test_config/torch/test_config_inference_single_device.yaml")
data = yaml.safe_load(p.read_text())["test_config"] or {}
ARCH = "<arch>"
out = []
for key, cfg in data.items():
    if cfg is None: continue
    # Top-level XFAIL
    if cfg.get("status") in ("KNOWN_FAILURE_XFAIL", "known_failure_xfail"):
        out.append((key, "top", cfg.get("reason"), cfg.get("required_pcc"),
                    cfg.get("assert_pcc")))
        continue
    # arch_overrides XFAIL — only count if it applies to ARCH
    ov = (cfg.get("arch_overrides") or {}).get(ARCH)
    if ov and ov.get("status") in ("KNOWN_FAILURE_XFAIL", "known_failure_xfail"):
        out.append((key, f"arch:{ARCH}", ov.get("reason"),
                    ov.get("required_pcc"), ov.get("assert_pcc")))
for row in out: print("\t".join("" if x is None else str(x) for x in row))
```

If `model_key` is supplied, skip enumeration and just resolve its entry — fail
fast if it is not currently `KNOWN_FAILURE_XFAIL` (top-level or `arch:<arch>`),
because re-verifying a non-XFAIL is not this skill's job.

### 2. Pick one

Apply `--pick` strategy. Echo the choice and the recorded `reason` so the user
sees what is about to run:

```
[pick] <model_key>   scope=<top|arch:n150>
       reason: <YAML reason or '<none>'>
```

### 3. Discover the test node id

```bash
pytest -q --collect-only tests/runner/test_models.py 2>&1 \
  | grep -F 'test_all_models_torch[<model_key>]'
```

If 0 matches → abort and print the loader/import hint (this means the YAML
entry references a model the runner cannot discover). If >1, take all.

### 4. Run pytest in the background

Run with `--runxfail`, a wall-clock cap, and tee to the log. Use `-svv` and
`--tb=long` so the log contains live model output and full tracebacks — this
is what makes the classification in step 5 trustworthy when a test silently
hangs or stalls partway through.

```bash
TT_XLA_ARCH=<arch> timeout <timeout> python -m pytest <node_ids> \
  --runxfail -svv --tb=long -p no:cacheprovider \
  --json-report --json-report-file=.claude/model_issue_pick/<safe_key>/result.json \
  2>&1 | tee .claude/model_issue_pick/<safe_key>/run.log
```

Flag rationale:
- **`-s`**: disable stdout capture so model-side prints, weight downloads,
  XLA compile traces, and tqdm progress reach the log in real time.
  Critical for diagnosing timeouts — without `-s`, a hung test produces a
  completely silent log and there is no signal to investigate.
- **`-vv`**: double-verbose. Shows full assertion diffs and longer node-id
  lines so the test summary is unambiguous.
- **`--tb=long`**: full traceback with surrounding source context for each
  frame. Required for the `xfail_changed` rule (step 5) to extract a useful
  new reason; `--tb=short` truncates the frames we need.

**Why `--runxfail` is required.** The model runner calls `pytest.xfail(reason)`
unconditionally in its finally path (see `tests/runner/test_utils.py:748`), so
without this flag *every* KNOWN_FAILURE_XFAIL run ends as `xfailed` regardless
of whether the model actually passed or failed. `--runxfail` makes
`pytest.xfail()` a no-op, so the run reports the test's real outcome — PASSED
when the model now works, FAILED with the real traceback when it doesn't.
This is what makes the verdict in step 5 trustworthy.

Invoke this through the `Bash` tool with `run_in_background: true`. Capture
the shell id. Do **not** poll on a sleep; the harness notifies on completion.
The user explicitly wants background execution — do not switch to foreground.

While waiting, briefly tell the user the pid/shell id and the log path so they
can `tail -f` if they want.

### 5. Classify the result

With `--runxfail` in effect the pytest exit code is the primary signal, and
the JSON report at `<workspace>/result.json` carries the runner-classified
`bringup_status` / `pcc` / `failing_reason` tags that disambiguate rule 2 vs
rule 3 (and the `now_incorrect_result` PCC fallback). Always prefer JSON
over stdout regex — the stdout log is for human review.

Rules in order:

| Order | Signal | Verdict | Recommendation |
|---|---|---|---|
| 1 | `exit_code == 124` (timeout sentinel) | `timeout` | Do not change status. Report and stop. |
| 2 | `exit_code == 0` AND `result.json` `tests[0].metadata.tags.bringup_status == "PASSED"` | `now_passing` | Flip status to `EXPECTED_PASSING`. Drop `reason`. |
| 3 | `exit_code == 0` AND `tags.bringup_status == "INCORRECT_RESULT"` (test passed only because `assert_pcc: false` or PCC threshold was lax) | `now_incorrect_result` | `EXPECTED_PASSING` with `assert_pcc: false` and an updated `required_pcc`. Record the observed PCC from `tags.pcc`. |
| 4 | `exit_code != 0` AND traceback present AND a normalized substring of the YAML `reason` appears in `tags.failing_reason.description` (or in the traceback region of the log, if no JSON) | `xfail_same` | No change. Recorded reason still reproduces. |
| 5 | `exit_code != 0` AND traceback present AND no normalized YAML reason substring is present | `xfail_changed` | Keep `KNOWN_FAILURE_XFAIL`; rewrite `reason` to `tags.failing_reason.description` (preferred) or the top-of-traceback. |
| 6 | `exit_code != 0` AND no traceback (collection error, import error, harness crash) | `runner_error` | Do not change status; surface the error. |

**JSON read recipe** (Python one-liner):
```python
import json; d = json.load(open(".claude/model_issue_pick/<safe_key>/result.json"))
tags = d["tests"][0]["metadata"]["tags"]
print(tags.get("bringup_status"), tags.get("pcc"), tags.get("pcc_threshold"))
print((tags.get("failing_reason") or {}).get("description"))
```

If `result.json` is missing or malformed, fall back to stdout: look for
`=== <N> passed in <T>s ===` (rule 2 candidate) / `=== <N> failed in <T>s ===`
(rules 4/5) and record `details.source: "stdout_fallback"`.

**Normalization for rules 4/5:** strip GitHub URLs, surrounding quotes, and
case-fold, then look for the longest non-URL token (≥ 6 chars) from the YAML
reason in the candidate string. Avoids false matches on generic words like
"PCC".

For `now_incorrect_result`, the observed `pcc` from JSON tags becomes the
new floor: suggest `required_pcc = floor(observed_pcc, 2 dp) - 0.01` so the
test passes with a small safety margin.

#### PCC guardrails

Before recommending a `required_pcc` change, compute the **same-family
median** from the YAML and reject proposals that drop too far below it.
This catches one-off regressions masquerading as legitimate threshold
adjustments.

```python
import yaml, re, statistics
from pathlib import Path
y = yaml.safe_load(Path("tests/runner/test_config/torch/test_config_inference_single_device.yaml").read_text())["test_config"]
FAMILY = "<family from model_key.split('/')[0]>"
peers = []
for k, cfg in y.items():
    if k == "<this model_key>" or cfg is None: continue
    if k.startswith(FAMILY + "/"):
        if isinstance(cfg.get("required_pcc"), (int, float)):
            peers.append(cfg["required_pcc"])
median = statistics.median(peers) if peers else None
```

Guardrail rules:

| Condition | Action |
|---|---|
| `len(peers) == 0` | Skip the guardrail — no peers to compare against. Proceed with the proposed PCC. Note `pcc_guardrail: "no_peers"` in the report. |
| Proposed PCC is **within 0.02** of the peer median | Proceed normally. `pcc_guardrail: "passed (median=<m>, proposed=<p>)"` |
| Proposed PCC is **more than 0.02 below** the peer median | **Warn** in the report and downgrade the recommendation: do not auto-apply even with `--apply`; instead emit `pcc_guardrail: "blocked (median=<m>, proposed=<p>)"` and surface a human-review prompt. Possible interpretations: a real PCC regression, an outlier model, or an incorrect peer set. The user decides. |

The guardrail only blocks under `--apply` mode. In dry-run mode the report
still includes the proposed PCC change but with the warning attached.

### 6. Write report and (optionally) apply

Write `report.md` containing:

```
# Re-verification report

Model key   : <model_key>
Scope       : <top|arch:n150>
YAML reason : <recorded reason>
Verdict     : <verdict from step 5>

## Provenance
tt-xla SHA       : <short sha of tt-xla HEAD>
tt-foundry SHA   : <short sha if submodule present, else "not a submodule">
Generated       : <YYYY-MM-DD HH:MM>
Pytest budget   : <timeout> s
Result classified from : json_report | stdout_fallback

## Evidence
<6–20 lines of relevant log excerpt: the bringup_status/guidance tag line,
the pytest summary line, and the first 5 lines of any traceback>

## Recommended YAML change
<diff-style snippet showing the existing entry and the proposed entry,
or "no change" for xfail_same/timeout/runner_error>
```

Print the verdict and the recommended change to the user. Do **not** edit the
YAML unless `--apply` was passed. When `--apply` is set:

- `now_passing`: replace the `status: KNOWN_FAILURE_XFAIL` line and remove the
  adjacent `reason:` line within the same block.
- `now_incorrect_result`: same status flip, plus add `assert_pcc: false` and a
  `required_pcc` line if not already present; preserve any existing PCC if it
  is already lower than the observed value.
- `xfail_changed`: keep `status: KNOWN_FAILURE_XFAIL`, rewrite the `reason:`
  value only. Quote with double quotes; escape interior `"` as `\"`.

Use the `Edit` tool for YAML mutations and target the smallest unique
surrounding block — never `replace_all`, since the same status string appears
on dozens of entries.

### 7. Terminal output

```
[model_issue_pick] <model_key>  arch=<arch>  verdict=<verdict>
  recorded reason : <truncated to 80 chars>
  observed signal : <one-line summary>
  log             : .claude/model_issue_pick/<safe_key>/run.log
  report          : .claude/model_issue_pick/<safe_key>/report.md
  yaml change     : <applied | dry-run (use --apply) | none>
```

## Hard rules

- **Background-only run.** Use `Bash` with `run_in_background: true` for the
  pytest invocation. The user asked for this explicitly. Do not switch to
  foreground "to wait."
- **Never demote on timeout.** A `timeout 600` hit is not evidence; report and
  stop. (Consistent with the project's rule that automated timeouts are not
  the source of truth for YAML status.)
- **Single entry per invocation.** Even if `--pick random` is given, run
  exactly one model. The orchestration of batches belongs to a loop/cron
  layer, not this skill.
- **No silent YAML edits.** Default mode is report-only; the user has to pass
  `--apply` to mutate the YAML.
- **Do not delete the workspace.** Leave `run.log` and `report.md` so the
  user can revisit. The skill is idempotent — re-running overwrites.

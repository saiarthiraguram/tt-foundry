---
name: potential_new_models
description: Suggests SOTA models that are good candidates for tt-forge-models / tt-xla bringup. Scans currently-covered model families under third_party/tt_forge_models/, then proposes newer, larger, or architecturally-adjacent SOTA models in the three priority areas (LLMs, video generation, vision/image models). Writes a plaintext .txt file with model name, reference URL, and rationale per candidate. Use when the user wants ideas for the next bringup, or asks "what should we add to tt-forge-models next?".
allowed-tools: Bash Read Write WebSearch WebFetch
---

# Potential New Models — Bringup Candidate Suggester

You propose SOTA models that should be considered for the next bringup wave
in `tt-forge-models` / `tt-xla`. Output is a plaintext `.txt` file the user
can hand to a planner or paste into a ticket.

## Invocation

`/potential_new_models [--out <path>] [--areas llm,video,vision] [--per-area N] [--no-web]`

- `--out` (default `.claude/potential_new_models/candidates.txt`): output file.
- `--areas` (default `llm,video,vision`): which focus areas to populate.
  Comma-separated subset of `llm`, `video`, `vision`. LLMs, video generation,
  and vision/image models are the priority areas — keep this default unless
  the user asks otherwise.
- `--per-area` (default `5`): how many candidates per area.
- `--no-web` (default off): skip WebSearch/WebFetch and rely solely on prior
  knowledge. Use when the user explicitly wants offline operation.

## Selection criteria

A model is a good candidate when **all** of the following hold:

1. **Not already covered.** Its family directory does not exist in
   `third_party/tt_forge_models/`. Treat the directory name as the family
   key — e.g. `llama` covers all Llama variants; `qwen_2_5_vl` is distinct
   from `qwen_3_vl`.
2. **SOTA or near-SOTA** in its category at the time of suggestion. Prefer
   models that are referenced in recent (≤ 12 month) leaderboards, benchmark
   papers, or widely-cited HuggingFace cards.
3. **Open weights available** — the model can actually be loaded from
   HuggingFace, the official repo, or a known mirror. Closed-API-only models
   (GPT-4.x, Gemini Pro, etc.) are NOT candidates.
4. **Plausible on a single n150 / single device** — multi-hundred-billion-
   parameter dense models that obviously cannot fit are NOT candidates; flag
   them as "future multi-chip target" instead of including them.
5. **Architecturally adjacent or strategically valuable.** Prefer at least
   one of:
   - A newer/larger variant of a family already in `tt-forge-models` (lower
     bringup lift, validates that existing compiler paths still work),
   - A widely-cited new architecture that the project does not yet cover at
     all (e.g. a new diffusion backbone, new tokenizer-free LLM),
   - A model that anchors a benchmark the team is tracking (MTEB, MMMU, VBench).

If a candidate fails any criterion, drop it; the .txt file should never
contain "we can't actually run this" entries.

## Steps

### 1. Scan currently-covered families

```bash
ls third_party/tt_forge_models/ \
  | grep -Ev '^(__pycache__|LICENSE|LICENSE_understanding\.txt|README\.md|__init__\.py|base\.py)$' \
  | sort > /tmp/_covered_families.txt
wc -l /tmp/_covered_families.txt
```

Read the list. This is the **exclusion set** for criterion 1. Also peek at
`tests/runner/test_config/torch/test_config_inference_single_device.yaml` to
identify families that exist as directories but whose specific variants
might still be uncovered (e.g. `llama` is covered but `llama-3.3` is not yet
a variant — those still count as a "covered family" for our purposes and
should NOT be suggested unless the new variant is architecturally novel).

### 2. Generate candidate lists per area

For each of `llm`, `video`, `vision` (filtered by `--areas`):

- If `--no-web` is NOT set: use `WebSearch` to find recent SOTA leaders in
  the area. Useful seed queries (adapt to the current date):
  - LLMs: `"open weights LLM" SOTA 2026 leaderboard MMLU OR GPQA OR LMArena`
  - Video gen: `"text-to-video" open weights 2026 VBench OR HuggingFace`
  - Vision: `"vision language model" OR "image generation" open weights 2026 MMMU OR ImageNet`
  Take 1–2 top results, then `WebFetch` the most authoritative page (Papers
  with Code, HuggingFace org page, project README) to confirm the model
  exists with open weights.
- If `--no-web` is set: rely only on prior knowledge — clearly mark these
  suggestions as "verify availability before bringup".

For each candidate, capture:
- **Name** (exactly as it appears on HuggingFace or the project README).
- **Family key** — the directory name it would live under in
  `tt-forge-models` (snake_case of the family, e.g. `llama_4`, `wan_2_2`,
  `intern_vl_3`).
- **Reference URL** — HuggingFace card, project repo, or paper.
- **Param scale** — approximate, used to confirm criterion 4.
- **Reason** — 1–2 sentences explaining what makes it a candidate. Reference
  criterion 5: is it a family extension, a novel architecture, or a benchmark
  anchor?

Filter against the exclusion set from step 1.

### 3. Write the .txt file

Plaintext, fixed-width framed sections. Do **not** use Markdown formatting
since the file is `.txt` and may be pasted into ticketing systems that don't
render Markdown.

```
================================================================================
Potential new model candidates for tt-forge-models / tt-xla bringup
Generated: <YYYY-MM-DD>
Already-covered families scanned from: third_party/tt_forge_models/
Total currently-covered families: <N>
Areas: <comma-separated areas>
================================================================================

[1] LLMs
--------------------------------------------------------------------------------
* <Model name 1>
  family key : <proposed dir name>
  reference  : <URL>
  size       : ~<N>B params
  reason     : <1-2 sentences>

* <Model name 2>
  ...

[2] Video generation
--------------------------------------------------------------------------------
* <Model name 1>
  ...

[3] Vision / image models
--------------------------------------------------------------------------------
* <Model name 1>
  ...

================================================================================
Notes
--------------------------------------------------------------------------------
- This list excludes families already present under third_party/tt_forge_models/.
- Param sizes are rough; verify against the model card before estimating
  device fit.
- Source date matters: SOTA shifts quickly in LLMs and video gen. Re-run this
  skill before a new bringup wave.
================================================================================
```

Sections for areas not in `--areas` should be omitted entirely (not left
empty). Create parent dirs for `--out` if missing.

### 4. Terminal output

```
[potential_new_models] areas=<...>  per_area=<N>  out=<path>
  covered families : <N>
  candidates total : <M>
  by area          : llm=<a> video=<b> vision=<c>
```

Print the first 3 candidate names per area inline as a sanity check.

## Hard rules

- **Read-only on the repo.** This skill only writes to `--out`; it never
  edits `tt-forge-models/`, YAML configs, or any source code.
- **No fabrication.** Every candidate must have a real reference URL. If
  WebSearch + WebFetch can't confirm a model exists with open weights,
  drop it from the list rather than guessing.
- **Open weights only.** Closed-API-only models are out of scope — bringup
  needs to load weights locally.
- **No duplicates.** Cross-check the family key against the
  `tt-forge-models` directory listing; skip if it matches even fuzzily
  (`llama_3` vs `llama` → covered).
- **Plaintext output.** The output file is `.txt`, not `.md`. Don't emit
  Markdown headers / tables / backticks in the file.

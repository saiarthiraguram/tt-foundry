---
name: vllm-model-bringup
description: Bring up a model on the vLLM-TT (Tenstorrent) integration. Use when adding and bringing up a model under vLLM on TT hardware and triggering its test in CI.
disable-model-invocation: false
argument-hint: model-name
---

# vLLM Model Bringup

## Goal

Bring up model `$1` on the vLLM-TT integration.

## Steps

### Step 1: Verify GitHub CLI auth

Bring-up needs the GitHub CLI (`gh`) authenticated — for raising the draft PR
and triggering/watching the CI run. Before anything else, check:

```bash
gh auth status
```

If it reports you are **not** logged in (non-zero exit / "not logged into any
GitHub hosts"), stop and ask the user to authenticate — they can run it in this
session with the `!` prefix:

```
! gh auth login
```

Do not proceed to Step 2 until `gh auth status` succeeds.

### Step 2: Gather requirements from the user

Before doing anything else, ask the user the following questions **as plain
text** (do NOT use the `AskUserQuestion` tool). Lead with the statement
**"Quick requirements check:"** and then ask exactly these four questions:

```
Quick requirements check:

1. Model name — Which vLLM model should be brought up? (HuggingFace model id, e.g. meta-llama/Llama-3.1-8B-Instruct.)
2. Inference mode — Single-inference (single device) or parallel-inference (multi-device / tensor-parallel)?
3. Model type — Text-based or multimodal?
4. Device (optional) — n150 (single) / n300-llmbox or galaxy (parallel). Leave blank to default from the inference mode.
```

Questions 1–3 are **mandatory**; question 4 (device) is **optional**. If the
user leaves the device blank, default to the device implied by the inference
mode (`n150` for single inference).

Do not proceed to later steps until the mandatory answers (model name, inference
mode, model type) are collected.

### Step 3: Create the test case

Add a **new test case at the end of**
`tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py`,
using `test_tensor_parallel_generation_galaxy_wh_6u_large` as the reference
(same `llm_args` shape, `assert_output_coherent`, `check_host_memory`).

**Follow the reference template exactly — do NOT add extra comments.** Keep the
`llm_args` shape identical to the template below and only change values that
Step 2 / Step 4 dictate.

**Name the test after the MODEL, not the device.** The convention is
`test_<parallelism>_generation_<model>` — e.g. `test_tensor_parallel_generation_mistral_small`,
`test_tensor_parallel_generation_gemma4_31b`, `test_tensor_parallel_generation_mistral_large`.
The device (`galaxy`, `n150`, `llmbox`) is expressed through the
`@pytest.mark.*` hardware marker, **never** baked into the function name. The
reference below happens to carry a device suffix (`..._galaxy_wh_6u_large`) —
that is legacy naming; do not copy it. Derive the suffix from the model id
(strip the org prefix and shorten, e.g. `mistralai/Mistral-Large-Instruct-2411`
→ `mistral_large`).

```python
llm_args = {
    "model": model_name,
    "max_num_batched_tokens": 4906,
    "max_num_seqs": 1,
    "max_model_len": 1024,
    "gpu_memory_utilization": 0.17,
    "additional_config": {
        "min_context_len": 1024,
        "enable_tensor_parallel": True,
        "experimental_weight_dtype": "bfp_bf8",
        "optimization_level": opt_level,
    },
}
```

Two things are decided by the answers from Step 2:

**a) Input — driven by model type.**

- **Text-based** → a plain prompt list, generated with `llm.generate(...)`:

  ```python
  prompts = [
      "I like taking walks in the",
  ]
  ...
  output_text = llm.generate(prompts, sampling_params)[0].outputs[0].text
  ```

- **Multimodal** → an image + chat message, generated with `llm.chat(...)`:

  ```python
  image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"
  messages = [
      {
          "role": "user",
          "content": [
              {
                  "type": "text",
                  "text": "What action do you think I should take in this situation?",
              },
              {"type": "image_url", "image_url": {"url": image_url}},
          ],
      },
  ]
  ...
  output_text = llm.chat(messages, sampling_params=sampling_params)[0].outputs[0].text
  ```

**b) Markers — driven by inference mode.**

- **Parallel inference** → include `@pytest.mark.tensor_parallel` and set
  `"enable_tensor_parallel": True` in `additional_config`.
- **Single inference** → **do not** add `@pytest.mark.tensor_parallel`, and
  omit `enable_tensor_parallel` from `additional_config`.

Keep `@pytest.mark.nightly` and add the hardware marker for the **Device** from
Step 2:

- `n150` → `@pytest.mark.single_device`
- `n300-llmbox` → `@pytest.mark.llmbox`
- `galaxy` → `@pytest.mark.galaxy_wh_6u`

Copy the reference test's `llm_args` as a starting point — the numeric engine
args (`max_model_len`, `max_num_batched_tokens`) are computed in Step 4 and
filled in there.

Reference test (multimodal, parallel):

```python
@pytest.mark.nightly
@pytest.mark.tensor_parallel
@pytest.mark.galaxy_wh_6u
@pytest.mark.parametrize(
    ["model_name", "opt_level"],
    [pytest.param("mistralai/Pixtral-Large-Instruct-2411", 0)],
)
def test_tensor_parallel_generation_galaxy_wh_6u_large(model_name: str, opt_level: int):
    image_url = "https://static.wikia.nocookie.net/essentialsdocs/images/7/70/Battle.png/revision/latest?cb=20220523172438"
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "What action do you think I should take in this situation?",
                },
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    sampling_params = vllm.SamplingParams(temperature=0.8, top_p=0.95, max_tokens=32)
    llm_args = {
        "model": model_name,
        "max_num_batched_tokens": 4906,
        "max_num_seqs": 1,
        "max_model_len": 1024,
        "gpu_memory_utilization": 0.17,
        "additional_config": {
            "min_context_len": 1024,
            "enable_tensor_parallel": True,
            "experimental_weight_dtype": "bfp_bf8",
            "optimization_level": opt_level,
        },
    }
    llm = vllm.LLM(**llm_args)

    output_text = llm.chat(messages, sampling_params=sampling_params)[0].outputs[0].text
    print("output: ", output_text)
    assert_output_coherent(output_text)

    check_host_memory(model_name)
```

### Step 4: Compute the engine args and update the test case

Once the test case from Step 3 exists, feed **the actual input/prompt from that
test** into the calculator and use its output to fill in the numeric engine
args. `max_model_len` and `max_num_batched_tokens` vary model-to-model, but they
are **derivable from `model_name` + the test's prompt/input** — do not
hand-guess them.

Derive the calculator inputs from the test you just wrote:

- `--prompt` — pass the **exact prompt text** from the test; the calculator
  tokenizes it with the model's own tokenizer for an exact count. **Prefer this
  over `--prompt-tokens N`** — do not hand-count / eyeball token counts. Use
  `--prompt-tokens N` only as a fallback when the repo has no HF tokenizer
  (Mistral `params.json`-only repos).
- `--output-tokens` — the `max_tokens` from the test's `SamplingParams`.
- `--num-images` — number of images in the test input (0 for text-only).
- `--max-num-seqs` — the `max_num_seqs` set in `llm_args`.

Run the calculator:

```bash
python .claude/skills/vllm-model-bringup/scripts/calc_engine_args.py MODEL_NAME \
    --prompt "exact prompt text from the test" \
    --output-tokens N [--num-images N] [--max-num-seqs N]
```

Then **write the calculator's suggested values back into the Step-3 test case**,
updating `max_model_len` and `max_num_batched_tokens` in `llm_args` to match.
Also set `additional_config["min_context_len"]` equal to `max_model_len` (the
reference keeps them in lockstep, e.g. both `1024`; for a short text prompt both
land at `64`) — do not leave the template's `1024` when `max_model_len` drops.

How the values are derived from the HF config:

- **text vs multimodal** — presence of a `vision_config` / `text_config`.
- **ceiling for `max_model_len`** — `max_position_embeddings` (top-level, or
  under `text_config` for multimodal). Never exceed this.
- **vision tokens per image** — read `mm_tokens_per_image` / `image_seq_length`
  if the config exposes it; otherwise the processor is run on a dummy image and
  the image placeholder tokens are counted (ground truth); last resort is the
  patch-grid estimate `(image_size / patch_size)^2`.

Formulas the script applies (rounded up to a multiple of 32):

- `max_model_len   = prompt_tokens + num_images*tokens_per_image + output_tokens`
  (capped at `max_position_embeddings`)
- `max_num_batched_tokens = max_model_len * max_num_seqs`

Caveats to tell the user:

- `max_num_batched_tokens` is a **starting point**. `TTPlatform` overrides it for
  MLA models and when `prefill_chunk_size` is set (see
  `integrations/vllm_plugin/vllm_tt/platform.py`); for multimodal, raise it if the
  image prefill OOMs.
- `gpu_memory_utilization` is tuned separately (it's the KV-cache budget; TT
  weights live outside it, so small values like 0.01–0.17 are normal).

### Step 5: Raise the branch (draft PR)

Commit **only the new test case** from Steps 3–4, and leave stray working-tree
files out (logs, scratch dirs, the local `.claude/skills/` tooling).

Branch name is `model_bringup_<model_name>`, where `<model_name>` is the short
model suffix used for the test (org prefix stripped, shortened) — e.g.
`mistralai/Mistral-Large-Instruct-2411` → `model_bringup_mistral_large`.

```bash
git checkout -b model_bringup_<model_name>

git add tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py

git commit -m "$(cat <<'EOF'
[vLLM] Bring up <HF model id> on <device> (<inference mode>)

Add a nightly <parallelism> generation test for <model> on <device>.

Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>
EOF
)"

git push -u origin model_bringup_<model_name>

gh pr create --draft \
    --title "[vLLM] Bring up <HF model id> on <device> (<inference mode>)" \
    --body "$(cat <<'EOF'
## What
Bring up <HF model id> on the vLLM-TT integration.

## Changes
- Add `test_<parallelism>_generation_<model>` (device marker `<marker>`).
- Engine args from the calculator: max_model_len=<N>, max_num_batched_tokens=<N> (prompt=<N>, output=<N>, images=<N>).

## Testing
Run Test Single on `<device>` (draft — pending green run).
EOF
)"
```

Keep it a **draft** until the CI run is green.

### Step 6: Trigger the test in Run Test Single

Run the single bring-up test with the **Run Test Single** workflow — it targets
one test directly, so no preset/matrix edit is involved. Set:

- **Use workflow from** — the branch raised in Step 5
  (`model_bringup_<model_name>`).
- **Path passed to pytest** (`dir`) — the full pytest nodeid of the test you
  created: `tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::<test_function_name>`
  (e.g. `...::test_tensor_parallel_generation_mistral_large`).
- **Choose runners for running test** (`runs_on`) — the device from Step 2:
  `n150` / `n300-llmbox` / `galaxy-wh-6u`.
- **Number of parallel runners** (`parallel_groups`) — `1` for a single test.
- **Install vllm-tt wheel in the test job** (`install_vllm_wheel`) — **always
  tick this checkbox / set it to `true`.** The vLLM bring-up test needs the
  vllm-tt (+ upstream vllm) wheel installed in the test job. It nominally
  auto-enables when `dir` contains `vllm_plugin`, but do **not** rely on that —
  enable it explicitly on every trigger.

Trigger it either way (against the pushed branch):

- **GitHub UI** — Actions → **Run Test Single** → *Run workflow* → fill the
  fields above.
- **CLI**:

  ```bash
  gh workflow run "Run Test Single" --ref model_bringup_<model_name> \
      -f dir="tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py::<test_function_name>" \
      -f runs_on="galaxy-wh-6u" \
      -f parallel_groups="1" \
      -f install_vllm_wheel=true
  ```

  (Confirm current input keys with `gh workflow view "Run Test Single" --yaml`.)

After dispatching, grab the run URL/id and watch it:

```bash
gh run list --workflow "Run Test Single" --branch model_bringup_<model_name> --limit 1
gh run watch <run-id>
```

Keep the PR in **draft** until this run is green; then flip it out of draft.

### Step 7: Report the summary

Close out the bring-up with a short summary for the user, pulling the links from
the previous steps:

- **Test** — the function name and file
  (`test_<parallelism>_generation_<model>` in
  `tests/integrations/vllm_plugin/generative/test_tensor_parallel_generation.py`),
  the model id, device/inference mode, and the calculator-derived engine args
  (`max_model_len`, `max_num_batched_tokens`, and the prompt/output/image token
  counts they came from).
- **Draft PR link** — the URL from Step 5's `gh pr create` (or fetch it with
  `gh pr view --json url -q .url`).
- **CI run link** — the Run Test Single URL from Step 6 (or
  `gh run list --workflow "Run Test Single" --branch model_bringup_<model_name> --limit 1`).

Example:

```
Bring-up summary — <HF model id>
- Test:    test_<parallelism>_generation_<model>  (<device>, <inference mode>)
           engine args: max_model_len=<N>, max_num_batched_tokens=<N>
- Draft PR: https://github.com/tenstorrent/tt-xla/pull/<n>
- CI run:   https://github.com/tenstorrent/tt-xla/actions/runs/<id>
```

Remind the user the PR stays a **draft** until the CI run is green.

## Notes

- vLLM integration lives in `integrations/vllm_plugin/vllm_tt/`.
- vLLM integration tests live in `tests/integrations/vllm_plugin/`.
#!/usr/bin/env python3
"""Compute suggested vLLM-TT engine args from a HuggingFace model + workload.

Formulas (rounded up to a multiple of 32):

    max_model_len = prompt_tokens + num_images * tokens_per_image + output_tokens
                    # capped at max_position_embeddings

    max_num_batched_tokens = max_model_len * max_num_seqs
                    # starting point; raise if multimodal image prefill OOMs

Usage:
    python calc_engine_args.py MODEL_NAME \\
        [--prompt-tokens N | --prompt "text"] \\
        [--output-tokens N] [--num-images N] [--max-num-seqs N]

    # or import:
    from calc_engine_args import compute_engine_args
    args = compute_engine_args("mistralai/...", prompt_tokens=20, output_tokens=32, num_images=1)
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any, Optional


class _Cfg:
    """Thin attr-access wrapper over a dict (and nested dicts)."""

    def __init__(self, data: dict):
        self._data = data
        for k, v in data.items():
            if isinstance(v, dict):
                setattr(self, k, _Cfg(v))
            else:
                setattr(self, k, v)

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)


def _get(cfg: Any, *names: str, default: Any = None) -> Any:
    for n in names:
        if isinstance(cfg, dict):
            if n in cfg and cfg[n] is not None:
                return cfg[n]
        elif hasattr(cfg, n) and getattr(cfg, n) is not None:
            return getattr(cfg, n)
    return default


def round_up(n: int, mult: int = 32) -> int:
    return ((int(n) + mult - 1) // mult) * mult


def load_model_config(model_name: str) -> tuple[Any, str]:
    """Load HF config.json, or Mistral-format params.json as a fallback."""
    from transformers import AutoConfig

    try:
        return AutoConfig.from_pretrained(model_name, trust_remote_code=True), "config.json"
    except Exception as hf_err:  # noqa: BLE001
        try:
            from huggingface_hub import hf_hub_download

            path = hf_hub_download(model_name, "params.json")
            with open(path) as f:
                data = json.load(f)
            return _Cfg(data), "params.json"
        except Exception as mist_err:  # noqa: BLE001
            raise RuntimeError(
                f"Could not load config for {model_name}: "
                f"HF config.json ({hf_err}); params.json ({mist_err})"
            ) from mist_err


def max_position(cfg: Any) -> Optional[int]:
    top = _get(cfg, "max_position_embeddings")
    if top:
        return int(top)
    tc = _get(cfg, "text_config")
    if tc is not None:
        val = _get(tc, "max_position_embeddings")
        return int(val) if val else None
    return None


def vision_config(cfg: Any) -> Any:
    """HF uses vision_config; Mistral params.json uses vision_encoder."""
    return _get(cfg, "vision_config", "vision_encoder")


def is_multimodal(cfg: Any) -> bool:
    return (
        vision_config(cfg) is not None
        or _get(cfg, "text_config") is not None
    )


def count_prompt_tokens(model_name: str, prompt: str) -> int:
    """Tokenize prompt text with the model's tokenizer (text tokens only)."""
    from transformers import AutoTokenizer

    try:
        tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return len(tok.encode(prompt, add_special_tokens=True))
    except Exception:
        # Mistral-format repos often have no HF tokenizer; approximate.
        # ~1 token per ~4 chars is a rough English lower bound; prefer --prompt-tokens.
        approx = max(1, (len(prompt) + 3) // 4)
        print(
            f"warning: no HF tokenizer for {model_name}; "
            f"approximating prompt_tokens={approx} from char length "
            f"(pass --prompt-tokens for an exact count)",
            file=sys.stderr,
        )
        return approx


def tokens_per_image(model_name: str, cfg: Any) -> tuple[Optional[int], str]:
    """Return (tokens_per_image, how_it_was_obtained)."""
    direct = _get(cfg, "mm_tokens_per_image", "image_seq_length")
    if direct:
        return int(direct), "config field"

    proc_err = "processor not attempted"
    try:
        from PIL import Image
        from transformers import AutoProcessor

        proc = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
        img = Image.new("RGB", (896, 896), (128, 128, 128))
        vc = vision_config(cfg)
        image_token_id = _get(cfg, "image_token_index", "image_token_id") or _get(
            vc, "image_token_id", "image_token_index"
        )
        out = proc(text="<placeholder>", images=img, return_tensors="pt")
        ids = out["input_ids"][0].tolist()
        if image_token_id is not None:
            n = sum(1 for t in ids if t == image_token_id)
            if n:
                return n, "counted via processor (dummy 896x896 image)"
        proc_err = "no image_token_id match in processor output"
    except Exception as exc:  # noqa: BLE001
        proc_err = str(exc)

    vc = vision_config(cfg)
    if vc is not None:
        isz = _get(vc, "image_size")
        psz = _get(vc, "patch_size")
        if isz and psz:
            est = (int(isz) // int(psz)) ** 2
            return (
                est,
                f"estimated (image_size/patch_size)^2 = ({isz}/{psz})^2 "
                f"(upper bound; ignores patch merging); processor: {proc_err}",
            )
    return None, f"could not determine ({proc_err})"


@dataclass
class EngineArgsSuggestion:
    max_model_len: int
    max_num_seqs: int
    max_num_batched_tokens: int
    prompt_tokens: int
    output_tokens: int
    num_images: int
    tokens_per_image: int
    image_tokens_total: int
    max_position_embeddings: Optional[int]
    multimodal: bool
    tokens_per_image_source: str

    def as_llm_kwargs(self) -> dict[str, int]:
        """Dict keys matching vllm.LLM(...) / test llm_args."""
        return {
            "max_model_len": self.max_model_len,
            "max_num_seqs": self.max_num_seqs,
            "max_num_batched_tokens": self.max_num_batched_tokens,
        }


def compute_engine_args(
    model_name: str,
    *,
    prompt_tokens: Optional[int] = None,
    prompt: Optional[str] = None,
    output_tokens: int = 32,
    num_images: int = 0,
    max_num_seqs: int = 1,
    cfg: Any = None,
) -> EngineArgsSuggestion:
    """Derive suggested engine args for a model + workload.

    Provide either ``prompt_tokens`` or ``prompt`` (tokenized automatically).
    Defaults to 64 text tokens if neither is given.
    """
    config_source = "caller"
    if cfg is None:
        cfg, config_source = load_model_config(model_name)

    if prompt is not None and prompt_tokens is not None:
        raise ValueError("pass only one of prompt_tokens or prompt")
    if prompt is not None:
        prompt_tokens = count_prompt_tokens(model_name, prompt)
    elif prompt_tokens is None:
        prompt_tokens = 64

    mm = is_multimodal(cfg)
    ceiling = max_position(cfg)
    # Stash for main() printing; harmless if unused by callers.
    setattr(cfg, "_config_source", config_source)

    tpi = 0
    tpi_src = "n/a (no images)"
    if mm and num_images > 0:
        tpi_val, tpi_src = tokens_per_image(model_name, cfg)
        if tpi_val is None:
            tpi = 0
            tpi_src = f"UNKNOWN — {tpi_src}; treating as 0 (set manually)"
        else:
            tpi = tpi_val

    image_tokens_total = tpi * num_images
    raw_len = prompt_tokens + image_tokens_total + output_tokens
    max_model_len = round_up(raw_len)
    if ceiling is not None:
        max_model_len = min(max_model_len, ceiling)

    # Prefill budget for one scheduler step; scale with concurrent sequences.
    max_num_batched_tokens = round_up(max_model_len * max_num_seqs)

    return EngineArgsSuggestion(
        max_model_len=max_model_len,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        prompt_tokens=prompt_tokens,
        output_tokens=output_tokens,
        num_images=num_images,
        tokens_per_image=tpi,
        image_tokens_total=image_tokens_total,
        max_position_embeddings=ceiling,
        multimodal=mm,
        tokens_per_image_source=tpi_src,
    )


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Suggest max_model_len / max_num_seqs / max_num_batched_tokens"
    )
    ap.add_argument("model_name")
    ap.add_argument(
        "--prompt-tokens",
        type=int,
        default=None,
        help="text tokens in the prompt (excludes image tokens)",
    )
    ap.add_argument(
        "--prompt",
        type=str,
        default=None,
        help="prompt text; tokenized to set prompt_tokens",
    )
    ap.add_argument(
        "--output-tokens",
        type=int,
        default=32,
        help="SamplingParams.max_tokens",
    )
    ap.add_argument(
        "--num-images",
        type=int,
        default=0,
        help="images per request (multimodal)",
    )
    ap.add_argument(
        "--max-num-seqs",
        type=int,
        default=1,
        help="concurrent sequences (batch size)",
    )
    ap.add_argument(
        "--json",
        action="store_true",
        help="print machine-readable JSON instead of the paste block",
    )
    args = ap.parse_args(argv)

    suggestion = compute_engine_args(
        args.model_name,
        prompt_tokens=args.prompt_tokens,
        prompt=args.prompt,
        output_tokens=args.output_tokens,
        num_images=args.num_images,
        max_num_seqs=args.max_num_seqs,
    )

    if args.json:
        print(json.dumps(asdict(suggestion), indent=2))
        return 0

    print(f"model:            {args.model_name}")
    print(f"multimodal:       {suggestion.multimodal}")
    print(f"max_position_emb: {suggestion.max_position_embeddings}  (hard ceiling)")
    print(
        f"workload:         same as test "
        f"(1 image + prompt text + max_tokens={args.output_tokens})"
    )
    print(f"prompt_tokens:    {suggestion.prompt_tokens}")
    print(f"output_tokens:    {suggestion.output_tokens}")
    print(f"num_images:       {suggestion.num_images}")
    if suggestion.num_images:
        print(
            f"tokens/image:     {suggestion.tokens_per_image}  "
            f"({suggestion.tokens_per_image_source})"
        )
        print(f"image_tokens:     {suggestion.image_tokens_total}")
    print(
        f"raw length:       {suggestion.prompt_tokens} + "
        f"{suggestion.image_tokens_total} + {suggestion.output_tokens} = "
        f"{suggestion.prompt_tokens + suggestion.image_tokens_total + suggestion.output_tokens}"
    )
    print("-" * 50)
    print("Suggested engine args (paste into llm_args):")
    for k, v in suggestion.as_llm_kwargs().items():
        print(f'    "{k}": {v},')
    print("-" * 50)
    print("Notes:")
    print("  * max_num_batched_tokens is a STARTING point. TTPlatform may override")
    print("    it for MLA / chunked prefill; raise it if image prefill OOMs.")
    print("  * Existing tests may use a lower max_model_len for DRAM and a higher")
    print("    max_num_batched_tokens for image prefill — tune after the first run.")
    print("  * gpu_memory_utilization is tuned separately (KV-cache budget).")
    return 0


if __name__ == "__main__":
    sys.exit(main())

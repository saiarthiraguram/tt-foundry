#!/usr/bin/env python3
"""Compute weight_fit.json from parameter count and per-arch DRAM budgets."""

from __future__ import annotations

import argparse
import json
import math
import sys

ARCH_DRAM_GIB = {"n150": 12, "p150": 32}
BYTES_PER_PARAM_FP32 = 4
BYTES_PER_PARAM_BF16 = 2
BUDGET_FACTOR = 0.85


def dram_bytes(gib: int) -> int:
    return int(gib * (1024**3))


def budget_bytes(gib: int) -> int:
    return math.floor(BUDGET_FACTOR * dram_bytes(gib))


def build_per_arch(num_params: int, archs: list[str]) -> tuple[dict, list[str]]:
    weight_fp32 = num_params * BYTES_PER_PARAM_FP32
    weight_bf16 = num_params * BYTES_PER_PARAM_BF16
    per_arch: dict = {}
    eligible: list[str] = []

    for arch in archs:
        gib = ARCH_DRAM_GIB.get(arch)
        if gib is None:
            continue
        bud = budget_bytes(gib)
        fits_fp32 = weight_fp32 <= bud
        fits_bf16 = weight_bf16 <= bud
        weight_predicted = not (fits_fp32 or fits_bf16)
        per_arch[arch] = {
            "dram_gib": gib,
            "dram_bytes": dram_bytes(gib),
            "budget_bytes": bud,
            "fits_fp32": fits_fp32,
            "fits_bf16": fits_bf16,
            "weight_predicted": weight_predicted,
        }
        eligible.append(arch)

    return per_arch, eligible, weight_fp32, weight_bf16


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--num-params", type=int, required=True)
    parser.add_argument("--hf-repo", default="")
    parser.add_argument(
        "--param-source",
        default="loader",
        choices=["loader", "config", "name_heuristic"],
    )
    parser.add_argument(
        "--component",
        default=None,
        help="If set, emit pipeline components[] entry with this name.",
    )
    parser.add_argument("--activation-class", default="unknown")
    parser.add_argument("--p150-only", action="store_true")
    parser.add_argument("--test-path", default="")
    parser.add_argument("--archs", default="n150,p150")
    parser.add_argument("--output", default="-", help="Output path or '-' for stdout.")
    args = parser.parse_args()

    archs = [a.strip() for a in args.archs.split(",") if a.strip()]
    per_arch, eligible, weight_fp32, weight_bf16 = build_per_arch(
        args.num_params, archs
    )

    if args.p150_only and "p150" in archs:
        eligible = ["p150"]

    base_fields = {
        "num_params": args.num_params,
        "weight_bytes_fp32": weight_fp32,
        "weight_bytes_bf16": weight_bf16,
        "activation_class": args.activation_class,
        "eligible_archs": eligible,
        "p150_only": args.p150_only,
        "per_arch": per_arch,
        "supported_archs": [],
    }
    if args.test_path:
        base_fields["test_path"] = args.test_path

    if args.component:
        comp = {"name": args.component, **base_fields}
        doc = {
            "model_key": args.model_key,
            "hf_repo": args.hf_repo,
            "param_estimate_source": args.param_source,
            "components": [comp],
        }
    else:
        doc = {
            "model_key": args.model_key,
            "hf_repo": args.hf_repo,
            "param_estimate_source": args.param_source,
            **base_fields,
        }

    text = json.dumps(doc, indent=2) + "\n"
    if args.output == "-":
        sys.stdout.write(text)
    else:
        with open(args.output, "w", encoding="utf-8") as fh:
            fh.write(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

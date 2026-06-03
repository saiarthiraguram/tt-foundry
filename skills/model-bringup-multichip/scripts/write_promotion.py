#!/usr/bin/env python3
"""Write promotion.json from bringup state and weight_fit.json."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


def short_sha(repo_path: str = ".") -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", repo_path, "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def normalize_arch_result(value: object, arch: str, dtype_ladder: dict) -> dict:
    if isinstance(value, dict):
        return value
    return {
        "result": str(value),
        "dtype_last": dtype_ladder.get(arch, "bf16"),
        "log": f"logs/iter_{arch}_run.log",
    }


def pick_component(weight_fit: dict, component: str | None) -> dict:
    if "components" in weight_fit:
        comps = weight_fit["components"]
        if component:
            for c in comps:
                if c.get("name") == component:
                    return c
        return comps[0] if comps else {}
    return weight_fit


def suggest_chip_count(weight_bytes_bf16: int) -> int:
    if weight_bytes_bf16 <= 0:
        return 4
    per_device_12g = 12 * (1024**3) * 0.85
    needed = weight_bytes_bf16 / per_device_12g
    if needed <= 2:
        return 2
    if needed <= 4:
        return 4
    if needed <= 8:
        return 8
    return 32


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bringup-dir", required=True)
    parser.add_argument("--model-key", required=True)
    parser.add_argument("--component", default=None)
    parser.add_argument("--tt-xla-root", default=".")
    args = parser.parse_args()

    bringup = Path(args.bringup_dir)
    state_path = bringup / "state.json"
    if not state_path.exists():
        print(f"error: missing {state_path}", file=sys.stderr)
        return 1

    state = json.loads(state_path.read_text(encoding="utf-8"))
    weight_fit_path = bringup / "weight_fit.json"
    weight_fit = (
        json.loads(weight_fit_path.read_text(encoding="utf-8"))
        if weight_fit_path.exists()
        else {}
    )

    arch_results_raw = state.get("arch_results") or {}
    dtype_ladder = state.get("details", {}).get("dtype_ladder") or {}
    eligible = state.get("arch_queue") or weight_fit.get("eligible_archs") or []

    promo_arch = {
        arch: normalize_arch_result(val, arch, dtype_ladder)
        for arch, val in arch_results_raw.items()
    }

    comp = pick_component(weight_fit, args.component)
    component_name = args.component or comp.get("name", "model")
    weight_bytes_bf16 = int(comp.get("weight_bytes_bf16") or 0)
    chip_count = suggest_chip_count(weight_bytes_bf16)

    promotion = {
        "model_key": args.model_key,
        "component": component_name,
        "reason": "weight_bound_all_eligible_arches",
        "eligible_archs_tried": list(eligible),
        "arch_results": promo_arch,
        "weight_bytes_bf16": weight_bytes_bf16,
        "suggested_multichip_arch": "n300-llmbox",
        "suggested_chip_count": chip_count,
        "tt_xla_sha": short_sha(args.tt_xla_root),
        "created_at": int(time.time()),
    }

    out_path = bringup / "promotion.json"
    out_path.write_text(json.dumps(promotion, indent=2) + "\n", encoding="utf-8")
    sys.stdout.write(json.dumps(promotion, indent=2) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

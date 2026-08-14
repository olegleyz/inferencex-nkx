#!/usr/bin/env python3
"""Reject target-specific dispatches outside reviewed NKX/Lepton recipe sets."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from resolve_cluster_target import load_targets


DEEPSEEK_IMAGE = "vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-4ba0a72"
QWEN35_FP8_IMAGE = "lmsysorg/sglang:nightly-dev-cu13-20260709-074bb928"
MINIMAX_M3_EAGLE3_IMAGE = (
    "vllm/vllm-openai:nightly-5e35a6f4f9bbc217c599692157ca985c894373f7"
)
MINIMAX_M3_STANDARD_IMAGE = (
    "vllm/vllm-openai:nightly-4080263bb2c5d10deac17aaeb88e0823bc35bca9"
)

MINIMAX_M3_STANDARD_CONTRACT = {
    "targets": {"lepton-gb300"},
    "expected": {
        "image": MINIMAX_M3_STANDARD_IMAGE,
        "model": "MiniMaxAI/MiniMax-M3-MXFP8",
        "model-prefix": "minimaxm3",
        "precision": "fp8",
        "framework": "dynamo-vllm",
        "runner": "gb300-nv",
        "isl": 8192,
        "osl": 1024,
        "spec-decoding": "none",
        "run-eval": False,
    },
    "recipe_prefix": "recipes/vllm/minimax-m3-gb300-fp8/8k1k/",
    "recipes": {
        "1p1d-dep2-dep8-8k1k.yaml",
        "1p1d-dep2-tep8-8k1k.yaml",
        "1p2d-dep2-tep8-8k1k.yaml",
        "2p1d-dep2-dep8-8k1k.yaml",
        "2p2d-dep2-tep8-8k1k.yaml",
        "2p4d-dep2-tep4-8k1k.yaml",
        "3p1d-dep2-dep16-8k1k.yaml",
        "3p1d-dep2-dep8-8k1k.yaml",
        "6p1d-dep2-dep8-8k1k.yaml",
    },
    "concurrencies": {
        "1p1d-dep2-dep8-8k1k.yaml": [256],
        "1p1d-dep2-tep8-8k1k.yaml": [128],
        "1p2d-dep2-tep8-8k1k.yaml": [32, 64, 128],
        "2p1d-dep2-dep8-8k1k.yaml": [512],
        "2p2d-dep2-tep8-8k1k.yaml": [16],
        "2p4d-dep2-tep4-8k1k.yaml": [4],
        "3p1d-dep2-dep16-8k1k.yaml": [512],
        "3p1d-dep2-dep8-8k1k.yaml": [1024],
        "6p1d-dep2-dep8-8k1k.yaml": [2048],
    },
}

MINIMAX_M3_EAGLE3_CONTRACT = {
    "expected": {
        "image": MINIMAX_M3_EAGLE3_IMAGE,
        "model": "MiniMaxAI/MiniMax-M3-MXFP8",
        "model-prefix": "minimaxm3",
        "precision": "fp8",
        "framework": "dynamo-vllm",
        "runner": "gb300-nv",
        "isl": 8192,
        "osl": 1024,
        "spec-decoding": "mtp",
        "run-eval": False,
    },
    "recipe_prefix": "recipes/vllm/minimax-m3-gb300-fp8/8k1k/mtp/",
    "recipes": {
        "1p1d-dep2-dep8-eagle3-c64-8k1k.yaml",
        "1p1d-dep2-tp4-eagle3-c1-8k1k.yaml",
        "1p1d-dep2-tp4-eagle3-c8-8k1k.yaml",
        "1p1d-dep2-tp8-eagle3-c1-8k1k.yaml",
        "1p1d-dep2-tp8-eagle3-c4-8k1k.yaml",
        "1p1d-dep2-tp8-eagle3-c8-8k1k.yaml",
        "2p1d-dep2-dep8-eagle3-c512-8k1k.yaml",
        "3p1d-dep2-dep8-eagle3-c256-8k1k.yaml",
        "4p1d-dep2-dep8-eagle3-c1024-8k1k.yaml",
        "4p1d-dep2-dep8-eagle3-c2048-8k1k.yaml",
        "6p1d-dep2-dep8-eagle3-c2048-8k1k.yaml",
    },
    "concurrencies": {
        "1p1d-dep2-dep8-eagle3-c64-8k1k.yaml": [64],
        "1p1d-dep2-tp4-eagle3-c1-8k1k.yaml": [1],
        "1p1d-dep2-tp4-eagle3-c8-8k1k.yaml": [8],
        "1p1d-dep2-tp8-eagle3-c1-8k1k.yaml": [1],
        "1p1d-dep2-tp8-eagle3-c4-8k1k.yaml": [4],
        "1p1d-dep2-tp8-eagle3-c8-8k1k.yaml": [8],
        "2p1d-dep2-dep8-eagle3-c512-8k1k.yaml": [512],
        "3p1d-dep2-dep8-eagle3-c256-8k1k.yaml": [256],
        "4p1d-dep2-dep8-eagle3-c1024-8k1k.yaml": [1024],
        "4p1d-dep2-dep8-eagle3-c2048-8k1k.yaml": [2048],
        "6p1d-dep2-dep8-eagle3-c2048-8k1k.yaml": [2048],
    },
}

CONTRACTS = {
    "deepseek-ai/DeepSeek-V4-Pro": {
        "expected": {
            "image": DEEPSEEK_IMAGE,
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "model-prefix": "dsv4",
            "precision": "fp4",
            "framework": "dynamo-vllm",
            "runner": "gb300-nv",
            "isl": 8192,
            "osl": 1024,
            "run-eval": False,
        },
        "recipe_prefix": "recipes/vllm/deepseek-v4/8k1k/",
        "recipes": {
            "disagg-gb300-1p6d-dep4-tp4.yaml",
            "disagg-gb300-1p9d-tep4-tp4.yaml",
            "disagg-gb300-4p1d-dep4-dep8-24-c4096.yaml",
            "disagg-gb300-5p1d-dep4-dep8-28-c4096.yaml",
            "disagg-gb300-6p1d-dep4-dep8-32-c4096.yaml",
            "disagg-gb300-7p2d-dep4-dep16.yaml",
        },
    },
    "Qwen/Qwen3.5-397B-A17B-FP8": {
        "expected": {
            "image": QWEN35_FP8_IMAGE,
            "model": "Qwen/Qwen3.5-397B-A17B-FP8",
            "model-prefix": "qwen3.5",
            "precision": "fp8",
            "framework": "dynamo-sglang",
            # This is the upstream logical runner. The isolated target maps it
            # to the private gb300-nv self-hosted runner after validation.
            "runner": "gb300",
            "isl": 8192,
            "osl": 1024,
            "run-eval": True,
        },
        "recipe_prefix": "recipes/sglang/qwen3.5/gb300-fp8/8k1k/",
        "recipes": {
            "1p1d-tp4-tp4.yaml",
            "4p1d-dep4-dep16.yaml",
            "8p1d-dep4-dep16.yaml",
        },
    },
    "MiniMaxAI/MiniMax-M3-MXFP8": MINIMAX_M3_EAGLE3_CONTRACT,
}


def _recipe(row: dict[str, Any]) -> str:
    settings = row.get("prefill", {}).get("additional-settings", [])
    matches = [
        value.removeprefix("CONFIG_FILE=").split(":", 1)[0]
        for value in settings
        if isinstance(value, str) and value.startswith("CONFIG_FILE=")
    ]
    if len(matches) != 1:
        raise RuntimeError("every target entry must select exactly one recipe")
    return matches[0]


def validate(target: str, matrix: Any, *, readiness_only: bool = False) -> int:
    """Validate a generated matrix and return its number of entries."""
    if target not in load_targets():
        raise RuntimeError(f"unsupported target cluster: {target}")
    if not isinstance(matrix, list) or not matrix:
        raise RuntimeError("target-specific matrix must contain at least one entry")
    models = {row.get("model") for row in matrix if isinstance(row, dict)}
    model = next(iter(models), None)
    if len(models) != 1 or not isinstance(model, str) or model not in CONTRACTS:
        rendered = sorted(repr(value) for value in models)
        raise RuntimeError(f"unreviewed target matrix model set: {rendered!r}")
    contract = CONTRACTS[model]
    if model == "MiniMaxAI/MiniMax-M3-MXFP8":
        spec_modes = {row.get("spec-decoding") for row in matrix}
        if spec_modes == {"none"}:
            contract = MINIMAX_M3_STANDARD_CONTRACT
        elif spec_modes == {"mtp"}:
            contract = MINIMAX_M3_EAGLE3_CONTRACT
        else:
            raise RuntimeError(f"unreviewed MiniMax spec-decoding set: {spec_modes!r}")
    is_standard_minimax = contract is MINIMAX_M3_STANDARD_CONTRACT
    allowed_targets = contract.get("targets")
    if allowed_targets is not None and target not in allowed_targets:
        raise RuntimeError(f"reviewed contract is not enabled for target {target}")
    recipes = contract["recipes"]
    if readiness_only:
        readiness_recipes = {
            "Qwen/Qwen3.5-397B-A17B-FP8": "1p1d-tp4-tp4.yaml",
        }
        if model == "MiniMaxAI/MiniMax-M3-MXFP8":
            readiness_recipes[model] = (
                "1p1d-dep2-tep8-8k1k.yaml"
                if is_standard_minimax
                else "1p1d-dep2-tp4-eagle3-c1-8k1k.yaml"
            )
        if model not in readiness_recipes or len(matrix) != 1:
            raise RuntimeError("readiness target must be one reviewed probe")
        expected = dict(contract["expected"])
        if model == "Qwen/Qwen3.5-397B-A17B-FP8":
            expected["run-eval"] = False
        contract = {
            **contract,
            "expected": expected,
            "recipes": {readiness_recipes[model]},
        }
        recipes = contract["recipes"]
    if len(matrix) > len(recipes):
        raise RuntimeError("target-specific matrix exceeds the reviewed recipe set")

    seen: set[str] = set()
    for row in matrix:
        if not isinstance(row, dict):
            raise RuntimeError("matrix entries must be objects")
        expected = contract["expected"]
        mismatches = {
            key: (row.get(key), value)
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches or "prefill" not in row:
            raise RuntimeError(f"unreviewed target matrix entry: {mismatches}")
        recipe = _recipe(row)
        if not recipe.startswith(contract["recipe_prefix"]):
            raise RuntimeError(f"recipe is outside the reviewed directory: {recipe}")
        name = recipe.rsplit("/", 1)[-1]
        if name not in recipes or name in seen:
            raise RuntimeError(f"unreviewed or duplicate target recipe: {name}")
        expected_concurrencies = contract.get("concurrencies", {}).get(name)
        if not readiness_only and expected_concurrencies is not None:
            if row.get("conc") != expected_concurrencies:
                raise RuntimeError(
                    f"unreviewed concurrency set for {name}: {row.get('conc')!r}"
                )
        if readiness_only:
            allowed = [128] if is_standard_minimax else [1]
            if row.get("conc") != allowed:
                raise RuntimeError(
                    f"readiness target must use input concurrency {allowed!r}"
                )
        seen.add(name)
    return len(matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--readiness-only", action="store_true")
    args = parser.parse_args()
    count = validate(
        args.target, json.load(sys.stdin), readiness_only=args.readiness_only
    )
    print(f"validated target={args.target} entries={count}")


if __name__ == "__main__":
    main()

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
    recipes = contract["recipes"]
    if readiness_only:
        if model != "Qwen/Qwen3.5-397B-A17B-FP8" or len(matrix) != 1:
            raise RuntimeError("readiness target must be the single Qwen3.5 probe")
        contract = {
            **contract,
            "expected": {**contract["expected"], "run-eval": False},
            "recipes": {"1p1d-tp4-tp4.yaml"},
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
        if readiness_only and row.get("conc") != [1]:
            raise RuntimeError("readiness target must use concurrency [1]")
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

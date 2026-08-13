#!/usr/bin/env python3
"""Reject target-specific dispatches outside the reviewed DSv4 recipe set."""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from resolve_cluster_target import load_targets


IMAGE = "vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-4ba0a72"
RECIPES = {
    "disagg-gb300-1p6d-dep4-tp4.yaml",
    "disagg-gb300-1p9d-tep4-tp4.yaml",
    "disagg-gb300-4p1d-dep4-dep8-24-c4096.yaml",
    "disagg-gb300-5p1d-dep4-dep8-28-c4096.yaml",
    "disagg-gb300-6p1d-dep4-dep8-32-c4096.yaml",
    "disagg-gb300-7p2d-dep4-dep16.yaml",
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


def validate(target: str, matrix: Any) -> int:
    """Validate a generated matrix and return its number of entries."""
    if target not in load_targets():
        raise RuntimeError(f"unsupported target cluster: {target}")
    if not isinstance(matrix, list) or not matrix:
        raise RuntimeError("target-specific matrix must contain at least one entry")
    if len(matrix) > len(RECIPES):
        raise RuntimeError("target-specific matrix exceeds the reviewed recipe set")

    seen: set[str] = set()
    for row in matrix:
        if not isinstance(row, dict):
            raise RuntimeError("matrix entries must be objects")
        expected = {
            "image": IMAGE,
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "model-prefix": "dsv4",
            "precision": "fp4",
            "framework": "dynamo-vllm",
            "runner": "gb300-nv",
            "isl": 8192,
            "osl": 1024,
        }
        mismatches = {
            key: (row.get(key), value)
            for key, value in expected.items()
            if row.get(key) != value
        }
        if mismatches or "prefill" not in row or row.get("run-eval") is True:
            raise RuntimeError(f"unreviewed target matrix entry: {mismatches}")
        recipe = _recipe(row)
        if not recipe.startswith("recipes/vllm/deepseek-v4/8k1k/"):
            raise RuntimeError(f"recipe is outside the reviewed directory: {recipe}")
        name = recipe.rsplit("/", 1)[-1]
        if name not in RECIPES or name in seen:
            raise RuntimeError(f"unreviewed or duplicate target recipe: {name}")
        seen.add(name)
    return len(matrix)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    args = parser.parse_args()
    count = validate(args.target, json.load(sys.stdin))
    print(f"validated target={args.target} entries={count}")


if __name__ == "__main__":
    main()

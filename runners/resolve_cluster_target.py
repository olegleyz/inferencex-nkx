#!/usr/bin/env python3
"""Resolve a reviewed benchmark target to its fixed runner and profile."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


TARGETS_PATH = Path(__file__).with_name("cluster_profiles") / "targets.json"


def load_targets(path: Path = TARGETS_PATH) -> dict[str, dict[str, Any]]:
    """Load the source-controlled target map."""
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError("cluster target map must be an object")
    return value


def resolve(
    target: str,
    legacy_profile: str = "",
    targets: dict[str, dict[str, Any]] | None = None,
) -> dict[str, str]:
    """Resolve target/profile inputs, preserving the empty upstream path."""
    targets = targets or load_targets()
    if not target:
        if not legacy_profile:
            return {
                "target-cluster": "",
                "runner-label": "",
                "cluster-profile": "",
                "artifact-suffix": "",
                "model-stage-result-file": "",
            }
        matches = [
            name
            for name, item in targets.items()
            if item.get("cluster_profile") == legacy_profile
        ]
        if len(matches) != 1:
            raise RuntimeError(f"unsupported legacy cluster profile: {legacy_profile}")
        target = matches[0]

    if target not in targets:
        raise RuntimeError(f"unsupported target cluster: {target}")
    item = targets[target]
    profile = str(item["cluster_profile"])
    if legacy_profile and legacy_profile != profile:
        raise RuntimeError(
            f"target/profile mismatch: target={target} profile={legacy_profile}"
        )
    return {
        "target-cluster": target,
        "runner-label": str(item["runner_label"]),
        "cluster-profile": profile,
        "artifact-suffix": target,
        "model-stage-result-file": str(item["model_stage_receipt"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", default="")
    parser.add_argument("--legacy-profile", default="")
    parser.add_argument("--github-output", type=Path)
    args = parser.parse_args()
    result = resolve(args.target, args.legacy_profile)
    if args.github_output:
        with args.github_output.open("a", encoding="utf-8") as stream:
            for key, value in result.items():
                if "\n" in value:
                    raise RuntimeError(f"unsafe newline in resolved {key}")
                stream.write(f"{key}={value}\n")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()

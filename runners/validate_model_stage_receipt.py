#!/usr/bin/env python3
"""Validate a committed ModelStage receipt against a fixed cluster target."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

from resolve_cluster_target import load_targets


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"expected a JSON object: {path}")
    return value


def _manifest_digest(manifest: dict[str, Any]) -> str:
    body = dict(manifest)
    body.pop("manifestDigest", None)
    payload = json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    return f"sha256:{hashlib.sha256(payload).hexdigest()}"


def validate(
    target: str,
    model_repository: str,
    receipt: dict[str, Any],
    manifest: dict[str, Any],
) -> str:
    """Validate receipt identity, completeness, paths, and shared manifest."""
    targets = load_targets()
    if target not in targets:
        raise RuntimeError(f"unsupported target cluster: {target}")
    target_config = targets[target]
    models = target_config.get("models")
    if models is None:
        expected = target_config
    elif not isinstance(models, dict) or model_repository not in models:
        raise RuntimeError(
            f"model {model_repository!r} is not staged for target {target}"
        )
    else:
        expected = {**target_config, **models[model_repository]}
    results = receipt.get("nodes", {}).get("results", [])
    nodes = [item.get("node") for item in results if isinstance(item, dict)]
    required = receipt.get("nodes", {}).get("required")
    verified = receipt.get("nodes", {}).get("verified")
    if not (
        receipt.get("state") == "ready"
        and receipt.get("clusterId") == expected["staging_cluster_id"]
        and receipt.get("model", {}).get("repository") == expected["model_repository"]
        and receipt.get("model", {}).get("revision") == expected["model_revision"]
        and receipt.get("verification", {}).get("mode") == "full-sha256"
        and receipt.get("verification", {}).get("requireAllTargets") is True
        and isinstance(required, int)
        and required == verified == len(results) == len(nodes) == len(set(nodes))
        and all(
            isinstance(item, dict) and item.get("state") in {"copied", "reused"}
            for item in results
        )
        and receipt.get("manifestDigest") == expected["manifest_digest"]
        and receipt.get("sharedPath") == expected["shared_path"]
        and receipt.get("localPath") == expected["local_path"]
    ):
        raise RuntimeError(f"model-stage receipt does not match target {target}")

    exact = expected.get("expected_nodes_exact")
    minimum = expected.get("expected_nodes_minimum")
    if exact is not None and required != exact:
        raise RuntimeError(f"target {target} requires exactly {exact} staged nodes")
    if minimum is not None and required < minimum:
        raise RuntimeError(f"target {target} requires at least {minimum} staged nodes")

    digest = expected["manifest_digest"]
    if not (
        manifest.get("manifestDigest") == digest == _manifest_digest(manifest)
        and manifest.get("repository") == expected["model_repository"]
        and manifest.get("revision") == expected["model_revision"]
    ):
        raise RuntimeError("shared model manifest identity or digest mismatch")
    return str(expected["local_path"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", required=True)
    parser.add_argument("--model-repository", required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--shared-manifest-output", type=Path, required=True)
    parser.add_argument("--github-env", type=Path)
    args = parser.parse_args()
    receipt = _read_object(args.receipt)
    shared_manifest_path = Path(receipt["sharedPath"]) / ".benchops-model-manifest.json"
    manifest = _read_object(shared_manifest_path)
    model_path = validate(args.target, args.model_repository, receipt, manifest)
    args.shared_manifest_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if args.github_env:
        values = {
            "MODEL_PATH_OVERRIDE": model_path,
            "MODEL_STAGE_MANIFEST_DIGEST": str(receipt["manifestDigest"]),
        }
        with args.github_env.open("a", encoding="utf-8") as stream:
            for key, value in values.items():
                if "\n" in value:
                    raise RuntimeError(f"unsafe newline in {key}")
                stream.write(f"{key}={value}\n")
    print(
        f"validated target={args.target} nodes={receipt['nodes']['verified']} "
        f"path={model_path} digest={receipt['manifestDigest']}"
    )


if __name__ == "__main__":
    main()

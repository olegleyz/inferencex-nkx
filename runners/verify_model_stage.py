#!/usr/bin/env python3
"""Read-only BenchOps ModelStage verification on one Slurm worker."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
from typing import Any


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


def verify(receipt_path: Path) -> str:
    """Validate receipt identity, marker, manifest, and every local file size."""
    receipt = _read_object(receipt_path)
    node = os.environ.get("SLURMD_NODENAME") or os.uname().nodename.split(".")[0]
    expected_nodes = {
        item.get("node")
        for item in receipt.get("nodes", {}).get("results", [])
        if isinstance(item, dict)
    }
    if node not in expected_nodes:
        raise RuntimeError(f"Slurm worker {node!r} is absent from the staging receipt")

    local_path = Path(receipt["localPath"])
    shared_path = Path(receipt["sharedPath"])
    expected_digest = receipt["manifestDigest"]
    repository = receipt["model"]["repository"]
    revision = receipt["model"]["revision"]

    manifest = _read_object(shared_path / ".benchops-model-manifest.json")
    marker = _read_object(local_path / ".benchops-model-complete.json")
    claimed_digest = manifest.get("manifestDigest")
    if not (
        claimed_digest == expected_digest == _manifest_digest(manifest)
        and manifest.get("repository") == repository
        and manifest.get("revision") == revision
        and marker.get("verified") is True
        and marker.get("repository") == repository
        and marker.get("revision") == revision
        and marker.get("manifestDigest") == expected_digest
    ):
        raise RuntimeError(f"model identity or manifest mismatch on {node}")

    entries = manifest.get("files")
    if not isinstance(entries, list):
        raise RuntimeError("manifest files must be a list")
    paths = [item.get("path") for item in entries if isinstance(item, dict)]
    safe_paths = (
        len(paths) == len(entries)
        and len(paths) == len(set(paths))
        and all(
            isinstance(relative, str)
            and relative
            and not PurePosixPath(relative).is_absolute()
            and ".." not in PurePosixPath(relative).parts
            for relative in paths
        )
    )
    total_bytes = sum(
        item.get("size", -1) for item in entries if isinstance(item, dict)
    )
    if not (
        safe_paths
        and manifest.get("totalFileCount") == len(entries)
        and manifest.get("totalBytes") == total_bytes
    ):
        raise RuntimeError("unsafe or internally inconsistent model manifest")

    for item in entries:
        candidate = local_path / item["path"]
        if not candidate.is_file() or candidate.stat().st_size != item["size"]:
            raise RuntimeError(f"missing or size-mismatched model file: {candidate}")

    return (
        f"verified node={node} path={local_path} files={len(entries)} "
        f"bytes={total_bytes} digest={expected_digest}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--receipt", type=Path, required=True)
    args = parser.parse_args()
    print(verify(args.receipt))


if __name__ == "__main__":
    main()

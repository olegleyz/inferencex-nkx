#!/usr/bin/env python3
"""Reject srt-slurm runs whose worker processes failed during startup."""

from __future__ import annotations

import argparse
import re
from pathlib import Path


FATAL_PATTERNS = (
    re.compile(r"EngineCore failed to start\."),
    re.compile(r"Worker failed with error"),
    re.compile(r"CUDA_ERROR_[A-Z_]+"),
    re.compile(r"(?:torch\.)?OutOfMemoryError"),
    re.compile(r"CUDA out of memory", re.IGNORECASE),
    re.compile(r"Segmentation fault", re.IGNORECASE),
)


def find_startup_failures(logs_dir: Path) -> dict[Path, list[str]]:
    """Return matching fatal lines keyed by worker log path."""
    failures: dict[Path, list[str]] = {}
    for path in sorted(logs_dir.glob("*_w*.out")):
        matches: list[str] = []
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            if any(pattern.search(line) for pattern in FATAL_PATTERNS):
                matches.append(line)
                if len(matches) == 5:
                    break
        if matches:
            failures[path] = matches
    return failures


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("logs_dir", type=Path)
    args = parser.parse_args()
    failures = find_startup_failures(args.logs_dir)
    if not failures:
        print(f"Worker startup validation passed: {args.logs_dir}")
        return
    print("Worker startup validation failed:")
    for path, lines in failures.items():
        print(f"  {path.name}")
        for line in lines:
            print(f"    {line}")
    raise SystemExit(1)


if __name__ == "__main__":
    main()

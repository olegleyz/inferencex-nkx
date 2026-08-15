#!/usr/bin/env bash

# Gracefully stop every GitHub Actions listener in the Lepton GB300 pool.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_POOL_ROOT="${ACTIONS_RUNNER_POOL_ROOT:-/scratch/fsw/users/oleizerov/.github-actions}"

runner_roots=(
    "$RUNNER_POOL_ROOT/gb300l-nv"
    "$RUNNER_POOL_ROOT/gb300l-nv_01"
    "$RUNNER_POOL_ROOT/gb300l-nv_02"
)

for runner_root in "${runner_roots[@]}"; do
    ACTIONS_RUNNER_ROOT="$runner_root" \
        "$SCRIPT_DIR/stop_github_actions_runner.sh"
done

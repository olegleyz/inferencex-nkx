#!/usr/bin/env bash

# Start every configured GitHub Actions listener in the Lepton GB300 pool.
# Runner registration remains a separate, explicit operation; this script only
# starts persistent runner roots that already contain a valid .runner file.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER_POOL_ROOT="${ACTIONS_RUNNER_POOL_ROOT:-/scratch/fsw/users/oleizerov/.github-actions}"

runner_roots=(
    "$RUNNER_POOL_ROOT/gb300l-nv"
    "$RUNNER_POOL_ROOT/gb300l-nv_01"
    "$RUNNER_POOL_ROOT/gb300l-nv_02"
)

for runner_root in "${runner_roots[@]}"; do
    if [[ ! -f "$runner_root/.runner" ]]; then
        echo "Configured runner root is not registered: $runner_root" >&2
        exit 1
    fi

    ACTIONS_RUNNER_ROOT="$runner_root" \
        "$SCRIPT_DIR/start_github_actions_runner.sh"
done

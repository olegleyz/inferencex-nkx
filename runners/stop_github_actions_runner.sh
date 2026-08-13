#!/usr/bin/env bash

# Gracefully stop a persistent GitHub Actions listener before its host login
# pod exits, allowing GitHub to release the runner session before restart.

set -euo pipefail

RUNNER_ROOT="${ACTIONS_RUNNER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PID_FILE="$RUNNER_ROOT/runner.pid"

if [[ ! -f "$PID_FILE" ]]; then
    exit 0
fi

pid="$(<"$PID_FILE")"
if ! kill -0 "$pid" 2>/dev/null; then
    exit 0
fi

kill -INT -- "-$pid" 2>/dev/null || kill -INT "$pid"
for _ in {1..20}; do
    if ! kill -0 -- "-$pid" 2>/dev/null && ! kill -0 "$pid" 2>/dev/null; then
        exit 0
    fi
    sleep 1
done

# Pod termination remains authoritative after the graceful wait expires.
exit 0

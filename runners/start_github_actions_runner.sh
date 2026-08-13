#!/usr/bin/env bash

# Start an already configured GitHub Actions runner from persistent storage.
# The Lepton LoginSet invokes this as the non-root benchmark user after each
# login-pod restart. Registration remains a separate, explicit operation.

set -euo pipefail

RUNNER_ROOT="${ACTIONS_RUNNER_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
cd "$RUNNER_ROOT"

test -x ./run.sh
test -f .runner

if [[ -f runner.pid ]] && kill -0 "$(<runner.pid)" 2>/dev/null; then
    exit 0
fi

nohup setsid ./run.sh > runner.log 2>&1 </dev/null &
echo "$!" > runner.pid

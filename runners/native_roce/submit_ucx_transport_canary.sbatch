#!/usr/bin/env bash
#SBATCH --account=nvidia
#SBATCH --partition=batch
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --exclusive
#SBATCH --time=00:15:00
#SBATCH --job-name=inferencex-minimax-ucx-canary
#SBATCH --output=/scratch/fsw/users/oleizerov/.inferencex/runtime-canaries/minimax-ucx-submit-%j.log

set -euo pipefail

: "${INFERENCEX_UCX_OVERLAY:?INFERENCEX_UCX_OVERLAY is required}"
: "${INFERENCEX_UCX_CANARY_IMAGE:?INFERENCEX_UCX_CANARY_IMAGE is required}"
export INFERENCEX_UCX_CANARY_LOG_DIR="${INFERENCEX_UCX_CANARY_LOG_DIR:-/scratch/fsw/users/oleizerov/.inferencex/runtime-canaries/minimax-ucx-1.22.0-${SLURM_JOB_ID}}"

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run_ucx_transport_canary.sh"

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

export INFERENCEX_UCX_OVERLAY="${INFERENCEX_UCX_OVERLAY:-/scratch/fsw/users/oleizerov/.inferencex/runtime-overlays/ucx-1.22.0-minimax-nixl-1.3.1}"
export INFERENCEX_UCX_CANARY_IMAGE="${INFERENCEX_UCX_CANARY_IMAGE:-/scratch/fsw/users/oleizerov/.benchops/images/vllm_vllm-openai_nightly-4080263bb2c5d10deac17aaeb88e0823bc35bca9.sqsh}"
export INFERENCEX_UCX_CANARY_LOG_DIR="${INFERENCEX_UCX_CANARY_LOG_DIR:-/scratch/fsw/users/oleizerov/.inferencex/runtime-canaries/minimax-ucx-1.22.0-${SLURM_JOB_ID}}"
export SLURM_EXPORT_ENV=ALL

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${script_dir}/run_ucx_transport_canary.sh"

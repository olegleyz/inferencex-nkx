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
export INFERENCEX_UCX_CANARY_IMAGE="${INFERENCEX_UCX_CANARY_IMAGE:-/scratch/fsw/users/oleizerov/.benchops/images/vllm_vllm-openai_nightly-5e35a6f4f9bbc217c599692157ca985c894373f7.sqsh}"
export INFERENCEX_UCX_CANARY_LOG_DIR="${INFERENCEX_UCX_CANARY_LOG_DIR:-/scratch/fsw/users/oleizerov/.inferencex/runtime-canaries/minimax-ucx-1.22.0-${SLURM_JOB_ID}}"
export SLURM_EXPORT_ENV=ALL

readonly canary_script="/scratch/fsw/users/oleizerov/.inferencex/sources/0460956b1/runners/native_roce/run_ucx_transport_canary.sh"
readonly canary_script_sha256="d228bb76f0a33f92ba60887a3220fb218a9adac73607274bd8a0cbd28a703f59"
[[ "$(sha256sum "$canary_script" | awk '{print $1}')" == "$canary_script_sha256" ]] || {
    echo "native-RoCE canary script checksum mismatch: $canary_script" >&2
    exit 1
}
exec bash "$canary_script"

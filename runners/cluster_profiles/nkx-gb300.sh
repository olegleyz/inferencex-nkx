#!/usr/bin/env bash

# NKX-managed GB300 Slurm cluster inputs. This profile changes scheduler,
# storage, architecture/runtime-cache, and pinned artifact identity only; the
# reviewed InferenceX DeepSeek recipe remains the source of model topology and
# engine arguments.

export SLURM_PARTITION="gpu"
export SLURM_ACCOUNT="nvidia"
export INFERENCEX_EXPECTED_SLURM_CLUSTER="nkx-slinky-gb300-dev-01"
export INFERENCEX_REQUIRE_PARTITION_RECEIPT_COVERAGE="1"

export AIPERF_MMAP_CACHE_HOST_PATH="/scratch/fsw/users/oleizerov/.benchops/datasets"
export HF_HUB_CACHE_HOST_PATH="/scratch/fsw/users/oleizerov/.benchops/cache/hf-hub"
export DYNAMO_WHEELS_CACHE_HOST_PATH="/scratch/fsw/users/oleizerov/.benchops/cache/dynamo-wheels"
export SQUASH_CACHE_HOST_PATH="/scratch/fsw/users/oleizerov/.benchops/images"

export SRT_SLURM_HOME_PATH="/home/oleizerov"
export SRT_SLURM_SHARED_ROOT="/scratch/fsw"
export SRT_SLURM_CPUS_PER_TASK="140"

# Exact srt-slurm revision and image artifact used by BenchOps job 2024.
export SRT_SLURM_DSV4_REF="758becd9d18dcab1fb722abc1875d73ee81a20cb"
export IMAGE_SQUASH_SHA256="c8dc9884c5c863170f2207840e44a18c92af292c9829ad48c3d89b3cd87ddaff"
export NGINX_SQUASH_SHA256="61e003876ea0b3b78c5745e261056544fe4e997e7c79b73d62c31d7d28483a1a"

# Effective job-2024 control-plane settings. kv_both and all engine arguments
# remain in the checked-in reviewed recipe and are not rewritten here.
export SRT_SLURM_ETCD_LEASE_TTL="600"
export SRT_SLURM_HEALTH_MAX_ATTEMPTS="720"

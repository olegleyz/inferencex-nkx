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
export NGINX_SQUASH_SHA256="61e003876ea0b3b78c5745e261056544fe4e997e7c79b73d62c31d7d28483a1a"

# Runtime identities and compatibility settings are launch-contract inputs,
# not properties shared by every model on the cluster. Keep them scoped to the
# exact reviewed model/framework contract so one reproduction cannot silently
# rewrite another model's recipe.
case "${MODEL_PREFIX:-}-${PRECISION:-}-${FRAMEWORK:-}" in
    dsv4-fp4-dynamo-vllm)
        # Exact artifacts and effective settings from BenchOps job 2024.
        export SRT_SLURM_DSV4_REF="758becd9d18dcab1fb722abc1875d73ee81a20cb"
        export IMAGE_SQUASH_SHA256="c8dc9884c5c863170f2207840e44a18c92af292c9829ad48c3d89b3cd87ddaff"
        export SRT_SLURM_CPUS_PER_TASK="140"
        export SRT_SLURM_ETCD_LEASE_TTL="600"
        export SRT_SLURM_HEALTH_MAX_ATTEMPTS="720"
        ;;
    qwen3.5-fp8-dynamo-sglang)
        # Exact srt-slurm commit and effective recipe adaptations observed in
        # the successful NKX BenchOps Qwen sweep (Slurm jobs 1421-1554).
        export SRT_SLURM_QWEN35_FP8_REF="3435776cd6db4c14f8b771ff7a3976deb62fe133"
        export IMAGE_SQUASH_SHA256="2ebe26d3323b61dcb8612a8c9b20f24aec1e57b3fcb61bf50b72c1e715851af0"
        export SRT_SLURM_CPUS_PER_TASK="16"
        export SRT_SLURM_ETCD_LEASE_TTL="120"
        export SRT_SLURM_HEALTH_MAX_ATTEMPTS="720"
        ;;
    minimaxm3-fp8-dynamo-vllm)
        # The short-lived Docker Hub nightly tag used by the successful
        # upstream sweep was deleted by vLLM's 14-build retention policy.
        # vLLM first published the identical aarch64 image to this immutable
        # commit-qualified Public ECR manifest before copying it to Docker Hub.
        export IMAGE_IMPORT_REGISTRY="public.ecr.aws"
        export IMAGE_IMPORT_REPOSITORY="q9t5s3a7/vllm-release-repo"
        export IMAGE_IMPORT_TAG="5e35a6f4f9bbc217c599692157ca985c894373f7-aarch64"
        export IMAGE_IMPORT_REFERENCE="public.ecr.aws#q9t5s3a7/vllm-release-repo:5e35a6f4f9bbc217c599692157ca985c894373f7-aarch64"
        export IMAGE_IMPORT_MANIFEST_SHA256="sha256:41442db2591d6bfb8dc219561f18deed55aaf5b95f910e5d9145186043d8eb94"
        export SRT_SLURM_MINIMAX_M3_REF="c180328b98c3793ca84a1e24a030f90545eb7d5d"
        ;;
esac

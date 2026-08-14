#!/usr/bin/env bash

# Lepton-managed GB300 Slurm inputs. This target is intentionally separate
# from nkx-gb300: it has a distinct Slurm controller, runner label, partition,
# node-local storage root, and ConnectX device mapping.

export SLURM_PARTITION="batch"
export SLURM_ACCOUNT="nvidia"
export INFERENCEX_EXPECTED_SLURM_CLUSTER="nkx-slinky-dev-02"
export INFERENCEX_REQUIRE_PARTITION_RECEIPT_COVERAGE="1"
export INFERENCEX_REJECT_WORKER_STARTUP_FAILURES="1"

export AIPERF_MMAP_CACHE_HOST_PATH="/scratch/fsw/users/oleizerov/.benchops/datasets"
export HF_HUB_CACHE_HOST_PATH="/scratch/fsw/users/oleizerov/.benchops/cache/hf-hub"
export DYNAMO_WHEELS_CACHE_HOST_PATH="/scratch/fsw/users/oleizerov/.benchops/cache/dynamo-wheels"
export SQUASH_CACHE_HOST_PATH="/scratch/fsw/users/oleizerov/.benchops/images"

export SRT_SLURM_HOME_PATH="/scratch/fsw/users/oleizerov"
export SRT_SLURM_SHARED_ROOT="/scratch/fsw"
export SRT_SLURM_TIME_LIMIT="04:00:00"

export NGINX_SQUASH_SHA256="55ca6f0a6833f1f40bac832551c0e3e4be9fdfb17b391e29a6f81e04d5ac9372"

# Runtime identities and compatibility settings are part of each reviewed
# model/framework contract. Do not leak DeepSeek's vLLM fabric overlay into
# SGLang or future model recipes on the same cluster.
case "${MODEL_PREFIX:-}-${PRECISION:-}-${FRAMEWORK:-}" in
    dsv4-fp4-dynamo-vllm)
        # Effective settings in completed Lepton BenchOps jobs 588, 864, 918,
        # 924, 927, and 930. The HCA names are Lepton cluster inputs.
        export SRT_SLURM_DSV4_REF="758becd9d18dcab1fb722abc1875d73ee81a20cb"
        export IMAGE_SQUASH_SHA256="57c98992ead2f428d16ba06f8e5ba0b620db6b3b7adce81ba094db198426772d"
        export SRT_SLURM_CPUS_PER_TASK="140"
        export SRT_SLURM_ETCD_LEASE_TTL="600"
        export SRT_SLURM_HEALTH_MAX_ATTEMPTS="720"
        export SRT_SLURM_RUNTIME_ENV_JSON='{"NVSHMEM_ENABLE_NIC_PE_MAPPING":"1","NVSHMEM_HCA_LIST":"rocep161s0:1,rocep162s0:1,rocep172s0:1,rocep173s0:1,rocep190s0:1,rocep191s0:1,rocep201s0:1,rocep202s0:1"}'
        ;;
    qwen3.5-fp8-dynamo-sglang)
        # Exact source and effective timeouts from completed Lepton BenchOps
        # Qwen3.5 jobs, including job 513. The checksum was independently
        # computed from the Lepton-native SquashFS artifact before readiness.
        export SRT_SLURM_QWEN35_FP8_REF="3435776cd6db4c14f8b771ff7a3976deb62fe133"
        export IMAGE_SQUASH_SHA256="e3c9ce21fa6f0f363dfb1fb4aff7a3b8dd55803b0c214c649719919a75b7bbe9"
        export SRT_SLURM_ETCD_LEASE_TTL="600"
        export SRT_SLURM_HEALTH_MAX_ATTEMPTS="720"
        ;;
    minimaxm3-fp8-dynamo-vllm)
        export SRT_SLURM_ETCD_LEASE_TTL="600"
        export SRT_SLURM_HEALTH_MAX_ATTEMPTS="720"
        if [[ "${SPEC_DECODING:-none}" == "mtp" ]]; then
            # The deleted Docker Hub nightly is recovered from vLLM's
            # immutable, architecture-specific Public ECR publication.
            export IMAGE_IMPORT_REGISTRY="public.ecr.aws"
            export IMAGE_IMPORT_REPOSITORY="q9t5s3a7/vllm-release-repo"
            export IMAGE_IMPORT_TAG="5e35a6f4f9bbc217c599692157ca985c894373f7-aarch64"
            export IMAGE_IMPORT_REFERENCE="public.ecr.aws#q9t5s3a7/vllm-release-repo:5e35a6f4f9bbc217c599692157ca985c894373f7-aarch64"
            export IMAGE_IMPORT_MANIFEST_SHA256="sha256:41442db2591d6bfb8dc219561f18deed55aaf5b95f910e5d9145186043d8eb94"
            export SRT_SLURM_MINIMAX_M3_REF="c180328b98c3793ca84a1e24a030f90545eb7d5d"
        else
            # Exact Lepton image and srt-slurm revision from completed
            # BenchOps MiniMax jobs, including Slurm job 484.
            export IMAGE_SQUASH_SHA256="1ac422ddf87efdb3d9902e254dd7d56cc9ce9d152b59f2b7e9c0716595eab481"
            export SRT_SLURM_MINIMAX_M3_STANDARD_REF="deb1dfd9934398664f92d194169c183e009da83b"
            # This older srt-slurm revision reuses the first uv on PATH in its
            # Arm64 Slurm orchestrator. Pin a compute-architecture binary so it
            # cannot inherit the x86_64 GitHub runner installation.
            export SRT_SLURM_COMPUTE_UV_VERSION="0.12.4"
            export SRT_SLURM_COMPUTE_UV_SHA256="49d881b3403187e1f1789720881e77e4251ad4259d86c4844862657d2a35d13f"
        fi
        ;;
esac

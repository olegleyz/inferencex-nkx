#!/usr/bin/bash

# This script sets up the environment and launches multi-node benchmarks

set -exo pipefail

case "${INFERENCEX_CLUSTER_PROFILE:-}" in
    "") ;;
    nkx-gb300)
        # shellcheck source=runners/cluster_profiles/nkx-gb300.sh
        source "${GITHUB_WORKSPACE}/runners/cluster_profiles/nkx-gb300.sh"
        ;;
    lepton-gb300)
        # shellcheck source=runners/cluster_profiles/lepton-gb300.sh
        source "${GITHUB_WORKSPACE}/runners/cluster_profiles/lepton-gb300.sh"
        ;;
    *)
        echo "Unsupported InferenceX cluster profile: ${INFERENCEX_CLUSTER_PROFILE}" >&2
        exit 1
        ;;
esac

export SLURM_PARTITION="${SLURM_PARTITION:-batch_1}"
export SLURM_ACCOUNT="${SLURM_ACCOUNT:-benchmark}"
export ENROOT_ROOTFS_WRITABLE=1

# A target-specific job must be attached to the intended Slurm controller.
# This check runs before any srun/sbatch operation or cache preparation.
if [[ -n "${INFERENCEX_EXPECTED_SLURM_CLUSTER:-}" ]]; then
    ACTUAL_SLURM_CLUSTER=$(
        scontrol show config | awk -F= '/^ClusterName[[:space:]]*=/ {
            value=$2; gsub(/^[[:space:]]+|[[:space:]]+$/, "", value); print value
        }'
    )
    if [[ "$ACTUAL_SLURM_CLUSTER" != "$INFERENCEX_EXPECTED_SLURM_CLUSTER" ]]; then
        echo "Slurm target mismatch: expected=$INFERENCEX_EXPECTED_SLURM_CLUSTER actual=$ACTUAL_SLURM_CLUSTER" >&2
        exit 1
    fi
    if ! sinfo --noheader --partition="$SLURM_PARTITION" --format='%P' | grep -q .; then
        echo "Slurm partition is unavailable: cluster=$ACTUAL_SLURM_CLUSTER partition=$SLURM_PARTITION" >&2
        exit 1
    fi
fi

# Host-side directory holding aiperf's content-addressed dataset mmap cache.
# Bind-mounted into worker containers at /aiperf_mmap_cache via the
# default_mounts: block in srtslurm.yaml below; aiperf reads it via
# AIPERF_DATASET_MMAP_CACHE_DIR (set in each agentic recipe's benchmark.env).
# Without it, every run re-tokenizes and re-writes ~65 GB of mmap files
# per dataset on first use. 777 mode so all gharunner_X SLURM users can
# write to it.
export AIPERF_MMAP_CACHE_HOST_PATH="${AIPERF_MMAP_CACHE_HOST_PATH:-/data/home/sa-shared/gharunners/ai-perf-cache}"

export HF_HUB_CACHE_HOST_PATH="${HF_HUB_CACHE_HOST_PATH:-/data/home/sa-shared/gharunners/hf-hub-cache}"
mkdir -p "$HF_HUB_CACHE_HOST_PATH"

# Persistent dynamo source-build cache. srtctl's hash-pinned dynamo install
# (_hash_cached_source_install) caches the built wheel + src tarball at
# /configs/dynamo-wheels/<hash> with a .complete sentinel; on a warm cache the
# install is just `pip install` from the cache (no apt, no root). In CI /configs
# is the per-job srt-slurm checkout (cold every job → cold build needs apt +
# root, which the non-root server containers can't do), so persist and share
# the cache across jobs by bind-mounting this host dir at /configs/dynamo-wheels.
# Seed it once with a --container-remap-root build. 777 for multi-user runners.
export DYNAMO_WHEELS_CACHE_HOST_PATH="${DYNAMO_WHEELS_CACHE_HOST_PATH:-/data/home/sa-shared/gharunners/dynamo-wheels}"
mkdir -p "$DYNAMO_WHEELS_CACHE_HOST_PATH"

export MODEL_PATH=$MODEL

if [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp4" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp4"
    export MODEL_PATH=/scratch/models/DeepSeek-R1-0528-NVFP4-v2
    export SRT_SLURM_MODEL_PREFIX="dsr1"
elif [[ $MODEL_PREFIX == "dsr1" && $PRECISION == "fp8" ]]; then
    export SERVED_MODEL_NAME="deepseek-r1-fp8"
    export MODEL_PATH=/scratch/models/DeepSeek-R1-0528
    export SRT_SLURM_MODEL_PREFIX="dsr1-fp8"
elif [[ $MODEL_PREFIX == "dsv4" && $PRECISION == "fp4" ]]; then
    # The checked-in default remains the original alias. Reproduction runs
    # replace it below with the receipt's revision-qualified node-local path
    # only after read-only validation on all 16 GPU workers.
    export MODEL_PATH=/scratch/models/DeepSeek-V4-Pro
    export SRT_SLURM_MODEL_PREFIX="deepseek-v4-pro"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" && $FRAMEWORK == "dynamo-trt" ]]; then
    export SERVED_MODEL_NAME="glm-5-nvfp4"
    export MODEL_PATH=/scratch/models/GLM-5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/GLM-5-NVFP4"
elif [[ $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp4" ]]; then
    # SRT_SLURM_MODEL_PREFIX matches the model.path alias ("glm-5-fp4")
    # in our GLM-5.1 sglang recipes.
    export MODEL_PATH=/scratch/models/GLM-5.1-NVFP4
    export SRT_SLURM_MODEL_PREFIX="glm-5-fp4"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/GLM-5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="glm-5-fp4"
elif [[ $MODEL_PREFIX == "glm5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/scratch/models/GLM-5-FP8
    export SRT_SLURM_MODEL_PREFIX="glm-5-fp8"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M2.5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-nvfp4"
elif [[ $MODEL_PREFIX == "minimaxm2.5" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M2.5
    export SRT_SLURM_MODEL_PREFIX="minimax-m2.5-fp8"
elif [[ $MODEL_PREFIX == "minimaxm3" && $PRECISION == "fp8" ]]; then
    export MODEL_PATH=/data/models/MiniMax-M3-MXFP8
    export SRT_SLURM_MODEL_PREFIX="minimax-m3-mxfp8"
elif [[ $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    export MODEL_PATH=/scratch/models/Kimi-K2.5-NVFP4
    export SRT_SLURM_MODEL_PREFIX="nvidia/Kimi-K2.5-NVFP4"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp4" ]]; then
    # SRT_SLURM_MODEL_PREFIX must match the model.path alias used in our
    # Qwen3.5 sglang recipes (qwen3.5-fp4).
    export MODEL_PATH=/scratch/models/Qwen3.5-397B-A17B-NVFP4
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp4"
elif [[ $MODEL_PREFIX == "qwen3.5" && $PRECISION == "fp8" ]]; then
    # SRT_SLURM_MODEL_PREFIX must match the model.path alias used in our
    # Qwen3.5 sglang recipes (qwen3.5-fp8).
    export MODEL_PATH=/scratch/models/Qwen3.5-397B-A17B-FP8
    export SRT_SLURM_MODEL_PREFIX="qwen3.5-fp8"
else
    echo "Unsupported model: $MODEL_PREFIX-$PRECISION. Supported models are: dsr1-fp4, dsr1-fp8, dsv4-fp4, glm5-fp4, glm5-fp8, minimaxm2.5-fp4, minimaxm2.5-fp8, kimik2.5-fp4, qwen3.5-fp4, qwen3.5-fp8"
    exit 1
fi

# BenchOps may provide a revision-qualified, independently verified node-local
# path. Preserve every existing model default when no override is supplied.
export MODEL_PATH="${MODEL_PATH_OVERRIDE:-$MODEL_PATH}"

# A ready receipt is immutable evidence that BenchOps staged a pinned model.
# Immediately before launch, independently perform its read-only marker,
# manifest, and file-size checks on every exact worker named in each receipt.
verify_model_stage_receipt_all_nodes() {
    local label="$1" receipt="$2" preflight_node_count missing_nodes
    local -a receipt_nodes active_partition_nodes
    test -f "$receipt"
    mapfile -t receipt_nodes < <(jq -er '.nodes.results[].node' "$receipt")
    test "${#receipt_nodes[@]}" -gt 0
    preflight_node_count="${#receipt_nodes[@]}"
    if [[ "${INFERENCEX_REQUIRE_PARTITION_RECEIPT_COVERAGE:-0}" == "1" ]]; then
        mapfile -t active_partition_nodes < <(
            sinfo --noheader --Node --partition="$SLURM_PARTITION" --format='%N|%T' |
                awk -F'|' '$2 ~ /^(idle|allocated|mixed|completing)$/ {print $1}' |
                sort -u
        )
        test "${#active_partition_nodes[@]}" -gt 0
        missing_nodes=$(
            comm -23 \
                <(printf '%s\n' "${active_partition_nodes[@]}" | sort -u) \
                <(printf '%s\n' "${receipt_nodes[@]}" | sort -u)
        )
        if [[ -n "$missing_nodes" ]]; then
            echo "Active partition workers absent from $label staging receipt:" >&2
            echo "$missing_nodes" >&2
            exit 1
        fi
        preflight_node_count="${#active_partition_nodes[@]}"
    fi
    srun \
        --partition="$SLURM_PARTITION" \
        --account="$SLURM_ACCOUNT" \
        --nodes="$preflight_node_count" \
        --ntasks="$preflight_node_count" \
        --ntasks-per-node=1 \
        --gpus-per-node=1 \
        --exclusive \
        --time=30 \
        python3 "${GITHUB_WORKSPACE}/runners/verify_model_stage.py" \
            --receipt "$receipt"
}

if [[ -n "${MODEL_PATH_OVERRIDE:-}" ]]; then
    verify_model_stage_receipt_all_nodes \
        "base model" "${GITHUB_WORKSPACE}/model-stage-result.json"
fi

if [[ "$MODEL_PREFIX-${SPEC_DECODING:-}" == "minimaxm3-mtp" ]]; then
    if [[ -z "${SPECULATIVE_MODEL_PATH_OVERRIDE:-}" ]]; then
        echo "MiniMax M3 EAGLE3 requires a verified node-local draft model" >&2
        exit 1
    fi
    verify_model_stage_receipt_all_nodes \
        "speculative model" "${GITHUB_WORKSPACE}/speculative-model-stage-result.json"
elif [[ -n "${SPECULATIVE_MODEL_PATH_OVERRIDE:-}" ]]; then
    echo "Unexpected speculative-model path for $MODEL_PREFIX-${SPEC_DECODING:-none}" >&2
    exit 1
fi

NGINX_IMAGE="nginx:1.27.4"

# Squash files live on the Vast NFS storage; use the /data/ mount
# (not /home/sa-shared/) — both are the same backing storage but the
# /home/sa-shared/ mount has a chronic ELOOP / "Too many levels of
# symbolic links" bug from workflow worker NFS sessions on lockfiles
# AND data files. /data/ has a separate NFS client cache that isn't
# poisoned. See feedback_gb300_nfs_eloop_workaround for diagnosis.
SQUASH_CACHE_HOST_PATH="${SQUASH_CACHE_HOST_PATH:-/data/home/sa-shared/gharunners/squash}"
SQUASH_FILE="${SQUASH_CACHE_HOST_PATH}/$(echo "$IMAGE" | sed 's/[\/:@#]/_/g').sqsh"
NGINX_SQUASH_FILE="${SQUASH_CACHE_HOST_PATH}/$(echo "$NGINX_IMAGE" | sed 's/[\/:@#]/_/g').sqsh"

# Run the import on a compute node via srun, not on the login node:
# the login node is x86_64 while the compute nodes are aarch64, so the
# arm64 squash file has to be built on a compute node.
import_squash() {
    local squash="$1" image="$2" expected_sha256="${3:-}"
    local lock="${squash}.lock"
    srun --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" \
        --nodes=1 --ntasks=1 --gpus-per-node=1 --exclusive --time=180 bash -c "
        exec 9>\"$lock\"
        flock -w 600 9 || { echo 'Failed to acquire lock for $squash' >&2; exit 1; }
        if unsquashfs -l \"$squash\" > /dev/null 2>&1; then
            echo 'Squash file already exists and is valid, skipping import: $squash'
        else
            rm -f \"$squash\"
            enroot import -o \"$squash\" docker://$image
        fi
        if [ -n \"$expected_sha256\" ]; then
            actual_sha256=\$(sha256sum \"$squash\" | awk '{print \$1}')
            if [ \"\$actual_sha256\" != \"$expected_sha256\" ]; then
                echo 'SquashFS checksum mismatch for $image' >&2
                echo \"expected=$expected_sha256 actual=\$actual_sha256 path=$squash\" >&2
                exit 1
            fi
        fi
    "
}

verify_image_import_manifest() {
    if [[ -z "${IMAGE_IMPORT_MANIFEST_SHA256:-}" ]]; then
        return
    fi
    : "${IMAGE_IMPORT_REGISTRY:?missing IMAGE_IMPORT_REGISTRY}"
    : "${IMAGE_IMPORT_REPOSITORY:?missing IMAGE_IMPORT_REPOSITORY}"
    : "${IMAGE_IMPORT_TAG:?missing IMAGE_IMPORT_TAG}"
    expected_reference="${IMAGE_IMPORT_REGISTRY}#${IMAGE_IMPORT_REPOSITORY}:${IMAGE_IMPORT_TAG}"
    if [[ "${IMAGE_IMPORT_REFERENCE:-}" != "$expected_reference" ]]; then
        echo "Image import reference does not match its reviewed registry coordinates" >&2
        exit 1
    fi

    # Do not expose the short-lived anonymous registry token through xtrace.
    set +x
    registry_token=$(curl -fsSL \
        "https://${IMAGE_IMPORT_REGISTRY}/token/?service=${IMAGE_IMPORT_REGISTRY}&scope=repository:${IMAGE_IMPORT_REPOSITORY}:pull" |
        jq -er '.token')
    manifest_headers=$(curl -fsSI \
        -H "Authorization: Bearer ${registry_token}" \
        -H 'Accept: application/vnd.docker.distribution.manifest.v2+json' \
        "https://${IMAGE_IMPORT_REGISTRY}/v2/${IMAGE_IMPORT_REPOSITORY}/manifests/${IMAGE_IMPORT_TAG}")
    unset registry_token
    actual_manifest_sha256=$(awk \
        'tolower($1) == "docker-content-digest:" {gsub("\\r", "", $2); print $2}' \
        <<<"$manifest_headers")
    unset manifest_headers
    set -x
    if [[ "$actual_manifest_sha256" != "$IMAGE_IMPORT_MANIFEST_SHA256" ]]; then
        echo "OCI manifest mismatch for ${IMAGE_IMPORT_REFERENCE}" >&2
        echo "expected=${IMAGE_IMPORT_MANIFEST_SHA256} actual=${actual_manifest_sha256}" >&2
        exit 1
    fi
    echo "Verified immutable OCI manifest ${actual_manifest_sha256} for ${IMAGE_IMPORT_REFERENCE}"
}

verify_image_import_manifest
import_squash "$SQUASH_FILE" "${IMAGE_IMPORT_REFERENCE:-$IMAGE}" "${IMAGE_SQUASH_SHA256:-}"
import_squash "$NGINX_SQUASH_FILE" "$NGINX_IMAGE" "${NGINX_SQUASH_SHA256:-}"

export EVAL_ONLY="${EVAL_ONLY:-false}"

export ISL="$ISL"
export OSL="$OSL"

echo "Cloning srt-slurm repository..."
RUN_KEY=$(printf "%s" "${RESULT_FILENAME:-${RUNNER_NAME:-gb300-nv}}" | sha1sum | cut -c1-12)
SRT_REPO_DIR="${GITHUB_WORKSPACE}/srt-slurm-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUN_KEY}"
SRTCTL_SETUP_SCRIPT=""
rm -rf "$SRT_REPO_DIR"

if [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "qwen3.5" ]]; then
    # Qwen3.5 agentic uses NVIDIA/srt-slurm v1.0.22: the two features the
    # cquil11 fork was pinned for are merged upstream (present in v1.0.22) —
    #   - `srtctl apply --no-preflight` (skip the in-process model FS check):
    #     model.path resolves to /scratch/models/Qwen3.5-397B-A17B-NVFP4
    #     (compute-node-only NVMe), which the GHA runner pod can't stat, so
    #     the Path.is_dir() preflight would fail before sbatch is ever
    #     called. The engine still fails loudly at runtime if the path is
    #     genuinely missing on the compute node.
    #   - benchmark_stage propagates srun_options (container-remap-root must
    #     reach the agentic_srt.sh srun).
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout v1.0.22
    mkdir -p recipes/sglang/qwen3.5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5" \
        recipes/sglang/qwen3.5
elif [[ "$IS_AGENTIC" == "1" && $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "dsv4" ]]; then
    # DSv4 GB300 sglang agentic: NVIDIA/srt-slurm v1.0.10 has the nginx
    # client_max_body_size fix (>1 MiB agentic warmup bodies), the
    # session-affinity frontend, and the BenchmarkType.CUSTOM / extra_mount
    # schema these recipes need.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout v1.0.10
    mkdir -p recipes/sglang/deepseek-v4/agentic
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/deepseek-v4/agentic" \
        recipes/sglang/deepseek-v4/agentic
elif [[ "$IS_AGENTIC" == "1" ]]; then
    # Agentic recipes use NVIDIA/srt-slurm v1.0.36. This is the upstream
    # version validated in InferenceX PR #2302 and includes per-node DP,
    # matching Dynamo health counts, multi-node TP port handling, and
    # Mooncake compatibility. Keep it pinned so sweeps are reproducible.
    git clone --branch v1.0.36 --single-branch https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR" || exit 1
    cd "$SRT_REPO_DIR" || exit 1

    mkdir -p recipes/vllm/deepseek-v4/agentic || exit 1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4/agentic" \
        recipes/vllm/deepseek-v4/agentic || exit 1
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "dsv4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout "${SRT_SLURM_DSV4_REF:-aflowers/gb200-dsv4-recipes}"
    mkdir -p recipes/vllm/deepseek-v4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/deepseek-v4" recipes/vllm/deepseek-v4
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
    if [[ $PRECISION == "fp4" ]]; then
        mkdir -p recipes/sglang/glm5/gb300-fp4
        cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5/gb300-fp4" recipes/sglang/glm5/gb300-fp4
    fi
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5.1" && $PRECISION == "fp8" ]]; then
    # GLM-5.1 FP8 (gb300) recipes are version-controlled in-repo; overlay them
    # onto the pinned submission branch.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
    mkdir -p recipes/sglang/glm5.1
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/glm5.1" recipes/sglang/glm5.1
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "glm5.1" ]]; then
    # GLM-5.1 MTP recipe (recipes/gb300-fp4/glm5-mtp.yaml) lives on
    # NVIDIA/srt-slurm:main — check it out; no in-repo overlay needed.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
elif [[ $FRAMEWORK == "dynamo-sglang" && $MODEL_PREFIX == "qwen3.5" ]]; then
    # Overlay our version-controlled Qwen3.5 recipes onto the srt-slurm checkout.
    # fp8 recipes pin dynamo by commit hash (source install), which needs the
    # cargo/maturin bootstrap included in the srt-slurm v1.0.25 release — the
    # sa-submission-q2-2026 sglang install path assumes maturin ships in the
    # image, and the lmsysorg/sglang nightly-dev-cu13 image doesn't include it.
    # Same branch the identical gb200-fp8 recipes run on. fp4 recipes pin
    # dynamo by version (pip install) and stay on the submission branch they
    # were validated against.
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    if [[ $PRECISION == "fp8" ]]; then
        git checkout "${SRT_SLURM_QWEN35_FP8_REF:-v1.0.25}"
    else
        git checkout sa-submission-q2-2026
    fi
    mkdir -p recipes/sglang/qwen3.5
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5" recipes/sglang/qwen3.5
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "minimaxm3" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    if [[ "${SPEC_DECODING:-}" == "mtp" ]]; then
        git checkout "${SRT_SLURM_MINIMAX_M3_REF:-v1.0.38}"
    else
        git checkout "${SRT_SLURM_MINIMAX_M3_STANDARD_REF:-sa-submission-q2-2026}"
    fi
    mkdir -p recipes/vllm/minimax-m3-gb300-fp8
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3-gb300-fp8" recipes/vllm/minimax-m3-gb300-fp8
elif [[ $FRAMEWORK == "dynamo-vllm" && $MODEL_PREFIX == "kimik2.5" && $PRECISION == "fp4" ]]; then
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout main
    mkdir -p recipes/vllm/kimi-k2.5-fp4
    cp -rT "$GITHUB_WORKSPACE/benchmarks/multi_node/srt-slurm-recipes/vllm/kimi-k2.5-fp4" recipes/vllm/kimi-k2.5-fp4
elif [[ $FRAMEWORK == "dynamo-trt" && $MODEL_PREFIX == "dsv4" ]]; then
    # DSv4 dynamo-trt recipes use the HuggingFace model ID as model.path,
    # so override SRT_SLURM_MODEL_PREFIX to match the recipe's model path key.
    SRT_SLURM_MODEL_PREFIX="deepseek-ai/DeepSeek-V4-Pro"
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
else
    git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"
    cd "$SRT_REPO_DIR"
    git checkout sa-submission-q2-2026
fi

echo "Installing srtctl..."
export UV_INSTALL_DIR="$GITHUB_WORKSPACE/.local/bin"
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$UV_INSTALL_DIR:$PATH"

VENV_DIR="${GITHUB_WORKSPACE}/.venv-srt-${GITHUB_RUN_ID:-manual}-${GITHUB_RUN_ATTEMPT:-0}-${RUN_KEY}"
rm -rf "$VENV_DIR"
# --seed installs pip+setuptools+wheel into the venv. Without it, the
# upstream prefetch-ai-dynamo-wheel.sh script (called by srtctl when a
# recipe has dynamo.wheel set) fails with "No module named pip" because
# uv venv defaults to no-pip.
uv venv --seed "$VENV_DIR"
source "$VENV_DIR/bin/activate"
uv pip install -e .

if ! command -v srtctl &> /dev/null; then
    echo "Error: Failed to install srtctl"
    exit 1
fi
SRTCTL_BIN="$(command -v srtctl)"

echo "Configs available at: $SRT_REPO_DIR/"

# Create srtslurm.yaml for srtctl (used by both frameworks)
SRTCTL_ROOT="${SRT_REPO_DIR}"
echo "Creating srtslurm.yaml configuration..."
EXTRA_MOUNTS_YAML=""
if [[ -n "${SRT_SLURM_HOME_PATH:-}" ]]; then
    printf -v EXTRA_MOUNTS_YAML '%s  "%s": "%s"\n' \
        "$EXTRA_MOUNTS_YAML" "$SRT_SLURM_HOME_PATH" "$SRT_SLURM_HOME_PATH"
fi
if [[ -n "${SRT_SLURM_SHARED_ROOT:-}" ]]; then
    printf -v EXTRA_MOUNTS_YAML '%s  "%s": "%s"\n' \
        "$EXTRA_MOUNTS_YAML" "$SRT_SLURM_SHARED_ROOT" "$SRT_SLURM_SHARED_ROOT"
fi
cat > srtslurm.yaml <<EOF
# SRT SLURM Configuration for GB300

# Default SLURM settings
default_account: "${SLURM_ACCOUNT}"
default_partition: "${SLURM_PARTITION}"
default_time_limit: "4:00:00"

# Resource defaults
gpus_per_node: 4
network_interface: ""

# Path to srtctl repo root (where the configs live)
srtctl_root: "${SRTCTL_ROOT}"

# Cluster-level bind mounts applied to every worker container
# (see srtctl/core/runtime.py — get_srtslurm_setting("default_mounts")).
# Used here for aiperf's persistent mmap cache so the dataset isn't
# re-tokenized + re-written every job.
default_mounts:
${EXTRA_MOUNTS_YAML}  "${AIPERF_MMAP_CACHE_HOST_PATH}": "/aiperf_mmap_cache"
  "${HF_HUB_CACHE_HOST_PATH}": "/hf_hub_cache"
  # Warm dynamo source-build cache (nested over the auto /configs mount) so the
  # hash-pinned install is a cache hit (pip-only, no apt/root) on every job.
  "${DYNAMO_WHEELS_CACHE_HOST_PATH}": "/configs/dynamo-wheels"

# Model path aliases
model_paths:
  "${SRT_SLURM_MODEL_PREFIX}": "${MODEL_PATH}"
containers:
  dynamo-trtllm: ${SQUASH_FILE}
  dynamo-sglang: ${SQUASH_FILE}
  v0.5.11: ${SQUASH_FILE}
  v0.5.13.post1: ${SQUASH_FILE}
  "${IMAGE}": ${SQUASH_FILE}
  nginx-sqsh: ${NGINX_SQUASH_FILE}
use_segment_sbatch_directive: false
EOF

echo "Generated srtslurm.yaml:"
cat srtslurm.yaml

echo "Running make setup..."
make setup ARCH=aarch64

# Older reviewed srt-slurm revisions do not materialize a compute-architecture
# uv binary during `make setup`. When a cross-architecture target declares an
# immutable uv identity, add it under the srt-slurm checkout and put that path
# first in the submitted Slurm environment. The host srtctl entrypoint remains
# the absolute venv path captured above.
if [[ -n "${SRT_SLURM_COMPUTE_UV_VERSION:-}" || -n "${SRT_SLURM_COMPUTE_UV_SHA256:-}" ]]; then
    if [[ -z "${SRT_SLURM_COMPUTE_UV_VERSION:-}" || ! "${SRT_SLURM_COMPUTE_UV_SHA256:-}" =~ ^[0-9a-f]{64}$ ]]; then
        echo "Compute uv requires a version and lowercase SHA-256 digest" >&2
        exit 1
    fi
    COMPUTE_UV_DIR="${SRT_REPO_DIR}/bin"
    COMPUTE_UV_BIN="${COMPUTE_UV_DIR}/uv"
    COMPUTE_UV_TMP=$(mktemp -d)
    COMPUTE_UV_ARCHIVE="${COMPUTE_UV_TMP}/uv.tar.gz"
    curl -LsSf \
        "https://github.com/astral-sh/uv/releases/download/${SRT_SLURM_COMPUTE_UV_VERSION}/uv-aarch64-unknown-linux-gnu.tar.gz" \
        -o "$COMPUTE_UV_ARCHIVE"
    echo "${SRT_SLURM_COMPUTE_UV_SHA256}  ${COMPUTE_UV_ARCHIVE}" | sha256sum --check --status
    mkdir -p "$COMPUTE_UV_DIR"
    tar -xzf "$COMPUTE_UV_ARCHIVE" --strip-components=1 -C "$COMPUTE_UV_DIR"
    rm -rf "$COMPUTE_UV_TMP"
    if ! file "$COMPUTE_UV_BIN" | grep -q 'ARM aarch64'; then
        echo "Compute uv is not an Arm64 executable: $COMPUTE_UV_BIN" >&2
        exit 1
    fi
    chmod +x "$COMPUTE_UV_BIN"
    echo "Prepared checksum-pinned Arm64 uv: version=${SRT_SLURM_COMPUTE_UV_VERSION} sha256=${SRT_SLURM_COMPUTE_UV_SHA256}"
fi

# Export eval-related env vars for srt-slurm post-benchmark eval
export INFMAX_WORKSPACE="$GITHUB_WORKSPACE"

echo "Submitting job with srtctl..."

if [[ -z "$CONFIG_FILE" ]]; then
    echo "Error: CONFIG_FILE is not set. The srt-slurm path requires a CONFIG_FILE in additional-settings." >&2
    echo "Config: MODEL_PREFIX=${MODEL_PREFIX} PRECISION=${PRECISION} FRAMEWORK=${FRAMEWORK}" >&2
    exit 1
fi

# Override the job name in the config file with the runner name.
# CONFIG_FILE may carry a ":zip_override_...[i]" selector suffix that only
# `srtctl apply -f` parses; strip it to the real path for the sed. srtctl
# below still receives the full CONFIG_FILE (with selector).
CONFIG_PATH="${CONFIG_FILE%%:*}"
sed -i "s/^name:.*/name: \"${RUNNER_NAME}\"/" "$CONFIG_PATH"

# Cluster profiles may add scheduler/runtime inputs to the resolved recipe.
# This runs after the reviewed recipe is overlaid into the pinned srt-slurm
# checkout, and the generated config and Slurm script remain native artifacts.
if [[ -n "${SRT_SLURM_CPUS_PER_TASK:-}" || -n "${SRT_SLURM_TIME_LIMIT:-}" || -n "${SRT_SLURM_ETCD_LEASE_TTL:-}" || -n "${SRT_SLURM_HEALTH_MAX_ATTEMPTS:-}" || -n "${SRT_SLURM_RUNTIME_ENV_JSON:-}" || -n "${SPECULATIVE_MODEL_PATH_OVERRIDE:-}" ]]; then
    export CONFIG_PATH SRT_SLURM_CPUS_PER_TASK SRT_SLURM_TIME_LIMIT SRT_SLURM_ETCD_LEASE_TTL SRT_SLURM_HEALTH_MAX_ATTEMPTS SRT_SLURM_RUNTIME_ENV_JSON SPECULATIVE_MODEL_PATH_OVERRIDE
    python - <<'PY'
import os
import json
from pathlib import Path

import yaml

path = Path(os.environ["CONFIG_PATH"])
document = yaml.safe_load(path.read_text())
if not isinstance(document, dict):
    raise RuntimeError("resolved InferenceX recipe must be a mapping")

cpus = os.environ.get("SRT_SLURM_CPUS_PER_TASK")
if cpus:
    if not cpus.isdigit() or int(cpus) <= 0:
        raise RuntimeError("SRT_SLURM_CPUS_PER_TASK must be a positive integer")
    directives = document.setdefault("sbatch_directives", {})
    existing = directives.get("cpus-per-task")
    if existing not in (None, cpus, int(cpus)):
        raise RuntimeError("reviewed recipe already defines a different cpus-per-task")
    directives["cpus-per-task"] = cpus

time_limit = os.environ.get("SRT_SLURM_TIME_LIMIT")
if time_limit:
    parts = time_limit.split(":")
    if len(parts) != 3 or any(not part.isdigit() for part in parts):
        raise RuntimeError("SRT_SLURM_TIME_LIMIT must use HH:MM:SS")
    slurm = document.setdefault("slurm", {})
    if not isinstance(slurm, dict):
        raise RuntimeError("reviewed recipe slurm field must be a mapping")
    slurm["time_limit"] = time_limit

attempts = os.environ.get("SRT_SLURM_HEALTH_MAX_ATTEMPTS")
if attempts:
    if not attempts.isdigit() or int(attempts) <= 0:
        raise RuntimeError("SRT_SLURM_HEALTH_MAX_ATTEMPTS must be a positive integer")
    health = document.setdefault("health_check", {})
    health["max_attempts"] = int(attempts)

ttl = os.environ.get("SRT_SLURM_ETCD_LEASE_TTL")
if ttl:
    if not ttl.isdigit() or int(ttl) <= 0:
        raise RuntimeError("SRT_SLURM_ETCD_LEASE_TTL must be a positive integer")
    frontend = document.setdefault("frontend", {}).setdefault("env", {})
    frontend["ETCD_LEASE_TTL"] = ttl
    backend = document.setdefault("backend", {})
    for key in ("prefill_environment", "decode_environment"):
        backend.setdefault(key, {})["ETCD_LEASE_TTL"] = ttl

runtime_env_json = os.environ.get("SRT_SLURM_RUNTIME_ENV_JSON")
if runtime_env_json:
    runtime_env = json.loads(runtime_env_json)
    if not isinstance(runtime_env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in runtime_env.items()
    ):
        raise RuntimeError("SRT_SLURM_RUNTIME_ENV_JSON must contain string pairs")
    frontend = document.setdefault("frontend", {}).setdefault("env", {})
    backend = document.setdefault("backend", {})
    for key, value in runtime_env.items():
        frontend[key] = value
        for environment in ("prefill_environment", "decode_environment"):
            backend.setdefault(environment, {})[key] = value

speculative_model_path = os.environ.get("SPECULATIVE_MODEL_PATH_OVERRIDE")
if speculative_model_path:
    vllm_config = document.setdefault("backend", {}).get("vllm_config")
    if not isinstance(vllm_config, dict):
        raise RuntimeError("speculative-model override requires backend.vllm_config")
    for role in ("prefill", "decode"):
        role_config = vllm_config.get(role)
        if not isinstance(role_config, dict):
            raise RuntimeError(f"speculative-model override requires {role} config")
        raw = role_config.get("speculative-config")
        if not isinstance(raw, str):
            raise RuntimeError(f"speculative-model override requires {role} speculative-config")
        speculative = json.loads(raw)
        if speculative.get("method") != "eagle3" or speculative.get("model") != "Inferact/MiniMax-M3-EAGLE3-GQA":
            raise RuntimeError(f"unexpected reviewed {role} speculative model")
        speculative["model"] = speculative_model_path
        role_config["speculative-config"] = json.dumps(
            speculative, separators=(",", ":")
        )
    # srtctl mounts the primary model alias at /model. The EAGLE3 draft
    # remains an absolute path in speculative-config, so expose that one
    # independently verified directory read-only at the same container path.
    base_model_path = os.environ.get("MODEL_PATH")
    if not base_model_path or not base_model_path.startswith("/"):
        raise RuntimeError("speculative-model override requires absolute MODEL_PATH")
    base_mount = f"{base_model_path}:/model:ro"
    draft_mount = f"{speculative_model_path}:{speculative_model_path}:ro"
    extra_mount = document.setdefault("extra_mount", [])
    if not isinstance(extra_mount, list) or any(
        not isinstance(value, str) for value in extra_mount
    ):
        raise RuntimeError("reviewed recipe extra_mount must be a string list")
    for mount in (base_mount, draft_mount):
        if mount not in extra_mount:
            extra_mount.append(mount)

if os.environ.get("INFERENCEX_READINESS_ONLY") == "1":
    benchmark = document.setdefault("benchmark", {})
    benchmark["concurrencies"] = "1"
    benchmark["num_prompts_mult"] = 1
    benchmark["num_warmup_mult"] = 0

path.write_text(yaml.safe_dump(document, sort_keys=False))
PY
fi

# --no-preflight skips newer srtctl's pre-submit model-path stat, which runs on
# the GHA runner host (im-gb300-login-02, an x86 login node). It's required
# whenever model.path resolves to the node-local /scratch NVMe that the login
# node can't see:
#   - the agentic path (DSv4-Pro checkpoint),
#   - glm5.1, whose GLM-5.1-NVFP4 weights are prestaged on the compute-node
#     /scratch/models, and
#   - qwen3.5 fp8, whose weights are also on the compute-node /scratch/models
#     and which runs on srt-slurm:v1.0.25 (the release that has the preflight;
#     qwen3.5 fp4 runs on sa-submission-q2-2026, which has none).
# The engine still fails loudly at runtime if the path is genuinely missing on
# the compute node. Other fixed-seq-len recipes resolve model.path to a
# login-visible location, so keep the precheck enforced for them.
# srtctl itself runs on the x86_64 login runner, but its submitted sweep runs
# on aarch64 GPU workers. Keep the host entrypoint by absolute path while
# removing the host virtualenv from the inherited Slurm environment; otherwise
# uv on the compute node finds the x86_64 Python first and fails with ENOEXEC.
# Do this only after host-side recipe generation, which uses the venv's python.
deactivate
if [[ -n "${COMPUTE_UV_BIN:-}" ]]; then
    export PATH="${COMPUTE_UV_DIR}:$PATH"
fi
hash -r

SRTCTL_APPLY_ARGS=(
    -f "$CONFIG_FILE"
    --tags "gb300,${MODEL_PREFIX},${PRECISION},${ISL}x${OSL},infmax-$(date +%Y%m%d)"
)
if [[ -n "${MODEL_PATH_OVERRIDE:-}" ]]; then
    # The job-2024 srt-slurm revision predates --no-preflight and does not
    # perform that login-node path check. Newer revisions may expose the flag.
    # In both cases the all-node read-only verification above is authoritative.
    if "$SRTCTL_BIN" apply --help 2>&1 | grep -q -- '--no-preflight'; then
        SRTCTL_APPLY_ARGS+=(--no-preflight)
    else
        echo "Pinned srtctl has no --no-preflight option; using completed all-node model verification"
    fi
elif [[ "$IS_AGENTIC" == "1" || "$MODEL_PREFIX" == "glm5.1" || ( "$MODEL_PREFIX" == "qwen3.5" && "$PRECISION" == "fp8" ) ]]; then
    SRTCTL_APPLY_ARGS+=(--no-preflight)
fi
if [[ -n "$SRTCTL_SETUP_SCRIPT" ]]; then
    SRTCTL_APPLY_ARGS+=(--setup-script "$SRTCTL_SETUP_SCRIPT")
fi

SRTCTL_OUTPUT=$("$SRTCTL_BIN" apply "${SRTCTL_APPLY_ARGS[@]}" 2>&1)
echo "$SRTCTL_OUTPUT"

JOB_ID=$(echo "$SRTCTL_OUTPUT" | grep -oP '✅ Job \K[0-9]+' || echo "$SRTCTL_OUTPUT" | grep -oP 'Job \K[0-9]+')

set +x

if [ -z "$JOB_ID" ]; then
    echo "Error: Failed to extract JOB_ID from srtctl output"
    exit 1
fi

echo "Extracted JOB_ID: $JOB_ID"

# Use the JOB_ID to find the logs directory
# srtctl creates logs in outputs/JOB_ID/logs/
LOGS_DIR="outputs/$JOB_ID/logs"
LOG_FILE="$LOGS_DIR/sweep_${JOB_ID}.log"

# Snapshot worker logs on any exit path — normal completion, error,
# SIGTERM (gh run cancel sends this to the launcher), even SIGKILL of
# our parent. Without this trap, the cancel-time tar lives only in the
# main flow below (after `wait $POLL_PID`), so a manual `gh run cancel`
# during the tail wait skips it entirely and the
# `Upload server logs` workflow step finds nothing to upload.
# Idempotent: the main-flow tar at the bottom of this script is now a
# no-op because the trap already produced the artifact, but it stays
# for narrative continuity in normal (non-cancel) runs.
_snapshot_server_logs() {
    if [ -n "${LOGS_DIR:-}" ] && [ -d "$LOGS_DIR" ] && [ -n "${GITHUB_WORKSPACE:-}" ]; then
        # Copy + tar are independent best-effort; an in-flight write
        # from a worker .out file at SIGTERM time would otherwise abort
        # the whole script before either succeeds.
        cp -r "$LOGS_DIR" "$GITHUB_WORKSPACE/LOGS" 2>/dev/null || true
        tar czf "$GITHUB_WORKSPACE/multinode_server_logs.tar.gz" -C "$LOGS_DIR" . 2>/dev/null || true
    fi
    if [ -n "${GITHUB_WORKSPACE:-}" ]; then
        runtime_dir="${GITHUB_WORKSPACE}/srt-slurm-runtime"
        mkdir -p "$runtime_dir"
        cp srtslurm.yaml "$runtime_dir/srtslurm.yaml" 2>/dev/null || true
        cp "${CONFIG_PATH:-}" "$runtime_dir/effective-recipe.yaml" 2>/dev/null || true
        cp "outputs/${JOB_ID:-}/config.yaml" "$runtime_dir/config.yaml" 2>/dev/null || true
        cp "outputs/${JOB_ID:-}/sbatch_script.sh" "$runtime_dir/sbatch_script.sh" 2>/dev/null || true
        git -C "${SRT_REPO_DIR:-.}" rev-parse HEAD > \
            "$runtime_dir/srt-slurm-commit.txt" 2>/dev/null || true
        git -C "$GITHUB_WORKSPACE" rev-parse HEAD > \
            "$runtime_dir/inferencex-commit.txt" 2>/dev/null || true
        env | LC_ALL=C sort | grep -E \
            '^(AIPERF_|CUDA_|DYNAMO_|ENROOT_|HF_HUB_|IMAGE=|IMAGE_IMPORT_|IMAGE_SQUASH_|INFERENCEX_|MODEL=|MODEL_PATH|NCCL_|NIXL_|NVSHMEM_|SLURM_|SPECULATIVE_MODEL_|SQUASH_|SRT_|TORCH_|UCX_|VLLM_)' \
            > "$runtime_dir/launcher-environment.txt" || true
    fi
}
trap _snapshot_server_logs EXIT

# Wait for log file to appear (also check job is still alive)
while ! ls "$LOG_FILE" &>/dev/null; do
    if ! squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; then
        echo "ERROR: Job $JOB_ID failed before creating log file"
        scontrol show job "$JOB_ID"
        exit 1
    fi
    echo "Waiting for JOB_ID $JOB_ID to begin and $LOG_FILE to appear..."
    sleep 5
done

# Poll for job completion in background
(
    while squeue -j "$JOB_ID" --noheader 2>/dev/null | grep -q "$JOB_ID"; do
        sleep 10
    done
) &
POLL_PID=$!

echo "Tailing LOG_FILE: $LOG_FILE"

# Stream the log file until job completes (-F follows by name, polls instead of inotify for NFS)
tail -F -s 2 -n+1 "$LOG_FILE" --pid=$POLL_PID 2>/dev/null

wait $POLL_PID

set -x

echo "Job $JOB_ID completed!"

# A failed data-parallel worker can leave enough surviving Dynamo endpoints to
# satisfy srt-slurm's aggregate health count. Reject that degraded topology
# before collecting a result when the cluster profile enables this guard.
if [[ "${INFERENCEX_REJECT_WORKER_STARTUP_FAILURES:-0}" == "1" ]]; then
    python3 "${GITHUB_WORKSPACE}/runners/validate_srt_worker_logs.py" "$LOGS_DIR"
fi

echo "Collecting results..."

if [ -d "$LOGS_DIR" ]; then
    echo "Found logs directory: $LOGS_DIR"
    # Tarball + LOGS copy are produced by the EXIT trap defined near
    # JOB_ID extraction (so cancel paths also get them); just log here.
    echo "multinode_server_logs.tar.gz will be (re)produced on script EXIT."
else
    echo "Warning: Logs directory not found at $LOGS_DIR"
fi

if [[ "${EVAL_ONLY:-false}" != "true" ]]; then
    if [ ! -d "$LOGS_DIR" ]; then
        exit 1
    fi

    # Find all result subdirectories
    RESULT_SUBDIRS=$(find "$LOGS_DIR" -maxdepth 1 -type d -name "*isl*osl*" 2>/dev/null)

    if [ -z "$RESULT_SUBDIRS" ]; then
        echo "Warning: No result subdirectories found in $LOGS_DIR"
    else
        # Process results from all configurations
        for result_subdir in $RESULT_SUBDIRS; do
            echo "Processing result subdirectory: $result_subdir"

            # Extract configuration info from directory name
            CONFIG_NAME=$(basename "$result_subdir")

            # Find all result JSON files
            RESULT_FILES=$(find "$result_subdir" -name "results_concurrency_*.json" 2>/dev/null)

            for result_file in $RESULT_FILES; do
                if [ -f "$result_file" ]; then
                    # Extract metadata from filename
                    # Files may be "results_concurrency_N_gpus_G_ctx_C_gen_D.json" (disagg) or "results_concurrency_N_gpus_G.json" (non-disagg)
                    filename=$(basename "$result_file")
                    concurrency=$(echo "$filename" | sed -n 's/results_concurrency_\([0-9]*\)_gpus_.*/\1/p')
                    gpus=$(echo "$filename" | sed -n 's/results_concurrency_[0-9]*_gpus_\([0-9][0-9]*\).*/\1/p')
                    ctx=$(echo "$filename" | sed -n 's/.*_ctx_\([0-9]*\)_gen_.*/\1/p')
                    gen=$(echo "$filename" | sed -n 's/.*_gen_\([0-9]*\)\.json/\1/p')

                    echo "Processing concurrency $concurrency with $gpus GPUs (ctx: $ctx, gen: $gen): $result_file"

                    if [ -n "$ctx" ] && [ -n "$gen" ]; then
                        WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}_ctx_${ctx}_gen_${gen}.json"
                    else
                        WORKSPACE_RESULT_FILE="$GITHUB_WORKSPACE/${RESULT_FILENAME}_${CONFIG_NAME}_conc${concurrency}_gpus_${gpus}.json"
                    fi
                    cp "$result_file" "$WORKSPACE_RESULT_FILE"

                    echo "Copied result file to: $WORKSPACE_RESULT_FILE"
                fi
            done
        done
    fi

    echo "All result files processed"
else
    echo "EVAL_ONLY=true: Skipping benchmark result collection"
fi

# Collect eval results if eval was requested
if [[ "${RUN_EVAL:-false}" == "true" || "${EVAL_ONLY:-false}" == "true" ]]; then
    EVAL_DIR="$LOGS_DIR/eval_results"
    if [ -d "$EVAL_DIR" ]; then
        echo "Extracting eval results from $EVAL_DIR"
        shopt -s nullglob
        for eval_file in "$EVAL_DIR"/*; do
            [ -f "$eval_file" ] || continue
            eval_dest="$GITHUB_WORKSPACE/$(basename "$eval_file")"
            rm -f "$eval_dest"
            if cp "$eval_file" "$eval_dest"; then
                echo "Copied eval artifact: $(basename "$eval_file")"
            else
                echo "WARNING: Failed to copy eval artifact, continuing: $(basename "$eval_file")"
            fi
        done
        shopt -u nullglob
    else
        echo "WARNING: RUN_EVAL=true but no eval results found at $EVAL_DIR"
    fi
fi

# Snapshot logs to GITHUB_WORKSPACE BEFORE cleanup, so the EXIT trap's
# `[ -d "$LOGS_DIR" ]` guard isn't already false by the time it fires
# (it runs AFTER the rm below, since EXIT traps are last-thing-before-exit).
# Without this inline call, R25 lost both 1p6d shards' logs.
_snapshot_server_logs

# Clean up srt-slurm outputs to prevent NFS silly-rename lock files
# from blocking the next job's checkout on this runner
echo "Cleaning up srt-slurm outputs..."
for i in 1 2 3 4 5; do
    rm -rf outputs 2>/dev/null && break
    echo "Retry $i/5: Waiting for NFS locks to release..."
    sleep 10
done
find . -name '.nfs*' -delete 2>/dev/null || true

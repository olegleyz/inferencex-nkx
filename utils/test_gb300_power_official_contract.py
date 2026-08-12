"""Static contract for the official GB300 dcgm-power lane.

Shares helpers with the GB200 contract; GB300 specifics are the default shared
squash cache path, the 19401 exporter port, and the preserved
v1.0.25/sa-submission non-power refs.
"""

import yaml

from test_gb200_power_official_contract import (
    REPO_ROOT,
    assert_exporter_provisioning,
    assert_pinned_clone_contract,
    assert_recipe_driven_detection,
)

RECIPE_PATH = REPO_ROOT / "benchmarks/multi_node/srt-slurm-recipes/sglang/qwen3.5/gb300-fp8/8k1k/1p1d-tp4-tp4.yaml"
LAUNCHER_PATH = REPO_ROOT / "runners/launch_gb300-nv.sh"
PREFLIGHT_PATH = REPO_ROOT / "runners/preflight_gb300_nkx.sh"


def test_recipe_declares_enabled_dcgm_power_lane():
    recipe = yaml.safe_load(RECIPE_PATH.read_text())

    telemetry = recipe["telemetry"]
    assert telemetry["enabled"] is True
    assert telemetry["provider"] == "dcgm-power"
    assert telemetry["required"] is True
    assert telemetry["dcgm_exporter"]["container_image"] == "dcgm-exporter"
    # 9401 is already bound by the cluster-level exporter on im-gb300 nodes.
    assert telemetry["dcgm_exporter"]["port"] == 19401

    assert recipe["benchmark"]["concurrencies"] == "1x2x4x8x16x32x64x128"
    assert recipe["resources"]["prefill_nodes"] == 1
    assert recipe["resources"]["decode_nodes"] == 1
    assert recipe["resources"]["gpus_per_node"] == 4


def test_launcher_detects_power_lane_from_recipe():
    assert_recipe_driven_detection(LAUNCHER_PATH.read_text())


def test_launcher_provisions_exporter_through_shared_squash_path():
    launcher = LAUNCHER_PATH.read_text()
    assert_exporter_provisioning(launcher)
    assert 'INFERENCEX_CACHE_ROOT:-/data/home/sa-shared/gharunners' in launcher
    assert 'DCGM_EXPORTER_SQSH="${SQUASH_DIR}/' in launcher
    assert 'srun --partition="$SLURM_PARTITION" --account="$SLURM_ACCOUNT" --exclusive --time=30' in launcher


def test_launcher_preserves_official_cluster_defaults_with_optional_overrides():
    launcher = LAUNCHER_PATH.read_text()
    assert 'SLURM_PARTITION:-batch_1' in launcher
    assert 'SLURM_ACCOUNT:-benchmark' in launcher
    assert 'MODEL_PATH_OVERRIDE:-$MODEL_PATH' in launcher
    assert 'SQUASH_DIR:-${INFERENCEX_CACHE_ROOT}/squash' in launcher


def test_launcher_maps_registry_and_enroot_spellings_to_the_same_squash():
    launcher = LAUNCHER_PATH.read_text()
    assert 'if [[ "$IMAGE" == *"#"* ]]' in launcher
    assert 'REGISTRY_IMAGE_ALIAS="  \\"${IMAGE//#//}\\": ${SQUASH_FILE}"' in launcher
    assert "${REGISTRY_IMAGE_ALIAS}" in launcher


def test_launcher_pins_power_producer():
    assert_pinned_clone_contract(LAUNCHER_PATH.read_text())


def test_non_power_lane_keeps_existing_ref_logic():
    launcher = LAUNCHER_PATH.read_text()
    assert 'git clone https://github.com/NVIDIA/srt-slurm.git "$SRT_REPO_DIR"' in launcher
    assert "git checkout v1.0.25" in launcher
    assert "git checkout sa-submission-q2-2026" in launcher


def test_launcher_supplies_arm64_uv_to_older_srt_slurm_jobs():
    launcher = LAUNCHER_PATH.read_text()
    assert "uv-aarch64-unknown-linux-gnu.tar.gz" in launcher
    assert "file \"$SRT_REPO_DIR/bin/uv\" | grep -q 'ARM aarch64'" in launcher
    assert "export PATH=\"${SRTCTL_SOURCE}/bin:$PATH\"" in launcher
    assert 'export UV_PROJECT_ENVIRONMENT="${SRTCTL_SOURCE}/.venv-compute"' in launcher
    assert "uv run --python 3.12 --no-dev --no-sync -m" in launcher


def test_launcher_prefetches_arm64_compute_environment_on_login_node():
    launcher = LAUNCHER_PATH.read_text()
    assert 'SRT_NEEDS_COMPUTE_ENV_PREFETCH=1' in launcher
    assert 'UV_PROJECT_ENVIRONMENT="$SRT_REPO_DIR/.venv-compute"' in launcher
    assert "uv sync --python /usr/bin/python3.12" in launcher
    assert "--python-platform aarch64-unknown-linux-gnu --no-dev" in launcher


def test_nkx_dsv4_lane_reuses_validated_cluster_contract():
    launcher = LAUNCHER_PATH.read_text()

    assert "deb1dfd9934398664f92d194169c183e009da83b" in launcher
    assert "deepseek-ai--DeepSeek-V4-Pro-ed9e8d533b48" in launcher
    assert "ed9e8d533b4866d9c92ba28f968d1905339bf0a3be5e1dcb5b506c88928318fa" in launcher
    assert "864739867846" in launcher
    assert 'SRT_UCX_TLS:-rc,cuda_ipc,cuda_copy,sm,self,tcp' in launcher
    assert 'SRT_TRTLLM_ENABLE_PDL:-0' in launcher
    assert 'NCCL_NET_PLUGIN: \\"none\\"' in launcher
    assert 'NVSHMEM_ENABLE_NIC_PE_MAPPING: \\"1\\"' in launcher
    assert "rocep161s0:1" in launcher
    assert "rocep202s0:1" in launcher
    assert '"$GITHUB_WORKSPACE/runners/preflight_gb300_nkx.sh"' in launcher


def test_nkx_preflight_checks_all_nodes_before_full_benchmark():
    preflight = PREFLIGHT_PATH.read_text()

    assert 'expected_nodes="${NKX_EXPECTED_GPU_NODES:-16}"' in preflight
    assert 'scontrol show topology' in preflight
    assert '/sys/class/infiniband/$hca/ports/1/link_layer' in preflight
    assert 'test -x /usr/bin/nvidia-imex-ctl' in preflight
    assert 'NCCL_NET_PLUGIN=none' in preflight
    assert 'collective=ok' in preflight
    assert 'GDRDMA' in preflight

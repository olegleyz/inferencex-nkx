#!/usr/bin/env bash
set -euo pipefail

readonly overlay="${INFERENCEX_UCX_OVERLAY:?INFERENCEX_UCX_OVERLAY is required}"
readonly image="${INFERENCEX_UCX_CANARY_IMAGE:?INFERENCEX_UCX_CANARY_IMAGE is required}"
readonly log_dir="${INFERENCEX_UCX_CANARY_LOG_DIR:?INFERENCEX_UCX_CANARY_LOG_DIR is required}"
readonly perftest="$overlay/bin/ucx_perftest"
readonly library_path="$overlay/lib:/usr/local/lib/python3.12/dist-packages/.nixl_cu13.mesonpy.libs"
readonly module_path="$overlay/lib/ucx"
readonly cross_host_size="${INFERENCEX_UCX_CANARY_CROSS_HOST_SIZE:-1048576}"
readonly iterations="${INFERENCEX_UCX_CANARY_ITERATIONS:-500}"
readonly warmup_iterations="${INFERENCEX_UCX_CANARY_WARMUP_ITERATIONS:-50}"

[[ "$cross_host_size" =~ ^[1-9][0-9]*$ ]] || {
    echo "INFERENCEX_UCX_CANARY_CROSS_HOST_SIZE must be a positive integer" >&2
    exit 2
}
[[ "$iterations" =~ ^[1-9][0-9]*$ ]] || {
    echo "INFERENCEX_UCX_CANARY_ITERATIONS must be a positive integer" >&2
    exit 2
}
[[ "$warmup_iterations" =~ ^[0-9]+$ ]] || {
    echo "INFERENCEX_UCX_CANARY_WARMUP_ITERATIONS must be a non-negative integer" >&2
    exit 2
}

mapfile -t nodes < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
[[ ${#nodes[@]} -eq 2 ]] || {
    echo "expected exactly two allocated nodes, found ${#nodes[@]}" >&2
    exit 2
}
mkdir -p "$log_dir"

run_pair() {
    local label="$1" server_node="$2" client_node="$3" port="$4"
    local transports="$5" size="$6"
    local server_log="$log_dir/${label}.server.log"
    local client_log="$log_dir/${label}.client.log"
    local prefix
    local -a common=(
        --nodes=1 --ntasks=1 --cpus-per-task=1 --gres=gpu:1 --exact
        --container-image="$image" --container-mounts=/scratch/fsw:/scratch/fsw
    )

    printf -v prefix \
        'export LD_LIBRARY_PATH=%q UCX_MODULE_DIR=%q UCX_TLS=%q UCX_LOG_LEVEL=info UCX_PROTO_INFO=y;' \
        "$library_path" "$module_path" "$transports"
    srun "${common[@]}" --nodelist="$server_node" \
        bash -lc "$prefix timeout 180s '$perftest' -t tag_bw -m cuda -s '$size' -n '$iterations' -w '$warmup_iterations' -p '$port' -v" \
        >"$server_log" 2>&1 &
    local server_pid=$!
    sleep 3

    printf -v prefix \
        'export LD_LIBRARY_PATH=%q UCX_MODULE_DIR=%q UCX_TLS=%q UCX_LOG_LEVEL=info UCX_PROTO_INFO=y;' \
        "$library_path" "$module_path" "$transports"
    local client_rc=0 server_rc=0
    timeout 190s srun "${common[@]}" --nodelist="$client_node" \
        bash -lc "$prefix timeout 180s '$perftest' '$server_node' -t tag_bw -m cuda -s '$size' -n '$iterations' -w '$warmup_iterations' -p '$port' -v" \
        >"$client_log" 2>&1 || client_rc=$?
    wait "$server_pid" || server_rc=$?
    [[ $client_rc -eq 0 && $server_rc -eq 0 ]] || {
        echo "$label failed: client=$client_rc server=$server_rc" >&2
        return 1
    }
}

srun --overlap --nodes=1 --ntasks=1 --nodelist="${nodes[0]}" \
    --container-image="$image" --container-mounts=/scratch/fsw:/scratch/fsw \
    bash -lc "export LD_LIBRARY_PATH='$library_path' UCX_MODULE_DIR='$module_path'; '$overlay/bin/ucx_info' -v" \
    >"$log_dir/runtime.log" 2>&1

# Exercise two local GPU pairs, then all four cross-node GPU pairs. Native
# policy excludes TCP; logs must prove cuda_ipc locally and rc_mlx5 remotely.
pair_pids=()
for pair in 0 1; do
    run_pair "same-host-$pair" "${nodes[0]}" "${nodes[0]}" "$((14337 + pair))" \
        '^tcp' 8388608 &
    pair_pids+=("$!")
done
for pid in "${pair_pids[@]}"; do
    wait "$pid"
done

pair_pids=()
for gpu in 0 1 2 3; do
    run_pair "cross-host-$gpu" "${nodes[0]}" "${nodes[1]}" "$((15337 + gpu))" \
        'rc,cuda' "$cross_host_size" &
    pair_pids+=("$!")
done
for pid in "${pair_pids[@]}"; do
    wait "$pid"
done

{
    echo "nodes=${nodes[*]}"
    echo "cross_host_size=$cross_host_size iterations=$iterations warmup_iterations=$warmup_iterations"
    grep -hE 'Library version:|UCX  INFO.*Version|cuda_ipc|rc_mlx5|FINAL' \
        "$log_dir"/*.log || true
} >"$log_dir/summary.log"
grep -q '1.22.0' "$log_dir/runtime.log"
grep -q 'cuda_ipc' "$log_dir/summary.log"
grep -q 'rc_mlx5' "$log_dir/summary.log"
if grep -Eqi 'Bad address|NIXL_ERR_BACKEND|CUDA_ERROR_|Segmentation fault' "$log_dir"/*.log; then
    echo "native-RoCE canary found a fatal transport signature" >&2
    exit 1
fi
echo "native-RoCE transport canary passed: $log_dir"

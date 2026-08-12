#!/usr/bin/env bash

set -euo pipefail

: "${SLURM_ACCOUNT:?SLURM_ACCOUNT is required}"
: "${SLURM_PARTITION:?SLURM_PARTITION is required}"
: "${NKX_MODEL_PATH:?NKX_MODEL_PATH is required}"
: "${NKX_IMAGE_SQUASH:?NKX_IMAGE_SQUASH is required}"

readonly expected_nodes="${NKX_EXPECTED_GPU_NODES:-16}"
readonly expected_model_bytes="${NKX_EXPECTED_MODEL_BYTES:-864739867846}"
readonly expected_model_digest="${NKX_EXPECTED_MODEL_DIGEST:-ed9e8d533b4866d9c92ba28f968d1905339bf0a3be5e1dcb5b506c88928318fa}"
readonly expected_hcas="${NKX_EXPECTED_HCAS:-rocep161s0 rocep162s0 rocep172s0 rocep173s0 rocep190s0 rocep191s0 rocep201s0 rocep202s0}"
readonly log_dir="${NKX_PREFLIGHT_LOG_DIR:-${GITHUB_WORKSPACE:-$PWD}/nkx-preflight}"

mkdir -p "$log_dir"

topology="$(scontrol show topology)"
topology_blocks="$(grep -c '^BlockName=' <<<"$topology")"
if [[ "$topology_blocks" -ne 1 ]]; then
  echo "Expected one GB300 topology block, found ${topology_blocks}:" >&2
  echo "$topology" >&2
  exit 1
fi
topology_nodes="$(sed -nE 's/.* Nodes=([^ ]+).*/\1/p' <<<"$topology")"
if [[ "$(scontrol show hostnames "$topology_nodes" | wc -l | tr -d ' ')" -ne "$expected_nodes" ]]; then
  echo "Topology block does not contain the expected ${expected_nodes} GPU nodes:" >&2
  echo "$topology" >&2
  exit 1
fi
echo "Validated one topology block containing ${expected_nodes} GPU nodes."

worker_check=$(cat <<'EOF'
set -euo pipefail
test "$(uname -m)" = aarch64
test "$(nvidia-smi -L | wc -l)" -eq 4
test -r "$NKX_MODEL_PATH/config.json"
test -r "$NKX_MODEL_PATH/model.safetensors.index.json"
test "$(cat "$NKX_MODEL_PATH/.benchops-model-digest")" = "$NKX_EXPECTED_MODEL_DIGEST"
test "$(du -sb "$NKX_MODEL_PATH" | awk '{print $1}')" = "$NKX_EXPECTED_MODEL_BYTES"
test "$(df -PB1 /scratch | awk 'NR == 2 {print $4}')" -gt 2000000000000
for hca in $NKX_EXPECTED_HCAS; do
  test "$(cat "/sys/class/infiniband/$hca/ports/1/link_layer")" = Ethernet
  test "$(cat "/sys/class/infiniband/$hca/ports/1/state" | awk '{print $2}')" = ACTIVE
done
test -x /usr/bin/nvidia-imex-ctl
test "$(nvidia-smi -q | grep -c 'GPU Recovery Action[[:space:]]*: None')" -eq 4
test "$(nvidia-smi --query-gpu=ecc.errors.uncorrected.volatile.total --format=csv,noheader,nounits | awk '$1 != 0 {bad++} END {print bad + 0}')" -eq 0
printf '%s host=ok model=ok gpus=4 roce=8 scratch_free=%s\n' \
  "$(hostname)" "$(df -h /scratch | awk 'NR == 2 {print $4}')"
EOF
)

export NKX_MODEL_PATH NKX_EXPECTED_MODEL_DIGEST="$expected_model_digest"
export NKX_EXPECTED_MODEL_BYTES="$expected_model_bytes" NKX_EXPECTED_HCAS="$expected_hcas"

if [[ "${NKX_SKIP_HOST_MODEL:-0}" != 1 ]]; then
  echo "Validating host, model, storage, GPU, RoCE, and IMEX state on ${expected_nodes} nodes..."
  srun \
  --account="$SLURM_ACCOUNT" \
  --partition="$SLURM_PARTITION" \
  --nodes="$expected_nodes" \
  --ntasks="$expected_nodes" \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --exclusive \
  --time=00:08:00 \
  --export=ALL \
    bash -c "$worker_check" </dev/null | tee "$log_dir/host-model.log"

  test "$(grep -c 'host=ok model=ok gpus=4 roce=8' "$log_dir/host-model.log")" -eq "$expected_nodes"
fi

runtime_check=$(cat <<'EOF'
set -euo pipefail
test "$(uname -m)" = aarch64
test "$(nvidia-smi -L | wc -l)" -eq 4
python3 - <<'PY'
import torch
import tensorrt_llm

assert torch.cuda.is_available()
assert torch.cuda.device_count() == 4
print(f"runtime=ok torch={torch.__version__} cuda={torch.version.cuda} trtllm={tensorrt_llm.__version__}")
PY
EOF
)

if [[ "${NKX_SKIP_RUNTIME:-0}" != 1 ]]; then
  echo "Validating the exact TensorRT-LLM image on ${expected_nodes} nodes..."
  srun \
  --account="$SLURM_ACCOUNT" \
  --partition="$SLURM_PARTITION" \
  --nodes="$expected_nodes" \
  --ntasks="$expected_nodes" \
  --ntasks-per-node=1 \
  --gpus-per-node=4 \
  --exclusive \
  --time=00:08:00 \
  --container-image="$NKX_IMAGE_SQUASH" \
  --no-container-entrypoint \
  --no-container-mount-home \
    bash -c "$runtime_check" </dev/null | tee "$log_dir/runtime.log"

  test "$(grep -c 'runtime=ok' "$log_dir/runtime.log")" -eq "$expected_nodes"
fi

collective_check=$(cat <<'EOF'
set -euo pipefail
export MASTER_PORT=29641
export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0
export NCCL_SOCKET_IFNAME=eth0
export NCCL_IB_HCA='=rocep161s0,rocep162s0,rocep172s0,rocep173s0,rocep190s0,rocep191s0,rocep201s0,rocep202s0'
export NCCL_NET_PLUGIN=none
python3 - <<'PY'
import os
import torch
import torch.distributed as dist

rank = int(os.environ["SLURM_PROCID"])
world = int(os.environ["SLURM_NTASKS"])
dist.init_process_group(
    "nccl",
    init_method=f"tcp://{os.environ['MASTER_ADDR']}:{os.environ['MASTER_PORT']}",
    rank=rank,
    world_size=world,
)
torch.cuda.set_device(0)
value = torch.tensor([rank + 1.0], device="cuda")
dist.all_reduce(value)
expected = world * (world + 1) / 2
assert value.item() == expected, (rank, value.item(), expected)
torch.cuda.synchronize()
dist.destroy_process_group()
print(f"rank={rank} collective=ok")
PY
EOF
)

echo "Running an exact-image all-node NCCL/RoCE collective canary..."
export MASTER_ADDR
MASTER_ADDR="$(sinfo --Node --noheader --partition="$SLURM_PARTITION" --format='%N' | sort -u | head -1)"
test -n "$MASTER_ADDR"
srun \
  --account="$SLURM_ACCOUNT" \
  --partition="$SLURM_PARTITION" \
  --nodes="$expected_nodes" \
  --ntasks="$expected_nodes" \
  --ntasks-per-node=1 \
  --gpus-per-node=1 \
  --exclusive \
  --time=00:08:00 \
  --export=ALL,MASTER_ADDR \
  --container-image="$NKX_IMAGE_SQUASH" \
  --no-container-entrypoint \
  --no-container-mount-home \
  bash -c "$collective_check" </dev/null 2>&1 | tee "$log_dir/collective.log"

test "$(grep -c 'collective=ok' "$log_dir/collective.log")" -eq "$expected_nodes"
grep -Eq 'NET/IB|NET.*IB' "$log_dir/collective.log"
grep -q 'GDRDMA' "$log_dir/collective.log"
if grep -Eq 'NET/Socket' "$log_dir/collective.log"; then
    echo "NCCL collective used the socket fallback instead of RoCE" >&2
    exit 1
fi
if grep -Eq 'libnccl-net-ofi|NET/Plugin: Could not find: ofi' "$log_dir/collective.log"; then
    echo "NCCL attempted to load the AWS OFI plugin on a ConnectX RoCE cluster" >&2
    exit 1
fi

echo "NKX GB300 preflight passed on ${expected_nodes} nodes."

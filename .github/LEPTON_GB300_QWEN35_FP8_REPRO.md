# Lepton GB300 Qwen3.5 FP8 reproduction

This document records the launch contract for running the native InferenceX
Qwen3.5 397B FP8 Dynamo-SGLang matrix on the isolated `lepton-gb300` target.
It applies only to Kubernetes context `gb300l`, Slurm cluster
`nkx-slinky-dev-02`, and partition `batch`; it must never route work to the
separate NKX-managed `gb300` cluster.

## Immutable model and runtime inputs

| Input | Value |
| --- | --- |
| InferenceX recipe baseline | `d089a9138c53d16c6388e4251a078fee8ca7bea6` |
| Model | `Qwen/Qwen3.5-397B-A17B-FP8` |
| Model revision | `ea5b4f81096f3901c91dea97f81324302495781d` |
| Model manifest | `sha256:3bc4f54f36992c01270313ae4a6b75cadfec4d58f6874282acbd6ada24ab7502` |
| Model bytes / files | `406,198,638,888` / `107` |
| Image | `lmsysorg/sglang:nightly-dev-cu13-20260709-074bb928` |
| Lepton SquashFS SHA-256 | `e3c9ce21fa6f0f363dfb1fb4aff7a3b8dd55803b0c214c649719919a75b7bbe9` |
| srt-slurm commit | `3435776cd6db4c14f8b771ff7a3976deb62fe133` (`v1.0.25`) |
| Dynamo source commit | `46520ca59afe992fb5ef61b3197b2316f8df9b2b` |
| Scenario | fixed sequence length, 8192 input / 1024 output tokens |
| Speculative decoding | none |

The checked-in recipes and `nvidia-master.yaml` configuration are the same
native InferenceX Qwen contract used for the NKX reproduction. Cluster inputs
are applied after recipe selection and remain visible in the resolved recipe,
generated Slurm script, environment artifact, and logs.

## Native matrix

| Recipe | Topology | Concurrencies | Nodes / GPUs |
| --- | --- | --- | --- |
| `1p1d-tp4-tp4.yaml` | 1 prefill TP4 + 1 decode TP4 | 1, 2, 4, 8, 16, 32, 64, 128 | 2 / 8 |
| `4p1d-dep4-dep16.yaml` | 4 prefill DEP4 + 1 decode DEP16 | 1024 | 8 / 32 |
| `8p1d-dep4-dep16.yaml` | 8 prefill DEP4 + 1 decode DEP16 | 2048, 4096 | 12 / 48 |

The normal generator retains InferenceX's native throughput and evaluation
split. Readiness alone selects the existing 1P1D recipe at concurrency 1,
disables evaluations, and sends one formal 8K/1K request.

## Explicit Lepton inputs

| Difference from upstream | Effective value | Concrete reason/evidence |
| --- | --- | --- |
| Private runner/target | `gb300l-nv`, `lepton-gb300` | Routes GitHub Actions only to the Lepton login runner. |
| Slurm identity | cluster `nkx-slinky-dev-02`, account `nvidia`, partition `batch` | Lepton scheduler configuration; verified before submission. |
| Shared storage | `/scratch/fsw/users/oleizerov/...` | Lepton-visible FSx cache and artifact roots. |
| Node-local model | `/raid/scratch/benchops-oleizerov/models/Qwen3.5-397B-A17B-FP8-ea5b4f81096f` | Revision-qualified path verified by a committed ModelStage receipt. |
| CPUs per task | no Qwen override | Completed Lepton reference job `513` had no `cpus-per-task` directive; DeepSeek's value `140` is not inherited. |
| Dynamo lease TTL | `600` seconds | Effective frontend and worker value in completed Lepton Qwen job `513`. |
| Health attempts | `720` at 10 seconds | Known-good Lepton slow-start allowance. |
| DeepSeek fabric overlay | absent | DeepSeek's vLLM-specific NVSHMEM HCA mapping is not injected into the Qwen SGLang recipe. |
| Image identity | Lepton SquashFS checksum above | The Lepton cache artifact is independently pinned; its SquashFS byte hash differs from the NKX cache artifact for the same image tag. |

No Qwen engine argument, topology, traffic shape, Mooncake/MNNVL setting, or
Dynamo revision is rewritten by the target adapter.

## Scheduler availability and staging policy

At the initial staging plan on 2026-08-14, Slurm reported 15 active GPU
workers, `ip-100-64-172-86` as `drained`, and `ip-100-64-207-81` as
`reserved`. No manual node exclusion was added. BenchOps commit `24b9b3d`
defines `targets.gpuNodes: ALL` as every active GPU worker, while the
InferenceX preflight independently rejects an active partition worker missing
from the receipt.

The receipt minimum is 12 workers because the largest native Qwen topology is
12 nodes. The actual receipt must still cover every active worker at dispatch
time. A drained or reserved node becoming active therefore blocks launch until
it has also been staged and verified.

## Prior Lepton reference ranges

The prior reviewed BenchOps records used the same model revision, image,
recipes, topology, traffic, and source baseline. Their output-token throughput
ranges are comparison evidence, not substitutes for native artifacts:

| Configuration | Concurrency | Prior output tokens/s |
| --- | ---: | ---: |
| 1P1D TP4/TP4 | 1 | 184.43–189.27 |
| 1P1D TP4/TP4 | 2 | 353.75–360.44 |
| 1P1D TP4/TP4 | 4 | 602.40–617.92 |
| 1P1D TP4/TP4 | 8 | 1,027.18–1,038.32 |
| 1P1D TP4/TP4 | 16 | 1,668.11–1,668.59 |
| 1P1D TP4/TP4 | 32 | 2,577.83–2,585.93 |
| 1P1D TP4/TP4 | 64 | 4,038.62–4,059.10 |
| 1P1D TP4/TP4 | 128 | 6,122.30–6,220.42 |
| 4P1D DEP4/DEP16 | 1024 | 34,246.44–34,281.87 |
| 8P1D DEP4/DEP16 | 2048 | 62,751.27–63,163.30 |
| 8P1D DEP4/DEP16 | 4096 | 66,705.05–66,819.79 |

## Model staging evidence

| Phase | Identifier | Result/evidence |
| --- | --- | --- |
| Declaration | BenchOps `afd9db7` | Pins the Lepton cluster, immutable HF revision, shared FSx root, `/raid` node-local root, all-active-worker policy, and 1 TiB reserve. |
| Active-node fix | BenchOps `24b9b3d` | Excludes only unavailable Slurm states from `ALL`; tests cover idle, allocated, drained, reserved, and non-GPU nodes. |
| Aggregate time-limit fix | BenchOps `b18c62d` | Adds the explicit four-hour limit already used by individual staging operations. |
| Initial copy | Slurm `1078` | Cancelled after discovering the previous 31-minute implicit limit; 4 atomic copies completed, while partial copies were never published. |
| Corrected copy | Slurm `1081` | Ready on 15/15 active workers: 11 copied, 4 reused, zero failed. |
| Independent audit | Slurm `1082` | Ready on 15/15; every copy reused only after full SHA-256 verification against manifest `sha256:3bc4f54f36992c01270313ae4a6b75cadfec4d58f6874282acbd6ada24ab7502`. |

Committed receipt:

```text
model-stages/receipts/qwen3.5-397b-a17b-fp8-lepton-gb300-ea5b4f81096f.json
```

## Dispatch contract

Focused readiness generator command:

```text
full-sweep --config-files configs/nvidia-master.yaml --runner-config configs/runners.yaml --multi-node --config-file recipes/sglang/qwen3.5/gb300-fp8/8k1k/1p1d-tp4-tp4.yaml --min-conc 1 --max-conc 1 --no-evals
```

Set `target-cluster=lepton-gb300`, `readiness-only=true`, and the committed
Lepton Qwen full-verification receipt. Do not dispatch until the receipt is
ready on all active workers.

After readiness succeeds, use the native matrix generator without
`--no-evals`:

```text
test-config --config-files configs/nvidia-master.yaml --runner-config configs/runners.yaml --config-keys qwen3.5-fp8-gb300-dynamo-sglang
```

Run IDs, Slurm IDs, artifact links, validation results, and comparisons are
added here as execution completes.

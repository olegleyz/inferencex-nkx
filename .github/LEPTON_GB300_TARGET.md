# Lepton GB300 InferenceX target

This fork supports the Lepton-managed Slurm cluster as an isolated benchmark
target named `lepton-gb300`. It is separate from `nkx-gb300`; selecting one
target cannot silently fall through to the other's runner, Slurm controller,
partition, storage, model receipt, or RoCE device mapping.

## Target contract

| Input | Value |
| --- | --- |
| Target | `lepton-gb300` |
| GitHub runner label | `gb300l-nv` |
| Registered runner | `gb300l-nv_00` |
| Slurm cluster | `nkx-slinky-dev-02` |
| Partition/account | `batch` / `nvidia` |
| CPUs per task | 140 |
| Shared root | `/scratch/fsw` |
| Node-local model | `/raid/scratch/benchops-oleizerov/models/DeepSeek-V4-Pro-b5968e9190ef` |
| Profile | `runners/cluster_profiles/lepton-gb300.sh` |

The runner uses Actions runner `2.336.0`, Linux x64 archive SHA-256
`04cf0be1aff4c3ec3554466c39124ca250e3effd8873bb7e8d68535aa9505d5d`.
Its registration and work directory are under the persistent Lepton user home:

```text
/scratch/fsw/users/oleizerov/.github-actions/gb300l-nv
```

The Lepton `LoginSet` starts and gracefully stops this non-root runner through
the source-controlled `start_github_actions_runner.sh` and
`stop_github_actions_runner.sh` helpers. The runner has Lepton-specific labels
only; it does not carry `gb300-nv` or `cluster:gb300-nkx`.

The live `nkx-slinky-dev-02-slurm-login-default` `LoginSet` has the following
idempotent lifecycle contract. The user fallback is necessary because the
Kubernetes `postStart` hook can run before SSSD resolves the user; the runner
itself still runs as UID 1763, never as root.

```yaml
postStart:
  exec:
    command:
      - /bin/bash
      - -lc
      - >-
        getent passwd oleizerov >/dev/null ||
        useradd --no-create-home --uid 1763 --gid 65534
        --home-dir /scratch/fsw/users/oleizerov --shell /bin/bash oleizerov;
        runuser -u oleizerov --
        /scratch/fsw/users/oleizerov/.github-actions/gb300l-nv/start-runner.sh
        || true; exit 0
preStop:
  exec:
    command:
      - /bin/bash
      - -lc
      - >-
        runuser -u oleizerov --
        /scratch/fsw/users/oleizerov/.github-actions/gb300l-nv/stop-runner.sh
        || true; exit 0
```

Runner registration is deliberately separate from pod startup because GitHub
registration tokens are short-lived. Install the pinned archive into the
persistent directory, verify the SHA-256 above, obtain a fresh repository
registration token, and configure the runner once with name `gb300l-nv_00`,
work directory `_work`, and labels
`slurm,gb300l-nv,cluster:gb300-lepton,gb300l`. Never store the registration
token in Git, the `LoginSet`, or the runner scripts.

## Fail-closed behavior

For a target-specific dispatch, the workflow and launcher require all of the
following before model preflight or benchmark submission:

1. the source-controlled target map resolves `lepton-gb300` to runner label
   `gb300l-nv` and profile `lepton-gb300`;
2. generated work is limited to the six reviewed DeepSeek V4 FP4
   Dynamo-vLLM recipes and the known-good image;
3. `scontrol show config` reports `ClusterName=nkx-slinky-dev-02`;
4. Slurm partition `batch` is visible;
5. the receipt identifies `nkx-slinky-lepton-gb300-dev-02`, the immutable
   model revision, Lepton paths, and Lepton manifest digest;
6. the receipt uses full SHA-256 verification and covers every currently
   active worker in the target partition;
7. one read-only validation task succeeds on every receipt node.

Any mismatch exits before `srtctl apply` or `sbatch`.

## Connectivity check

The `Cluster Target Connectivity` workflow verifies runner routing and Slurm
identity without allocating GPUs or submitting a benchmark. Run this before a
readiness or performance workflow after runner or cluster maintenance.

## Model readiness status

BenchOps commit `c1b99a1b48bcec7e9d5af27b7922c11a80115010` excludes
unavailable Slurm states from `gpuNodes: ALL` and uses a single all-target
allocation. Slurm job `1040` performed a complete SHA-256 audit of the 16
schedulable workers; all 16 copies matched and were reused. Worker
`ip-100-64-207-81` was in the separate `reserved` state and was not eligible.
The accepted receipt is committed at
`model-stages/receipts/deepseek-v4-pro-lepton-gb300-b5968e9190ef.json`.

If the reserved worker becomes active, workflow coverage validation will fail
until a new full-SHA receipt includes it. This is intentional.

## Native matrix

After receipt acceptance and a successful readiness point, use the same native
InferenceX selector used on NKX:

```bash
python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/nvidia-master.yaml \
  --config-keys dsv4-fp4-gb300-dynamo-vllm \
  --seq-lens 8k1k \
  --no-evals
```

Dispatch `e2e-tests.yml` with `target-cluster=lepton-gb300` and the committed
Lepton receipt. The target map selects that receipt automatically when the
optional receipt input is empty; an explicit override still has to pass the
same target-aware identity checks. The target changes only runner, scheduler/storage inputs, and
the Lepton RoCE mapping; it retains the original recipes and native matrix,
Slurm launcher, result processing, and artifact collection.

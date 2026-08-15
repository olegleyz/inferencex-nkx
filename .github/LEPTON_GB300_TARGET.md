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
| Registered runners | `gb300l-nv_00`, `gb300l-nv_01`, `gb300l-nv_02` |
| Slurm cluster | `nkx-slinky-dev-02` |
| Partition/account | `batch` / `nvidia` |
| CPUs per task | 140 |
| Slurm time limit | `04:00:00` (Lepton `batch` maximum) |
| Shared root | `/scratch/fsw` |
| Node-local model | `/raid/scratch/benchops-oleizerov/models/DeepSeek-V4-Pro-b5968e9190ef` |
| Profile | `runners/cluster_profiles/lepton-gb300.sh` |

The runner uses Actions runner `2.336.0`, Linux x64 archive SHA-256
`04cf0be1aff4c3ec3554466c39124ca250e3effd8873bb7e8d68535aa9505d5d`.
Each registration and work directory is isolated under the persistent Lepton
user home. The original `_00` listener retains its legacy directory so adding
the pool does not disturb existing jobs:

```text
/scratch/fsw/users/oleizerov/.github-actions/gb300l-nv     # gb300l-nv_00
/scratch/fsw/users/oleizerov/.github-actions/gb300l-nv_01  # gb300l-nv_01
/scratch/fsw/users/oleizerov/.github-actions/gb300l-nv_02  # gb300l-nv_02
```

The Lepton `LoginSet` starts and gracefully stops all three non-root listeners
through the source-controlled pool helpers. Each listener has the same
Lepton-specific routing labels and a unique runner name, work directory, PID,
log, and Slurm job name. None carries `gb300-nv` or `cluster:gb300-nkx`.

This matches upstream InferenceX's GB300 scheduling model: GitHub expands the
matrix into independent jobs, one listener submits one `srt-slurm` job and
waits for it, and Slurm decides which submitted jobs can run concurrently.
Three listeners therefore permit up to three matrix points to be submitted at
once; they do not bypass Slurm placement or capacity decisions.

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

Those legacy hook paths are persistent symbolic links to
`../start-gb300l-runner-pool.sh` and `../stop-gb300l-runner-pool.sh`. This
preserves the existing `LoginSet` specification and avoids a login-pod rollout
when the pool size changes. The generic single-listener helpers live beside
the pool scripts in `.github-actions/` and receive each isolated runner root
through `ACTIONS_RUNNER_ROOT`.

Runner registration is deliberately separate from pod startup because GitHub
registration tokens are short-lived. Install the pinned archive into the
persistent directory, verify the SHA-256 above, obtain a fresh repository
registration token, and configure each runner root once with its corresponding
name (`gb300l-nv_00`, `gb300l-nv_01`, or `gb300l-nv_02`), work directory
`_work`, and labels
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
8. every intended srt-slurm worker log is free of startup-fatal signatures;
   a surviving subset cannot publish a benchmark result as the requested
   topology.

Any mismatch exits before `srtctl apply` or `sbatch`.

The worker-log postcondition runs after Slurm completion and before result
collection. It was added after Lepton Slurm job `1063` lost one prefill worker
to GPU XID 94 (`ROBUST_CHANNEL_CONTAINED_ERROR`) while Dynamo's aggregate
endpoint count still allowed the benchmark to start. Slurm automatically
drained the affected node; benchmark tooling never undrains or repairs nodes.

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

## Completed DeepSeek V4 native matrix

The six-point DeepSeek V4 FP4 Dynamo-vLLM matrix completed on August 13-14,
2026. All points used the immutable model receipt above, image
`vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-4ba0a72`, native InferenceX
SA-Bench traffic, the checked-in srt-slurm recipes, and Lepton target
`gb300l-nv`. Results are split across the original matrix run and two
point-specific retries; every row records its exact source commit.

| Topology | Recipe | Conc. | Requests | Output tok/s | GitHub run/job | Slurm | Commit |
| --- | --- | ---: | ---: | ---: | --- | ---: | --- |
| 1P9D | `disagg-gb300-1p9d-tep4-tp4.yaml` | 18 | 180 | 1,289.99 | `31755963110` / `94631697897` | `1055` | `d92139a012969fa8f9fccf9994d88d89242b00e4` |
| 1P6D | `disagg-gb300-1p6d-dep4-tp4.yaml` | 192 | 1,920 | 6,665.76 | `31770235139` / `94674490288` | `1075` | `33688cffadfb9eff319d26f5cb348c5eb1d1d457` |
| 4P1D | `disagg-gb300-4p1d-dep4-dep8-24-c4096.yaml` | 4,096 | 40,960 | 30,291.90 | `31770236419` / `94674485533` | `1071` | `33688cffadfb9eff319d26f5cb348c5eb1d1d457` |
| 5P1D | `disagg-gb300-5p1d-dep4-dep8-28-c4096.yaml` | 4,096 | 40,960 | 37,835.49 | `31755963110` / `94631697888` | `1059` | `d92139a012969fa8f9fccf9994d88d89242b00e4` |
| 6P1D | `disagg-gb300-6p1d-dep4-dep8-32-c4096.yaml` | 4,096 | 40,960 | 44,759.77 | `31755963110` / `94631697946` | `1067` | `d92139a012969fa8f9fccf9994d88d89242b00e4` |
| 7P2D | `disagg-gb300-7p2d-dep4-dep16.yaml` | 3,072 | 30,720 | 50,858.75 | `31755963110` / `94631697896` | `1051` | `d92139a012969fa8f9fccf9994d88d89242b00e4` |

GitHub run URLs use
`https://github.com/olegleyz/inferencex-nkx/actions/runs/<RUN_ID>`. Each valid
job uploaded its native processed result, nested server-log archive, resolved
srt-slurm runtime configuration, and model-stage provenance. The four valid
jobs from the original matrix were independently scanned after extracting the
nested `multinode_server_logs.tar.gz`; the two retries passed both the online
post-Slurm guard and a second scan of their downloaded artifacts.

### Rejected attempts and corrective actions

| Attempt | Failure | Corrective action |
| --- | --- | --- |
| Original 4P1D, Slurm `1063` | GPU XID 94 on `ip-100-64-172-86` killed one prefill worker, but aggregate Dynamo endpoint counting let the benchmark continue with a degraded topology. | Treat the result as invalid, leave the automatically drained node untouched, add the fail-closed worker-log validator, and retry only 4P1D as Slurm `1071` on healthy nodes. |
| Original 1P6D job `94631697922` | The workflow definition came from `d92139a`, but its mutable branch checkout advanced to `bd2db8f`; the two revisions disagreed on the model-receipt validator arguments. No Slurm benchmark was submitted. | Pin `inputs.ref` to the full immutable commit and retry only 1P6D as Slurm `1075`. |
| First retry dispatches `31770196840` and `31770198110` | A manually expanded short SHA was not a real commit. | Cancel both runs before allocation and redispatch with the value returned by `git rev-parse HEAD`. |

The completed retries are GitHub runs
[`31770236419`](https://github.com/olegleyz/inferencex-nkx/actions/runs/31770236419)
and
[`31770235139`](https://github.com/olegleyz/inferencex-nkx/actions/runs/31770235139).

# NKX GB300 customizations and failure history

This document records the behavior added around the original InferenceX
DeepSeek V4 Dynamo-vLLM recipes, together with the failures encountered while
establishing the reproducible NKX execution path. It applies only to
`nkx-slinky-gb300-dev-01` (Kubernetes context `gb300`), not `gb300l`.

The accepted six-point run used GitHub Actions run
[`31665151409`](https://github.com/olegleyz/inferencex-nkx/actions/runs/31665151409)
at exact launch commit
`8d41115141aed811b848fbe9647ca4403dcc0bc6`. All six matrix jobs completed
successfully. No hidden node modifications were used.

## Customizations outside the original recipe

| Area | Customization | Why it was necessary | Benchmark effect |
| --- | --- | --- | --- |
| NKX profile | Added `runners/cluster_profiles/nkx-gb300.sh` with Slurm account `nvidia`, partition `gpu`, storage/cache paths, mounts, and 140 CPUs per task. | The upstream runner contained SemiAnalysis cluster values such as account `benchmark`, partition `batch_1`, and different shared paths. | Cluster adaptation only. |
| Model staging handoff | Added a committed, immutable BenchOps staging receipt and `MODEL_PATH_OVERRIDE`. | InferenceX does not stage models, while the model had to exist on every worker's local NVMe before submission. | Changes only the physical model path to the revision-qualified `/scratch/models/DeepSeek-V4-Pro-b5968e9190ef`. |
| Model validation | Added `runners/verify_model_stage.py` and an immediate read-only check on all 16 GPU workers. It validates the receipt, marker, manifest, and file sizes. | Prevents an expensive benchmark from starting with missing, partial, or incorrect local copies. | Pre-submit safety gate; no runtime model modification. |
| Slurm preflight allocation | The model preflight requests an unpinned 16-node allocation, then checks each actual `SLURMD_NODENAME` against the receipt. | NKX `topology/block` rejected some smaller explicitly pinned allocations. | Validation-only allocation; benchmark placement remains recipe-controlled. |
| Effective recipe overlay | The launcher adds `cpus-per-task: 140`, health limit `720`, and `ETCD_LEASE_TTL=600` after copying the reviewed recipe into the pinned srt-slurm checkout. | These were effective settings in successful BenchOps job `2024`, but were not fully represented by the checked-in recipe. | These are the only meaningful runtime-control overlays. They are preserved in the effective recipe artifact. |
| Artifact identity | Pinned srt-slurm commit `758becd9d18dcab1fb722abc1875d73ee81a20cb`, vLLM and nginx SquashFS SHA-256 digests, and validated cached images before launch. | An image tag or branch name alone is insufficient for reproducibility. | Identity check only; the known-good image is unchanged. |
| SquashFS handling | Parameterized the NKX image-cache path, added exact checksum checks, and explicitly imported on an aarch64 compute node under `flock`. | The GitHub runner/login node is x86_64; compute nodes are aarch64. Concurrent imports could also corrupt shared artifacts. | Preparation only. |
| `srtctl` compatibility | Feature-detected `srtctl apply --no-preflight`. The pinned job-2024 revision predates that option, so the original command is used after the all-node check. | Passing the newer flag unconditionally would fail; a login node also cannot inspect node-local `/scratch`. | No recipe or engine change. |
| Readiness mode | Added an explicitly labelled readiness mode: concurrency 1, no warmup, and one formal request. | Allowed the exact topology, image, and runtime to be tested before submitting the full workload. | Used only for readiness run `31659981532`; never reported as comparable throughput. |
| Provenance collection | Added capture of the effective recipe, `srtslurm.yaml`, generated Slurm script, launcher environment, source revisions, and server logs. | Native results previously lacked enough information for a peer to reconstruct the complete launch contract. | Observability only. |
| Public-fork authentication | Replaced a missing upstream PAT secret with workflow-scoped `${{ github.token }}` for repository checkouts and result collection. | The public fork does not possess the upstream repository secret. | Authentication only. |
| Workflow plumbing | Added inputs for the NKX cluster profile, staging receipt, and readiness mode. | These inputs had to reach the reusable multinode workflow without hardcoding them into recipes. | Orchestration only. |

The reproduction did **not** add a custom UCX/NVSHMEM binary overlay, change
KV roles, modify engine arguments, install packages interactively, or copy the
earlier TensorRT-LLM workarounds. The six-point matrix was selected through
InferenceX's native exact configuration selector, not through a custom
submission loop.

## Errors and corrective actions

| Phase | Error or symptom | Action taken | Outcome |
| --- | --- | --- | --- |
| Initial investigation | The first effort used Dynamo-TensorRT-LLM and a development TRT image instead of the known-good Dynamo-vLLM image. | Paused TensorRT-LLM and froze the exact BenchOps job `2024` vLLM contract. | Reproduction returned to the correct backend and image. |
| TRT run `31557290928` | `ibv_create_cq: Invalid argument` and `ibv_create_qp: Cannot allocate memory`; effective MNNVL behavior was wrong. | Built source-controlled GPU, RoCE, collective, and MNNVL canaries. | Preflight run `31558738676` passed, showing the cluster itself was not generally broken. This remained TRT-only evidence. |
| TRT Slurm job `2272` | Prefill registration failed after about 16 minutes of synchronous TRT autotuning. The image contained Dynamo 1.2.0 and its short lease expired. | Investigated the embedded runtime and attempted a longer TTL/runtime upgrade. | Identified the lease-expiry cause, but the workaround exposed another problem. |
| TRT Slurm job `2283` | Every MPI rank concurrently upgraded shared `/opt/dynamo/venv`; `pip` operations raced and corrupted the environment. | Abandoned the per-rank upgrade and established the rule that package preparation must never run concurrently from worker ranks. | TRT remained paused; no such upgrade was used in the vLLM reproduction. |
| Model-stage workflow design | The initial proposal expected BenchOps to be checked out by GitHub Actions, but BenchOps was only available in internal GitLab. | Removed benchmark-time staging. Staged separately, performed full SHA-256 verification, and committed only the immutable receipt. | Benchmark jobs neither access BenchOps nor modify model files. |
| Slurm model preflight | Smaller pinned allocations were rejected by NKX `topology/block`. | Switched the validation gate to an unpinned all-16-node allocation and verified the nodes returned by Slurm. | Reliable 16/16 validation without weakening node checks. |
| Public-fork checkout | The workflow referenced an upstream PAT secret unavailable in the fork. | Used `${{ github.token }}` for read-only checkout and result collection. | Checkout and artifact collection succeeded. |
| Pinned `srtctl` | The pinned job-2024 revision did not recognize `--no-preflight`. | Feature-detected the option and omitted it for the pinned revision after completing the stronger all-node check. | Original `srtctl apply` interface succeeded. |
| Readiness artifacts | A readiness result could be mistaken for a comparable benchmark result. | Added `readiness` to job labels and result filenames. | Readiness and performance artifacts are unambiguous. |
| First matrix dispatch `31665133763` | The expanded `inputs.ref` was mistyped. | Cancelled immediately and dispatched with the exact pushed SHA. | No benchmark Slurm job was submitted by the bad dispatch. |
| Monitoring | Teleport credentials expired while the matrix was running. | Monitored through GitHub Actions; after refresh, performed a direct sequential `sacct` audit. | The jobs continued unaffected; all six were confirmed `COMPLETED`, exit `0:0`. |
| Log warning | `[nvlink-bf16-patch] ... failed to patch` appeared in successful output. | Compared it with BenchOps job `2024`, where the same warning appeared repeatedly without causing failure. | Classified as expected and nonfatal rather than changing the environment. |
| Result variance | `1p9d` was +6.57% over the three-run BenchOps mean, while the other five points were within -0.11% to +0.76%. | Verified identical token counts and similar TPOT; documented the lower TTFT and variance instead of adjusting the recipe. | Valid result, but repeats would be appropriate before characterizing that point statistically. |

## Accepted outcome

The accepted matrix run was GitHub Actions run `31665151409`. All six native
InferenceX jobs succeeded, and a post-run Slurm accounting audit confirmed
every job as `COMPLETED` with exit code `0:0`. See
`.github/NKX_GB300_REPRO.md` for the complete launch contract, Slurm job IDs,
metrics, artifact identifiers, and hashes.

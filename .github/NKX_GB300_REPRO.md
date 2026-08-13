# NKX GB300 DeepSeek V4 reproduction

This run reproduces the completed BenchOps Slurm job `2024` through the
InferenceX workflow and native `srt-slurm` launcher. It targets only the
NKX-managed cluster `nkx-slinky-gb300-dev-01` (`gb300`); it is not valid for
the Lepton-managed `gb300l` cluster.

## Frozen baseline

| Input | Value |
| --- | --- |
| InferenceX base | `d089a9138c53d16c6388e4251a078fee8ca7bea6` |
| srt-slurm | `758becd9d18dcab1fb722abc1875d73ee81a20cb` |
| BenchOps model staging | `d933853` |
| Image | `vllm/vllm-openai:dsv4-megamoe-mxfp4-arm64-cu130-4ba0a72` |
| Image SquashFS SHA-256 | `c8dc9884c5c863170f2207840e44a18c92af292c9829ad48c3d89b3cd87ddaff` |
| nginx image | `nginx:1.27.4` |
| nginx SquashFS SHA-256 | `61e003876ea0b3b78c5745e261056544fe4e997e7c79b73d62c31d7d28483a1a` |
| Hugging Face model | `deepseek-ai/DeepSeek-V4-Pro` |
| Model revision | `b5968e9190ef611bbf34a7229255be88a0e937c1` |
| Model manifest SHA-256 | `sha256:5c7c518159b3c7d0780d947846669693b16fe79f114b2010dd29769647eaa40d` |
| Recipe | `recipes/vllm/deepseek-v4/8k1k/disagg-gb300-6p1d-dep4-dep8-32-c4096.yaml` |
| Topology | 6 prefill workers (TP4/EP4), 1 decode worker (TP8/EP8) |
| Traffic | ISL 8192, OSL 1024, concurrency 4096, 40,960 requests |
| Placement | one 16-node GB300 block; 9 allocated nodes including infrastructure |

The immutable evidence record is
`results/runs/nkx-slinky-gb300-dev-01--inference-deepseek-v4-pro-fp4--c4d58b208b2b--2024.json`
in the `ai-cluster-benchmarking` repository. Job `2024` completed all 40,960
requests at 44,736.48 output tokens/s.

## Explicit NKX inputs

The upstream recipe is retained. `runners/cluster_profiles/nkx-gb300.sh`
supplies only the cluster inputs recovered from job `2024`:

- Slurm partition `gpu`, account `nvidia`, and 140 CPUs per task;
- NKX shared cache, image, home, and `/scratch/fsw` mount paths;
- the exact srt-slurm commit and image SquashFS digest;
- the effective job-2024 health limit (720 attempts) and etcd lease TTL
  (600 seconds).

This public fork does not define upstream's repository PAT secret. Its
read-only benchmark and result-collector checkouts therefore use GitHub's
workflow-scoped `github.token`; this changes repository authentication only,
not benchmark source, recipe, image, model, or runtime behavior.

The model path is an equivalent physical input, not a recipe change. BenchOps
stages the pinned revision to
`/scratch/models/DeepSeek-V4-Pro-b5968e9190ef` on every GPU worker. The
workflow accepts a committed ready receipt, checks its cluster and model
identity, requires every selected node to be verified, checks the shared
manifest digest, and then passes only its `localPath` as
`MODEL_PATH_OVERRIDE`. The benchmark job never downloads, copies, repairs, or
rewrites model data.

The pinned job-2024 srt-slurm revision predates the newer
`srtctl apply --no-preflight` option and does not perform that login-node model
path check. The launcher therefore feature-detects the option. For this exact
revision it relies on the immediately preceding 16-node read-only validation
and invokes the original `srtctl apply` interface without the unsupported
flag.

NKX Slurm currently accepts an unpinned 16-node allocation but rejects some
smaller explicitly pinned allocations under `topology/block`. Both BenchOps
staging and the benchmark's immediate read-only preflight therefore request
all 16 nodes without `--nodelist`; one task runs per worker and validates its
actual `SLURMD_NODENAME` against the immutable receipt. Staging still limits
copy work to four workers at a time. The explicit full audit hashes all local
copies in one all-target allocation, one task per worker.

Job `2024`'s archived configuration and generated Slurm script show the
upstream `vllm-container-deps.sh` setup and do not expose an additional UCX
binary overlay path. This reproduction therefore does not add an unrecorded
UCX or NVSHMEM overlay. Runtime transport environment produced natively by
srt-slurm remains visible in the collected artifacts.

After normalizing only the job name, revision-qualified model path, Slurm CPU
directive, health/TTL inputs, and BenchOps-only compatibility variables, job
`2024`'s archived `config.yaml` and the checked-in recipe have the identical
canonical SHA-256
`8ca433b3410df77d8ab4923f5f7e8a4cab20cf2d74659ef06f46e79d69e1c5c5`.
This launcher sets `cpus-per-task` directly and retains the recipe's existing
`kv_both` roles, so it does not inject the old `BENCHOPS_CPUS_PER_TASK` or
`BENCHOPS_PRESERVE_KV_BOTH` compatibility variables into engine processes.

## Stage and verify before dispatch

From the pinned `ai-cluster-benchmarking` checkout, first review the plan and
then perform a complete checksum audit:

```bash
uv run benchops model stage \
  -f model-stages/deepseek-v4-pro-gb300.yaml \
  --plan \
  --output-json model-stage-plan-gb300.json

uv run benchops model stage \
  -f model-stages/deepseek-v4-pro-gb300.yaml \
  --output-json model-stage-result-gb300.json

uv run benchops model stage \
  -f model-stages/deepseek-v4-pro-gb300.yaml \
  --full-verify \
  --output-json model-stage-full-verify-gb300.json
```

Do not dispatch unless the final receipt has `state == "ready"`, identifies
the NKX cluster and exact model revision above, and reports
`verification.mode == "full-sha256"` and
`nodes.required == nodes.verified == 16`. The audited receipt used for this
reproduction is committed at
`model-stages/receipts/deepseek-v4-pro-gb300-b5968e9190ef.json`, so the
workflow run and exact source commit preserve it together. Slurm job `2397`
performed the initial 16-node staging (`4 reused`, `12 copied`); job `2400`
performed the final all-node full SHA-256 audit recorded in that receipt.

## Generate exactly one reviewed point

```bash
uv run --with 'pydantic>=2' --with pyyaml \
  python utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/nvidia-master.yaml \
  --model-prefix dsv4 \
  --framework dynamo-vllm \
  --precision fp4 \
  --seq-lens 8k1k \
  --multi-node \
  --min-conc 4096 \
  --max-conc 4096 \
  --config-file recipes/vllm/deepseek-v4/8k1k/disagg-gb300-6p1d-dep4-dep8-32-c4096.yaml \
  --no-evals
```

The `--config-file` selector is intentional: concurrency 4096 alone matches
three checked-in DeepSeek topologies. Generation must return exactly one
matrix entry before dispatch.

First dispatch the workflow from the pushed reproduction commit with
`cluster-profile=nkx-gb300`, `readiness-only=true`, and
`model-stage-result-file` pointing to the committed full-verification receipt.
This preserves the exact image, setup, 6P/1D topology, and transport path but
changes the traffic-only fields to concurrency 1, no warmup, and one formal
request. Its job display and result filename are explicitly suffixed
`readiness`; it is a readiness candidate, not a comparable benchmark result.

Only after that gate proves image and model access, multi-rank startup, model
initialization, worker registration, control-plane lease survival, selected
transports, and a successful request, dispatch the full point with
`readiness-only=false`. Preserve the workflow run URL, exact
commit SHA, generated matrix, resolved recipe, `srtslurm.yaml`, generated
Slurm script, effective environment, logs, and native result artifacts.

## Accepted reproduction

The source-controlled execution path above completed successfully at
InferenceX commit `72b394f4773ab7b3c416cc483119349a7f9be5b2`:

| Evidence | Identifier |
| --- | --- |
| Readiness workflow | [GitHub Actions run 31659981532](https://github.com/olegleyz/inferencex-nkx/actions/runs/31659981532) |
| Readiness Slurm job | `2407` (`COMPLETED`, 10/10 requests) |
| Full workflow | [GitHub Actions run 31661199748](https://github.com/olegleyz/inferencex-nkx/actions/runs/31661199748) |
| Full Slurm job | `2411` (`COMPLETED`, 40,960/40,960 requests) |
| Placement | the same nine nodes recorded by job `2024` |

The full run achieved 44,031.09 output tokens/s, 47.7867 requests/s, and
396,386.30 total tokens/s in 857.142 seconds. Against BenchOps job `2024`,
the output-token throughput differs by -1.58%, duration by +1.60%, mean TTFT
by -0.14%, and mean TPOT by +2.03%. Both runs processed the identical
302,018,346 input tokens and 37,740,883 output tokens.

The workflow's native artifacts preserve the model-stage receipt, aggregate
benchmark JSON, raw per-request result, complete frontend/prefill/decode logs,
effective recipe, cluster configuration, generated Slurm script, launcher
environment, and source revisions. Key artifact SHA-256 values are:

| Artifact | SHA-256 |
| --- | --- |
| Effective recipe | `6e473e3a6fa4a4365c8c55536bb48f80df55e8b2ca074cad44845f89b4e79d31` |
| Generated Slurm script | `6f0e32aa2e4d9b22048ae8a37db0eef7536f957cd0b785802cb5940a7f45cfd6` |
| Launcher environment | `dc736208198de91a226bf84e5f96c8a85b0c9a2cef800116e13af30efda98d66` |
| srt-slurm cluster configuration | `19797848c69701691e4669347a3dd775489f3b957144ae5c0846ef4d57966b89` |
| Aggregate benchmark result | `c316aec51e2172b9857e1671277cf31ed60c10e61e140d1a606eaaf4279d4b97` |
| Model-stage receipt | `ec750cc6fe832e19a73f043e9111d8c7f9b787cda1722fda477023466bf7a7bb` |
| Complete server-log archive | `9c708610b2b96e78a29998de652c76145f519370b585bcbc0ed6869ab8d78355` |

The only effective-recipe differences from job `2024` are the run name, the
revision-qualified node-local model path, and removal of the two BenchOps
bookkeeping variables `BENCHOPS_CPUS_PER_TASK` and
`BENCHOPS_PRESERVE_KV_BOTH`. Their behavior remains explicit in the generated
Slurm `cpus-per-task: 140` directive and the recipe's prefill/decode
`kv_role: kv_both` settings. All other differences are the concrete NKX
cluster inputs documented above; no benchmark-time model copy, package
upgrade, interactive node modification, or TensorRT-LLM path was used.

## Broader DeepSeek matrix

BenchOps submission
`nkx-slinky-gb300-dev-01-inferencex-20260810-205415037647-2f850127`
ran three repeats of each of the six reviewed configurations below. The first
InferenceX matrix reproduction runs one instance of each point. Repeats are a
separate follow-up after all six execution paths have passed readiness and one
full sweep.

| Configuration | Concurrency | Requests | Serving GPUs | Allocation nodes | BenchOps jobs | Mean output tokens/s |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| `1p6d-dep4-tp4` | 192 | 1,920 | 28 | 8 | `2131`, `2134`, `2137` | 6,911.09 |
| `1p9d-tep4-tp4` | 18 | 180 | 40 | 10 | `2122`, `2125`, `2128` | 1,284.42 |
| `4p1d-dep4-dep8` | 4,096 | 40,960 | 24 | 7 | `2140`, `2143`, `2146` | 30,428.89 |
| `5p1d-dep4-dep8` | 4,096 | 40,960 | 28 | 8 | `2149`, `2152`, `2155` | 37,536.85 |
| `6p1d-dep4-dep8` | 4,096 | 40,960 | 32 | 9 | `2158`, `2161`, `2164` | 44,419.36 |
| `7p2d-dep4-dep16` | 3,072 | 30,720 | 60 | 16 | `2167`, `2170`, `2173` | 50,132.01 |

InferenceX already has a native exact-key selector for this set. It avoids a
copied NKX master configuration and resolves directly from the original
`configs/nvidia-master.yaml` entry and its six checked-in recipes:

```bash
uv run --with 'pydantic>=2' --with pyyaml \
  python utils/matrix_logic/generate_sweep_configs.py test-config \
  --config-files configs/nvidia-master.yaml \
  --config-keys dsv4-fp4-gb300-dynamo-vllm \
  --seq-lens 8k1k \
  --no-evals
```

The GitHub Actions dispatch passes that command verbatim, the committed model
receipt, and `cluster-profile=nkx-gb300`. It retains InferenceX's normal matrix
fan-out and native Slurm scheduling behavior; there is no NKX-only matrix
serializer or separate submission loop.

The accepted readiness run `31659981532` and full run `31661199748` already
validated the shared image, model receipt, setup composition, transport path,
and native artifact collection. Dispatch the broader matrix with the normal
workflow defaults: `readiness-only=false`, `fail-fast=false`, and the native
matrix fan-out. Slurm remains authoritative for each recipe's allocation.

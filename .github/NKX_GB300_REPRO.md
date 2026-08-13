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

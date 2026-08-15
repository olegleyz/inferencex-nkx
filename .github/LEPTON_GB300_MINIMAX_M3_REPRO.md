# Lepton GB300 MiniMax M3 reproduction

This target runs MiniMax M3 MXFP8 through the normal InferenceX GitHub Actions,
`srt-slurm`, recipe, launcher, and native result-collection path on the separate
Lepton-managed `nkx-slinky-dev-02` Slurm cluster. It never targets the
NKX-managed `gb300` cluster.

## Frozen standard Dynamo-vLLM contract

- InferenceX source contract: `af285ca48da805cf5bd7419010333824fef2fd9b`
- Matrix: `configs/lepton-minimaxm3-standard-baseline.yaml`
- Image: `vllm/vllm-openai:nightly-4080263bb2c5d10deac17aaeb88e0823bc35bca9`
- Lepton SquashFS SHA-256:
  `1ac422ddf87efdb3d9902e254dd7d56cc9ce9d152b59f2b7e9c0716595eab481`
- srt-slurm: `deb1dfd9934398664f92d194169c183e009da83b`
- Model:
  `MiniMaxAI/MiniMax-M3-MXFP8@c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`
- Traffic: 8192 input tokens, 1024 output tokens
- Matrix size: nine native recipe jobs and 11 concurrency data points

The nine checked-in recipe files are byte-identical to the frozen InferenceX
source. The matrix retains upstream's `1p1d-dep2-tep8` point at concurrency
128 and `1p1d-dep2-dep8` point at concurrency 256.

The prior reviewed BenchOps selector is useful cluster evidence but is not the
matrix source: it omitted upstream's `1p1d-dep2-tep8` point and instead ran
`1p1d-dep2-dep8` at both concurrency 128 and 256. This native reproduction
deliberately removes that silent topology drift.

## Frozen EAGLE3 contract

- Successful upstream workflow: `30848195936`
- InferenceX recipe source:
  `5f738ee0b7a6f88c6064d42f814099643ecad773`
- Matrix: `configs/nkx-minimaxm3-eagle3-baseline.yaml`
- Declared image:
  `vllm/vllm-openai:nightly-5e35a6f4f9bbc217c599692157ca985c894373f7`
- Immutable original aarch64 OCI manifest:
  `public.ecr.aws/q9t5s3a7/vllm-release-repo@sha256:41442db2591d6bfb8dc219561f18deed55aaf5b95f910e5d9145186043d8eb94`
- srt-slurm: `c180328b98c3793ca84a1e24a030f90545eb7d5d`
  (`v1.0.38`)
- Base model: the same pinned MiniMax M3 MXFP8 revision above
- Draft model:
  `Inferact/MiniMax-M3-EAGLE3-GQA@96692486b5fd38ebf8fd2a5f6bb53427d30819a8`
- Matrix size: 11 native recipe jobs / data points

The 11 checked-in EAGLE3 recipe files are byte-identical to the successful
upstream workflow's source. They retain the original CUTLASS MSA decode
setting and do not include later unbenchmarked recipe changes.

## Explicit Lepton inputs

| Input | Frozen InferenceX value | Lepton value | Classification |
| --- | --- | --- | --- |
| Slurm target | InferenceX GB300 runner | cluster `nkx-slinky-dev-02`, partition `batch`, account `nvidia`, runner `gb300l-nv` | Scheduler input |
| Worker architecture | GB300 | aarch64, four GPUs per node | Hardware input |
| Base model path | logical recipe alias | `/raid/scratch/benchops-oleizerov/models/MiniMax-M3-MXFP8-c5454eb03678` | Revision-qualified physical path |
| Draft model path | Hugging Face ID in recipe | `/raid/scratch/benchops-oleizerov/models/MiniMax-M3-EAGLE3-GQA-96692486b5fd` | Revision-qualified physical path |
| Shared staging root | site-specific | `/scratch/fsw/users/oleizerov/models` | Storage input |
| Standard image storage | Docker image | checksum-pinned Lepton SquashFS cache | Equivalent materialized artifact |
| EAGLE3 image storage | deleted Docker Hub tag | original aarch64 Public ECR manifest | Equivalent immutable artifact location |
| Control-plane lease | recipe/runtime default | `ETCD_LEASE_TTL=600`, as used by completed Lepton BenchOps MiniMax jobs | Lepton runtime compatibility input |
| Readiness timeout | site-specific | 720 attempts at the upstream interval | Cluster startup input |

Both model paths must come from committed, full-SHA256 BenchOps ModelStage
receipts. The workflow performs read-only receipt, manifest, file-size, and
active-partition coverage checks before launching `srtctl`. Benchmark jobs do
not download, copy, repair, or rewrite either model.

Model staging uses BenchOps commit
`b18c62d20799ddd7173415ea901afbbecb55f0c5`. Both declarations retain
`targets.gpuNodes: ALL`; BenchOps resolves that value from Slurm's currently
schedulable GPU workers and skips unavailable states such as drained and
reserved. No node names or manual exclusion list are committed. The plan must
be reviewed immediately before staging, and the resulting receipt freezes the
exact workers that were required and verified for that execution.

On August 14, 2026, both dynamic plans resolved 15 schedulable `batch`
workers. Base-model staging Slurm job `1129` copied, checksum-verified, and
atomically published the model on all 15 workers; independent full-SHA256
audit job `1130` then verified 15/15 with zero failures. EAGLE3 staging job
`1132` copied and verified 15/15, followed by independent full-SHA256 audit
job `1133`, also 15/15 with zero failures. The committed receipts are:

- `model-stages/receipts/minimax-m3-mxfp8-lepton-gb300-c5454eb03678.json`
- `model-stages/receipts/minimax-m3-eagle3-gqa-lepton-gb300-96692486b5fd.json`

## Dispatch sequence

Do not dispatch until both model receipts are committed and every active
`batch` worker is covered. Run one standard readiness topology first, followed
by the standard matrix. Then run one EAGLE3 readiness topology followed by the
EAGLE3 matrix. Each readiness run keeps its reviewed topology but rewrites the
effective benchmark section to one request after native serving readiness.

For every run retain the exact fork commit, generated matrix, effective recipe,
generated Slurm script, launcher environment, both model receipts and shared
manifests where applicable, image identity, server logs, benchmark JSON, and
native aggregate artifact.

## August 15, 2026 completion record

The standard matrix completed all nine jobs (11 concurrency data points) in
GitHub Actions run `31841679846`. The EAGLE3 matrix completed all 11 recipe
points using the original matrix run plus individually isolated recoveries:

| Evidence | Purpose | Outcome |
| --- | --- | --- |
| GHA `31858721822`, Slurm `1219` | Native EAGLE3 readiness | Passed |
| GHA `31859994530` | Original 11-point EAGLE3 matrix | Five valid points retained |
| GHA `31865382304` | TP4 c1 recovery | Passed |
| GHA `31865383218` | TP8 c8 recovery | Passed |
| GHA `31865384083` | 3P c256 recovery | Passed |
| GHA `31865386256` | 4P c2048 recovery | Passed |
| GHA `31865387194` | 6P c2048 recovery | Passed |
| GHA `31876477928`, Slurm `1320` | 4P1D DEP2/DEP8 c1024 canonical recovery | Passed |

All canonical runs use fork commit
`bfe57ef8094c1c6f0efcf182520b69de1d5ce29a` or an earlier ancestor with the
same frozen recipe/runtime contract. GitHub Actions preserves the effective
recipe, generated Slurm script, fingerprints, server logs, benchmark JSON,
and native aggregate result under each run.

The final c1024 recovery completed all 10,240 measured requests in 397.22
seconds. It reported 23,761.43 output tokens/s, 25.78 requests/s, 26,410.63 ms
mean TTFT, and 12.08 ms mean TPOT. The matching upstream result was 23,674.58
output tokens/s, so the Lepton result is 0.37% higher. This comparison is for
the unchanged canonical run; the diagnostic probe below is not a published
matrix result.

### c1024 failure investigation

An earlier exact c1024 attempt, GHA `31870602301` / Slurm `1306`, initialized
and registered every worker but failed during the 2,048-request warmup. The
first fatal signature was a vLLM CUTLASS MSA assertion on decode DP0:
`assert query_fp8 is not None`. Later RPC, TCPStore, and lease messages were
cascading symptoms. Native RoCE/NIXL did not fail in that attempt.

Source-controlled canaries (Slurm `1299` and `1301`) passed the relevant
native UCX/CUDA registration shape, including on the previously affected
worker. A focused warmup-load probe was therefore added at commit
`bfe57ef8094c1c6f0efcf182520b69de1d5ce29a`. Its validator permits only the
reviewed Lepton EAGLE3 c1024 point. The probe preserves concurrency, topology,
models, image, native transport, and the original two-times warmup, while
reducing only the subsequent measured request multiplier from ten to one and
labeling the artifacts as diagnostic.

GHA `31873865150` / Slurm `1315` passed the exact 2,048-request warmup and its
1,024-request measurement without the assertion. That result justified one
unchanged canonical retry. GHA `31876477928` / Slurm `1320` then passed the
same warmup and the full 10,240-request measurement. The evidence classifies
the assertion as a nondeterministic vLLM runtime failure rather than a recipe,
model-content, or persistent RoCE configuration defect; no runtime setting,
transport fallback, model file, or cluster state was changed to obtain the
canonical success.

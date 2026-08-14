# NKX GB300 MiniMax M3 EAGLE3 reproduction

This target reproduces the 11-point MiniMax M3 8K/1K Dynamo-vLLM EAGLE3
throughput sweep from upstream InferenceX PR #2478. It runs only on the
NKX-managed `nkx-slinky-gb300-dev-01` Slurm cluster through the `nkx-gb300`
target. It does not target the separate Lepton cluster.

## Frozen upstream contract

- Successful upstream workflow: `30848195936`
- InferenceX recipe source commit:
  `5f738ee0b7a6f88c6064d42f814099643ecad773`
- Dedicated matrix key: `minimaxm3-fp8-gb300-dynamo-vllm-mtp`
- Matrix file: `configs/nkx-minimaxm3-eagle3-baseline.yaml`
- srt-slurm: `c180328b98c3793ca84a1e24a030f90545eb7d5d`
  (`v1.0.38`)
- Declared image:
  `vllm/vllm-openai:nightly-5e35a6f4f9bbc217c599692157ca985c894373f7`
- Base model:
  `MiniMaxAI/MiniMax-M3-MXFP8@c5454eb03678d8710e54a4e0fc681b9f3b4a3dba`
- EAGLE3 draft:
  `Inferact/MiniMax-M3-EAGLE3-GQA@96692486b5fd38ebf8fd2a5f6bb53427d30819a8`
- Traffic: 8192 input tokens, 1024 output tokens, native recipe warmup and
  request multipliers

The dedicated matrix is structurally identical to the single upstream config
block at the source commit. The 11 recipe YAML files under
`benchmarks/multi_node/srt-slurm-recipes/vllm/minimax-m3-gb300-fp8/8k1k/mtp/`
are byte-identical to that commit. In particular, they retain the CUTLASS MSA
decode setting and do not include the later, unbenchmarked `FULL_DECODE_ONLY`
change.

## Explicit NKX inputs

The upstream Docker Hub nightly tag is no longer available. vLLM's own
historical `push-nightly-builds.sh` shows that the architecture-specific image
was first published to AWS Public ECR and then copied to Docker Hub. The
original immutable aarch64 manifest remains available at:

```text
public.ecr.aws/q9t5s3a7/vllm-release-repo@sha256:41442db2591d6bfb8dc219561f18deed55aaf5b95f910e5d9145186043d8eb94
```

The NKX profile imports the commit-qualified ECR aarch64 tag using Enroot's
registry syntax and first resolves that tag through the Registry API. Launch
fails unless its live manifest is exactly the digest above. The checked-in
recipe and result identity retain the original Docker Hub tag. This is an
artifact-location adaptation, not a rebuilt or upgraded runtime.

Other NKX inputs are the source-controlled `gpu` partition, `nvidia` account,
shared cache roots under `/scratch/fsw/users/oleizerov`, aarch64 Enroot import,
and revision-qualified node-local model paths under `/scratch/models`.

| Input | Upstream baseline | NKX effective value | Classification |
| --- | --- | --- | --- |
| Slurm target | SemiAnalysis `batch_1` runner | `gpu`, account `nvidia`, cluster `nkx-slinky-gb300-dev-01` | Scheduler input |
| Container source | Deleted Docker Hub tag | Original aarch64 Public ECR manifest by digest | Equivalent immutable artifact location |
| Base model storage | `/data/models/MiniMax-M3-MXFP8` | `/scratch/models/MiniMax-M3-MXFP8-c5454eb03678` | Revision-qualified physical path |
| Draft model storage | Unpinned HF ID resolved at runtime | `/scratch/models/MiniMax-M3-EAGLE3-GQA-96692486b5fd` | Pinned, pre-staged physical path |
| Model mounts | Upstream base mount plus runtime HF access | Both verified directories mounted read-only | Staging enforcement |
| srt-slurm checkout | Tag `v1.0.38` | Tag commit `c180328b...` | Stronger source pin |
| Recipe runtime/topology | PR #2478 YAML | Byte-identical PR #2478 YAML before explicit path substitution | Unchanged |

Both models are staged separately. The workflow validates committed full-SHA256
receipts and then performs read-only marker, manifest, file-size, and node
coverage checks on every active GPU worker. Only the two verified local paths
are substituted into the effective recipe, and the generated srt-slurm
configuration mounts both directories read-only. Benchmark jobs do not
download, copy, repair, or rewrite model data.

## Upstream result caveat

The successful upstream logs declared and preflighted base revision
`c5454eb03678`, but their runtime fingerprint reported the pre-existing model
directory at revision `1c4e6a69f327`. That run continued despite the fingerprint
failure. This reproduction intentionally honors the recipe's declared immutable
revision `c5454eb...`; it does not reproduce the official runner's storage
drift. Any metric comparison must disclose this difference.

## Generate and validate the matrix

```bash
uv run --no-project --with pydantic --with pyyaml --python 3.12 \
  utils/matrix_logic/generate_sweep_configs.py full-sweep \
  --config-files configs/nkx-minimaxm3-eagle3-baseline.yaml \
  --no-evals --multi-node
```

The output must contain exactly 11 entries. The target validator rejects any
other model, image, runtime, recipe path, duplicate recipe, or readiness point.

## Dispatch order

Set `COMMIT` to the pushed immutable commit containing both ready receipts.
First dispatch only the TP4 concurrency-1 readiness point:

```bash
gh api -X POST \
  /repos/olegleyz/inferencex-nkx/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='codex/nkx-vllm-repro' \
  -f "inputs[ref]=$COMMIT" \
  -f 'inputs[test-name]=NKX GB300 MiniMax M3 EAGLE3 TP4 c1 readiness' \
  -f 'inputs[target-cluster]=nkx-gb300' \
  -f 'inputs[model-stage-result-file]=model-stages/receipts/minimax-m3-mxfp8-gb300-c5454eb03678.json' \
  -f 'inputs[speculative-model-stage-result-file]=model-stages/receipts/minimax-m3-eagle3-gqa-gb300-96692486b5fd.json' \
  -F 'inputs[readiness-only]=true' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files configs/nkx-minimaxm3-eagle3-baseline.yaml --no-evals --multi-node --config-file recipes/vllm/minimax-m3-gb300-fp8/8k1k/mtp/1p1d-dep2-tp4-eagle3-c1-8k1k.yaml'
```

Do not dispatch the matrix unless readiness proves both model receipts, image
import, package identity, worker registration, one inference request, and native
artifact collection. Then dispatch the exact 11 points:

```bash
gh api -X POST \
  /repos/olegleyz/inferencex-nkx/actions/workflows/e2e-tests.yml/dispatches \
  -f ref='codex/nkx-vllm-repro' \
  -f "inputs[ref]=$COMMIT" \
  -f 'inputs[test-name]=NKX GB300 MiniMax M3 EAGLE3 baseline matrix' \
  -f 'inputs[target-cluster]=nkx-gb300' \
  -f 'inputs[model-stage-result-file]=model-stages/receipts/minimax-m3-mxfp8-gb300-c5454eb03678.json' \
  -f 'inputs[speculative-model-stage-result-file]=model-stages/receipts/minimax-m3-eagle3-gqa-gb300-96692486b5fd.json' \
  -f 'inputs[generate-cli-command]=full-sweep --config-files configs/nkx-minimaxm3-eagle3-baseline.yaml --no-evals --multi-node'
```

For each run, retain the exact commit, generated matrix, effective recipe,
generated Slurm script, launcher environment, model receipts and manifests,
server logs, benchmark results, package fingerprint, and native aggregate
artifact.

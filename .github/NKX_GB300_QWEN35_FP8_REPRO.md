# NKX GB300 Qwen3.5 FP8 reproduction

This document records the source-controlled launch contract for reproducing
the published Qwen3.5 397B FP8 Dynamo-SGLang result on the NKX-managed GB300
Slurm cluster. The target is only `nkx-gb300` / Slurm cluster
`nkx-slinky-gb300-dev-01`; it does not select the separate Lepton target.

## Immutable model and runtime inputs

| Input | Value |
| --- | --- |
| InferenceX baseline | `d089a9138c53d16c6388e4251a078fee8ca7bea6` |
| Model | `Qwen/Qwen3.5-397B-A17B-FP8` |
| Model revision | `ea5b4f81096f3901c91dea97f81324302495781d` |
| Model stage definition | BenchOps commit `f690dd4` |
| Model manifest | `sha256:3bc4f54f36992c01270313ae4a6b75cadfec4d58f6874282acbd6ada24ab7502` |
| Full-verification receipt | `model-stages/receipts/qwen3.5-397b-a17b-fp8-gb300-ea5b4f81096f.json` |
| Image | `lmsysorg/sglang:nightly-dev-cu13-20260709-074bb928` |
| Image SquashFS SHA-256 | `2ebe26d3323b61dcb8612a8c9b20f24aec1e57b3fcb61bf50b72c1e715851af0` |
| srt-slurm commit | `3435776cd6db4c14f8b771ff7a3976deb62fe133` (`v1.0.25`) |
| Dynamo source commit | `46520ca59afe992fb5ef61b3197b2316f8df9b2b` |
| Scenario | fixed sequence length, 8192 input / 1024 output tokens |
| Speculative decoding | none |

The three checked-in recipe files and the selected `nvidia-master.yaml`
configuration block are byte-identical between the successful BenchOps source
revision above and this reproduction branch before cluster adaptation.

## Native matrix

| Recipe | Topology | Concurrencies | Nodes / GPUs |
| --- | --- | --- | --- |
| `1p1d-tp4-tp4.yaml` | 1 prefill TP4 + 1 decode TP4 | 1, 2, 4, 8, 16, 32, 64, 128 | 2 / 8 |
| `4p1d-dep4-dep16.yaml` | 4 prefill DEP4 + 1 decode DEP16 | 1024 | 8 / 32 |
| `8p1d-dep4-dep16.yaml` | 8 prefill DEP4 + 1 decode DEP16 | 2048, 4096 | 12 / 48 |

The normal InferenceX generator marks these entries for evaluation as well as
throughput. The full workflow therefore retains its native throughput and
eval job split. Readiness is the sole exception: it selects the existing
1P1D recipe at concurrency 1 with `--no-evals` and sends one formal request.

## Explicit NKX inputs and known-good adaptations

| Difference from upstream | Effective value | Reason/evidence |
| --- | --- | --- |
| Private runner/target | `gb300-nv`, `nkx-gb300` | Routes the public logical `gb300` runner to the private NKX login runner. |
| Slurm identity | account `nvidia`, partition `gpu` | NKX scheduler configuration. The launcher verifies cluster name and partition before submission. |
| Storage/cache roots | `/scratch/fsw/users/oleizerov/.benchops/...` | NKX shared filesystem paths for image, HF, Dynamo-wheel, and result caches. |
| Model path | `/scratch/models/Qwen3.5-397B-A17B-FP8-ea5b4f81096f` | Revision-qualified node-local copy from the committed ModelStage receipt. |
| CPUs per task | `16` | Exact effective value in successful BenchOps Qwen Slurm scripts (for example job `1508`); it is intentionally not DeepSeek's `140`. |
| Dynamo lease TTL | `120` seconds | Exact effective value in the successful Qwen recipe lock and worker logs; it is intentionally not DeepSeek's `600`. |
| Health attempts | `720` at 10 seconds | Known-good slow-model startup allowance recorded in the successful BenchOps effective recipe. |
| Image identity | SquashFS checksum above | Prevents a mutable image tag from silently changing the runtime artifact. |
| srt-slurm identity | immutable commit above | Prevents the `v1.0.25` tag from being the only runtime identity. |

No Qwen engine arguments are rewritten. The original recipe remains the source
of topology, SGLang arguments, Mooncake/MNNVL configuration, traffic shape,
and Dynamo commit. The workflow only resolves the recipe's model alias to the
verified node-local path and records the effective recipe, generated Slurm
script, source commits, environment, logs, and native result artifacts.

## Dispatch commands

Both commands must use the exact pushed reproduction commit and the committed
full-verification receipt. Do not dispatch while the receipt is absent or any
of its 16 required nodes is unverified.

Focused readiness generator command:

```text
full-sweep --config-files configs/nvidia-master.yaml --runner-config configs/runners.yaml --multi-node --config-file recipes/sglang/qwen3.5/gb300-fp8/8k1k/1p1d-tp4-tp4.yaml --min-conc 1 --max-conc 1 --no-evals
```

Set `target-cluster=nkx-gb300`, `readiness-only=true`, and
`model-stage-result-file=model-stages/receipts/qwen3.5-397b-a17b-fp8-gb300-ea5b4f81096f.json`.

Normal native matrix generator command:

```text
test-config --config-files configs/nvidia-master.yaml --runner-config configs/runners.yaml --config-keys qwen3.5-fp8-gb300-dynamo-sglang
```

Set the same target and receipt, with `readiness-only=false`. Do not use
`--no-evals` for the normal matrix.

## Run evidence

| Phase | Identifier | Result/evidence |
| --- | --- | --- |
| Shared download | initial stage record `20260814T024222956384Z` | The Teleport client stream expired after download, but the remote process continued and atomically published all 107 files (406,198,638,888 bytes). No incomplete directory was published. |
| Node-local stage | Slurm `2439` | Ready on 16/16 required NKX GPU workers; 16 copied, 16 verified, zero failed. |
| Independent full audit | Slurm `2440` | Ready on 16/16; every existing copy reused only after full SHA-256 verification against manifest `sha256:3bc4f54f36992c01270313ae4a6b75cadfec4d58f6874282acbd6ada24ab7502`. |
| Initial readiness | GitHub Actions `31769246166`, Slurm `2444` | Serving succeeded, including one 8K/1K request, but post-processing failed because the one-request result had `std_tpot_ms=0` and `process_result.py` inverted every TPOT statistic. Native logs and the raw SA-Bench result were preserved. |
| Result-processing fix | commit `9ca0c63f0dfcc1fdb29f32c665c258913fd61291` | Zero-valued TPOT statistics remain recorded but are not inverted into undefined interactivity values. All 32 result-processing tests passed, and the exact captured readiness JSON replayed successfully. |
| Green readiness | GitHub Actions `31770432333`, job `94675083957`, Slurm `2448` | Success at the exact fix commit. Receipt validation, original 1P TP4 + 1D TP4 recipe, model initialization, worker registration, one 8K/1K request, result processing, aggregation, success-rate calculation, and native artifact upload all passed. Artifacts: result `9208215437`, server logs `9208215749`, resolved srt-slurm configuration `9208216074`, and aggregated result `9208220165`. |

Full-run GitHub Actions IDs, benchmark Slurm job IDs, conclusions, artifact
links, and performance comparisons are added here after execution.

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


RUNNERS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNERS))

import resolve_cluster_target  # noqa: E402
import validate_cluster_target_matrix  # noqa: E402
import validate_model_stage_receipt  # noqa: E402


class ResolveClusterTargetTests(unittest.TestCase):
    def test_lepton_routes_to_distinct_runner(self) -> None:
        result = resolve_cluster_target.resolve("lepton-gb300")
        self.assertEqual(result["runner-label"], "gb300l-nv")
        self.assertEqual(result["cluster-profile"], "lepton-gb300")
        self.assertEqual(
            result["model-stage-result-file"],
            "model-stages/receipts/deepseek-v4-pro-lepton-gb300-b5968e9190ef.json",
        )

    def test_lepton_declares_qwen_receipt_separately(self) -> None:
        target = resolve_cluster_target.load_targets()["lepton-gb300"]
        qwen = target["models"]["Qwen/Qwen3.5-397B-A17B-FP8"]
        self.assertEqual(
            qwen["model_stage_receipt"],
            "model-stages/receipts/"
            "qwen3.5-397b-a17b-fp8-lepton-gb300-ea5b4f81096f.json",
        )
        self.assertTrue(qwen["local_path"].startswith("/raid/scratch/"))

    def test_mismatched_profile_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "target/profile mismatch"):
            resolve_cluster_target.resolve("lepton-gb300", "nkx-gb300")

    def test_legacy_nkx_profile_resolves(self) -> None:
        result = resolve_cluster_target.resolve("", "nkx-gb300")
        self.assertEqual(result["target-cluster"], "nkx-gb300")


class ClusterProfileTests(unittest.TestCase):
    def _lepton_environment(self, model_contract: str) -> dict[str, str]:
        script = RUNNERS / "cluster_profiles" / "lepton-gb300.sh"
        model_prefix, precision, framework = model_contract.split("-", 2)
        command = (
            f"MODEL_PREFIX={model_prefix!r}; PRECISION={precision!r}; "
            f"FRAMEWORK={framework!r}; source {str(script)!r}; "
            "python3 -c 'import json,os; print(json.dumps(dict(os.environ)))'"
        )
        result = subprocess.run(
            ["bash", "-c", command],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)

    def test_lepton_qwen_does_not_inherit_deepseek_runtime_overlay(self) -> None:
        environment = self._lepton_environment("qwen3.5-fp8-dynamo-sglang")
        self.assertEqual(
            environment["SRT_SLURM_QWEN35_FP8_REF"],
            "3435776cd6db4c14f8b771ff7a3976deb62fe133",
        )
        self.assertEqual(
            environment["IMAGE_SQUASH_SHA256"],
            "e3c9ce21fa6f0f363dfb1fb4aff7a3b8dd55803b0c214c649719919a75b7bbe9",
        )
        self.assertEqual(environment["SRT_SLURM_ETCD_LEASE_TTL"], "600")
        self.assertNotIn("SRT_SLURM_CPUS_PER_TASK", environment)
        self.assertNotIn("SRT_SLURM_RUNTIME_ENV_JSON", environment)

    def test_lepton_deepseek_keeps_reviewed_runtime_overlay(self) -> None:
        environment = self._lepton_environment("dsv4-fp4-dynamo-vllm")
        self.assertEqual(environment["SRT_SLURM_CPUS_PER_TASK"], "140")
        self.assertIn("NVSHMEM_HCA_LIST", environment["SRT_SLURM_RUNTIME_ENV_JSON"])


class TargetMatrixTests(unittest.TestCase):
    def _row(self) -> dict[str, object]:
        return {
            "image": validate_cluster_target_matrix.DEEPSEEK_IMAGE,
            "model": "deepseek-ai/DeepSeek-V4-Pro",
            "model-prefix": "dsv4",
            "precision": "fp4",
            "framework": "dynamo-vllm",
            "runner": "gb300-nv",
            "isl": 8192,
            "osl": 1024,
            "run-eval": False,
            "prefill": {
                "additional-settings": [
                    "CONFIG_FILE=recipes/vllm/deepseek-v4/8k1k/"
                    "disagg-gb300-1p6d-dep4-tp4.yaml"
                ]
            },
        }

    def test_reviewed_recipe_is_accepted(self) -> None:
        self.assertEqual(
            validate_cluster_target_matrix.validate("lepton-gb300", [self._row()]),
            1,
        )

    def test_changed_image_is_rejected(self) -> None:
        row = self._row()
        row["image"] = "example.invalid/changed:latest"
        with self.assertRaisesRegex(RuntimeError, "unreviewed target matrix entry"):
            validate_cluster_target_matrix.validate("lepton-gb300", [row])

    def test_qwen35_fp8_native_matrix_is_accepted(self) -> None:
        rows = []
        for recipe in (
            "1p1d-tp4-tp4.yaml",
            "4p1d-dep4-dep16.yaml",
            "8p1d-dep4-dep16.yaml",
        ):
            rows.append(
                {
                    "image": validate_cluster_target_matrix.QWEN35_FP8_IMAGE,
                    "model": "Qwen/Qwen3.5-397B-A17B-FP8",
                    "model-prefix": "qwen3.5",
                    "precision": "fp8",
                    "framework": "dynamo-sglang",
                    "runner": "gb300",
                    "isl": 8192,
                    "osl": 1024,
                    "run-eval": True,
                    "prefill": {
                        "additional-settings": [
                            "CONFIG_FILE=recipes/sglang/qwen3.5/gb300-fp8/8k1k/"
                            + recipe
                        ]
                    },
                }
            )
        for target in ("nkx-gb300", "lepton-gb300"):
            with self.subTest(target=target):
                self.assertEqual(
                    validate_cluster_target_matrix.validate(target, rows), 3
                )

    def test_qwen35_fp8_single_readiness_probe_is_accepted(self) -> None:
        row = {
            "image": validate_cluster_target_matrix.QWEN35_FP8_IMAGE,
            "model": "Qwen/Qwen3.5-397B-A17B-FP8",
            "model-prefix": "qwen3.5",
            "precision": "fp8",
            "framework": "dynamo-sglang",
            "runner": "gb300",
            "isl": 8192,
            "osl": 1024,
            "run-eval": False,
            "conc": [1],
            "prefill": {
                "additional-settings": [
                    (
                        "CONFIG_FILE=recipes/sglang/qwen3.5/gb300-fp8/8k1k/"
                        "1p1d-tp4-tp4.yaml"
                    )
                ]
            },
        }
        self.assertEqual(
            validate_cluster_target_matrix.validate(
                "nkx-gb300", [row], readiness_only=True
            ),
            1,
        )


class ReceiptTests(unittest.TestCase):
    def test_target_receipt_and_manifest_are_bound_together(self) -> None:
        manifest = {
            "repository": "deepseek-ai/DeepSeek-V4-Pro",
            "revision": "revision",
            "files": [],
            "totalFileCount": 0,
            "totalBytes": 0,
        }
        body = json.dumps(
            manifest, sort_keys=True, separators=(",", ":")
        ).encode()
        digest = f"sha256:{hashlib.sha256(body).hexdigest()}"
        manifest["manifestDigest"] = digest
        target = {
            "staging_cluster_id": "cluster-a",
            "model_repository": manifest["repository"],
            "model_revision": manifest["revision"],
            "manifest_digest": digest,
            "shared_path": "/shared/model",
            "local_path": "/local/model",
            "expected_nodes_exact": 1,
        }
        receipt = {
            "state": "ready",
            "clusterId": "cluster-a",
            "model": {
                "repository": manifest["repository"],
                "revision": manifest["revision"],
            },
            "verification": {"mode": "full-sha256", "requireAllTargets": True},
            "manifestDigest": digest,
            "sharedPath": "/shared/model",
            "localPath": "/local/model",
            "nodes": {
                "required": 1,
                "verified": 1,
                "results": [{"node": "worker-0", "state": "reused"}],
            },
        }
        with patch.object(
            validate_model_stage_receipt,
            "load_targets",
            return_value={"test": target},
        ):
            self.assertEqual(
                validate_model_stage_receipt.validate(
                    "test", manifest["repository"], receipt, manifest
                ),
                "/local/model",
            )

            receipt["clusterId"] = "cluster-b"
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                validate_model_stage_receipt.validate(
                    "test", manifest["repository"], receipt, manifest
                )

    def test_receipt_must_match_selected_model(self) -> None:
        with patch.object(
            validate_model_stage_receipt,
            "load_targets",
            return_value={
                "test": {
                    "models": {
                        "owner/model-a": {
                            "model_repository": "owner/model-a"
                        }
                    }
                }
            },
        ), self.assertRaisesRegex(RuntimeError, "is not staged"):
            validate_model_stage_receipt.validate(
                "test", "owner/model-b", {}, {}
            )


if __name__ == "__main__":
    unittest.main()

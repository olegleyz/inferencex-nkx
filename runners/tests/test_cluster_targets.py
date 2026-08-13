from __future__ import annotations

import hashlib
import json
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

    def test_mismatched_profile_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "target/profile mismatch"):
            resolve_cluster_target.resolve("lepton-gb300", "nkx-gb300")

    def test_legacy_nkx_profile_resolves(self) -> None:
        result = resolve_cluster_target.resolve("", "nkx-gb300")
        self.assertEqual(result["target-cluster"], "nkx-gb300")


class TargetMatrixTests(unittest.TestCase):
    def _row(self) -> dict[str, object]:
        return {
            "image": validate_cluster_target_matrix.IMAGE,
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
                validate_model_stage_receipt.validate("test", receipt, manifest),
                "/local/model",
            )

            receipt["clusterId"] = "cluster-b"
            with self.assertRaisesRegex(RuntimeError, "does not match"):
                validate_model_stage_receipt.validate("test", receipt, manifest)


if __name__ == "__main__":
    unittest.main()

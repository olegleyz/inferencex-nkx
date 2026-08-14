from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path


RUNNERS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RUNNERS))

import validate_srt_worker_logs  # noqa: E402


class ValidateSrtWorkerLogsTests(unittest.TestCase):
    def test_clean_worker_logs_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "node_prefill_w0.out").write_text("Model ready\n")
            self.assertEqual(validate_srt_worker_logs.find_startup_failures(root), {})

    def test_cuda_startup_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "node_prefill_w3.out"
            path.write_text(
                "EngineCore failed to start.\n"
                "RuntimeError: CUDA_ERROR_LAUNCH_FAILED\n"
            )
            failures = validate_srt_worker_logs.find_startup_failures(root)
            self.assertEqual(list(failures), [path])
            self.assertEqual(len(failures[path]), 2)

    def test_non_worker_logs_are_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "benchmark.out").write_text("CUDA_ERROR_LAUNCH_FAILED\n")
            self.assertEqual(validate_srt_worker_logs.find_startup_failures(root), {})

    def test_native_kv_transport_failure_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "node_decode_w0.out"
            path.write_text(
                "ibv_reg_mr(address=0x1) failed: Bad address\n"
                "NIXL_ERR_BACKEND: transfer_setup_failed\n"
                "Request has invalid KV blocks\n"
            )
            failures = validate_srt_worker_logs.find_startup_failures(root)
            self.assertEqual(list(failures), [path])
            self.assertEqual(len(failures[path]), 3)


if __name__ == "__main__":
    unittest.main()

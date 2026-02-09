#!/usr/bin/env python3

import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable or "python3"


class AnalysisScriptTests(unittest.TestCase):
    def test_evaluate_mape_compare_and_leave_one_out(self):
        with tempfile.TemporaryDirectory() as td:
            measured = os.path.join(td, "measured.csv")
            with open(measured, "w") as f:
                f.write(
                    "workload,gpu,problem_size,stage,tile_size,measured_speedup\n"
                    "gemm,A40,2048,1,64,1.0\n"
                    "gemm,A40,2048,2,64,1.1\n"
                    "gemm,H100_SXM5,2048,1,64,1.0\n"
                    "gemm,H100_SXM5,2048,2,64,1.2\n"
                )

            subprocess.check_call(
                [
                    PY,
                    os.path.join(ROOT, "scripts", "evaluate_mape.py"),
                    "--measured",
                    measured,
                    "--compare-l2-modes",
                    "--leave-one-out",
                ]
            )

    def test_anomaly_analysis_topk(self):
        with tempfile.TemporaryDirectory() as td:
            measured = os.path.join(td, "measured.csv")
            out = os.path.join(td, "anomaly.csv")
            with open(measured, "w") as f:
                f.write(
                    "workload,gpu,problem_size,stage,tile_size,measured_speedup\n"
                    "gemm,A40,2048,1,64,1.0\n"
                    "gemm,A40,2048,2,64,1.1\n"
                    "stencil,A40,1048576,1,256,1.0\n"
                    "stencil,A40,1048576,2,256,1.2\n"
                )

            subprocess.check_call(
                [
                    PY,
                    os.path.join(ROOT, "scripts", "anomaly_analysis.py"),
                    "--measured",
                    measured,
                    "--top-k",
                    "2",
                    "--output",
                    out,
                ]
            )

            self.assertTrue(os.path.exists(out))

    def test_run_nsight_dry_run(self):
        with tempfile.TemporaryDirectory() as td:
            subprocess.check_call(
                [
                    PY,
                    os.path.join(ROOT, "scripts", "run_nsight_profiling.py"),
                    "--dry-run",
                    "--build-dir",
                    "build",
                    "--output-dir",
                    td,
                    "--n-values",
                    "2048",
                    "--lengths",
                    "1048576",
                    "--tile-sizes",
                    "16",
                    "--stages",
                    "2",
                ]
            )


if __name__ == "__main__":
    unittest.main()

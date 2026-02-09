#!/usr/bin/env python3

import csv
import json
import os
import subprocess
import sys
import tempfile
import unittest


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = sys.executable or "python3"


class DataPipelineTests(unittest.TestCase):
    def test_build_measured_speedup_csv(self):
        with tempfile.TemporaryDirectory() as td:
            raw = os.path.join(td, "raw.csv")
            out = os.path.join(td, "measured.csv")

            with open(raw, "w") as f:
                f.write(
                    "workload,gpu,problem_size,variant,stage,tile_size,time_ms,gflops,correct,max_abs_error\n"
                    "gemm,A40,1024,V1,1,16,10.0,0,1,0\n"
                    "gemm,A40,1024,V3,2,16,8.0,0,1,0\n"
                    "gemm,A40,1024,V3,3,16,5.0,0,1,0\n"
                )

            subprocess.check_call(
                [
                    PY,
                    os.path.join(ROOT, "scripts", "build_measured_speedup_csv.py"),
                    "--raw",
                    raw,
                    "--output",
                    out,
                ]
            )

            with open(out, "r") as f:
                rows = list(csv.DictReader(f))

            self.assertEqual(len(rows), 3)
            speeds = {int(r["stage"]): float(r["measured_speedup"]) for r in rows}
            self.assertAlmostEqual(speeds[1], 1.0, places=6)
            self.assertAlmostEqual(speeds[2], 1.25, places=6)
            self.assertAlmostEqual(speeds[3], 2.0, places=6)

    def test_extract_l2_params(self):
        with tempfile.TemporaryDirectory() as td:
            inp = os.path.join(td, "pointer.csv")
            out_json = os.path.join(td, "l2.json")

            with open(inp, "w") as f:
                f.write(
                    "gpu,size_bytes,elements,stride,iterations,cycles_per_load,ns_per_load\n"
                    "A40,1048576,262144,32,1000,120,80\n"
                    "A40,2097152,524288,32,1000,125,84\n"
                    "A40,4194304,1048576,32,1000,130,87\n"
                    "A40,8388608,2097152,32,1000,180,120\n"
                    "A40,16777216,4194304,32,1000,350,230\n"
                    "A40,33554432,8388608,32,1000,390,255\n"
                )

            subprocess.check_call(
                [
                    PY,
                    os.path.join(ROOT, "scripts", "extract_l2_params.py"),
                    "--input",
                    inp,
                    "--output-json",
                    out_json,
                ]
            )

            with open(out_json, "r") as f:
                data = json.load(f)

            self.assertIn("A40", data)
            self.assertGreater(data["A40"]["l2_eff_mb"], 0.0)
            self.assertGreater(data["A40"]["dram_latency_cycles"], data["A40"]["l2_latency_cycles"])


if __name__ == "__main__":
    unittest.main()

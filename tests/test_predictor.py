#!/usr/bin/env python3

import os
import sys
import unittest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from predictor import predict_one
from triton_prefilter import prune_num_stages


class PredictorTests(unittest.TestCase):
    def test_stage1_is_baseline(self):
        pred = predict_one("gemm", "A40", 2048, 1, tile_size=64)
        self.assertTrue(pred["valid"])
        self.assertAlmostEqual(pred["pred_speedup"], 1.0, places=6)

    def test_v100_has_no_cp_async(self):
        pred = predict_one("gemm", "V100", 2048, 2, tile_size=64)
        self.assertFalse(pred["valid"])
        self.assertEqual(pred["reason"], "gpu_has_no_cp_async")
        self.assertAlmostEqual(pred["pred_speedup"], 1.0, places=6)

    def test_prefilter_keeps_stage1_on_v100(self):
        out = prune_num_stages("gemm", "V100", 2048, [1, 2, 3, 4], epsilon=0.03, tile_size=64)
        self.assertEqual(out["kept"], [1])

    def test_w_conc_over_l2_grows_with_stage(self):
        p2 = predict_one("gemm", "A40", 4096, 2, tile_size=64)
        p3 = predict_one("gemm", "A40", 4096, 3, tile_size=64)
        self.assertTrue(p3["w_conc_over_l2"] >= p2["w_conc_over_l2"])

    def test_l2_mode_plumbed(self):
        pred = predict_one("gemm", "A40", 2048, 2, tile_size=64, l2_mode="nominal")
        self.assertEqual(pred["l2_mode"], "nominal")


if __name__ == "__main__":
    unittest.main()

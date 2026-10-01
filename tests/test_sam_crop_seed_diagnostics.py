from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from tools.sam_crop_seed_diagnostics import compute_seed_resampling_diagnostics, main


class SeedResamplingDiagnosticsTests(unittest.TestCase):
    def test_constant_oversized_rectangular_seed_preserves_original(self):
        seed = np.ones((1200, 1500), dtype=bool)
        before = seed.copy()
        receipt, proxy = compute_seed_resampling_diagnostics(seed)
        self.assertEqual(proxy.shape, seed.shape)
        self.assertTrue(np.array_equal(seed, before))
        self.assertTrue(np.array_equal(proxy, seed))
        self.assertFalse(proxy.flags.writeable)
        self.assertEqual(receipt["mask_conditioning_side"], 1152)
        self.assertEqual(receipt["low_mask_side"], 288)
        self.assertEqual(receipt["low_grid_native_proxy_threshold_0"]["iou"], 1.0)
        self.assertFalse(receipt["gpu_kernel_equivalence_claimed"])
        self.assertIn("not an actual tracker", receipt["scope"])
        json.dumps(receipt, allow_nan=False)

    def test_thin_component_proxy_change_is_reported_without_quality_claim(self):
        seed = np.zeros((1200, 1600), dtype=bool)
        seed[300:380, 300:380] = True
        seed[600:900, 805] = True
        receipt, proxy = compute_seed_resampling_diagnostics(seed)
        metrics = receipt["low_grid_native_proxy_threshold_0"]
        self.assertEqual(metrics["reference_components_8"], 2)
        self.assertEqual(metrics["reference_components_without_projected_overlap"], 1)
        self.assertGreater(metrics["false_negative_pixels"], 0)
        self.assertTrue(proxy[330, 330])
        self.assertFalse(proxy[750, 805])
        self.assertIn("does not measure", receipt["interpretation"])
        self.assertIn("prove tracker information loss", receipt["interpretation"])

    def test_bfloat16_emulation_is_explicit_and_uses_cpu_interpolation(self):
        seed = np.zeros((100, 180), dtype=bool)
        seed[25:75, 30:150] = True
        receipt, proxy = compute_seed_resampling_diagnostics(seed, conditioning_dtype="bfloat16")
        self.assertTrue(receipt["bfloat16_quantization_emulated"])
        self.assertEqual(receipt["interpolation_compute_dtype"], "float32_cpu")
        self.assertEqual(proxy.shape, seed.shape)
        self.assertGreater(receipt["low_grid_native_proxy_threshold_0"]["iou"], 0.9)

    def test_empty_prompt_and_pixel_budget_are_explicit_errors(self):
        with self.assertRaisesRegex(ValueError, "unavailable coverage"):
            compute_seed_resampling_diagnostics(np.zeros((32, 32), dtype=bool))
        with self.assertRaisesRegex(ValueError, "pixel budget"):
            compute_seed_resampling_diagnostics(np.ones((32, 32), dtype=bool), max_native_pixels=512)
        with self.assertRaisesRegex(ValueError, "Boolean"):
            compute_seed_resampling_diagnostics(np.ones((32, 32), dtype=np.uint8))

    def test_cli_retains_seed_and_writes_receipt_and_proxy(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            seed = np.zeros((128, 160), dtype=bool)
            seed[40:90, 50:110] = True
            np.savez_compressed(root / "native.npz", seed=seed)
            self.assertEqual(main([
                "--seed", str(root / "native.npz"), "--output-json", str(root / "proxy.json"),
                "--output-proxy-mask", str(root / "proxy.npy"),
            ]), 0)
            with np.load(root / "native.npz", allow_pickle=False) as loaded:
                self.assertTrue(np.array_equal(loaded["seed"], seed))
            receipt = json.loads((root / "proxy.json").read_text())
            self.assertTrue(receipt["research_only"])
            self.assertEqual(np.load(root / "proxy.npy", allow_pickle=False).shape, seed.shape)


if __name__ == "__main__":
    unittest.main()

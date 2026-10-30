"""Contract coverage for the self-contained weighted CPU/GPU sweep policies."""
from __future__ import annotations

from collections import Counter
import importlib
import importlib.util
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest import mock

import numpy as np


POLICIES = Path(__file__).resolve().parents[1] / "XTA/examples/external_augmentations"
PROFILES = ("light", "baseline", "heavy", "superheavy")


def policy_module(backend: str, profile: str):
    return importlib.import_module(f"XTA.examples.external_augmentations.{backend}_{profile}")


def representative_seeds(module):
    """Exercise each selected preset, elastic replay, and its rare CLAHE stage."""
    found = {}
    for seed in range(10000):
        params = module.CPUAugmentation._sample_parameters(seed, 48, 56)
        profile = params["profile"]
        found.setdefault((profile, "ordinary"), seed)
        if params["elastic"]:
            found.setdefault((profile, "elastic"), seed)
        if params["clahe_clip_limit"] is not None:
            found.setdefault((profile, "clahe"), seed)
        if len(found) == 12:
            break
    return tuple(found.values())


class SweepProfileTests(unittest.TestCase):
    def test_weighted_selection_is_shared_and_reproducible(self):
        cpu = policy_module("CPU", "sweep")
        gpu = policy_module("GPU", "sweep")
        choices = [cpu._select_profile(seed) for seed in range(20000)]
        self.assertEqual(choices, [gpu._select_profile(seed) for seed in range(20000)])
        self.assertEqual(choices, [cpu._select_profile(seed) for seed in range(20000)])
        counts = Counter(choices)
        for profile, probability in zip(PROFILES, (.4, .3, .2, .1)):
            self.assertAlmostEqual(counts[profile] / len(choices), probability, delta=.015)

    def test_sampled_parameters_match_the_selected_shipped_preset(self):
        cpu = policy_module("CPU", "sweep")
        gpu = policy_module("GPU", "sweep")
        for seed in representative_seeds(cpu):
            with self.subTest(seed=seed):
                actual = cpu.CPUAugmentation._sample_parameters(seed, 48, 56)
                self.assertEqual(actual, gpu.GPUAugmentation._sample_parameters(seed, 48, 56))
                profile = actual.pop("profile")
                actual.pop("noise_multiplier_range")
                expected = policy_module("CPU", profile).CPUAugmentation._sample_parameters(seed, 48, 56)
                self.assertEqual(actual, expected)

    def test_cpu_outputs_match_selected_preset_and_preserve_channels_and_empty_masks(self):
        module = policy_module("CPU", "sweep")
        image = (np.arange(48 * 56, dtype=np.uint16).reshape(48, 56) % 256).astype(np.uint8)
        mask = np.zeros((48, 56), dtype=np.uint8)
        mask[12:36, 14:42] = 1
        for channels in (1, 5):
            source = image if channels == 1 else np.stack([image] * channels, axis=-1)
            for seed in representative_seeds(module):
                with self.subTest(channels=channels, seed=seed):
                    sweep = module.build_augmentation()
                    preset = policy_module("CPU", module._select_profile(seed)).build_augmentation()
                    sweep.set_random_seed(seed)
                    preset.set_random_seed(seed)
                    actual, expected = sweep(image=source, mask=mask), preset(image=source, mask=mask)
                    np.testing.assert_array_equal(actual["image"], expected["image"])
                    np.testing.assert_array_equal(actual["mask"], expected["mask"])
                    self.assertEqual(actual["image"].shape, source.shape)
                    self.assertEqual(actual["mask"].dtype, np.uint8)
                    self.assertTrue(set(np.unique(actual["mask"])).issubset({0, 1}))
                    sweep.set_random_seed(seed)
                    empty = sweep(image=source, mask=np.zeros_like(mask))
                    np.testing.assert_array_equal(empty["image"], actual["image"])
                    self.assertFalse(np.any(empty["mask"]))

    def test_policies_load_from_individual_copies(self):
        from XTA.pta_augmentation import load_augmentation_definition, load_gpu_augmentation_definition

        with tempfile.TemporaryDirectory() as directory:
            for backend, loader in (("CPU", load_augmentation_definition), ("GPU", load_gpu_augmentation_definition)):
                path = Path(directory) / f"{backend}_sweep.py"
                shutil.copyfile(POLICIES / path.name, path)
                loaded = loader(str(path))
                self.assertEqual(loaded.path.resolve(), path.resolve())
                self.assertTrue(loaded.content_sha256)

    @unittest.skipUnless(os.environ.get("XTA_RUN_EXTERNAL_AUGMENTATION_CUDA") == "1",
                         "set XTA_RUN_EXTERNAL_AUGMENTATION_CUDA=1 with a reserved GPU")
    def test_gpu_matches_selected_presets_and_is_invariant_to_batch_splitting(self):
        import torch

        self.assertTrue(torch.cuda.is_available())
        module = policy_module("GPU", "sweep")
        cpu = policy_module("CPU", "sweep")
        chosen = {}
        for seed in representative_seeds(cpu):
            chosen.setdefault(module._select_profile(seed), seed)
        seeds = (None, *(chosen[profile] for profile in PROFILES))
        image = (np.arange(48 * 56, dtype=np.uint16).reshape(48, 56) % 256).astype(np.uint8)
        mask = np.zeros((48, 56), dtype=np.uint8)
        mask[12:36, 14:42] = 1
        with mock.patch.dict(os.environ, {"PTA_GPU_TORCH_COMPILE": "0"}):
            sweep = module.build_gpu_augmentation(device="cuda:0", batch_size=16)
            images, masks = sweep.apply_batch_many(images=(image, 255 - image), masks=(mask, mask),
                                                   seeds=(seeds, tuple(reversed(seeds))), output_size=(48, 56))
            offset = 0
            for source, source_seeds in ((image, seeds), (255 - image, tuple(reversed(seeds)))):
                for seed in source_seeds:
                    profile = "light" if seed is None else module._select_profile(seed)
                    preset = policy_module("GPU", profile).build_gpu_augmentation(device="cuda:0", batch_size=1)
                    expected_image, expected_mask = preset.apply_batch(image=source, mask=mask,
                                                                       seeds=(seed,), output_size=(48, 56))
                    self.assertTrue(torch.equal(images[offset], expected_image[0]))
                    self.assertTrue(torch.equal(masks[offset], expected_mask[0]))
                    single_image, single_mask = sweep.apply_batch(image=source, mask=mask,
                                                                  seeds=(seed,), output_size=(48, 56))
                    self.assertTrue(torch.equal(images[offset], single_image[0]))
                    self.assertTrue(torch.equal(masks[offset], single_mask[0]))
                    offset += 1


if __name__ == "__main__":
    unittest.main()

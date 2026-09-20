from __future__ import annotations

import ast
from collections import Counter
import copy
import importlib
import io
import math
import os
import random
import re
import sys
import tokenize
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = ROOT / "XTA" / "examples" / "external_augmentations"
GPU_PROFILES = (
    ("light", EXAMPLES / "GPU_light.py"),
    ("baseline", EXAMPLES / "GPU_baseline.py"),
    ("heavy", EXAMPLES / "GPU_heavy.py"),
    ("superheavy", EXAMPLES / "GPU_superheavy.py"),
)
CPU_PROFILES = tuple(
    (profile, EXAMPLES / f"CPU_{profile}.py")
    for profile, _path in GPU_PROFILES
)


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _class_method(path: Path, class_name: str, method_name: str) -> ast.FunctionDef:
    for node in _tree(path).body:
        if not isinstance(node, ast.ClassDef) or node.name != class_name:
            continue
        for member in node.body:
            if isinstance(member, ast.FunctionDef) and member.name == method_name:
                return member
    raise AssertionError(f"{path}: missing {class_name}.{method_name}")


def _isolated_sample_function(
    path: Path,
    *,
    class_name: str,
) -> object:
    function = copy.deepcopy(
        _class_method(path, class_name, "_sample_parameters")
    )
    function.name = "sample_parameters"
    function.decorator_list = []
    policy_tree = _tree(path)
    helpers = [copy.deepcopy(node) for node in policy_tree.body
               if (isinstance(node, ast.FunctionDef)
                   and node.name in {'_subseed', '_sample_bit_depth', '_sample_clahe'})
               or (isinstance(node, ast.Assign)
                   and any(isinstance(target, ast.Name) and target.id.startswith(('BIT_DEPTH', 'CLAHE_'))
                           for target in node.targets))]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__",
                names=[ast.alias(name="annotations")],
                level=0,
            ),
            ast.Import(names=[ast.alias(name="random")]),
            *helpers,
            function,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    namespace: dict[str, object] = {}
    exec(compile(module, str(path), "exec"), namespace)
    return namespace["sample_parameters"]


def _selectors(sample: dict[str, object]) -> tuple[object, ...]:
    return (
        int(sample["d4"]),
        bool(float(sample["scale"]) < 1.0),
        bool(sample["elastic"]),
        bool(float(sample["brightness"]) != 1.0),
        bool(float(sample["blur_sigma"]) != 0.0),
        int(sample["noise_family"]),
        bool(float(sample["salt_pepper_amount"]) != 0.0),
    )


def _magnitudes(sample: dict[str, object], *, height: int, width: int) -> tuple[float, ...]:
    noise_strength = float(sample["noise_strength"])
    noise_magnitude = (
        abs(noise_strength - 1.0)
        if int(sample["noise_family"]) == 2
        else noise_strength
    )
    return (
        abs(float(sample["rotation"])),
        abs(float(sample["scale"]) - 1.0),
        abs(float(sample["translate_x"])) / float(width),
        abs(float(sample["translate_y"])) / float(height),
        abs(float(sample["shear_x"])),
        abs(float(sample["shear_y"])),
        abs(float(sample["brightness"]) - 1.0),
        float(sample["blur_sigma"]),
        noise_magnitude,
        float(sample["salt_pepper_amount"]),
    )


class ExternalAugmentationExampleTests(unittest.TestCase):
    def test_gpu_files_have_one_supported_export_and_no_versioned_definition_names(self) -> None:
        supported_exports = {
            "custom_transforms",
            "augmentation",
            "build_augmentation",
            "build_gpu_augmentation",
        }
        version_token = re.compile(r"\bv\d+(?:\.\d+)*\b|Augments[_A-Za-z]*V?\d+", re.IGNORECASE)
        versioned_name = re.compile(r"(?:^|_)v\d+(?:_|$)|V\d+")

        for profile, path in GPU_PROFILES:
            with self.subTest(profile=profile):
                source = path.read_text(encoding="utf-8")
                tree = ast.parse(source, filename=str(path))
                top_level_names = {
                    node.name
                    for node in tree.body
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))
                }
                self.assertEqual(top_level_names & supported_exports, {"build_gpu_augmentation"})
                self.assertIn("GPUAugmentation", top_level_names)
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        self.assertIsNone(versioned_name.search(node.name), node.name)
                    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        docstring = ast.get_docstring(node, clean=False) or ""
                        self.assertIsNone(version_token.search(docstring), docstring)
                comments = (
                    token.string
                    for token in tokenize.generate_tokens(io.StringIO(source).readline)
                    if token.type == tokenize.COMMENT
                )
                for comment in comments:
                    self.assertIsNone(version_token.search(comment), comment)

    def test_gpu_profiles_preserve_branch_selection_and_increase_magnitudes(self) -> None:
        height, width = 257, 383
        samplers = [
            _isolated_sample_function(path, class_name="GPUAugmentation")
            for _profile, path in GPU_PROFILES
        ]
        for seed in range(2048):
            samples = [
                sampler(seed, height, width)  # type: ignore[operator]
                for sampler in samplers
            ]
            self.assertTrue(all(_selectors(sample) == _selectors(samples[0]) for sample in samples[1:]))
            magnitudes = [
                _magnitudes(sample, height=height, width=width)
                for sample in samples
            ]
            for field_values in zip(*magnitudes):
                self.assertTrue(
                    all(left <= right + 1e-12 for left, right in zip(field_values, field_values[1:])),
                    (seed, field_values),
                )

    def test_cpu_profiles_match_their_gpu_parameter_graphs(self) -> None:
        for (profile, gpu_path), (_cpu_profile, cpu_path) in zip(
            GPU_PROFILES,
            CPU_PROFILES,
        ):
            gpu_sample = _isolated_sample_function(
                gpu_path,
                class_name="GPUAugmentation",
            )
            cpu_sample = _isolated_sample_function(
                cpu_path,
                class_name="CPUAugmentation",
            )

            for seed in range(512):
                with self.subTest(profile=profile, seed=seed):
                    self.assertEqual(
                        cpu_sample(seed, 257, 383),  # type: ignore[operator]
                        gpu_sample(seed, 257, 383),  # type: ignore[operator]
                    )

    def test_profile_bit_depth_frequencies_and_clahe_defaults(self) -> None:
        for (profile, path), probability, depths, clips in zip(
            CPU_PROFILES, (.2, .3, .4, .5), ((8,), (4, 8), (2, 4, 8), (1, 2, 4, 8)),
            ((1., 2.), (1., 4.), (2., 6.), (3., 8.)),
        ):
            sampler = _isolated_sample_function(path, class_name="CPUAugmentation")
            random.seed(1337)
            state = random.getstate()
            samples = [sampler(seed, 96, 112) for seed in range(20000)]
            self.assertEqual(random.getstate(), state)
            counts = Counter(sample['bit_depth'] for sample in samples)
            with self.subTest(profile=profile):
                self.assertEqual(set(counts), set(depths) | {None})
                self.assertAlmostEqual(counts[None] / len(samples), 1 - probability, delta=.012)
                for depth in depths:
                    self.assertAlmostEqual(counts[depth] / len(samples), probability / len(depths), delta=.012)
                active = [sample['clahe_clip_limit'] for sample in samples if sample['clahe_clip_limit'] is not None]
                self.assertAlmostEqual(len(active) / len(samples), .01, delta=.003)
                self.assertTrue(all(clips[0] <= value <= clips[1] for value in active))
                self.assertAlmostEqual(sum(sample['elastic'] for sample in samples) / len(samples), .3, delta=.012)

    def test_cpu_profiles_are_seeded_deterministic_and_preserve_empty_masks(self) -> None:
        loaded_cv2 = sys.modules.get("cv2")
        if loaded_cv2 is not None and type(loaded_cv2).__name__ == "_StubModule":
            self.skipTest("CPU example execution requires real OpenCV, not import stubs")
        try:
            import cv2  # noqa: F401
        except ModuleNotFoundError as exc:  # pragma: no cover - dependency-gated host
            if exc.name != "cv2":
                raise
            self.skipTest(f"CPU example dependencies are unavailable: {exc}")

        height, width = 64, 80
        image = np.arange(height * width, dtype=np.uint16).reshape(height, width)
        image = np.asarray(image % 256, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        for profile, path in CPU_PROFILES:
            with self.subTest(profile=profile):
                module = importlib.import_module(
                    f"XTA.examples.external_augmentations.{path.stem}"
                )
                policy = module.build_augmentation()
                policy.set_random_seed(12345)
                first = policy(image=image, mask=mask)
                policy.set_random_seed(12345)
                second = policy(image=image, mask=mask)

                np.testing.assert_array_equal(first["image"], second["image"])
                np.testing.assert_array_equal(first["mask"], second["mask"])
                self.assertEqual(first["image"].shape, image.shape)
                self.assertEqual(first["image"].dtype, np.uint8)
                self.assertEqual(first["mask"].dtype, np.uint8)
                self.assertEqual(int(np.count_nonzero(first["mask"])), 0)

    @unittest.skipUnless(
        os.environ.get("XTA_RUN_EXTERNAL_AUGMENTATION_CUDA", "").strip() == "1",
        "set XTA_RUN_EXTERNAL_AUGMENTATION_CUDA=1 on a CUDA PyTorch host",
    )
    def test_gpu_profiles_execute_deterministically_on_cuda(self) -> None:
        try:
            import torch
        except ModuleNotFoundError as exc:  # pragma: no cover - hardware gate
            self.fail(f"CUDA example gate was enabled without PyTorch: {exc}")
        self.assertTrue(torch.cuda.is_available(), "CUDA example gate requires torch.cuda")

        height, width = 96, 112
        image = np.arange(height * width, dtype=np.uint16).reshape(height, width)
        image = np.asarray(image % 256, dtype=np.uint8)
        mask = np.zeros((height, width), dtype=np.uint8)
        mask[24:72, 28:84] = 1
        seeds = (None, 7, 410, 586)

        outputs: list[object] = []
        with mock.patch.dict(os.environ, {"PTA_GPU_TORCH_COMPILE": "0"}):
            for profile, path in GPU_PROFILES:
                with self.subTest(profile=profile):
                    module = importlib.import_module(
                        f"XTA.examples.external_augmentations.{path.stem}"
                    )
                    policy = module.build_gpu_augmentation(
                        device="cuda:0",
                        batch_size=len(seeds),
                    )
                    first_images, first_masks = policy.apply_batch(
                        image=image,
                        mask=mask,
                        seeds=seeds,
                        output_size=(height, width),
                    )
                    second_images, second_masks = policy.apply_batch(
                        image=image,
                        mask=mask,
                        seeds=seeds,
                        output_size=(height, width),
                    )
                    torch.cuda.synchronize()
                    self.assertEqual(
                        tuple(first_images.shape),
                        (len(seeds), 1, height, width),
                    )
                    self.assertEqual(
                        tuple(first_masks.shape),
                        (len(seeds), height, width),
                    )
                    self.assertEqual(first_images.dtype, torch.uint8)
                    self.assertEqual(first_masks.dtype, torch.uint8)
                    self.assertTrue(torch.equal(first_images, second_images))
                    self.assertTrue(torch.equal(first_masks, second_masks))
                    self.assertLessEqual(int(first_masks.max().item()), 1)
                    outputs.append(first_images.detach().cpu())

        self.assertTrue(
            all(not torch.equal(left, right) for left, right in zip(outputs, outputs[1:])),
            "adjacent magnitude profiles unexpectedly produced identical seeded batches",
        )


class GPUAugmentationGaussianMathTests(unittest.TestCase):
    """Check GPU policy math on CPU without constructing a CUDA policy."""

    @classmethod
    def setUpClass(cls) -> None:
        loaded_torch = sys.modules.get("torch")
        if loaded_torch is not None and type(loaded_torch).__name__ == "_StubModule":
            raise unittest.SkipTest("Gaussian execution requires real PyTorch, not import stubs")
        try:
            import torch
        except ModuleNotFoundError as exc:
            if exc.name != "torch":
                raise
            raise unittest.SkipTest("Gaussian execution requires PyTorch") from exc

        cls.torch = torch
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, cls.previous_threads)
        cls.modules = [
            (
                profile,
                importlib.import_module(f"XTA.examples.external_augmentations.{path.stem}"),
            )
            for profile, path in GPU_PROFILES
        ]

    def _square_reference(self, sample: object, sigma: float) -> object:
        # Build the isotropic 2D density independently of the policy's 1D kernel.
        radius = max(1, int(math.ceil(3.0 * max(0.05, sigma))))
        coords = np.arange(-radius, radius + 1, dtype=np.float64)
        kernel = np.exp(
            -(coords[:, None] ** 2 + coords[None, :] ** 2)
            / (2.0 * max(0.05, sigma) ** 2)
        )
        kernel /= kernel.sum()
        channels = int(sample.shape[1])
        weight = self.torch.from_numpy(kernel.astype(np.float32))
        weight = weight.view(1, 1, *weight.shape).repeat(channels, 1, 1, 1)
        pad_mode = "reflect" if min(sample.shape[-2:]) > radius else "replicate"
        padded = self.torch.nn.functional.pad(
            sample, (radius, radius, radius, radius), mode=pad_mode
        )
        return self.torch.nn.functional.conv2d(padded, weight, groups=channels)

    def test_blur_matches_square_gaussian_for_channels_and_boundary_modes(self) -> None:
        cases = (
            ((2, 3, 29, 31), 5.0),
            ((1, 1, 9, 17), 1.25),
            ((1, 3, 7, 31), 5.0),
            ((1, 2, 31, 7), 5.0),
            ((1, 3, 1, 2), 8.0),
            ((1, 1, 49, 53), 8.0),
            ((1, 3, 15, 17), 5.0),
            ((1, 3, 3, 4), 0.1),
        )
        generator = self.torch.Generator(device="cpu").manual_seed(1729)
        for shape, sigma in cases:
            sample = self.torch.rand(shape, generator=generator)
            expected = self._square_reference(sample, sigma)
            for profile, module in self.modules:
                with self.subTest(profile=profile, shape=shape, sigma=sigma):
                    actual = module._blur_sample(sample, sigma)
                    self.assertEqual(tuple(actual.shape), shape)
                    self.torch.testing.assert_close(
                        actual, expected, rtol=4e-6, atol=4e-6
                    )

    def test_disabled_blur_preserves_the_input_tensor(self) -> None:
        sample = self.torch.arange(12, dtype=self.torch.float32).reshape(1, 1, 3, 4)
        for profile, module in self.modules:
            for sigma in (0.0, 0.05):
                with self.subTest(profile=profile, sigma=sigma):
                    self.assertIs(module._blur_sample(sample, sigma), sample)

    def test_elastic_field_preserves_seed_and_normalizes_per_axis_rms(self) -> None:
        seed = 31891
        for (profile, module), fraction in zip(
            self.modules, (.005, .01, .015, .02)
        ):
            policy = module.GPUAugmentation.__new__(module.GPUAugmentation)
            policy.device = self.torch.device("cpu")
            for height, width in ((27, 31), (7, 31), (31, 7), (1, 2), (512, 768)):
                with self.subTest(profile=profile, shape=(height, width)):
                    actual = policy._elastic_displacement(seed, height, width)
                    if min(height, width) < 2:
                        self.assertFalse(bool(actual.any()))
                    else:
                        rms = actual.double().square().mean((-2, -1)).sqrt()
                        self.torch.testing.assert_close(rms, self.torch.full_like(rms, fraction * min(height, width)),
                                                        rtol=2e-4, atol=1e-6)
                        self.torch.testing.assert_close(actual.mean((-2, -1)), self.torch.zeros(2),
                                                        rtol=0, atol=3e-4)
                    self.assertTrue(
                        self.torch.equal(
                            actual, policy._elastic_displacement(seed, height, width)
                        )
                    )


class ExternalAugmentationIntensityMathTests(unittest.TestCase):
    """Execute the standalone policy math against independent intensity fixtures."""

    @classmethod
    def setUpClass(cls):
        for dependency in ('cv2', 'torch'):
            loaded = sys.modules.get(dependency)
            if loaded is not None and type(loaded).__name__ == '_StubModule':
                raise unittest.SkipTest(f'Intensity math requires real {dependency}')
        try:
            import cv2
            import torch
        except ModuleNotFoundError as exc:
            raise unittest.SkipTest(f'Intensity math dependencies are unavailable: {exc}') from exc
        cls.cv2, cls.torch = cv2, torch
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(1)
        cls.addClassCleanup(torch.set_num_threads, previous_threads)
        cls.modules = [(profile,
                        importlib.import_module(f'XTA.examples.external_augmentations.CPU_{profile}'),
                        importlib.import_module(f'XTA.examples.external_augmentations.GPU_{profile}'))
                       for profile, _ in CPU_PROFILES]

    @staticmethod
    def u8(image):
        return np.rint(np.clip(image, 0, 1) * 255).astype(np.uint8)

    def quantize_pair(self, cpu, gpu, source, eligible, bits):
        before = source.copy()
        expected = cpu._adaptive_bit_depth_numpy(source, eligible, bits)
        actual = gpu._adaptive_bit_depth_torch(self.torch.from_numpy(source.copy()),
                                              self.torch.from_numpy(eligible.copy()), bits).numpy()
        np.testing.assert_array_equal(self.u8(actual), self.u8(expected))
        np.testing.assert_allclose(actual, expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(actual[~eligible], 0)
        np.testing.assert_array_equal(source, before)
        return actual

    def test_eight_bit_range_stretch_excludes_protected_background(self):
        for profile, cpu, gpu in self.modules:
            with self.subTest(profile=profile):
                source = np.array([0, 10, 20, 30, 40], np.float32) / 255
                result = self.quantize_pair(cpu, gpu, source, source > 0, 8)
                np.testing.assert_array_equal(self.u8(result), [0, 0, 85, 170, 255])
                source[0] = .98
                result = self.quantize_pair(cpu, gpu, source, np.array([False, True, True, True, True]), 8)
                np.testing.assert_array_equal(self.u8(result), [0, 0, 85, 170, 255])
                source = np.array([0, 80, 98, 125], np.float32) / 255
                result = self.quantize_pair(cpu, gpu, source, source > 0, 8)
                np.testing.assert_array_equal(self.u8(result), [0, 0, 102, 255])

    def test_quantile_palettes_monotonicity_and_indivisible_histogram_ties(self):
        fixtures = (np.arange(256, dtype=np.float32),
                    np.array([0] + [17] * 900 + [18] * 2 + [19], dtype=np.float32),
                    np.array([0, 40] + [255] * 999, dtype=np.float32))
        for profile, cpu, gpu in self.modules:
            for bins in fixtures:
                source = bins / 255
                for bits in (1, 2, 4):
                    with self.subTest(profile=profile, bits=bits, values=len(source)):
                        output = self.u8(self.quantize_pair(cpu, gpu, source, source > 0, bits))
                        palette = np.arange(1 << bits) * (255 // ((1 << bits) - 1))
                        self.assertTrue(set(output) <= set(palette))
                        self.assertEqual((int(output.min()), int(output.max())), (0, 255))
                        self.assertTrue(np.all(np.diff(output.astype(int)) >= 0))
                        for value in np.unique(bins):
                            self.assertEqual(len(np.unique(output[bins == value])), 1)
                        if len(source) == 256:
                            np.testing.assert_array_equal(np.unique(output), palette)

    def test_empty_constant_and_channel_shared_quantization(self):
        for profile, cpu, gpu in self.modules:
            for bits in (1, 2, 4, 8):
                with self.subTest(profile=profile, bits=bits):
                    for value in (0., .333):
                        source = np.full((3, 7, 9), value, np.float32)
                        eligible = source > 0
                        eligible[:, :2] = False
                        result = self.quantize_pair(cpu, gpu, source, eligible, bits)
                        np.testing.assert_array_equal(result, np.where(eligible, source, 0))
                    source = np.arange(1, 121, dtype=np.float32).reshape(5, 8, 3) / 255
                    normal = self.quantize_pair(cpu, gpu, source, source > 0, bits)
                    reverse = self.quantize_pair(cpu, gpu, source[..., ::-1].copy(),
                                                 (source[..., ::-1] > 0).copy(), bits)
                    np.testing.assert_array_equal(normal, reverse[..., ::-1])
                    batch = np.array([[0, 10, 20, 30, 40], [0, 100, 120, 140, 160]], np.float32) / 255
                    independent = [self.quantize_pair(cpu, gpu, sample, sample > 0, bits) for sample in batch]
                    np.testing.assert_array_equal(independent[0], independent[1])

    def test_clahe_matches_opencv_for_gray_and_small_uneven_tiles(self):
        rng = np.random.default_rng(31267)
        for shape in ((128, 176), (91, 117), (80, 83), (3, 5), (1, 19), (1, 1)):
            source = rng.integers(0, 256, shape, dtype=np.uint8)
            sample = source.astype(np.float32) / 255
            for profile, cpu, gpu in self.modules:
                for clip in (1., 2.37, 4., 8.):
                    with self.subTest(shape=shape, profile=profile, clip=clip):
                        expected = self.cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(source)
                        actual = self.u8(cpu._clahe_numpy(sample, clip))
                        tensor = self.torch.from_numpy(sample[None].copy())
                        torch_result = self.u8(gpu._clahe_torch(tensor, clip).numpy()[0])
                        np.testing.assert_array_equal(torch_result, actual)
                        difference = np.abs(actual.astype(int) - expected.astype(int))
                        self.assertLessEqual(difference.max(), 1)
                        self.assertLess(difference.mean(), .02)

    def test_clahe_pools_context_channels_and_keeps_samples_independent(self):
        rng = np.random.default_rng(91)
        source = rng.random((17, 19, 3), dtype=np.float32)
        before = source.copy()
        for profile, cpu, gpu in self.modules:
            with self.subTest(profile=profile):
                normal = cpu._clahe_numpy(source, 3.25)
                reverse = cpu._clahe_numpy(source[..., ::-1], 3.25)
                np.testing.assert_array_equal(normal, reverse[..., ::-1])
                single = cpu._clahe_numpy(source[..., 0], 3.25)
                repeated = cpu._clahe_numpy(np.repeat(source[..., :1], 3, axis=2), 3.25)
                np.testing.assert_array_equal(repeated, np.repeat(single[..., None], 3, axis=2))
                batch = self.torch.from_numpy(np.stack((source.transpose(2, 0, 1), source[::-1].transpose(2, 0, 1))).copy())
                together = gpu._clahe_torch(batch, 3.25)
                for index in range(2):
                    self.torch.testing.assert_close(together[index], gpu._clahe_torch(batch[index], 3.25), rtol=0, atol=0)
                np.testing.assert_array_equal(self.u8(together[0].numpy().transpose(1, 2, 0)), self.u8(normal))
                np.testing.assert_array_equal(source, before)

    def test_cpu_elastic_has_expected_rms_and_replay_contract(self):
        for (profile, cpu, _), fraction in zip(self.modules, (.005, .01, .015, .02)):
            self.assertEqual(cpu.CPUAugmentation.tta_replay_contract, 'opencv-affine-elastic-v1')
            for height, width in ((1, 19), (27, 31), (128, 192), (512, 768)):
                with self.subTest(profile=profile, shape=(height, width)):
                    field = cpu.CPUAugmentation._elastic_displacement(2718, height, width)
                    np.testing.assert_array_equal(field, cpu.CPUAugmentation._elastic_displacement(2718, height, width))
                    if min(height, width) < 2:
                        self.assertFalse(np.any(field))
                    else:
                        rms = np.sqrt(np.mean(field.astype(np.float64) ** 2, axis=(0, 1)))
                        np.testing.assert_allclose(rms, fraction * min(height, width), rtol=2e-4)
                        np.testing.assert_allclose(field.mean(axis=(0, 1)), 0, atol=3e-4)

    def test_photometry_preserves_original_zeros_and_never_changes_labels(self):
        for profile, cpu, _ in self.modules:
            image = (np.arange(32 * 48).reshape(32, 48) % 255 + 1).astype(np.uint8)
            image[:4] = 0
            mask = np.zeros(image.shape, np.uint8)
            mask[9:25, 13:32] = 1
            policy = cpu.build_augmentation()
            params = policy._sample_parameters(18, *image.shape)
            params.update(noise_family=0, noise_strength=.5, salt_pepper_amount=.2,
                          clahe_clip_limit=4., bit_depth=2)
            out = policy._apply_intensity_noise(image.astype(np.float32) / 255, seed=18, params=params)
            with self.subTest(profile=profile):
                self.assertTrue(np.all(out[image == 0] == 0))
                self.assertTrue(set(self.u8(out).ravel()) <= {0, 85, 170, 255})
                with mock.patch.object(policy, '_sample_parameters', return_value=params):
                    augmented = policy(image=image, mask=mask)
                    with mock.patch.object(policy, '_apply_intensity_noise', side_effect=lambda image, **kwargs: image):
                        spatial = policy(image=image, mask=mask)
                np.testing.assert_array_equal(augmented['mask'], spatial['mask'])


if __name__ == "__main__":
    unittest.main()

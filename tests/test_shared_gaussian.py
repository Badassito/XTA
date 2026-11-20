from __future__ import annotations

import itertools
import math
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


from XTA import assembly
from XTA import pta
from XTA.gaussian import binary_gaussian_pass


def _replicate_edge_gaussian(
    input: np.ndarray,
    *,
    sigma: float,
    output: np.ndarray | None = None,
    mode: str,
    cval: float,
    truncate: float,
) -> np.ndarray:
    """Tiny independent reference backend for environments without SciPy."""
    if mode != "nearest" or float(truncate) != 4.0:
        raise AssertionError((mode, cval, truncate))
    source = np.asarray(input, dtype=np.float32).copy()
    radius = int(round(float(truncate) * float(sigma)))
    coordinates = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (coordinates / float(sigma)) ** 2)
    kernel /= np.sum(kernel)
    result = np.zeros(source.shape, dtype=np.float32)
    shape = tuple(int(value) for value in source.shape)
    for target in itertools.product(*(range(size) for size in shape)):
        total = 0.0
        for offsets in itertools.product(range(-radius, radius + 1), repeat=3):
            source_index = tuple(
                min(max(int(target[axis] + offsets[axis]), 0), shape[axis] - 1)
                for axis in range(3)
            )
            weight = math.prod(float(kernel[offset + radius]) for offset in offsets)
            total += float(source[source_index]) * weight
        result[target] = np.float32(total)
    if output is not None:
        output[...] = result
        return output
    return result


def _serial_runner(_phase: str, count: int, operation: object) -> None:
    for index in range(int(count)):
        operation(index)  # type: ignore[operator]


class SharedGaussianPrimitiveTests(unittest.TestCase):
    def test_threshold_is_inclusive_at_exactly_half(self) -> None:
        probabilities = np.asarray(
            [[[0.49999997, 0.5], [0.50000006, 0.0]]],
            dtype=np.float32,
        )

        def fixed_filter(input: np.ndarray, **kwargs: object) -> np.ndarray:
            self.assertEqual(kwargs["mode"], "nearest")
            self.assertEqual(kwargs["cval"], 0.0)
            self.assertEqual(kwargs["truncate"], 4.0)
            return probabilities.copy()

        actual = binary_gaussian_pass(
            np.zeros(probabilities.shape, dtype=np.uint8),
            sigma=1.25,
            gaussian_filter=fixed_filter,
        )
        np.testing.assert_array_equal(
            actual,
            np.asarray([[[0, 1], [1, 0]]], dtype=np.uint8),
        )

    def test_pta_tta_exact_equality_at_volume_faces_across_passes(self) -> None:
        source = np.zeros((5, 5, 5), dtype=np.uint8)
        source[:, 1:4, 1:4] = 1
        source[2, 0, 4] = 1
        sigma = 0.8

        expected_first = (
            _replicate_edge_gaussian(
                source.astype(np.float32),
                sigma=sigma,
                mode="nearest",
                cval=0.0,
                truncate=4.0,
            )
            >= 0.5
        ).astype(np.uint8)
        expected_second = (
            _replicate_edge_gaussian(
                expected_first.astype(np.float32),
                sigma=sigma,
                mode="nearest",
                cval=0.0,
                truncate=4.0,
            )
            >= 0.5
        ).astype(np.uint8)
        # The bar crossing both z faces keeps its centre at the faces, and the
        # isolated face voxel is removed.
        self.assertEqual(int(expected_first[0, 2, 2]), 1)
        self.assertEqual(int(expected_first[4, 2, 2]), 1)
        self.assertEqual(int(expected_first[2, 0, 4]), 0)
        # The bar's corner columns at the z faces separate the boundary modes:
        # replicate-edge gives about 0.561 there, a constant-zero boundary 0.420.
        self.assertEqual(int(expected_first[0, 1, 1]), 1)
        self.assertEqual(int(expected_first[4, 1, 1]), 1)
        self.assertFalse(np.array_equal(expected_first, source))

        filter_inputs: list[np.ndarray] = []

        def recording_filter(input: np.ndarray, **kwargs: object) -> np.ndarray:
            filter_inputs.append(np.asarray(input, dtype=np.float32).copy())
            return _replicate_edge_gaussian(input, **kwargs)  # type: ignore[arg-type]

        pta_mask = source.copy()
        tta_mask = source.copy()
        with (
            mock.patch.object(pta.ndi, "gaussian_filter", side_effect=recording_filter),
            mock.patch.object(
                assembly.ndi, "gaussian_filter", side_effect=recording_filter
            ),
            mock.patch.object(assembly, "gaussian_smoothing_gpu_enabled", return_value=False),
            mock.patch.object(
                assembly,
                "allocate_workspace_array",
                side_effect=lambda **kwargs: np.empty(kwargs["shape"], dtype=kwargs["dtype"]),
            ),
            mock.patch.object(assembly, "close_memmap_array"),
            mock.patch.object(assembly, "choose_slice_parallel_workers", return_value=1),
            mock.patch.object(
                assembly,
                "parallel_for_indices_chunked",
                side_effect=lambda total, function, **_kwargs: [
                    function(index) for index in range(int(total))
                ],
            ),
            tempfile.TemporaryDirectory() as tmp,
        ):
            pta_stats = pta.apply_gaussian_smoothing(
                pta_mask,
                sigma=sigma,
                passes=2,
                warnings=pta.WarningLog(),
            )
            tta_stats = assembly.apply_gaussian_smoothing_inplace(
                tta_mask,
                sigma=sigma,
                passes=2,
                temp_dir=Path(tmp),
                reserve_bytes=0,
                workers=1,
            )

        np.testing.assert_array_equal(pta_mask, expected_second)
        np.testing.assert_array_equal(tta_mask, expected_second)
        np.testing.assert_array_equal(pta_mask, tta_mask)
        self.assertEqual(len(pta_stats), 2)
        self.assertEqual(tta_stats["passes_completed"], 2)
        self.assertEqual(tta_stats["backend"], "cpu_scipy_ndimage")

        self.assertEqual(len(filter_inputs), 4)
        np.testing.assert_array_equal(filter_inputs[0], source.astype(np.float32))
        np.testing.assert_array_equal(filter_inputs[1], expected_first.astype(np.float32))
        np.testing.assert_array_equal(filter_inputs[2], source.astype(np.float32))
        np.testing.assert_array_equal(filter_inputs[3], expected_first.astype(np.float32))

    def test_workspace_alias_preserves_per_slice_change_observation(self) -> None:
        source = np.zeros((3, 3, 3), dtype=np.uint8)
        source[1, 1, 1] = 1
        workspace = np.empty(source.shape, dtype=np.float32)
        changes: list[tuple[int, int, int]] = []

        def observe(z: int, old: np.ndarray, new: np.ndarray) -> None:
            changes.append(
                (
                    int(z),
                    int(np.count_nonzero(new & ~old)),
                    int(np.count_nonzero(old & ~new)),
                )
            )

        binary_gaussian_pass(
            source,
            sigma=0.8,
            gaussian_filter=_replicate_edge_gaussian,
            float_workspace=workspace,
            destination=source,
            slice_runner=_serial_runner,
            observe_slice=observe,
        )

        self.assertEqual([item[0] for item in changes], [0, 1, 2])
        self.assertGreater(sum(item[2] for item in changes), 0)
        self.assertEqual(sum(item[1] for item in changes), 0)

    def test_rod_crossing_faces_is_not_eroded_at_faces(self) -> None:
        from scipy import ndimage as ndi

        source = np.zeros((40, 40, 40), dtype=np.uint8)
        source[:, 10:30, 10:30] = 1

        actual = binary_gaussian_pass(source, sigma=3.0, gaussian_filter=ndi.gaussian_filter)

        middle = int(np.count_nonzero(actual[20]))
        self.assertGreater(middle, 0)
        self.assertEqual(int(np.count_nonzero(actual[0])), middle)
        self.assertEqual(int(np.count_nonzero(actual[39])), middle)

    def test_structure_not_touching_face_matches_interior(self) -> None:
        # The replicated face slice is empty, so a structure one slice in from
        # the face is filtered exactly as in the interior.  A mirror boundary
        # would reflect it across the face and fabricate face foreground.
        from scipy import ndimage as ndi

        yy, xx = np.mgrid[:32, :32]
        disc = ((yy - 16) ** 2 + (xx - 16) ** 2) <= 64
        near_face = np.zeros((24, 32, 32), dtype=np.uint8)
        near_face[1:3][:, disc] = 1
        interior = np.zeros_like(near_face)
        interior[11:13][:, disc] = 1

        near_face_result = binary_gaussian_pass(near_face, sigma=1.0, gaussian_filter=ndi.gaussian_filter)
        interior_result = binary_gaussian_pass(interior, sigma=1.0, gaussian_filter=ndi.gaussian_filter)

        self.assertGreater(int(np.count_nonzero(near_face_result)), 0)
        self.assertEqual(int(np.count_nonzero(near_face_result[0])), 0)
        np.testing.assert_array_equal(near_face_result[:14], interior_result[10:])
        self.assertEqual(int(np.count_nonzero(near_face_result[14:])), 0)


if __name__ == "__main__":
    unittest.main()

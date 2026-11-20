from __future__ import annotations

import multiprocessing as mp
import queue
import tempfile
import traceback
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


from XTA import media


def _spawn_categorical_resize(result_queue: object, out_path: str) -> None:
    """Exercise the public helper from a freshly spawned interpreter."""
    try:
        source = np.asarray(
            [
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
            ],
            dtype=np.uint8,
        )
        result = media.resize_categorical_volume_to_processing_cube_uint8(
            source,
            (3, 3, 3),
            Path(out_path),
            workers=2,
            prefer_memory=False,
            reserve_bytes=0,
        )
        result_queue.put(("ok", str(result.dtype), np.asarray(result).tolist()))
    except BaseException:
        result_queue.put(("error", traceback.format_exc(), None))


class CategoricalProcessingCubeResizeTests(unittest.TestCase):
    @staticmethod
    def _nearest_indices(in_count: int, out_count: int) -> np.ndarray:
        return np.asarray(
            [((2 * index + 1) * in_count) // (2 * out_count) for index in range(out_count)],
            dtype=np.intp,
        )

    def test_shape_match_returns_the_input_without_allocating(self) -> None:
        source = np.asarray(
            [
                [[0, 1], [1, 0]],
                [[1, 0], [0, 1]],
            ],
            dtype=np.uint8,
        )
        with mock.patch.object(media, "allocate_workspace_array") as allocate:
            result = media.resize_categorical_volume_to_processing_cube_uint8(
                source,
                source.shape,
                Path("unused.uint8"),
            )

        self.assertIs(result, source)
        allocate.assert_not_called()

    def test_center_ties_follow_the_exact_integer_rule(self) -> None:
        # 4 -> 7 output centers land on source cell boundaries at coordinates
        # 3 and 5 ((2i+1)*4/14 = 2.0 and 3.14); the integer floor takes the
        # upper cell deterministically.
        np.testing.assert_array_equal(
            media._center_aligned_nearest_source_indices(4, 7),
            np.asarray([0, 0, 1, 2, 2, 3, 3], dtype=np.intp),
        )
        np.testing.assert_array_equal(
            media._center_aligned_nearest_source_indices(1, 4),
            np.asarray([0, 0, 0, 0], dtype=np.intp),
        )
        np.testing.assert_array_equal(
            media._center_aligned_nearest_source_indices(4, 1),
            np.asarray([2], dtype=np.intp),
        )
        with self.assertRaises(ValueError):
            media._center_aligned_nearest_source_indices(0, 3)

    def test_all_axes_use_center_aligned_nearest_and_output_is_binary_uint8(self) -> None:
        source = np.zeros((4, 4, 5), dtype=np.uint8)
        for t_idx in range(source.shape[0]):
            for y_idx in range(source.shape[1]):
                for x_idx in range(source.shape[2]):
                    if (11 * t_idx + 5 * y_idx + 3 * x_idx) % 7 in {0, 2, 5}:
                        source[t_idx, y_idx, x_idx] = np.uint8(255)

        out_shape = (7, 7, 8)
        allocated = np.empty(out_shape, dtype=np.uint8)
        out_path = Path("categorical_cube.uint8")
        with (
            mock.patch.object(
                media,
                "allocate_workspace_array",
                return_value=allocated,
            ) as allocate,
            mock.patch.object(
                media.cv2,
                "resize",
                side_effect=AssertionError("categorical resize must not call cv2.resize"),
            ),
        ):
            result = media.resize_categorical_volume_to_processing_cube_uint8(
                source,
                out_shape,
                out_path,
                workers=4,
                prefer_memory=False,
                reserve_bytes=1234,
            )

        source_t = self._nearest_indices(source.shape[0], out_shape[0])
        source_y = self._nearest_indices(source.shape[1], out_shape[1])
        source_x = self._nearest_indices(source.shape[2], out_shape[2])
        expected = (
            source[np.ix_(source_t, source_y, source_x)] != 0
        ).astype(np.uint8)

        self.assertIs(result, allocated)
        np.testing.assert_array_equal(result, expected)
        self.assertEqual(result.dtype, np.dtype(np.uint8))
        self.assertLessEqual(set(np.unique(result).tolist()), {0, 1})
        np.testing.assert_array_equal(result[0], expected[0])
        np.testing.assert_array_equal(result[-1], expected[-1])
        allocate.assert_called_once_with(
            shape=out_shape,
            dtype=np.uint8,
            path=out_path,
            desc="Categorical processing-cube volume (center-aligned nearest)",
            prefer_memory=False,
            reserve_bytes=1234,
            initialize_zero=False,
        )

    def test_labels_stay_registered_with_the_intensity_cube(self) -> None:
        # Each output row's label slice must be the native slice nearest to the
        # gray row's sampling position, read back from a ramp. One categorical
        # resize per index bit recovers the label slice of every output row.
        cases = (
            ((5, 1, 1), (28, 1, 1)),        # T-only OpenCV slab
            ((50, 1, 1), (285, 1, 1)),      # T-only OpenCV slab, many natives
            ((5, 6, 30), (29, 29, 30)),     # general T+XY path
        )
        in_memory = mock.patch.object(
            media, "allocate_workspace_array",
            side_effect=lambda **kwargs: np.empty(kwargs["shape"], dtype=kwargs["dtype"]),
        )
        with in_memory:
            for in_shape, out_shape in cases:
                in_t, out_t = in_shape[0], out_shape[0]
                step = 255 // (in_t - 1)
                ramp = np.broadcast_to(
                    (np.arange(in_t, dtype=np.uint8) * np.uint8(step))[:, None, None], in_shape,
                ).copy()
                gray = media.resize_volume_to_processing_cube_gray8(
                    ramp, out_shape, Path("gray.uint8"), workers=1,
                )
                position = np.asarray(gray, dtype=np.float64)[:, 0, 0] / step
                labels = np.zeros(out_t, dtype=np.int64)
                for bit in range(int(in_t - 1).bit_length()):
                    selected = ((np.arange(in_t) >> bit) & 1).astype(bool)
                    mask = np.zeros(in_shape, dtype=np.uint8)
                    mask[selected] = 1
                    resized = media.resize_categorical_volume_to_processing_cube_uint8(
                        mask, out_shape, Path("mask.uint8"), workers=1,
                    )
                    labels |= np.asarray(resized, dtype=np.int64)[:, 0, 0] << bit
                # uint8 rounding and OpenCV fixed-point weights blur the read-back
                # position by up to one gray step; exact ties (distance 0.5) accept
                # either neighbouring slice. Endpoint labels miss by 0.7-0.8.
                distance = np.abs(labels - position)
                self.assertLessEqual(
                    float(distance.max()), 0.5 + 1.0 / step + 1e-9,
                    f"{in_shape}->{out_shape}: rows {np.flatnonzero(distance > 0.5 + 1.0 / step).tolist()}",
                )

    def test_public_helper_executes_under_spawn(self) -> None:
        context = mp.get_context("spawn")
        result_queue = context.Queue()
        try:
            with tempfile.TemporaryDirectory() as temp_dir:
                process = context.Process(
                    target=_spawn_categorical_resize,
                    args=(result_queue, str(Path(temp_dir) / "spawn_cube.uint8")),
                )
                process.start()
                try:
                    status, detail, values = result_queue.get(timeout=45.0)
                except queue.Empty:
                    process.terminate()
                    process.join(timeout=10.0)
                    self.fail("spawned categorical resize returned no result")
                process.join(timeout=45.0)
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=10.0)
                    self.fail("spawned categorical resize did not exit")
        finally:
            result_queue.close()
            result_queue.join_thread()

        self.assertEqual(process.exitcode, 0)
        self.assertEqual(status, "ok", detail)
        self.assertEqual(detail, "uint8")
        spawned = np.asarray(values, dtype=np.uint8)
        self.assertEqual(spawned.shape, (3, 3, 3))
        self.assertLessEqual(set(np.unique(spawned).tolist()), {0, 1})


if __name__ == "__main__":
    unittest.main()

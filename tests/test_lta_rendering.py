from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


from XTA.geometry import expand_views_into_tta_variants, get_view_infos
from XTA.lta_rendering import (
    LtaPhysicalViewCacheRef,
    build_lta_rendered_view,
    implicit_rgb,
    render_native_tile_window,
    union_tile_chunk_into_view,
)
from XTA.runtime import close_memmap_array


class LtaRenderingTests(unittest.TestCase):
    def test_implicit_rgb_is_exact_and_rejects_non_uint8(self) -> None:
        gray = np.arange(12, dtype=np.uint8).reshape(3, 4)
        rgb = implicit_rgb(gray)

        self.assertEqual(rgb.shape, (3, 4, 3))
        np.testing.assert_array_equal(rgb[:, :, 0], gray)
        np.testing.assert_array_equal(rgb[:, :, 1], gray)
        np.testing.assert_array_equal(rgb[:, :, 2], gray)
        with self.assertRaisesRegex(ValueError, "uint8"):
            implicit_rgb(gray.astype(np.float32))

    def test_file_backed_cache_ref_renders_native_tile_and_revalidates(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.gray8.dat"
            source = np.memmap(path, dtype=np.uint8, mode="w+", shape=(3, 5, 6))
            source[:] = np.arange(source.size, dtype=np.uint8).reshape(source.shape)
            source.flush()
            stat = path.stat()
            del source
            cache = LtaPhysicalViewCacheRef(
                path=path,
                shape=(3, 5, 6),
                dtype="uint8",
                physical_view_id="transverse",
                identity_sha256="a" * 64,
                size_bytes=90,
                mtime_ns=stat.st_mtime_ns,
            )

            frames = render_native_tile_window(
                cache,
                frame_start=1,
                frame_stop=3,
                tile_xyxy=(2, 1, 6, 5),
            )

            self.assertEqual([frame.size for frame in frames], [(4, 4), (4, 4)])
            pixels = np.asarray(frames[0])
            self.assertEqual(pixels.shape, (4, 4, 3))
            np.testing.assert_array_equal(pixels[:, :, 0], pixels[:, :, 1])
            with path.open("ab") as handle:
                handle.write(b"x")
            with self.assertRaisesRegex(RuntimeError, "changed"):
                cache.revalidate()

    def test_render_failure_keeps_retained_cache_view_readable(self) -> None:
        retained = []

        def retain_then_fail(frame):
            retained.append(frame)
            raise RuntimeError("intentional renderer failure")

        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.gray8.dat"
            source = np.memmap(path, dtype=np.uint8, mode="w+", shape=(1, 3, 4))
            source[:] = np.arange(12, dtype=np.uint8).reshape(source.shape)
            source.flush()
            stat = path.stat()
            del source
            cache = LtaPhysicalViewCacheRef(
                path=path,
                shape=(1, 3, 4),
                dtype="uint8",
                physical_view_id="transverse",
                identity_sha256="a" * 64,
                size_bytes=12,
                mtime_ns=stat.st_mtime_ns,
            )

            with mock.patch("XTA.lta_rendering.implicit_rgb", new=retain_then_fail):
                with self.assertRaisesRegex(RuntimeError, "intentional renderer failure"):
                    render_native_tile_window(
                        cache, frame_start=0, frame_stop=1, tile_xyxy=(0, 0, 4, 3)
                    )
            self.assertEqual(int(retained[0][2, 3]), 11)
            retained.clear()

    def test_scaled_window_matches_rgb_area_resize_and_expands_only_model_frames(self) -> None:
        from XTA.lta_dynamic_crops import resize_crop_frame
        from XTA.lta_rendering import reference_existing_physical_view_cache

        gray = np.random.default_rng(71).integers(0, 256, (3, 20, 22), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.gray8.dat"
            path.write_bytes(gray.tobytes())
            cache = reference_existing_physical_view_cache(path, shape=gray.shape,
                physical_view_id="transverse", source_identity="area-parity")
            # Integer AREA/box rounding, fractional downsampling, enlargement,
            # and the unchanged-size path all preserve identical RGB pixels.
            for tile, side in (((3, 2, 18, 17), 5), ((4, 3, 18, 17), 8),
                               ((2, 1, 14, 13), 15), ((2, 1, 14, 13), 12)):
                with self.subTest(tile=tile, side=side):
                    expanded_shapes = []
                    def observe_rgb(frame):
                        expanded_shapes.append(frame.shape)
                        return implicit_rgb(frame)
                    with mock.patch("XTA.lta_rendering.implicit_rgb", side_effect=observe_rgb):
                        frames = render_native_tile_window(cache, frame_start=1, frame_stop=3,
                            tile_xyxy=tile, model_side=side)
                    self.assertEqual(expanded_shapes, [(side, side)] * 2)
                    self.assertTrue(frames.source_cache_mapping_retired)
                    x0, y0, x1, y1 = tile
                    for index, frame in enumerate(frames, start=1):
                        # Match the former worker path: native RGB PIL pixels
                        # followed by resize_crop_frame with INTER_AREA.
                        reference = resize_crop_frame(implicit_rgb(gray[index, y0:y1, x0:x1]), side=side)
                        np.testing.assert_array_equal(np.asarray(frame), reference)

    def test_scaled_compact_window_keeps_global_crop_offsets(self) -> None:
        gray = np.random.default_rng(9).integers(0, 256, (12, 14), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "cache.compact.dat"
            path.write_bytes(gray.tobytes())
            cache = LtaPhysicalViewCacheRef(path=path, shape=(5, 30, 40), dtype="uint8",
                physical_view_id="transverse", identity_sha256="a" * 64,
                size_bytes=gray.size, mtime_ns=path.stat().st_mtime_ns,
                frame_crops=((3, 5, 7, 17, 21, 0),))
            frames = render_native_tile_window(cache, frame_start=3, frame_stop=4,
                tile_xyxy=(9, 8, 19, 16), model_side=5)
            from XTA.lta_dynamic_crops import resize_crop_frame
            reference = resize_crop_frame(implicit_rgb(gray[3:11, 2:12]), side=5)
            np.testing.assert_array_equal(np.asarray(frames[0]), reference)
            self.assertTrue(frames.source_cache_mapping_retired)

    def test_tile_chunk_union_uses_global_frame_and_xy_offsets(self) -> None:
        destination = np.zeros((4, 6, 7), dtype=np.uint8)
        chunk = np.zeros((2, 3, 4), dtype=np.uint8)
        chunk[0, 1, 2] = 1
        chunk[1, 2, 3] = 1

        union_tile_chunk_into_view(
            destination,
            chunk,
            frame_start=1,
            tile_xyxy=(2, 2, 6, 5),
        )

        self.assertEqual(int(destination[1, 3, 4]), 1)
        self.assertEqual(int(destination[2, 4, 5]), 1)
        self.assertEqual(int(destination.sum()), 2)

    @unittest.skipIf(type(__import__("sys").modules.get("cv2")).__name__ == "_StubModule", "requires OpenCV")
    def test_transverse_identity_render_mask_restore_and_native_projection(self) -> None:
        volume = np.arange(3 * 4 * 4, dtype=np.uint8).reshape(3, 4, 4)
        physical = get_view_infos(
            T=3,
            H=4,
            W=4,
            cartesian_views=("transverse",),
            azimuthal_views=(),
            azimuthal_azimuth_angles=(),
            tilt_groups=(),
        )[0]
        runtime = expand_views_into_tta_variants((physical,), (0.0,))[0]

        with tempfile.TemporaryDirectory() as temp_dir:
            rendered = build_lta_rendered_view(
                volume,
                runtime,
                temp_dir=Path(temp_dir),
                output_size=4,
            )
            frame = rendered.render_frame_rgb(1)
            model_mask = np.zeros((4, 4), dtype=bool)
            model_mask[2, 1] = True
            view_masks = rendered.restore_model_masks_to_view({1: model_mask})
            native = rendered.project_view_masks_to_native(
                view_masks,
                out_path=Path(temp_dir) / "native.u8.dat",
            )
            native_copy = np.array(native, copy=True)
            close_memmap_array(native)
            del native

        self.assertEqual(rendered.raster_plan.mode.value, "lta")
        self.assertEqual(frame.shape, (4, 4, 3))
        np.testing.assert_array_equal(frame[:, :, 0], volume[1])
        self.assertEqual(int(view_masks[1, 2, 1]), 1)
        np.testing.assert_array_equal(native_copy, view_masks)


if __name__ == "__main__":
    unittest.main()

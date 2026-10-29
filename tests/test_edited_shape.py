"""Edited geometry, cell-center alignment and actual streamed NRRD validation."""
from __future__ import annotations

import gzip
import hashlib
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy import ndimage

from tools import xta_edited_shape as shape


def render(preview, full, **kwargs):
    result = np.zeros(full, np.uint8)
    for t, y, rows in shape.iter_edited_shape_slabs(preview, full, **kwargs):
        result[t, y:y + len(rows)] = rows
    return result


def read_nrrd(path, full):
    with path.open('rb') as raw:
        header = []
        while True:
            line = raw.readline()
            if line == b'\n':
                break
            header.append(line)
        data = gzip.GzipFile(fileobj=raw).read()
    return b''.join(header).decode('ascii'), np.frombuffer(data, np.uint8).reshape(full)


def dense_policy2_reference(preview, full, block_mib=64, workers=1):
    """Original dense interpolation, before skipping trusted background."""
    field = shape.signed_distance_field(preview, full)
    rows = max(1, min(full[1], int(block_mib * shape.MIB // (full[2] * 10 * workers))))
    scale = np.asarray(preview.shape, np.float64) / np.asarray(full, np.float64)
    offset = .5 * scale + .5
    x_indices = shape._cell_indices(preview.shape[2], full[2])
    result = np.empty(full, np.uint8)
    for t in range(full[0]):
        for y in range(0, full[1], rows):
            n = min(rows, full[1] - y)
            values = ndimage.affine_transform(
                field, np.diag(scale), offset=offset + scale * np.array([t, y, 0]),
                output_shape=(1, n, full[2]), output=np.float32,
                order=1, mode='constant', cval=-float(max(full)), prefilter=False)
            source_t = (2*t + 1)*preview.shape[0]//(2*full[0])
            source_y = shape._cell_indices(preview.shape[1], full[1], y, y+n)
            trusted = preview[source_t][np.ix_(source_y, x_indices)]
            result[t, y:y+n] = (values[0] > 0) & trusted.astype(bool)
    return result


class EditedShapeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_axis_widths_have_no_two_voxel_expansion(self):
        a = np.zeros((9, 10, 11), np.uint8)
        a[2:7, 3:8, 4:9] = 1
        full = (45, 50, 55)
        b = render(a, full)
        for line, start, stop in ((b[:, 27, 32], 10, 35),
                                  (b[22, :, 32], 15, 40),
                                  (b[22, 27, :], 20, 45)):
            np.testing.assert_array_equal(np.flatnonzero(line), np.arange(start, stop))

    def test_noninteger_anisotropic_cuboid_physical_faces(self):
        a = np.zeros((9, 10, 11), np.uint8)
        a[2:7, 3:8, 4:9] = 1
        full = (38, 57, 69)
        b = render(a, full)
        for line, low_n, n, lo, hi in (
                (b[:, 31, 40], 9, 38, 2, 7),
                (b[19, :, 40], 10, 57, 3, 8),
                (b[19, 31, :], 11, 69, 4, 9)):
            centers = (np.arange(n) + .5) * low_n / n
            np.testing.assert_array_equal(line, (centers > lo) & (centers < hi))

    def test_cell_face_ties_use_exact_integer_mapping(self):
        # F2's 388 -> 1930 T mapping has exact cell-face ties. Floating point
        # multiplication by a precomputed ratio can incorrectly round left.
        indices = shape._cell_indices(388, 1930)
        self.assertEqual(indices[482], 97)
        expected = [((2*j+1)*388)//(2*1930) for j in range(1930)]
        np.testing.assert_array_equal(indices, expected)

    def test_full_foreground_stops_at_volume_edges(self):
        a = np.ones((4, 5, 6), np.uint8)
        b = render(a, (20, 25, 30))
        # Distance interpolation rounds multi-axis corners, while planar faces
        # terminate at their original physical voxel faces without expansion.
        self.assertTrue(b[:, 12, 15].all())
        self.assertTrue(b[10, :, 15].all())
        self.assertTrue(b[10, 12, :].all())
        self.assertTrue(b[np.ix_(np.arange(2, 20, 5), np.arange(2, 25, 5),
                                np.arange(2, 30, 5))].all())

    def test_empty_remains_empty(self):
        self.assertFalse(render(np.zeros((3, 4, 5), np.uint8), (15, 20, 25)).any())

    def test_identity_grid_is_exact(self):
        rng = np.random.default_rng(2)
        a = rng.integers(0, 2, (8, 9, 10), dtype=np.uint8)
        np.testing.assert_array_equal(render(a, a.shape), a)

    def test_one_voxel_thin_foreground_retains_centers(self):
        a = np.zeros((7, 8, 9), np.uint8)
        a[3, 2:6, 4] = 1
        path = self.root / 'thin.nrrd'
        meta = shape.write_edited_shape(path, a, (35, 40, 45))
        audit = meta['reconstruction']['center_roundtrip']
        self.assertEqual(audit['sampled_voxels'], a.size)
        self.assertEqual(audit['false_positive_voxels'], 0)
        self.assertEqual(audit['false_negative_voxels'], 0)

    def test_one_cell_gap_separates_nearby_cuboids(self):
        a = np.zeros((9, 9, 15), np.uint8)
        a[2:7, 2:7, 2:7] = 1
        a[2:7, 2:7, 8:13] = 1
        b = render(a, (45, 45, 75))
        self.assertFalse(b[:, :, 35:40].any())
        self.assertEqual(ndimage.label(b)[1], 2)

    def test_diagonally_separated_objects_stay_separated(self):
        a = np.zeros((9, 12, 12), np.uint8)
        a[2:7, 2:5, 2:5] = 1
        a[2:7, 6:9, 6:9] = 1
        b = render(a, (45, 60, 60))
        self.assertEqual(ndimage.label(b)[1], 2)

    def test_corner_touch_does_not_add_face_connection(self):
        a = np.zeros((9, 12, 12), np.uint8)
        a[2:7, 2:5, 2:5] = 1
        a[2:7, 5:8, 5:8] = 1
        b = render(a, (45, 60, 60))
        self.assertEqual(ndimage.label(b)[1], 2)

    def test_signed_distance_cannot_bridge_edited_background_cells(self):
        # Unguarded signed-distance interpolation connects the two components
        # through cells explicitly edited to background, despite center fidelity.
        a = np.zeros((6, 6, 6), np.uint8)
        points = np.array([[3, 1, 2], [3, 2, 2], [3, 2, 3], [4, 1, 3], [4, 2, 2]])
        a[tuple(points.T)] = 1
        b = render(a, (30, 30, 30))
        trusted = a.repeat(5, axis=0).repeat(5, axis=1).repeat(5, axis=2)
        self.assertFalse(np.any((b != 0) & (trusted == 0)))
        self.assertEqual(ndimage.label(a)[1], 2)
        self.assertEqual(ndimage.label(b)[1], 2)
        meta = shape.write_edited_shape(self.root / 'separate.nrrd', a, (30, 30, 30))
        audit = meta['reconstruction']['separation_audit']
        self.assertEqual(audit['source_foreground_components'], 2)
        self.assertEqual(audit['output_foreground_components'], 2)
        self.assertEqual(audit['foreground_voxels_in_edited_background_cells'], 0)

    def test_remote_join_cannot_create_local_background_shortcut(self):
        a = np.zeros((7, 7, 7), np.uint8)
        points = np.array([[3, 1, 2], [3, 2, 2], [3, 2, 3], [4, 1, 3], [4, 2, 2],
                           [4, 1, 4], [4, 1, 5], [4, 2, 5], [4, 3, 5], [4, 4, 5],
                           [4, 4, 4], [4, 4, 3], [4, 4, 2], [3, 4, 2], [3, 3, 2]])
        a[tuple(points.T)] = 1
        b = render(a, (35, 35, 35))
        trusted = a.repeat(5, axis=0).repeat(5, axis=1).repeat(5, axis=2)
        self.assertFalse(np.any((b != 0) & (trusted == 0)))
        self.assertEqual(ndimage.label(a)[1], 1)
        self.assertEqual(ndimage.label(b)[1], 1)
        # The former bridge was across these edited background cells; every
        # native voxel in each remains background, even though the remote path
        # means a foreground-component-count check alone would miss this defect.
        for t, y, x in np.argwhere(a == 0):
            self.assertFalse(b[t*5:(t+1)*5, y*5:(y+1)*5, x*5:(x+1)*5].any())

    def test_tunnel_stays_open(self):
        a = np.zeros((9, 11, 11), np.uint8)
        a[1:8, 2:9, 2:9] = 1
        a[:, 5, 5] = 0
        b = render(a, (45, 55, 55))
        # The square tunnel becomes smooth around its corners, but an unbroken
        # background path remains through every foreground cross-section.
        self.assertFalse(b[:, 27, 27].any())
        background, _ = ndimage.label(~b.astype(bool))
        self.assertEqual(background[0, 27, 27], background[-1, 27, 27])
        self.assertEqual(ndimage.label(b)[1], 1)

    def test_empty_remote_region_cannot_gain_false_positives(self):
        a = np.zeros((10, 10, 10), np.uint8)
        a[1:4, 1:4, 1:4] = 1
        b = render(a, (50, 50, 50))
        self.assertFalse(b[25:].any())
        self.assertFalse(b[:, 25:].any())
        self.assertFalse(b[:, :, 25:].any())
        self.assertEqual(ndimage.label(b)[1], 1)

    def test_smooth_diagonal_surfaces_have_no_isolated_foreground(self):
        z, y, x = np.indices((17, 19, 21))
        a = (((z - 8) ** 2 + (y - 9) ** 2 + (x - 10) ** 2) <= 36).astype(np.uint8)
        b = render(a, (85, 95, 105))
        self.assertEqual(ndimage.label(b)[1], 1)
        # Every transverse sphere section is a single solid object; no small
        # false positive objects or interior holes arise from interpolation.
        for plane in b:
            if plane.any():
                self.assertEqual(ndimage.label(plane)[1], 1)
                np.testing.assert_array_equal(ndimage.binary_fill_holes(plane), plane)

    def test_boundary_foreground_is_not_extended_or_clipped(self):
        a = np.zeros((5, 6, 7), np.uint8)
        a[:, :, :3] = 1
        b = render(a, (25, 30, 35))
        np.testing.assert_array_equal(b[12, 15], np.arange(35) < 15)
        self.assertTrue(b[0, 15, 2:13].all())

    def test_streaming_thread_and_block_sizes_are_identical(self):
        a = np.zeros((7, 8, 9), np.uint8)
        a[1:6, 2:7, 3:8] = 1
        full = (31, 37, 43)
        expected = render(a, full, block_mib=1)
        for workers in (1, 2, 4):
            actual = render(a, full, block_mib=.001, workers=workers)
            np.testing.assert_array_equal(actual, expected)

    def test_sparse_fast_paths_match_dense_policy2_at_random_and_tied_coordinates(self):
        rng = np.random.default_rng(1382)
        for case in range(100):
            low = tuple(int(n) for n in rng.integers(2, 8, size=3))
            full = tuple(int(n * rng.integers(1, 6) + rng.integers(0, 2)) for n in low)
            a = (rng.random(low) < rng.uniform(0, 1)).astype(np.uint8)
            workers = (1, 4, 128)[case % 3]
            budget = (.001, .01, 1)[case % 3]
            with self.subTest(case=case, low=low, full=full, workers=workers):
                expected = dense_policy2_reference(a, full, budget, workers)
                actual = render(a, full, block_mib=budget, workers=workers)
                np.testing.assert_array_equal(actual, expected)

    def test_background_blocks_skip_interpolation_and_sparse_blocks_crop_only_trailing_area(self):
        a = np.zeros((8, 8, 8), np.uint8)
        a[0, 0, 0] = 1
        full = (40, 40, 40)
        affine = ndimage.affine_transform
        with patch.object(ndimage, 'affine_transform', wraps=affine) as interpolation:
            actual = render(a, full)
        np.testing.assert_array_equal(actual, dense_policy2_reference(a, full))
        self.assertEqual(interpolation.call_count, 5)
        pixels = sum(np.prod(call.kwargs['output_shape']) for call in interpolation.call_args_list)
        self.assertLess(pixels, np.prod(full) // 100)

    def test_128_worker_plan_has_explicit_useful_buffer_budget(self):
        full = (1930, 3064, 3023)
        plan = shape.block_plan(full, block_mib=4096, workers=128)
        self.assertGreaterEqual(plan['row_block_height'], 1000)
        self.assertLess(plan['scheduled_blocks'], 6000)
        self.assertLessEqual(plan['row_block_height'] * full[2] * 10 * 128, 4096 * shape.MIB)

    def test_written_geometry_binary_hashes_and_real_center_audit(self):
        a = np.zeros((6, 7, 8), np.uint8)
        a[1:5, 1:6, 2:7] = 1
        full = (29, 37, 39)
        path = self.root / 'edited.nrrd'
        meta = shape.write_edited_shape(path, a, full, block_mib=.002, workers=2)
        header, b = read_nrrd(path, full)
        self.assertIn('sizes: 39 37 29', header)
        self.assertIn('space directions: (1,0,0) (0,1,0) (0,0,1)', header)
        self.assertIn('space origin: (0,0,0)', header)
        self.assertEqual(set(np.unique(b)), {0, 1})
        self.assertEqual(meta['foreground_voxels'], int(b.sum()))
        self.assertEqual(meta['binary_payload_sha256'], hashlib.sha256(b.tobytes()).hexdigest())
        self.assertEqual(meta['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        self.assertEqual(meta['bytes'], path.stat().st_size)
        self.assertEqual(meta['reconstruction']['preview_pitch_native_tyx'], [29/6, 37/7, 39/8])
        self.assertEqual(meta['reconstruction']['source_foreground_voxels'], int(a.sum()))
        topology = meta['reconstruction']['separation_audit']
        self.assertEqual(topology['connectivity'], 6)
        self.assertEqual(topology['source_foreground_components'], 1)
        self.assertEqual(topology['output_foreground_components'], 1)
        self.assertEqual(topology['foreground_voxels_in_edited_background_cells'], 0)
        indices = [np.floor((np.arange(l)+.5)*n/l).astype(int) for l, n in zip(a.shape, full)]
        np.testing.assert_array_equal(b[np.ix_(*indices)], a)

    def test_writer_refuses_center_class_changes_and_removes_temporary(self):
        a = np.zeros((3, 3, 3), np.uint8)
        a[1, 1, 1] = 1
        path = self.root / 'bad.nrrd'
        def corrupt(*args, **kwargs):
            for t in range(15):
                yield t, 0, np.zeros((15, 15), np.uint8)
        with patch.object(shape, '_blocks', corrupt):
            with self.assertRaisesRegex(ValueError, 'center roundtrip failed'):
                shape.write_edited_shape(path, a, (15, 15, 15))
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_writer_does_not_overwrite_existing_input(self):
        path = self.root / 'exists.nrrd'
        path.write_bytes(b'original')
        with self.assertRaises(FileExistsError):
            shape.write_edited_shape(path, np.ones((2, 2, 2), np.uint8), (10, 10, 10))
        self.assertEqual(path.read_bytes(), b'original')

    def test_optional_fast_gzip_preserves_identical_binary_payload_and_audits(self):
        a = np.zeros((5, 6, 7), np.uint8)
        a[1:4, 1:5, 2:6] = 1
        full = (25, 30, 35)
        available = ('stdlib', 'isal') if shape._IGzipFile is not None else ('stdlib',)
        previous = None
        for backend in available:
            with self.subTest(backend=backend), patch.dict(os.environ, {'XTA_NATIVE_GZIP_BACKEND': backend}):
                path = self.root / (backend + '.nrrd')
                meta = shape.write_edited_shape(path, a, full, workers=2)
                _, payload = read_nrrd(path, full)
                self.assertEqual(meta['reconstruction']['gzip_backend'], backend)
                if previous is not None:
                    np.testing.assert_array_equal(payload, previous[0])
                    self.assertEqual(meta['binary_payload_sha256'], previous[1]['binary_payload_sha256'])
                    self.assertEqual(meta['reconstruction']['center_roundtrip'], previous[1]['reconstruction']['center_roundtrip'])
                    self.assertEqual(meta['reconstruction']['separation_audit'], previous[1]['reconstruction']['separation_audit'])
                previous = payload, meta

    def test_explicit_missing_fast_gzip_is_not_silently_retried(self):
        with patch.object(shape, '_IGzipFile', None), patch.dict(os.environ, {'XTA_NATIVE_GZIP_BACKEND': 'isal'}):
            with self.assertRaisesRegex(RuntimeError, 'requires the optional isal'):
                shape._gzip_type()

    def test_gzip_crc_damage_is_rejected_by_each_backend(self):
        a = np.ones((3, 3, 3), np.uint8)
        path = self.root / 'crc.nrrd'
        with patch.dict(os.environ, {'XTA_NATIVE_GZIP_BACKEND': 'stdlib'}):
            shape.write_edited_shape(path, a, (15, 15, 15))
        damaged = bytearray(path.read_bytes())
        damaged[-8] ^= 1
        path.write_bytes(damaged)
        available = ('stdlib', 'isal') if shape._IGzipFile is not None else ('stdlib',)
        for backend in available:
            with self.subTest(backend=backend), patch.dict(os.environ, {'XTA_NATIVE_GZIP_BACKEND': backend}):
                with self.assertRaises((gzip.BadGzipFile, EOFError, OSError)):
                    shape._validate_written(path, a, (15, 15, 15), 64)

    def test_actual_written_background_center_false_positive_is_rejected(self):
        a = np.zeros((3, 3, 3), np.uint8)
        path = self.root / 'unexpected.nrrd'
        def corrupt(*args, **kwargs):
            for t in range(15):
                yield t, 0, np.ones((15, 15), np.uint8)
        with patch.object(shape, '_blocks', corrupt):
            with self.assertRaisesRegex(ValueError, '27 false positives, 0 false negatives'):
                shape.write_edited_shape(path, a, (15, 15, 15))
        self.assertFalse(path.exists())

    def test_off_center_background_intrusion_is_rejected(self):
        a = np.zeros((3, 3, 3), np.uint8)
        a[1, 1, 1] = 1
        generated = render(a, (15, 15, 15))
        generated[0, 0, 0] = 1  # Does not coincide with any audited source center.
        path = self.root / 'off-center-intrusion.nrrd'
        blocks = [(t, 0, generated[t]) for t in range(15)]
        with patch.object(shape, '_blocks', return_value=iter(blocks)):
            with self.assertRaisesRegex(ValueError, 'entered 1 edited background cells'):
                shape.write_edited_shape(path, a, (15, 15, 15))
        self.assertFalse(path.exists())

    def test_split_with_unchanged_voxel_centers_is_rejected(self):
        a = np.ones((2, 2, 2), np.uint8)
        generated = np.ones((10, 10, 10), np.uint8)
        generated[4] = 0  # Source centers are at native indices 2 and 7.
        path = self.root / 'split.nrrd'
        blocks = [(t, 0, generated[t]) for t in range(10)]
        with patch.object(shape, '_blocks', return_value=iter(blocks)):
            with self.assertRaisesRegex(ValueError, '1 edited, 2 reconstructed'):
                shape.write_edited_shape(path, a, (10, 10, 10))
        self.assertFalse(path.exists())

    def test_truncated_native_output_cannot_be_published(self):
        a = np.ones((3, 3, 3), np.uint8)
        path = self.root / 'short.nrrd'
        with patch.object(shape, '_blocks', return_value=iter([
                (0, 0, np.ones((15, 15), np.uint8))])):
            with self.assertRaisesRegex(ValueError, 'Truncated written NRRD payload'):
                shape.write_edited_shape(path, a, (15, 15, 15))
        self.assertFalse(path.exists())

    def test_insufficient_native_grid_for_thin_object_refuses_publication(self):
        a = np.zeros((3, 3, 3), np.uint8)
        a[1, 1, 1] = 1
        path = self.root / 'undersampled.nrrd'
        with self.assertRaisesRegex(ValueError, 'center roundtrip failed'):
            shape.write_edited_shape(path, a, (4, 4, 4))
        self.assertFalse(path.exists())
        self.assertEqual(list(self.root.iterdir()), [])

    def test_streaming_component_counter_matches_dense_oracle_500_random_masks(self):
        rng = np.random.default_rng(704)
        for i in range(500):
            dims = tuple(int(n) for n in rng.integers(1, 9, size=3))
            a = (rng.random(dims) < rng.uniform(.05, .95)).astype(np.uint8)
            with self.subTest(case=i, shape=dims):
                self.assertEqual(shape._component_count(a), ndimage.label(a)[1])

    def test_guarded_reconstruction_preserves_500_random_edited_masks(self):
        rng = np.random.default_rng(892)
        for i in range(500):
            a = (rng.random((5, 5, 5)) < rng.uniform(.1, .9)).astype(np.uint8)
            b = render(a, (25, 25, 25))
            trusted = a.repeat(5, axis=0).repeat(5, axis=1).repeat(5, axis=2)
            with self.subTest(case=i):
                self.assertFalse(np.any((b != 0) & (trusted == 0)))
                np.testing.assert_array_equal(b[2::5, 2::5, 2::5], a)
                self.assertEqual(ndimage.label(a)[1], ndimage.label(b)[1])

    def test_dangling_symlink_destination_cannot_be_replaced(self):
        path = self.root / 'dangling.nrrd'
        try:
            path.symlink_to(self.root / 'missing-target.nrrd')
        except OSError as exc:
            self.skipTest(f'This Windows account cannot create symlinks: {exc}')
        with self.assertRaises(FileExistsError):
            shape.write_edited_shape(path, np.ones((2, 2, 2), np.uint8), (10, 10, 10))
        self.assertTrue(path.is_symlink())

    def test_destination_created_during_generation_is_not_overwritten(self):
        path = self.root / 'racing.nrrd'
        a = np.ones((2, 2, 2), np.uint8)
        validate = shape._validate_written
        def create_destination(*args, **kwargs):
            result = validate(*args, **kwargs)
            path.write_bytes(b'concurrent destination')
            return result
        with patch.object(shape, '_validate_written', side_effect=create_destination):
            with self.assertRaises(FileExistsError):
                shape.write_edited_shape(path, a, (10, 10, 10))
        self.assertEqual(path.read_bytes(), b'concurrent destination')
        self.assertEqual(list(self.root.iterdir()), [path])

    def test_bad_inputs_are_rejected(self):
        for preview, full, kwargs in (
                (np.zeros((2, 2)), (10, 10, 10), {}),
                (np.ones((2, 2, 2)) * 2, (10, 10, 10), {}),
                (np.ones((2, 2, 2)) * np.nan, (10, 10, 10), {}),
                (np.ones((2, 2, 2)), (1, 10, 10), {}),
                (np.ones((2, 2, 2)), (10, 10, 10), {'workers': 0}),
                (np.ones((2, 2, 2)), (10, 10, 10), {'block_mib': 0})):
            with self.subTest(full=full, kwargs=kwargs):
                with self.assertRaises(ValueError):
                    list(shape.iter_edited_shape_slabs(preview, full, **kwargs))


if __name__ == '__main__':
    unittest.main()

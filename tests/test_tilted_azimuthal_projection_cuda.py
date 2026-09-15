"""Frozen composed-CPU oracle, CPU LUT parity, and opt-in CUDA parity.

GPU cases require XTA_TEST_TILTED_AZIMUTHAL_CUDA=1 after the caller owns the GPU
marker. Importing this module and running the oracle/LUT tests do not use CUDA.
"""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import math
import os
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from XTA import backprojection as bp, geometry
from XTA.config import TiltedViewGroup


def tilted_views(angles=(30., 45.), raster=0, shape=(5, 7, 9)):
    axes = ('transverse', 'sagittal', 'coronal')
    return [view for view in geometry.get_view_infos(*shape, cartesian_views=(),
        azimuthal_views=tuple('tilted_' + axis for axis in axes),
        azimuthal_azimuth_angles=(45., 45., 45.),
        tilt_groups=(TiltedViewGroup(axes, tuple(angles), ('vertical', 'horizontal')),),
        azimuthal_native_raster=raster) if geometry.is_tilted_azimuthal_view(view)]


def source_for(view, pattern='sparse', *, seed=0, processing_size=None):
    shape = (view.num_slices, view.src_h, view.src_w)
    if processing_size is not None:
        shape = (view.num_slices, processing_size, processing_size)
    if pattern == 'empty':
        return np.zeros(shape, np.uint8)
    if pattern == 'dense':
        return np.full(shape, 255, np.uint8)
    rng = np.random.default_rng(seed)
    return (rng.random(shape) < .19).astype(np.uint8) * np.uint8(7)


class FrozenComposedCpuOracle:
    """The pre-GPU CPU compose arithmetic; intentionally independent of new LUTs.

    The existing Azimuthal sampling/map and processing-grid helpers remain the
    authoritative first stage. The shear, round-to-even, output-axis mapping and
    per-frame source-row OR below are frozen from the old nested compose function.
    """

    def __init__(self, source, view, output_shape, occupancy=None):
        self.source = np.asarray(source)
        self.view = view
        self.shape = tuple(map(int, output_shape))
        self.tilted = geometry.azimuthal_source_tilted_view(view)
        self.grid = bp.resolve_azimuthal_processing_grid(source, view)
        samples, _ = bp.build_azimuthal_backprojection_plan(view)
        plane_h, plane_w = self.tilted.src_h, self.tilted.src_w
        dense = bp._azimuthal_dense_map_for_processing(
            bp.build_dense_azimuthal_backprojection_map(view, samples, out_shape_hw=(plane_h, plane_w)),
            self.grid)
        valid = np.flatnonzero(np.asarray(dense.valid_mask, dtype=bool).reshape(-1))
        self.valid_v = np.ascontiguousarray((valid // plane_w).astype(np.int32))
        self.valid_u = np.ascontiguousarray((valid % plane_w).astype(np.int32))
        self.angle = np.ascontiguousarray(np.asarray(dense.source_idx_map, np.int32).reshape(-1)[valid])
        self.column = np.ascontiguousarray(np.asarray(dense.u_idx_map, np.int32).reshape(-1)[valid])
        self.row_occupancy = None
        if occupancy is not None:
            candidate = np.asarray(occupancy, dtype=bool).reshape(-1)
            if candidate.size == self.grid.processing_h:
                self.row_occupancy = candidate

    @property
    def frame_count(self):
        return int(self.tilted.num_slices)

    @staticmethod
    def map_axis(values, in_len, out_len):
        if int(in_len) == int(out_len):
            return values.astype(np.int32, copy=False)
        mapped = (values.astype(np.int64, copy=False) * int(out_len)) // int(in_len)
        return np.minimum(mapped, int(out_len) - 1).astype(np.int32, copy=False)

    def coordinates(self, frame):
        rows = bp._azimuthal_processing_rows_for_output(self.grid, self.frame_count, int(frame))
        if rows.size <= 0 or (self.row_occupancy is not None and not np.any(self.row_occupancy[rows])):
            return None
        hit = np.zeros(self.angle.size, bool)
        for row in rows.tolist():
            if self.row_occupancy is None or self.row_occupancy[row]:
                hit |= np.asarray(self.source[self.angle, row, self.column], np.uint8) != 0
        if not np.any(hit):
            return None
        selected = np.flatnonzero(hit)
        vv, uu = self.valid_v[selected], self.valid_u[selected]
        vertical = self.tilted.tilt_direction == 'vertical'
        axis = vv if vertical else uu
        axis_center = float((self.tilted.src_h - 1) / 2. if vertical else (self.tilted.src_w - 1) / 2.)
        tangent = float(math.tan(math.radians(float(self.tilted.tilt_angle_deg))))
        stack_float = float(geometry.tilted_frame_center(self.tilted, int(frame))) + (
            float(tangent) * (axis.astype(np.float32, copy=False) - float(axis_center)))
        ss = np.rint(stack_float).astype(np.int32, copy=False)
        inside = (ss >= 0) & (ss < int(geometry.tilted_stack_axis_length(self.tilted)))
        if not np.any(inside):
            return None
        ss, vv, uu = ss[inside], vv[inside], uu[inside]
        work_t, work_h, work_w = self.tilted.full_t, self.tilted.full_h, self.tilted.full_w
        out_t, out_h, out_w = self.shape
        base = geometry.tilted_base_view_name(self.tilted)
        if base == 'transverse':
            coordinates = (self.map_axis(ss, work_t, out_t), self.map_axis(vv, work_h, out_h), self.map_axis(uu, work_w, out_w))
        elif base == 'sagittal':
            coordinates = (self.map_axis(vv, work_t, out_t), self.map_axis(ss, work_h, out_h), self.map_axis(uu, work_w, out_w))
        else:
            coordinates = (self.map_axis(vv, work_t, out_t), self.map_axis(uu, work_h, out_h), self.map_axis(ss, work_w, out_w))
        return tuple(np.ascontiguousarray(axis, dtype=np.int32) for axis in coordinates)

    def dense(self, start=0, stop=None):
        output = np.zeros(self.shape, np.uint8)
        for frame in range(int(start), self.frame_count if stop is None else int(stop)):
            coordinates = self.coordinates(frame)
            if coordinates is not None:
                output[coordinates] = 1
        return output

    def packed(self, start=0, stop=None):
        return np.packbits(self.dense(start, stop), axis=2, bitorder='big')


def cpu_lut_output(source, plan, start=0, stop=None):
    """Interpret the new integer LUT on CPU without calling either projector."""
    output = np.zeros(plan.output_shape, np.uint8)
    points = np.asarray(plan.points)
    if not len(points):
        return output
    for frame in range(start, plan.frame_count if stop is None else stop):
        rows = plan.rows[plan.row_offsets[frame]:plan.row_offsets[frame + 1]]
        hit = np.zeros(len(points), bool)
        for row in rows:
            hit |= np.asarray(source[points[:, 0], int(row), points[:, 1]], np.uint8) != 0
        stack = plan.stack_map[frame, points[:, 2]]
        selected = hit & (stack >= 0)
        if plan.base_id == 0:
            coordinates = (stack[selected], points[selected, 3], points[selected, 4])
        elif plan.base_id == 1:
            coordinates = (points[selected, 3], stack[selected], points[selected, 4])
        else:
            coordinates = (points[selected, 3], points[selected, 4], stack[selected])
        output[coordinates] = 1
    return output


class FrozenTiltedAzimuthalCpuTests(unittest.TestCase):
    def test_frozen_oracle_matches_existing_dense_and_packed_cpu_operators(self):
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()), \
                contextlib.redirect_stderr(io.StringIO()), \
                mock.patch.dict(os.environ, {'YOLO_TTA_GPU_TILTED_AZIMUTHAL_BACKPROJECT': '0'}), \
                mock.patch.object(bp, 'allocate_workspace_array', side_effect=lambda **kw: np.zeros(kw['shape'], kw['dtype'])), \
                mock.patch.object(bp, '_numba_or_tilted_azimuthal_coordinates_into_packed', None):
            for index, view in enumerate(tilted_views() + tilted_views((45.,), shape=(6, 8, 10))):
                for shape in ((5, 7, 9), (3, 4, 5), (7, 9, 13)):
                    source = source_for(view, seed=index)
                    occupancy = np.any(source, axis=(0, 2))
                    expected = FrozenComposedCpuOracle(source, view, shape, occupancy).dense()
                    for sink_only in (False, True):
                        seen, output = [], np.zeros(shape, np.uint8)
                        def collect(start, block):
                            seen.extend(range(start, start + len(block)))
                            output[start:start + len(block)] = block
                        with self.subTest(view=view.name, shape=shape, sink_only=sink_only):
                            actual = bp._backproject_tilted_azimuthal_volume_to_volume(source, view,
                                Path(td) / 'oracle.dat', 'oracle', prefer_memory=True, reserve_bytes=0,
                                workers=1, out_shape_tyx=shape, known_row_occupancy=occupancy,
                                known_slice_bboxes=None, projection_block_callback=collect, sink_only=sink_only)
                            np.testing.assert_array_equal(output if sink_only else actual, expected)
                            self.assertEqual(seen, list(range(shape[0])))

    def test_frozen_oracle_preserves_row_padded_big_bit_layout(self):
        view = tilted_views((30.,))[0]
        oracle = FrozenComposedCpuOracle(source_for(view, 'dense'), view, (7, 9, 13))
        packed = oracle.packed()
        self.assertEqual(packed.shape, (7, 9, 2))
        self.assertEqual(packed.size % 4, 2)
        self.assertFalse(np.any(packed[:, :, -1] & np.uint8(0b111)))
        np.testing.assert_array_equal(np.unpackbits(packed, axis=2, count=13, bitorder='big'), oracle.dense())


class TiltedAzimuthalPlanParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from XTA.tilted_azimuthal_projection import build_tilted_azimuthal_plan
        cls.build = staticmethod(build_tilted_azimuthal_plan)

    def assert_parity(self, source, view, shape, occupancy=None, bboxes=None):
        oracle = FrozenComposedCpuOracle(source, view, shape, occupancy)
        source_before = source.copy()
        plan = self.build(source, view, shape, known_row_occupancy=occupancy, known_slice_bboxes=bboxes)
        self.assertEqual(tuple(plan.source_shape), source.shape)
        self.assertEqual(tuple(plan.output_shape), shape)
        self.assertEqual(plan.frame_count, oracle.frame_count)
        self.assertEqual(plan.points.dtype, np.int32)
        self.assertEqual(plan.points.shape[1], 5)
        self.assertEqual(plan.stack_map.dtype, np.int32)
        self.assertEqual(plan.rows.dtype, np.int32)
        self.assertEqual(plan.row_offsets.dtype, np.int64)
        for array in (plan.points, plan.stack_map, plan.rows, plan.row_offsets):
            self.assertTrue(array.flags.c_contiguous)
            self.assertFalse(array.flags.writeable)
        np.testing.assert_array_equal(source, source_before)
        np.testing.assert_array_equal(cpu_lut_output(source, plan), oracle.dense())
        for start in (0, oracle.frame_count // 2, oracle.frame_count):
            np.testing.assert_array_equal(cpu_lut_output(source, plan, start), oracle.dense(start))
        return plan

    def test_all_bases_signs_directions_rounding_and_target_scaling(self):
        for index, view in enumerate(tilted_views() + tilted_views((45.,), shape=(6, 8, 10))):
            for shape in ((5, 7, 9), (3, 4, 5), (7, 9, 13)):
                for pattern in ('empty', 'dense', 'sparse'):
                    source = source_for(view, pattern, seed=index)
                    for occupancy in (None, np.any(source, axis=(0, 2))):
                        with self.subTest(view=view.name, shape=shape, pattern=pattern, occupancy=occupancy is not None):
                            self.assert_parity(source, view, shape, occupancy)

    def test_nonuniform_angle_plan_and_source_row_or_resampling(self):
        for view in tilted_views((30.,)):
            stack = geometry.azimuthal_source_tilted_view(view).num_slices
            for rows in (3, 2 * stack + 1):
                for angles in ((0., 17., 71., 139.), (0., 61., 103., 163.), (0., 139., 17., 71.)):
                    varied = replace(view, src_h=rows, num_slices=4, azimuths_deg=angles)
                    source = source_for(varied, seed=rows)
                    with self.subTest(view=view.name, rows=rows, angles=angles):
                        plan = self.assert_parity(source, varied, (7, 9, 13), np.any(source, axis=(0, 2)))
                        if rows > stack:
                            self.assertTrue(np.any(np.diff(plan.row_offsets) > 1))

    def test_canonical_square_processing_grid_and_ignored_stale_metadata(self):
        if not isinstance(getattr(bp.cv2, '__version__', None), str):
            self.skipTest('Real OpenCV required for fixed-point canonical-grid oracle')
        with mock.patch.dict(os.environ, {'YOLO_TTA_DELAY_NATIVE_EXPANSION': '1'}):
            for index, view in enumerate(tilted_views((30.,))):
                source = source_for(view, seed=index, processing_size=3)
                with self.subTest(view=view.name):
                    self.assert_parity(source, view, (7, 9, 13))
                    self.assert_parity(source, view, (7, 9, 13), np.zeros(99, bool), np.full((view.num_slices, 4), -1))

    def test_matching_row_metadata_is_trusted_but_valid_boxes_remain_only_validated(self):
        view = tilted_views((30.,))[0]
        source = source_for(view, 'dense')
        # The established composed CPU path validates boxes but does not use
        # them to prune gathers, even when all supplied boxes are empty.
        self.assert_parity(source, view, (7, 9, 13), bboxes=np.zeros((view.num_slices, 4), np.int64))
        for row in (None, source.shape[1] // 2):
            occupancy = np.zeros(source.shape[1], bool)
            if row is not None:
                occupancy[row] = True
            plan = self.assert_parity(source, view, (7, 9, 13), occupancy)
            if row is None:
                self.assertEqual(len(plan.rows), 0)
                self.assertFalse(np.any(cpu_lut_output(source, plan)))

    def test_basis_impulses_cover_nonboolean_source_values_and_collisions(self):
        for view in tilted_views((45.,), raster=3) + tilted_views((45.,), raster=3, shape=(6, 8, 10)):
            source = source_for(view, 'empty')
            for index in range(source.size):
                source.fill(0)
                source.flat[index] = 255 if index % 2 else 2
                with self.subTest(view=view.name, index=index):
                    self.assert_parity(source, view, (3, 4, 5))

    def test_host_plan_budget_refuses_before_geometry_sized_map_allocation(self):
        from XTA.tilted_azimuthal_projection import TiltedAzimuthalPlanUnavailable
        view = tilted_views((30.,))[0]
        source = source_for(view)
        with mock.patch.object(bp, 'build_dense_azimuthal_backprojection_map',
                               side_effect=AssertionError('geometry allocation entered')):
            with self.assertRaises(TiltedAzimuthalPlanUnavailable):
                self.build(source, view, (7, 9, 13), max_plan_bytes=1)
            oversized = replace(view, full_t=2 ** 31)
            with self.assertRaises(TiltedAzimuthalPlanUnavailable):
                self.build(source, oversized, (7, 9, 13))

    def test_backend_cpu_contract_retains_row_padding_and_prefix_without_cuda(self):
        from XTA.tilted_azimuthal_projection_cuda import (
            _validate_tilted_azimuthal_contract, _validate_initial_packed,
        )
        view = tilted_views((30.,))[0]
        source = source_for(view)
        plan = self.build(source, view, (7, 9, 13))
        contract = _validate_tilted_azimuthal_contract(source, plan, 2 * 9 * 13, source.shape[0] * source.shape[2])
        self.assertEqual(contract.packed_bytes, 7 * 9 * 2)
        self.assertEqual(contract.packed_words, 32)
        self.assertEqual(contract.band_rows, 1)
        self.assertEqual(contract.max_block_depth, 2)
        prefix = FrozenComposedCpuOracle(source, view, plan.output_shape).packed(0, 1)
        np.testing.assert_array_equal(_validate_initial_packed(prefix, 1, contract), prefix)
        with self.assertRaises(ValueError):
            _validate_initial_packed(prefix.reshape(-1), 1, contract)
        with self.assertRaises(ValueError):
            _validate_initial_packed(None, 1, contract)


@unittest.skipUnless(os.environ.get('XTA_TEST_TILTED_AZIMUTHAL_CUDA') == '1',
                     'Opt-in GPU parity requires ownership of the shared GPU marker')
class TiltedAzimuthalCudaParityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from XTA.tilted_azimuthal_projection import build_tilted_azimuthal_plan
        from XTA.tilted_azimuthal_projection_cuda import TiltedAzimuthalCudaProjector
        cls.build = staticmethod(build_tilted_azimuthal_plan)
        cls.projector = TiltedAzimuthalCudaProjector

    def project(self, source, plan, *, initial_packed=None, first_frame=0):
        stage = self.projector(source, plan, device_index=0, initial_packed=initial_packed,
                               first_frame=first_frame, reserve_bytes=0, upload_bytes=1024, block_bytes=1024)
        try:
            midpoint = first_frame + (plan.frame_count - first_frame) // 2
            for start, stop in ((first_frame, midpoint), (midpoint, plan.frame_count)):
                if start < stop:
                    stage.accumulate(start, stop)
            return stage.project(0, plan.output_shape[0])
        finally:
            stage.close()

    def test_cuda_matches_frozen_oracle_across_orientation_and_shape_matrix(self):
        for index, view in enumerate(tilted_views() + tilted_views((45.,), shape=(6, 8, 10))):
            for shape in ((5, 7, 9), (3, 4, 5), (7, 9, 13)):
                for pattern in ('empty', 'dense', 'sparse'):
                    source = source_for(view, pattern, seed=index)
                    occupancy = np.any(source, axis=(0, 2)) if index % 2 else None
                    plan = self.build(source, view, shape, known_row_occupancy=occupancy)
                    with self.subTest(view=view.name, shape=shape, pattern=pattern):
                        np.testing.assert_array_equal(self.project(source, plan), FrozenComposedCpuOracle(source, view, shape, occupancy).dense())

    def test_cuda_promotion_preserves_cpu_row_padded_big_bit_bytes(self):
        for index, view in enumerate(tilted_views((30.,))):
            varied = replace(view, src_h=geometry.azimuthal_source_tilted_view(view).num_slices * 2 + 1,
                             azimuths_deg=(0., 17., 71., 139.))
            source = source_for(varied, seed=index)
            shape = (7, 9, 13)
            oracle = FrozenComposedCpuOracle(source, varied, shape)
            plan = self.build(source, varied, shape)
            for first_frame in (0, 1, oracle.frame_count // 2, oracle.frame_count):
                initial = oracle.packed(0, first_frame)
                saved = initial.copy()
                with self.subTest(view=view.name, first_frame=first_frame):
                    np.testing.assert_array_equal(self.project(source, plan, initial_packed=initial, first_frame=first_frame), oracle.dense())
                    np.testing.assert_array_equal(initial, saved)

    def test_cuda_basis_impulses_and_canonical_square_processing(self):
        for view in tilted_views((45.,), raster=3, shape=(6, 8, 10)):
            source = source_for(view, 'empty')
            plan = self.build(source, view, (3, 4, 5))
            for index in range(source.size):
                source.fill(0)
                source.flat[index] = 2 if index % 2 else 255
                with self.subTest(view=view.name, index=index):
                    np.testing.assert_array_equal(self.project(source, plan), FrozenComposedCpuOracle(source, view, (3, 4, 5)).dense())
        with mock.patch.dict(os.environ, {'YOLO_TTA_DELAY_NATIVE_EXPANSION': '1'}):
            for index, view in enumerate(tilted_views((30.,))):
                source = source_for(view, seed=index, processing_size=3)
                plan = self.build(source, view, (7, 9, 13))
                with self.subTest(view=view.name, canonical=True):
                    np.testing.assert_array_equal(self.project(source, plan), FrozenComposedCpuOracle(source, view, (7, 9, 13)).dense())

    def test_cuda_requires_ordered_complete_accumulation_before_publication(self):
        view = tilted_views((30.,))[0]
        source = source_for(view)
        plan = self.build(source, view, (7, 9, 13))
        stage = self.projector(source, plan, device_index=0, reserve_bytes=0)
        try:
            with self.assertRaises((RuntimeError, ValueError)):
                stage.project(0, 1)
            with self.assertRaises((RuntimeError, ValueError)):
                stage.accumulate(1, 2)
            stage.accumulate(0, plan.frame_count)
            with self.assertRaises((RuntimeError, ValueError)):
                stage.accumulate(0, 1)
            np.testing.assert_array_equal(stage.project(0, plan.output_shape[0]),
                                          FrozenComposedCpuOracle(source, view, plan.output_shape).dense())
        finally:
            stage.close()

    def test_cuda_raw_and_packed_cvol_roundtrips_after_projector_closes(self):
        from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore, CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT
        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            for index, view in enumerate(tilted_views((45.,), shape=(6, 8, 10))):
                for pattern in ('empty', 'dense', 'sparse'):
                    source = source_for(view, pattern, seed=index)
                    shape = (7, 9, 13)
                    oracle = FrozenComposedCpuOracle(source, view, shape)
                    plan = self.build(source, view, shape)
                    first_frame = plan.frame_count // 2
                    stage = self.projector(source, plan, device_index=0, reserve_bytes=0,
                        block_bytes=2 * shape[1] * shape[2], upload_bytes=source.shape[0] * source.shape[2],
                        initial_packed=oracle.packed(0, first_frame), first_frame=first_frame)
                    try:
                        stage.accumulate(first_frame, plan.frame_count)
                        blocks = {packed: [stage.project_encoded(start, min(2, shape[0] - start), packed=packed)
                                           for start in range(0, shape[0], 2)] for packed in (False, True)}
                    finally:
                        stage.close()
                    for packed in (False, True):
                        with self.subTest(view=view.name, pattern=pattern, packed=packed):
                            writer = IncrementalRawBBoxMaskStoreWriter(shape=shape,
                                store_dir=Path(td) / f'{index}-{pattern}-{packed}',
                                format_name=INTERNAL_PACKED_CVOL_FORMAT if packed else CVOL_FORMAT,
                                desc='Tilted Azimuthal CUDA oracle')
                            try:
                                for block in reversed(blocks[packed]):
                                    self.assertFalse(block.payload.flags.writeable)
                                    writer.consume_encoded_block(block.first_z, block.records, block.payload, packed=packed)
                                stats = writer.finalize()
                                expected = oracle.dense()
                                self.assertEqual(stats['foreground_voxels'], int(expected.sum()))
                                with contextlib.closing(RawBBoxMaskStore.open(writer.store_dir, mmap_payload=True)) as store:
                                    decoded = np.empty(shape, np.uint8)
                                    for frame in range(shape[0]):
                                        store.fill_decoded_slice_into(frame, decoded[frame])
                                np.testing.assert_array_equal(decoded, expected)
                            finally:
                                writer.discard()

    def test_cuda_cluster_shape_last_voxel_exceeds_signed32_packed_byte_address(self):
        from XTA.tilted_azimuthal_projection import TiltedAzimuthalProjectionPlan
        shape = (1931, 3064, 3022)
        source = np.ones((1, 1, 1), np.uint8)

        def readonly(values, dtype):
            array = np.asarray(values, dtype=dtype)
            array.flags.writeable = False
            return array

        plan = TiltedAzimuthalProjectionPlan(source_shape=source.shape, output_shape=shape,
            points=readonly([[0, 0, 0, shape[1] - 1, shape[2] - 1]], np.int32),
            stack_map=readonly([[shape[0] - 1]], np.int32),
            row_offsets=readonly([0, 1], np.int64), rows=readonly([0], np.int32), base_id=0)
        packed_bytes = shape[0] * shape[1] * ((shape[2] + 7) // 8)
        self.assertGreater(packed_bytes - 1, np.iinfo(np.int32).max)
        self.assertGreater(math.prod(shape) - 1, np.iinfo(np.uint32).max)
        # Allocate the real ~2.083-GiB device bitset, but only one host source
        # byte and one ~9-MiB host output plane. No full host volume or prefix.
        stage = self.projector(source, plan, device_index=0, reserve_bytes=0,
                               upload_bytes=4096, block_bytes=shape[1] * shape[2])
        try:
            self.assertEqual(stage.contract.packed_bytes, packed_bytes)
            stage.accumulate(0, 1)
            last = stage.project(shape[0] - 1, 1)
            self.assertEqual(last.shape, (1, shape[1], shape[2]))
            self.assertEqual(last[0, -1, -1], 1)
            self.assertEqual(int(np.count_nonzero(last)), 1)
            for packed in (False, True):
                encoded = stage.project_encoded(shape[0] - 1, 1, packed=packed)
                self.assertEqual(encoded.first_z, shape[0] - 1)
                self.assertEqual(len(encoded.records), 1)
                record = encoded.records[0]
                self.assertEqual((record.z, record.y0, record.y1, record.x0, record.x1),
                                 (shape[0] - 1, shape[1] - 1, shape[1], shape[2] - 1, shape[2]))
                self.assertEqual((record.foreground, record.size), (1, 1))
                np.testing.assert_array_equal(encoded.payload, np.asarray([1], np.uint8))
        finally:
            stage.close()


if __name__ == '__main__':
    unittest.main()

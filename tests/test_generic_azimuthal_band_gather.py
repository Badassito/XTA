"""Upright Azimuthal sink behavior and restricted singleton gather coverage.

Expected output uses the established whole-cross helper, with a separate physical
FOV oracle for dense masks. These tests do not import the benchmark candidate.
"""
from __future__ import annotations

import hashlib
from dataclasses import replace

import numpy as np
import pytest

from XTA import backprojection as bp, geometry


def make_view(base):
    return next(view for view in geometry.get_view_infos(7, 9, 11, cartesian_views=(),
        azimuthal_views=(base,), azimuthal_azimuth_angles=(45.,))
        if view.family == 'azimuthal')


def whole_cross_expected(source, view, shape):
    """Independent execution schedule: complete one orientation plane at a time."""
    stack_count, plane_shape = bp._azimuthal_output_stack_and_plane_shape(view, shape)
    grid = bp.resolve_azimuthal_processing_grid(source, view)
    plan, _ = bp.build_azimuthal_backprojection_plan(view)
    mapping = bp._azimuthal_dense_map_for_processing(
        bp.build_dense_azimuthal_backprojection_map(view, plan, out_shape_hw=plane_shape), grid)
    expected = np.zeros(shape, np.uint8)
    for index in range(stack_count):
        rows = bp._azimuthal_processing_rows_for_output(grid, stack_count, index)
        plane = bp._azimuthal_backproject_plane(source, mapping, rows)
        if view.azimuthal_base_view == 'sagittal':
            expected[:, index, :] = plane
        else:
            expected[:, :, index] = plane
    return expected


class Capture:
    def __init__(self, fail_at=None):
        self.blocks = []
        self.firsts = []
        self.fail_at = fail_at
        self.aborted = []

    def __call__(self, first, block):
        if self.fail_at is not None and len(self.blocks) == self.fail_at:
            raise RuntimeError('injected band consumer failure')
        # Keep references deliberately: a producer must not overwrite a previously
        # published band while computing its next one.
        self.firsts.append(first)
        self.blocks.append(block)

    def abort(self, reason):
        self.aborted.append(reason)


def project(source, view, shape, consumer, tmp_path, workers=2):
    return bp.backproject_azimuthal_volume_to_volume(source, view, tmp_path / 'unused.u8',
        'generic sink behavior', workers=workers, out_shape_tyx=shape, reserve_bytes=0,
        projection_block_callback=consumer, sink_only=True)


@pytest.fixture(autouse=True)
def cpu_environment(monkeypatch):
    # Upright nontransverse projection cannot select CUDA, but this also keeps
    # accidental unrelated GPU dispatch out of this behavior suite.
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_DELAY_NATIVE_EXPANSION', '1')
    monkeypatch.setenv('YOLO_TTA_PROJECTION_CALLBACK_BLOCK', '7')


@pytest.mark.parametrize('base', ['sagittal', 'coronal'])
@pytest.mark.parametrize('mode', ['singleton', 'contracted', 'upscaled', 'reduced'])
@pytest.mark.parametrize('pattern', ['empty', 'sparse', 'numeric'])
def test_sink_matches_whole_cross_oracle_and_retains_borrowed_input(tmp_path, base, mode, pattern):
    view = make_view(base)
    view = replace(view, azimuths_deg=(1., 37., 88., 151.))
    shape = [17, 9, 11]
    stack_axis = 1 if base == 'sagittal' else 2
    shape[stack_axis] = (max(1, view.src_h // 2) if mode == 'contracted' else
                         2 * view.src_h + 1 if mode == 'upscaled' else view.src_h)
    shape = tuple(shape)
    raster = (5, 5) if mode == 'reduced' else (view.src_h, view.src_w)
    data = np.zeros((view.num_slices, *raster), np.uint8)
    if pattern == 'sparse':
        data[0, 1::3, 1::2] = 1
        data[-1, -2:, -2:] = 1
    elif pattern == 'numeric':
        data[:] = np.random.default_rng(631).choice(
            np.array([0, 1, 2, 4, 128, 255], np.uint8), size=data.shape)
        data[:, 1::3] = 0
    # Exercise read-only, noncontiguous borrowed inputs in every pattern.
    source = data[:, :, ::-1]
    source.flags.writeable = False
    before = hashlib.sha256(source.tobytes()).hexdigest()
    expected = whole_cross_expected(source, view, shape)
    capture = Capture()
    result = project(source, view, shape, capture, tmp_path)
    assert isinstance(result, bp.SinkOnlyProjectionResult)
    assert result.shape == shape
    assert capture.firsts == [0, 7, 14]
    assert [len(block) for block in capture.blocks] == [7, 7, 3]
    assert not capture.aborted
    np.testing.assert_array_equal(np.concatenate(capture.blocks), expected)
    assert hashlib.sha256(source.tobytes()).hexdigest() == before


@pytest.mark.parametrize('base', ['sagittal', 'coronal'])
def test_dense_mask_has_exact_closed_physical_fov_and_exterior_background(tmp_path, base):
    view = make_view(base)
    shape = (17, 9, 11)
    source = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
    _, plane = bp._azimuthal_output_stack_and_plane_shape(view, shape)
    work_h, work_w = geometry.azimuthal_plane_shape(view)
    yy, xx = np.indices(plane, dtype=np.float64)
    dy = ((2 * yy + 1 - plane[0]) * work_h / (2 * plane[0])
          + (work_h - 1) * .5 - view.center_y)
    dx = ((2 * xx + 1 - plane[1]) * work_w / (2 * plane[1])
          + (work_w - 1) * .5 - view.center_x)
    distance = np.hypot(dx, dy)
    radius = view.roi_radius if view.roi_radius > 0 else max(0., (view.diameter - 1) / 2.)
    limit = radius + .5
    inside = distance <= limit + 8 * np.finfo(np.float64).eps * np.maximum(distance, abs(limit))
    expected = np.broadcast_to(inside[:, None, :] if base == 'sagittal'
                               else inside[:, :, None], shape)
    assert expected.any() and not expected.all()
    capture = Capture()
    project(source, view, shape, capture, tmp_path)
    np.testing.assert_array_equal(np.concatenate(capture.blocks), expected.astype(np.uint8))


@pytest.mark.parametrize('dtype', [np.uint16, np.float32, np.int16])
def test_numeric_compatibility_keeps_existing_uint8_cast_and_or(tmp_path, dtype):
    view = make_view('sagittal')
    shape = (17, 4, 11)
    source = (np.arange(view.num_slices * view.src_h * view.src_w)
              .reshape(view.num_slices, view.src_h, view.src_w) * 17).astype(dtype)
    expected = whole_cross_expected(source, view, shape)
    capture = Capture()
    project(source, view, shape, capture, tmp_path)
    np.testing.assert_array_equal(np.concatenate(capture.blocks), expected)


def test_callback_failure_aborts_once_and_clean_retry_uses_unchanged_source(tmp_path):
    view = make_view('coronal')
    source = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
    shape = (17, 9, 11)
    expected = whole_cross_expected(source, view, shape)
    consumer = Capture(fail_at=1)
    with pytest.raises(RuntimeError, match='injected band consumer failure'):
        project(source, view, shape, consumer, tmp_path, workers=3)
    assert consumer.firsts == [0]
    assert len(consumer.aborted) == 1
    np.testing.assert_array_equal(consumer.blocks[0], expected[:7])
    assert np.all(source == 1)
    retry = Capture()
    project(source, view, shape, retry, tmp_path, workers=3)
    np.testing.assert_array_equal(np.concatenate(retry.blocks), expected)


def test_empty_plan_preserves_empty_publication_before_source_validation(tmp_path, monkeypatch):
    view = make_view('sagittal')
    monkeypatch.setattr(bp, 'build_azimuthal_backprojection_plan', lambda view: ([], {}))
    consumer = Capture()
    # The original operator does not resolve source geometry when no angles exist.
    project(np.zeros((0,), np.uint8), view, (17, 9, 11), consumer, tmp_path)
    np.testing.assert_array_equal(np.concatenate(consumer.blocks), np.zeros((17, 9, 11), np.uint8))


@pytest.mark.parametrize('source,match', [
    (np.zeros((4, 5), np.uint8), '3D'),
    (np.zeros((4, 3, 5), np.uint8), 'square'),
])
def test_source_grid_errors_precede_callback_publication(tmp_path, source, match):
    consumer = Capture()
    with pytest.raises(ValueError, match=match):
        project(source, make_view('sagittal'), (17, 9, 11), consumer, tmp_path)
    assert consumer.blocks == []


@pytest.mark.parametrize('base', ['sagittal', 'coronal'])
def test_multirow_contraction_keeps_established_whole_cross_reduction(tmp_path, monkeypatch, base):
    view = make_view(base)
    shape = (17, 4, 11) if base == 'sagittal' else (17, 9, 4)
    source = np.ones((view.num_slices, view.src_h, view.src_w), np.uint8)
    expected = whole_cross_expected(source, view, shape)
    original = bp._azimuthal_backproject_plane
    observed_row_counts = []

    def record_cross(*args, **kwargs):
        observed_row_counts.append(len(args[2]))
        return original(*args, **kwargs)

    monkeypatch.setattr(bp, '_azimuthal_backproject_plane', record_cross)
    capture = Capture()
    project(source, view, shape, capture, tmp_path)
    assert observed_row_counts and max(observed_row_counts) > 1
    np.testing.assert_array_equal(np.concatenate(capture.blocks), expected)


def test_one_multirow_mapping_keeps_entire_job_on_whole_cross_path(tmp_path, monkeypatch):
    view = make_view('sagittal')
    shape = (17, view.src_h, 11)
    source = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
    source[:, 1] = 1
    source[:, 2] = 2
    original_rows = bp._azimuthal_processing_rows_for_output

    def one_mixed_row(grid, stack_count, index):
        return (np.array([1, 2], dtype=np.int32) if index == 1
                else original_rows(grid, stack_count, index))

    monkeypatch.setattr(bp, '_azimuthal_processing_rows_for_output', one_mixed_row)
    expected = whole_cross_expected(source, view, shape)
    original_cross = bp._azimuthal_backproject_plane
    seen = []

    def record_cross(*args, **kwargs):
        seen.append(len(args[2]))
        return original_cross(*args, **kwargs)

    monkeypatch.setattr(bp, '_azimuthal_backproject_plane', record_cross)
    capture = Capture()
    project(source, view, shape, capture, tmp_path)
    assert len(seen) == view.src_h * 3
    assert seen.count(2) == 3
    np.testing.assert_array_equal(np.concatenate(capture.blocks), expected)

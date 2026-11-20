"""Every categorical restore takes the same center-aligned nearest sample on upsampled axes.

Dense CPU, sparse, GPU-lane, D1, confidence projection and confidence export restores
share ``media._center_aligned_nearest_source_indices``; shrinking XY keeps the area
footprint and shrinking T keeps the interval union.
"""
import contextlib
import sys
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import finalization, geometry, media, outputs
from XTA.media import _center_aligned_nearest_source_indices as center_table

# (source, output): every axis grows, then a mixed case (Y shrinks while X grows),
# where both axes take one center-aligned nearest sample.
SHAPES = [((3, 4, 5), (10, 6, 8)), ((3, 9, 5), (10, 6, 8))]


def _in_memory(**kwargs):
    return np.zeros(kwargs['shape'], dtype=kwargs['dtype'])


def _source(shape):
    rng = np.random.default_rng(int(np.prod(shape)))
    return (rng.random(shape) < 0.45).astype(np.uint8)


def _oracle(source, out_shape):
    rows = [center_table(int(n_in), int(n_out)) for n_in, n_out in zip(source.shape, out_shape)]
    return source[np.ix_(*rows)]


def test_upsampled_temporal_restore_takes_the_center_slice():
    # The endpoint rule gave round(3*99/399) = 1.
    assert outputs._restore_source_indices_for_output_z(100, 400, 3) == [0]
    assert [outputs._restore_source_indices_for_output_z(3, 10, z)[0] for z in range(10)] == \
        center_table(3, 10).tolist()


def test_center_table_samples_cell_centers_without_library_modes():
    for n_in, n_out in [(4, 6), (5, 8), (2048, 3072), (3, 10), (9, 6)]:
        coords = (np.arange(n_out) + 0.5) * n_in / n_out
        table = center_table(n_in, n_out)
        assert np.all((table <= coords) & (coords < table + 1))
        starts, stops = outputs._nrrd_sparse_resize_axis_map(n_in, n_out, False)
        np.testing.assert_array_equal(starts, table)
        np.testing.assert_array_equal(stops, table + 1)


@pytest.mark.parametrize('source_shape,out_shape', SHAPES)
def test_every_restore_implementation_agrees(source_shape, out_shape):
    from XTA.confidence_export import compact_score_plane
    from XTA.confidence_projection import _bounded_restore_slice, read_score_slice_in_output_shape
    from tests.reference_backends.d1_orthogonal_coverage import execute_orthogonal_coverage_reference
    source = _source(source_shape)
    expected = _oracle(source, out_shape)
    assert expected.any() and not expected.all()

    with mock.patch.object(media, 'allocate_workspace_array', side_effect=_in_memory):
        dense = np.array(media.restore_mask_volume_to_original_shape(
            source, out_shape, Path('restored.uint8'), workers=1))
    np.testing.assert_array_equal(dense, expected, err_msg='media dense restore')

    union = np.zeros(out_shape, np.uint8)
    finalization.assemble_view_volumes_into_native_union(
        union, {'transverse': source}, *source_shape, out_shape_tyx=out_shape, workers=1)
    np.testing.assert_array_equal(union, expected, err_msg='finalization CPU assembly')

    nrrd_dense = np.stack([outputs._read_layer_slice_in_output_shape(source, out_shape, z)
                           for z in range(out_shape[0])])
    np.testing.assert_array_equal(nrrd_dense, expected, err_msg='NRRD dense restore')

    sparse = np.zeros(out_shape, np.uint8)
    for z in range(out_shape[0]):
        for source_z in outputs._restore_source_indices_for_output_z(source_shape[0], out_shape[0], z):
            plane = source[source_z]
            ys, xs = np.nonzero(plane)
            bbox = (int(ys.min()), int(xs.min()), int(ys.max()) + 1, int(xs.max()) + 1)
            region = outputs._resize_sparse_binary_crop_to_output_region(
                plane[bbox[0]:bbox[2], bbox[1]:bbox[3]], source_shape=source_shape[1:],
                source_bbox=bbox, output_shape=out_shape[1:])
            if region is not None:
                y0, x0, y1, x1, crop = region
                sparse[z, y0:y1, x0:x1] |= crop
    np.testing.assert_array_equal(sparse, expected, err_msg='NRRD sparse restore')

    view = geometry.get_view_infos(*source_shape, cartesian_views=('transverse',))[0]
    d1 = execute_orthogonal_coverage_reference(source, view, out_shape)
    np.testing.assert_array_equal(d1, expected, err_msg='D1 coverage reference')

    scores = source * np.uint8(173)
    for z in range(out_shape[0]):
        np.testing.assert_array_equal(read_score_slice_in_output_shape(scores, out_shape, z),
                                      expected[z] * np.uint8(173), err_msg='confidence projection')
        np.testing.assert_array_equal(_bounded_restore_slice(scores, out_shape, z, 7),
                                      expected[z] * np.uint8(173), err_msg='bounded confidence restore')

    class _ScoreReader:
        def read_crop(self, source_z):
            return (0, source_shape[1], 0, source_shape[2], scores[source_z])

    everywhere = np.ones(out_shape[1:], np.uint8)
    for z in range(out_shape[0]):
        compact = compact_score_plane(_ScoreReader(), everywhere, source_shape=source_shape,
                                      target_shape=out_shape, z=z)
        np.testing.assert_array_equal(compact, expected[z] * np.uint8(173), err_msg='confidence export')


class _CudaShim:
    """Just enough of torch.cuda to run the final-fusion GPU lane on CPU tensors."""

    lane_streams = 0

    @staticmethod
    def is_available():
        return True

    @staticmethod
    def device_count():
        return 1

    @staticmethod
    def mem_get_info(_device):
        return (64 << 30, 64 << 30)

    @staticmethod
    def device(_device):
        return contextlib.nullcontext()

    @staticmethod
    def stream(_stream):
        return contextlib.nullcontext()

    @staticmethod
    def empty_cache():
        return None

    class Stream:
        def __init__(self, device=None):
            _CudaShim.lane_streams += 1

        def synchronize(self):
            return None

        def wait_event(self, _event):
            return None

    class Event:
        def __init__(self, **_kwargs):
            pass

        def record(self, _stream=None):
            return None

        def synchronize(self):
            return None


def test_gpu_final_fusion_lane_reproduces_the_cpu_restore(monkeypatch):
    torch = pytest.importorskip('torch')

    class _TorchShim:
        cuda = _CudaShim

        def __getattr__(self, name):
            return getattr(torch, name)

        @staticmethod
        def device(_name):
            return torch.device('cpu')

        @staticmethod
        def empty(*args, pin_memory=False, **kwargs):
            return torch.empty(*args, **kwargs)

    out_shape = (10, 9, 8)
    views = {
        'transverse': _source((4, 12, 5)),   # Y shrinks, X grows: nearest on both
        'sagittal': _source((12, 3, 10)),    # (Y, T, X): both XY axes shrink, area footprint
        'coronal': _source((5, 7, 4)),       # (X, T, Y): both XY axes grow
    }
    monkeypatch.setattr(finalization, 'gpu_slice_labeling_configured_devices', lambda: None)

    monkeypatch.setattr(finalization, 'fused_final_gpu_enabled', lambda: False)
    expected = np.zeros(out_shape, np.uint8)
    finalization.assemble_view_volumes_and_projected_layers_fused(
        expected, views, [], 4, 12, 5, out_shape_tyx=out_shape, workers=1)
    assert expected.any()

    monkeypatch.setattr(finalization, 'fused_final_gpu_enabled', lambda: True)
    _CudaShim.lane_streams = 0
    actual = np.zeros(out_shape, np.uint8)
    with mock.patch.dict(sys.modules, {'torch': _TorchShim()}):
        finalization.assemble_view_volumes_and_projected_layers_fused(
            actual, views, [], 4, 12, 5, out_shape_tyx=out_shape, workers=1)
    assert _CudaShim.lane_streams > 0, 'the GPU lane did not run'
    np.testing.assert_array_equal(actual, expected)

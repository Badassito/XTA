"""Azimuthal CPU SAM images: bounded parallel preparation is byte-identical to serial."""
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry
from XTA.config import resolve_tilted_view_groups
from XTA.sam_cyclic import build_cyclic_frame_addressing
from tests.test_sam_view_image_cache import context_for, demand

SHAPE = (24, 128, 128)  # 128*128 bytes is a 16 KiB multiple: auto take-gather applies.


@pytest.fixture(autouse=True)
def cpu_budget(monkeypatch):
    monkeypatch.setattr('XTA.workspace._cpu_count', lambda: 4)
    monkeypatch.delenv('YOLO_TTA_SAM_AZIMUTHAL_CPU_PREFETCH', raising=False)
    monkeypatch.delenv('YOLO_TTA_AZIMUTHAL_TAKE_GATHER', raising=False)
    monkeypatch.setenv('YOLO_TTA_SAM_TILTED_AZIMUTHAL_PREFETCH_MIN_WIDTH', '1')


def source_volume(shape=SHAPE, seed=20261029):
    return np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)


def azimuthal_view(shape=SHAPE, *, raster=0, base='transverse', tilted=False, angle=30.):
    if tilted:
        views = geometry.get_view_infos(*shape, cartesian_views=(),
            tilt_groups=resolve_tilted_view_groups([f'{base}:20:vertical']),
            azimuthal_views=(f'tilted_{base}',), azimuthal_azimuth_angles=(angle,),
            azimuthal_native_raster=raster)
    else:
        views = geometry.get_view_infos(*shape, cartesian_views=(), azimuthal_views=(base,),
            azimuthal_azimuth_angles=(angle,), azimuthal_native_raster=raster)
    return next(view for view in views if view.family == 'azimuthal')


@pytest.mark.parametrize('rows', (1, 7, 256))
@pytest.mark.parametrize('plane', ((64, 64), (128, 128), (61, 67)))
def test_take_gather_matches_advanced_index_values_and_strides(monkeypatch, rows, plane):
    monkeypatch.setenv('YOLO_TTA_AZIMUTHAL_TAKE_GATHER', '1')
    block = np.random.default_rng(rows).integers(0, 256, (rows, plane[0]*plane[1]), np.uint8)
    flat = np.random.default_rng(7).integers(0, block.shape[1], (53, 4)).astype(np.int64)
    expected = block[:, flat]
    actual = geometry._azimuthal_gather_contiguous_block(block, flat)
    np.testing.assert_array_equal(actual, expected)
    assert actual.shape == expected.shape and actual.strides == expected.strides
    samples = actual.astype(np.float32, copy=False)
    weights = np.random.default_rng(3).random((53, 4), np.float32)
    reference = np.einsum('tuk,uk->tu', expected.astype(np.float32, copy=False), weights)
    assert np.array_equal(np.einsum('tuk,uk->tu', samples, weights).view(np.uint32), reference.view(np.uint32))


def test_auto_take_gather_only_selects_set_aliasing_plane_strides(monkeypatch):
    assert geometry._azimuthal_take_gather_selected(512*512)
    assert geometry._azimuthal_take_gather_selected(16384)
    assert not geometry._azimuthal_take_gather_selected(3064*3023)
    assert not geometry._azimuthal_take_gather_selected(8192)
    monkeypatch.setenv('YOLO_TTA_AZIMUTHAL_TAKE_GATHER', '0')
    assert not geometry._azimuthal_take_gather_selected(512*512)
    monkeypatch.setenv('YOLO_TTA_AZIMUTHAL_TAKE_GATHER', '1')
    assert geometry._azimuthal_take_gather_selected(3064*3023)


@pytest.mark.parametrize('raster', (0, 16))  # 16 < stack folds the stack axis.
@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_native_azimuthal_frames_identical_with_and_without_take_gather(monkeypatch, raster, base):
    shape = (32, 128, 128) if base == 'transverse' else (128, 32, 128) if base == 'sagittal' else (128, 128, 32)
    source = source_volume(shape)
    view = azimuthal_view(shape, raster=raster, base=base, angle=17.)
    for frame in (0, 1, view.num_slices//2, view.num_slices-1):
        monkeypatch.setenv('YOLO_TTA_AZIMUTHAL_TAKE_GATHER', '0')
        expected = geometry.get_view_frame_by_index(source, view, frame)
        monkeypatch.setenv('YOLO_TTA_AZIMUTHAL_TAKE_GATHER', '1')
        actual = geometry.get_view_frame_by_index(source, view, frame)
        assert actual.dtype == expected.dtype and actual.tobytes() == expected.tobytes()


def build(tmp_path, monkeypatch, name, view, shape, frames, *, prefetch, addressing=None, source=None):
    monkeypatch.setenv('YOLO_TTA_SAM_AZIMUTHAL_CPU_PREFETCH', '1' if prefetch else '0')
    context = context_for(tmp_path/name, source_volume() if source is None else source)
    telemetry = SimpleNamespace(add=mock.Mock(), gauge=mock.Mock())
    monkeypatch.setattr('XTA.sam_integration.runtime_telemetry', lambda: telemetry)
    try:
        reference = context.image_provider(view, shape, demand(shape, frames, addressing))
        data = reference.open()
        try:
            payload = bytes(np.asarray(data))
        finally:
            data._mmap.close()
        metrics = {key: getattr(context, key) for key in ('rendered_frames', 'rendered_pixels',
            'native_sampling_calls', 'native_sampling_pixels', 'canonical_sampling_pixels',
            'image_cache_reused_pixels', 'source_materializations')}
        count = lambda key: sum(call.args[1] for call in telemetry.add.call_args_list if call.args[0] == key)
        return dict(payload=payload, identity=reference.identity_sha256, crops=reference.frame_crops,
            shape=reference.shape, metrics=metrics, prepared=count('sam.cpu_images.azimuthal_prefetch_records'),
            cohorts=count('sam.cpu_images.azimuthal_prefetch_cohorts'),
            declines=count('sam.cpu_images.azimuthal_prefetch_declines'))
    finally:
        context.close()


@pytest.mark.parametrize('case', ('transverse', 'fold', 'tilted', 'cyclic'))
def test_prefetched_cohort_bytes_identity_and_metrics_match_serial(tmp_path, monkeypatch, case):
    view = azimuthal_view(raster=16 if case == 'fold' else 0, tilted=case == 'tilted')
    shape = (view.num_slices, 40, 40)
    frames = {0: (0, 0, 40, 40), 1: (3, 5, 31, 37), 2: (0, 0, 40, 40), view.num_slices-1: (7, 0, 40, 23)}
    addressing = None
    if case == 'cyclic':
        addressing = dict(build_cyclic_frame_addressing(shape, 3))
        # Mirrored aliases of frames 0/1/2: N and N+2 are fully reused, N+1 only partially.
        n = view.num_slices
        frames.update({n: (0, 0, 40, 40), n+1: (0, 0, 40, 40), n+2: (11, 2, 29, 38)})
    serial = build(tmp_path, monkeypatch, 'serial', view, shape, frames, prefetch=False, addressing=addressing)
    parallel = build(tmp_path, monkeypatch, 'parallel', view, shape, frames, prefetch=True, addressing=addressing)
    assert serial['cohorts'] == serial['prepared'] == 0
    assert parallel['cohorts'] == 1 and parallel['prepared'] >= 2
    for key in ('payload', 'identity', 'crops', 'shape', 'metrics'):
        assert parallel[key] == serial[key], key
    # Every prepared record is exactly one serial native-plane sampling.
    assert parallel['prepared'] == serial['metrics']['native_sampling_calls']
    if case == 'cyclic':
        assert serial['metrics']['image_cache_reused_pixels'] > 0
        # Fully covered aliases never sample a plane on either route.
        assert parallel['prepared'] == len(frames)-2


def test_lazy_unmaterialized_source_materializes_once_like_serial(tmp_path, monkeypatch):
    view = azimuthal_view()
    shape = (view.num_slices, 32, 32)
    frames = {frame: (0, 0, 32, 32) for frame in range(4)}
    data = source_volume()

    class Lazy:
        _is_lazy_processing_cube = True
        streaming_backend = True
        def __init__(self):
            self.shape, self.dtype, self.ndim = data.shape, data.dtype, 3
            self._array = None
        @property
        def materialized(self):
            return self._array is not None
    lazy_sources = []
    def make():
        lazy = Lazy()
        lazy_sources.append(lazy)
        return lazy
    def ready(source):
        if getattr(source, '_is_lazy_processing_cube', False):
            source._array = data
    monkeypatch.setattr('XTA.media.wait_for_volume_ready', ready)
    serial = build(tmp_path, monkeypatch, 'serial', view, shape, frames, prefetch=False, source=make())
    parallel = build(tmp_path, monkeypatch, 'parallel', view, shape, frames, prefetch=True, source=make())
    assert parallel['cohorts'] == 1
    assert serial['metrics']['source_materializations'] == parallel['metrics']['source_materializations'] == 1
    assert parallel['payload'] == serial['payload'] and parallel['identity'] == serial['identity']


def test_toggle_scratch_and_tilted_width_gates_keep_serial_route(tmp_path, monkeypatch):
    view = azimuthal_view()
    shape = (view.num_slices, 40, 40)
    frames = {frame: (0, 0, 40, 40) for frame in range(4)}
    expected = build(tmp_path, monkeypatch, 'off', view, shape, frames, prefetch=False)
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', str(64*1024))
    tight = build(tmp_path, monkeypatch, 'tight', view, shape, frames, prefetch=True)
    assert tight['cohorts'] == 0 and tight['declines'] == 1 and tight['payload'] == expected['payload']
    monkeypatch.delenv('YOLO_TTA_SAM_RENDER_MAX_BYTES')
    monkeypatch.setenv('YOLO_TTA_SAM_TILTED_AZIMUTHAL_PREFETCH_MIN_WIDTH', '1024')
    tilted = azimuthal_view(tilted=True)
    narrow = build(tmp_path, monkeypatch, 'narrow', tilted, (tilted.num_slices, 40, 40), frames, prefetch=True)
    assert narrow['cohorts'] == 0


def test_worker_failure_propagates_and_refunds_private_cache(tmp_path, monkeypatch):
    view = azimuthal_view()
    shape = (view.num_slices, 40, 40)
    original = geometry.get_view_frame_by_index
    def failing(source, sampled_view, index, view_frames=None):
        if int(index) == 2:
            raise RuntimeError('native azimuthal failure')
        return original(source, sampled_view, index, view_frames)
    monkeypatch.setattr(geometry, 'get_view_frame_by_index', failing)
    monkeypatch.setenv('YOLO_TTA_SAM_AZIMUTHAL_CPU_PREFETCH', '1')
    context = context_for(tmp_path, source_volume())
    try:
        with pytest.raises(RuntimeError, match='native azimuthal failure'):
            context.image_provider(view, shape, demand(shape, {frame: (0, 0, 40, 40) for frame in range(5)}))
        assert not context._image_builds and not context._cache_entries
        cache_dir = context.temp_dir/'sam_image_cache'
        assert not cache_dir.exists() or not any(cache_dir.iterdir())
    finally:
        context.close()


def test_donor_cache_keeps_serial_reuse_route(tmp_path, monkeypatch):
    view = azimuthal_view()
    shape = (view.num_slices, 40, 40)
    monkeypatch.setenv('YOLO_TTA_SAM_AZIMUTHAL_CPU_PREFETCH', '1')
    context = context_for(tmp_path, source_volume())
    telemetry = SimpleNamespace(add=mock.Mock(), gauge=mock.Mock())
    monkeypatch.setattr('XTA.sam_integration.runtime_telemetry', lambda: telemetry)
    try:
        context.image_provider(view, shape, demand(shape, {0: (0, 0, 20, 40), 1: (0, 0, 20, 40)}))
        cohorts = lambda: sum(call.args[1] for call in telemetry.add.call_args_list
                              if call.args[0] == 'sam.cpu_images.azimuthal_prefetch_cohorts')
        assert cohorts() == 1
        context.image_provider(view, shape, demand(shape, {0: (0, 0, 40, 40), 1: (0, 0, 40, 40)}))
        assert cohorts() == 1 and context.image_cache_reused_pixels > 0
    finally:
        context.close()

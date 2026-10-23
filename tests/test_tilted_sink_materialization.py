"""Tilted layers publish exact ordered CVOL planes without a dense destination."""
import gc

import numpy as np
import pytest

from XTA import assembly, backprojection, geometry
from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore
from XTA.projection_coverage import iter_destination_samples
from XTA.runtime import (
    defer_retired_memmap_directory_cleanup,
    wait_for_retired_memmap_directory_cleanup,
)


def _fixture():
    view = geometry._build_tilted_view_infos(
        9, 11, 13, tilt_views=('transverse',), tilt_angles=(30.,),
        tilt_directions=('horizontal',),
    )[0]
    source = np.zeros((view.num_slices, 7, 7), np.uint8)
    source[::2, 1:6, 2:7] = 1
    shape = (13, 15, 17)
    expected = np.zeros(shape, np.uint8)
    for destination, frames, rows, columns in iter_destination_samples(view, source.shape, shape):
        expected.reshape(-1)[destination[source[frames, rows, columns] != 0]] = 1
    return view, source, shape, expected


def _materialize(tmp_path, monkeypatch, *, internal=False):
    view, source, shape, expected = _fixture()
    monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'numpy')
    monkeypatch.setattr(assembly, 'final_source_output_shape', lambda: shape)
    monkeypatch.setattr(assembly, 'nrrd_layer_sink', lambda: None)
    original = source.copy()
    ref = assembly.materialize_nrrd_view_layer(
        source, model_name='test', view=view, source='fullframe', mask_kind='yolo',
        stage='pre_interpolation', temp_dir=tmp_path, workers=3,
        known_has_foreground=True, submit_to_sink=False, internal_packbits_store=internal,
    )
    assert ref is not None and ref.shape == shape
    store = RawBBoxMaskStore.open(ref.path, mmap_payload=True)
    try:
        actual = np.stack([store.decode_slice(z) for z in range(shape[0])])
        meta = dict(store.meta)
    finally:
        store.close()
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(source, original)
    # Dense fallback retires its mapping without invalidating live NumPy views;
    # Windows unlinks the backing after the last owner and deferred cleanup settle.
    gc.collect()
    for raw_path in tmp_path.rglob('*.orthogonal.u8.dat'):
        defer_retired_memmap_directory_cleanup(raw_path.parent)
        wait_for_retired_memmap_directory_cleanup(raw_path.parent)
    assert not list(tmp_path.rglob('*.orthogonal.u8.dat'))
    return ref, meta


@pytest.mark.parametrize('internal', [False, True], ids=['nrrd-cvol', 'internal-packbits'])
def test_tilted_materialization_streams_ordered_planes_without_dense_allocation(
    tmp_path, monkeypatch, internal,
):
    writer_type = assembly.IncrementalRawBBoxMaskStoreWriter
    covered = []

    class RecordingWriter(writer_type):
        def __call__(self, z0, block):
            covered.extend(range(z0, z0 + len(block)))
            return super().__call__(z0, block)

        def consume_empty_range(self, z0, count):
            covered.extend(range(z0, z0 + count))
            return super().consume_empty_range(z0, count)

    monkeypatch.setattr(assembly, 'IncrementalRawBBoxMaskStoreWriter', RecordingWriter)
    monkeypatch.setattr(backprojection, 'allocate_workspace_array',
        lambda **kwargs: pytest.fail('Tilted sink allocated a full dense destination'))
    monkeypatch.setattr(assembly, 'write_raw_bbox_mask_store',
        lambda *args, **kwargs: pytest.fail('Tilted sink rescanned a dense destination'))
    ref, meta = _materialize(tmp_path, monkeypatch, internal=internal)
    assert covered == list(range(ref.shape[0]))
    assert meta['projection_payload_fusion'] == 'tilted_native_destination_pull'
    assert meta['projection_sink_only'] is True
    assert meta['projection_geometry_contract'] == 'xta.native_destination_pull/1'
    if internal:
        assert ref.storage_format == INTERNAL_PACKED_CVOL_FORMAT


@pytest.mark.parametrize('failure_point', ['write', 'finalize'])
def test_tilted_sink_failure_discards_partial_store_and_retries_exact_dense_projection(
    tmp_path, monkeypatch, failure_point,
):
    writer_type = assembly.IncrementalRawBBoxMaskStoreWriter
    calls = {'aborted': 0, 'discarded': 0, 'dense_encodes': 0}
    dense_encode = assembly.write_raw_bbox_mask_store

    class FailedWriter(writer_type):
        def __call__(self, z0, block):
            super().__call__(z0, block)
            if failure_point == 'write':
                raise OSError('injected tilted sink write failure')

        def finalize(self):
            if failure_point == 'finalize':
                raise OSError('injected tilted sink finalization failure')
            return super().finalize()

        def abort(self, reason):
            calls['aborted'] += 1
            return super().abort(reason)

        def discard(self):
            calls['discarded'] += 1
            return super().discard()

    def encode(*args, **kwargs):
        calls['dense_encodes'] += 1
        return dense_encode(*args, **kwargs)

    monkeypatch.setattr(assembly, 'IncrementalRawBBoxMaskStoreWriter', FailedWriter)
    monkeypatch.setattr(assembly, 'write_raw_bbox_mask_store', encode)
    _, meta = _materialize(tmp_path, monkeypatch)
    assert meta['projection_sink_only'] is False
    assert calls['aborted'] >= 1
    assert calls['discarded'] == 1
    assert calls['dense_encodes'] == 1

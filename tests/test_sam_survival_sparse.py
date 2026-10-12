"""Sparse support auditing keeps dense-reader counts and validation semantics."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from XTA import interpolation, tta_outputs
from XTA.artifact_archive import publish_directory, reference
from XTA.interpolation import NrrdLayerRef, RawBBoxMaskStore
from XTA.reconciliation_runtime import RuntimeLayer


def sparse_layer(tmp_path, *, packed=True, archive=False, key='forward'):
    value = np.zeros((5, 7, 11), np.uint8)
    value[0, 2:4, 1:4] = 1
    value[0, 2, 2] = 0  # Foreground in the final bbox hole is not selected support.
    value[4, 6, 8:10] = 1
    format_name = (interpolation.INTERNAL_PACKED_CVOL_FORMAT if packed
                   else interpolation.CVOL_FORMAT)
    path = tmp_path / f'{key}.cvol'
    interpolation.write_raw_bbox_mask_store(value, path, format_name=format_name, workers=1)
    if archive:
        virtual = Path(reference(tmp_path / 'source.tar', f'layers/{key}.cvol'))
        publish_directory(path, virtual)
        path = virtual
    layer = NrrdLayerRef(key=key, name=key, path=path, shape=value.shape,
        storage_format=format_name, model_name='detector', mask_kind='bridge',
        interpolation_backend='sam', interpolation_direction='forward',
        proposal_selection_status='policy_selected', sam_run_ids=('selected-run',),
        # Proposal metadata does not remap the already-projected component backing.
        native_transform={'kind': 'radial', 'native_shape_tyx': [11, 5, 7],
                          'source_shape_tyx': list(value.shape), 'view_name': 'radial'})
    return layer, value


@pytest.fixture(autouse=True)
def support_only(monkeypatch):
    # Independent adjacent tests exercise topology and proposal identity checks.
    monkeypatch.setattr(tta_outputs, 'measure_sam_final_connections', lambda *_a, **_k: {})


@pytest.mark.parametrize('packed', (False, True))
@pytest.mark.parametrize('archive', (False, True))
def test_native_crops_match_counts_without_dense_reads_and_validate_planes_once(
        tmp_path, monkeypatch, packed, archive):
    layer, value = sparse_layer(tmp_path, packed=packed, archive=archive)
    final = np.ones_like(value)
    final[0, 2, 1] = 0
    reads = []
    original_any = np.any

    def observed_any(array, *args, **kwargs):
        if isinstance(array, np.ndarray) and array.shape == final.shape[1:]:
            reads.append(array.shape)
        return original_any(array, *args, **kwargs)

    monkeypatch.setattr(np, 'any', observed_any)
    monkeypatch.setattr(RuntimeLayer, 'read_slab', lambda *_a: pytest.fail('dense support read'))
    other = replace(layer, key='backward', interpolation_direction='backward')
    receipts = tta_outputs.measure_bridge_output_survival([layer, other, layer], final)
    assert len(receipts) == 2
    assert len(reads) == final.shape[0]
    assert sum(item['final_binary_validation_planes'] for item in receipts) == final.shape[0]
    for item in receipts:
        assert item['selected_voxels'] == np.count_nonzero(value) == 7
        assert item['retained_voxels'] == np.count_nonzero(value & final) == 6
        assert item['removed_voxels'] == 1
        assert item['support_survival'] == 'later_filtered'
        assert item['source_grid_planes_read'] == 5
        assert item['source_grid_full_planes_read'] == 0
        assert item['source_grid_bbox_crops_read'] == 2
        assert item['support_read_strategy'] == 'native_bbox_crops'
        assert item['sam_run_ids'] == ['selected-run']
        assert item['connection_survival'] == 'not_assessed'


@pytest.mark.parametrize('frame', (0, 2))
def test_covered_plane_still_validates_full_final_plane_outside_support(tmp_path, frame):
    layer, value = sparse_layer(tmp_path)
    final = np.zeros_like(value)
    final[frame, 0, 0] = 2
    with pytest.raises(ValueError, match='binary 0/1'):
        tta_outputs.measure_bridge_output_survival([layer], final)


def test_empty_extent_preserves_untouched_validation_domain(tmp_path, monkeypatch):
    layer, value = sparse_layer(tmp_path)
    layer = replace(layer, segment_extent_ijk=(0, -1, 0, -1, 0, -1),
                    segment_extent_shape_tyx=value.shape)
    monkeypatch.setattr(RawBBoxMaskStore, 'open', lambda *_a, **_k: pytest.fail('empty payload read'))
    final = np.full_like(value, 2)
    item = tta_outputs.measure_bridge_output_survival([layer], final)[0]
    assert item['selected_voxels'] == item['retained_voxels'] == 0
    assert item['source_grid_planes_read'] == item['final_binary_validation_planes'] == 0
    assert item['support_read_strategy'] == 'empty_extent'


@pytest.mark.parametrize('output_shape', ((7, 9, 13), (3, 4, 6)))
def test_resized_backings_keep_shared_restoration_reader(tmp_path, monkeypatch, output_shape):
    layer, _ = sparse_layer(tmp_path)
    owner = RuntimeLayer(layer, output_shape)
    try:
        oracle = owner.read_slab(0, output_shape[0])
    finally:
        owner.close()
    final = np.ones(output_shape, np.uint8)
    final[::2, ::2, ::2] = 0
    original = RuntimeLayer.read_slab
    calls = []

    def observed(self, z0, z1):
        calls.append((z0, z1))
        return original(self, z0, z1)

    monkeypatch.setattr(RuntimeLayer, 'read_slab', observed)
    item = tta_outputs.measure_bridge_output_survival([layer], final)[0]
    assert len(calls) == output_shape[0]
    assert item['selected_voxels'] == np.count_nonzero(oracle)
    assert item['retained_voxels'] == np.count_nonzero(np.logical_and(oracle, final))
    assert item['support_read_strategy'] == 'restored_full_planes'
    assert item['source_grid_full_planes_read'] == output_shape[0]
    assert item['source_grid_bbox_crops_read'] == 0


def test_same_shape_dense_backing_stays_dense(tmp_path, monkeypatch):
    layer, value = sparse_layer(tmp_path)
    path = tmp_path / 'dense.dat'
    value.tofile(path)
    layer = replace(layer, path=path, storage_format='raw_u8')
    original = RuntimeLayer.read_slab
    calls = []

    def observed(self, z0, z1):
        calls.append((z0, z1))
        return original(self, z0, z1)

    monkeypatch.setattr(RuntimeLayer, 'read_slab', observed)
    item = tta_outputs.measure_bridge_output_survival([layer], value.copy())[0]
    assert len(calls) == value.shape[0]
    assert item['selected_voxels'] == item['retained_voxels'] == 7
    assert item['support_read_strategy'] == 'restored_full_planes'


def test_native_crop_corruption_is_rejected_and_owner_closed(tmp_path, monkeypatch):
    layer, value = sparse_layer(tmp_path)
    original_init, original_close = RuntimeLayer.__init__, RuntimeLayer.close
    closed = []

    def corrupted(self, ref, shape):
        original_init(self, ref, shape)
        self.source.index = self.source.index.copy()
        self.source.index[0]['y1'] = shape[1] + 1

    def observed_close(self):
        closed.append(self)
        original_close(self)

    monkeypatch.setattr(RuntimeLayer, '__init__', corrupted)
    monkeypatch.setattr(RuntimeLayer, 'close', observed_close)
    with pytest.raises(ValueError, match='invalid bbox'):
        tta_outputs.measure_bridge_output_survival([layer], value.copy())
    assert len(closed) == 1
    assert closed[0].source is None

"""Tile configurations analyze the complete canonical observation grid before joining a parent."""
from __future__ import annotations

import numpy as np

from XTA import assembly, geometry


def _settings(view, tmp_path):
    return dict(model_name='detector', view=view, config_id='seam', temp_dir=tmp_path,
                interpolate=9, interpolation_walk_back=1, interpolation_candidates=1,
                interpolate_passes=1, interpolate_min_radius=0., interpolation_search_angle=65.,
                slice_workers=1, interpolation_task_workers=1)


def test_real_sdf_connects_endpoints_across_overlapping_detector_tiles(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_GPU_SLICE_LABELING', '0')
    original = np.zeros((11, 25, 34), np.uint8)
    original[1, 10:15, 8:13] = 1
    original[9, 10:15, 19:24] = 1
    view = geometry.get_view_infos(*original.shape, cartesian_views=('transverse',))[0]
    # The footprints overlap four columns, but neither one contains both endpoints.
    canonical = np.zeros_like(original)
    for left, right in ((0, 18), (14, 34)):
        tile = original[:, :, left:right]
        assert sum(bool(plane.any()) for plane in tile) == 1
        canonical[:, :, left:right] |= tile
    np.testing.assert_array_equal(canonical, original)
    result = assembly.finalize_independent_tile_config(tile_accumulator_mm=canonical, **_settings(view, tmp_path))
    assert result.interpolation_stats[0]['accepted_connections'] == 2
    assert result.interpolation_stats[0]['added_voxels'] == 177
    assert np.count_nonzero(result.final_accumulator_mm & ~original) == 177
    assert not result.nrrd_layers


def test_empty_independent_config_needs_no_canvas_or_interpolator(tmp_path, monkeypatch):
    from dataclasses import replace
    from XTA import confidence_tiles
    view = replace(geometry.get_view_infos(4, 6, 9, cartesian_views=('transverse',))[0],
                   augmentation_base_view='transverse')
    layers, retired = [], []
    def empty_layer(**fields):
        layers.append(fields)
        return 'empty-ref'
    monkeypatch.setattr(assembly, 'materialize_empty_policy_layer', empty_layer)
    monkeypatch.setattr(confidence_tiles, 'retire_tile_confidence', lambda *identity: retired.append(identity))
    result = assembly.finalize_independent_tile_config(
        tile_accumulator_mm=None, nrrd_layers_enabled=True, **_settings(view, tmp_path))
    assert result.final_accumulator_mm is None and result.interpolation_stats == []
    assert result.nrrd_layers == ['empty-ref']
    assert (layers[0]['stage'], layers[0]['tile_acceptance']) == ('seam_pre_tile_interpolation', 'independent')
    assert retired == [('detector', view.name, 'seam')]
    from XTA.cuda_d1 import _nrrd_layer_name
    name = _nrrd_layer_name(view=view, source='tile', mask_kind='yolo', pass_index=0,
                            tile_config_id='seam', tile_acceptance='independent')
    assert 'independent observation' in name and 'accepted by' not in name

    empty = np.zeros((4, 6, 9), np.uint8)
    result = assembly.finalize_independent_tile_config(tile_accumulator_mm=empty, **_settings(view, tmp_path))
    assert result.final_accumulator_mm is None and result.interpolation_stats == []
    assert not empty.any()


def test_simultaneous_model_observation_refs_retain_independent_bytes(tmp_path, monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from XTA.interpolation import RawBBoxMaskStore

    shape = (3, 7, 9)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    monkeypatch.setattr(assembly, 'nrrd_layer_sink', lambda: None)
    def analyze(item):
        model_name, column = item
        canvas = np.zeros(shape, np.uint8)
        canvas[1, 2, column] = 1
        options = _settings(view, tmp_path)
        options.update(model_name=model_name, interpolate=0, nrrd_layers_enabled=True)
        result = assembly.finalize_independent_tile_config(tile_accumulator_mm=canvas, **options)
        [observed] = result.nrrd_layers
        return observed, canvas

    with ThreadPoolExecutor(max_workers=2) as workers:
        products = list(workers.map(analyze, [('first', 3), ('second', 4)]))
    assert products[0][0].path != products[1][0].path
    for layer, expected in products:
        assert layer.tile_acceptance == 'independent'
        reader = RawBBoxMaskStore.open(layer.path, mmap_payload=True)
        try:
            actual = np.stack([reader.decode_slice(z) for z in range(shape[0])])
        finally:
            reader.close()
        np.testing.assert_array_equal(actual != 0, expected != 0)

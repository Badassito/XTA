"""The selected interpolation support keeps the existing whole-component tile gate."""
from __future__ import annotations

import numpy as np
import pytest

from XTA import assembly


@pytest.mark.parametrize('backend', ('sdf', 'sam'))
def test_parent_then_selected_bridge_admits_whole_original_components(backend):
    # Both backends present binary support to the same gate. A single support hit
    # admits the complete component, including its pixels beyond that support.
    parent_crop = (2, 8, 3, 10)
    original = np.zeros((2, 6, 7), dtype=np.uint8)
    original[0, 0:2, 0:2] = 1
    original[0, 3:5, 4:7] = 1
    original[1, 2:4, 1:3] = 1
    parent = np.zeros((2, 10, 12), dtype=np.uint8)
    parent[0, 2, 3] = 1
    selected_bridge = np.zeros_like(parent)
    selected_bridge[0, 5, 7] = 1
    accepted = np.zeros_like(parent)
    direct_category = np.zeros_like(parent)
    bridge_category = np.zeros_like(parent)
    parent_before = parent.copy()
    bridge_before = selected_bridge.copy()
    residual = original.copy()

    direct = assembly.gate_tile_components_against_support_inplace(
        residual, parent, parent_crop=parent_crop,
        accepted_total_mm=accepted, accepted_category_mm=direct_category,
        retain_rejected_components=True, workers=1, desc=f'{backend}: direct',
    )
    assert direct == dict(accepted_components=1, rejected_components=2,
                         accepted_voxels=4, rejected_voxels=10)
    assert int(residual.sum()) == 10
    bridge = assembly.gate_tile_components_against_support_inplace(
        residual, selected_bridge, parent_crop=parent_crop,
        accepted_total_mm=accepted, accepted_category_mm=bridge_category,
        retain_rejected_components=False, workers=1, desc=f'{backend}: bridge',
    )
    assert bridge == dict(accepted_components=1, rejected_components=1,
                         accepted_voxels=6, rejected_voxels=4)
    assert int(accepted.sum()) == 10
    assert int(direct_category.sum()) == 4
    assert int(bridge_category.sum()) == 6
    assert int(residual.sum()) == 6
    assert np.array_equal(accepted, direct_category | bridge_category)
    assert np.array_equal(parent, parent_before)
    assert np.array_equal(selected_bridge, bridge_before)


def test_empty_selected_sam_bridge_preserves_direct_parent_admission():
    original = np.zeros((1, 7, 7), dtype=np.uint8)
    original[0, 0:2, 0:2] = 1
    original[0, 5:7, 5:7] = 1
    parent = np.zeros_like(original)
    parent[0, 0, 0] = 1
    raw_rejected_sam = np.zeros_like(original)
    raw_rejected_sam[0, 6, 6] = 1
    selected = np.zeros_like(original)
    accepted = np.zeros_like(original)
    residual = original.copy()
    assembly.gate_tile_components_against_support_inplace(
        residual, parent, parent_crop=(0, 7, 0, 7),
        accepted_total_mm=accepted, retain_rejected_components=True,
    )
    before = accepted.copy()
    result = assembly.gate_tile_components_against_support_inplace(
        residual, selected, parent_crop=(0, 7, 0, 7),
        accepted_total_mm=accepted, retain_rejected_components=False,
    )
    assert result['accepted_components'] == 0
    assert result['rejected_components'] == 1
    assert np.array_equal(accepted, before)
    assert int(accepted.sum()) == 4
    assert not residual.any()
    # This fixture would be rescued by rejected raw evidence; supplying only the
    # selected bridge union therefore matters independently of gate semantics.
    assert np.any(raw_rejected_sam & original)


def test_selected_bridge_hit_preserves_eight_connected_diagonal_component():
    tile = np.zeros((1, 5, 5), dtype=np.uint8)
    tile[0, 1, 1] = tile[0, 2, 2] = tile[0, 3, 3] = 1
    support = np.zeros_like(tile)
    support[0, 1, 1] = 1
    accepted = np.zeros_like(tile)
    result = assembly.gate_tile_components_against_support_inplace(
        tile, support, parent_crop=(0, 5, 0, 5),
        accepted_total_mm=accepted, retain_rejected_components=False,
    )
    assert result['accepted_components'] == 1
    assert result['accepted_voxels'] == 3
    assert int(accepted.sum()) == 3

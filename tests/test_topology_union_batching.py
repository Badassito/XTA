"""Tiny boundary union batches retain complete topology and measured timing."""
from __future__ import annotations

from collections import deque
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import runtime, topology


def _coordinate_graph_labels(mask, *, wrap=False):
    """Independent voxel graph with the native mirrored temporal seam."""
    depth, height, width = mask.shape
    labels = np.zeros(mask.shape, np.uint32)
    count = 0
    for z, y, x in np.ndindex(mask.shape):
        if not mask[z, y, x] or labels[z, y, x]:
            continue
        count += 1
        labels[z, y, x] = count
        pending = deque([(z, y, x)])
        while pending:
            az, ay, ax = pending.popleft()
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if not (dz or dy or dx):
                            continue
                        bz, by, bx = az + dz, ay + dy, ax + dx
                        if wrap and (bz < 0 or bz >= depth):
                            bz %= depth
                            bx = width - 1 - bx
                        if not (0 <= bz < depth and 0 <= by < height and 0 <= bx < width):
                            continue
                        if mask[bz, by, bx] and not labels[bz, by, bx]:
                            labels[bz, by, bx] = count
                            pending.append((bz, by, bx))
    return labels, count


def _first_occurrence_ids(labels):
    mapping = {0: 0}
    result = np.zeros(labels.shape, np.uint32)
    for index, value in enumerate(labels.flat):
        value = int(value)
        if value not in mapping:
            mapping[value] = len(mapping)
        result.flat[index] = mapping[value]
    return result


@pytest.mark.parametrize('cap', [1, 2, 3, 1_048_576])
@pytest.mark.parametrize('sparse', [False, True])
@pytest.mark.parametrize('case', ['two_columns', 'mirrored_seam_only'])
def test_boundary_chunking_matches_independent_complete_voxel_graph(tmp_path, cap, sparse, case):
    source = np.zeros((12, 16, 16), np.uint8)
    wrap = case == 'mirrored_seam_only'
    if wrap:
        source[0, 5, 13] = source[-1, 5, 2] = 1
    else:
        source[:, 2:4, 2:4] = 1
        source[:, 10:12, 10:12] = 1
    expected, expected_count = _coordinate_graph_labels(source, wrap=wrap)
    assert expected_count == (1 if wrap else 2)
    original = source.copy()
    stats = {}
    store, paths = None, []
    try:
        with mock.patch.dict(os.environ, {
            'YOLO_TTA_TOPOLOGY_SLAB_SLICES': '4',
            'YOLO_TTA_TOPOLOGY_SLAB_WORKERS': '2',
            'YOLO_TTA_TOPOLOGY_UNION_BATCH_CODES': str(cap),
        }), mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False):
            store, count, paths = topology.label_foreground_volume_streaming(source,
                tmp_path/'labels', prefer_memory=True, reserve_bytes=0, workers=2,
                compact_relabel=False, sparse_local_labels=sparse, wrap_axis=wrap,
                component_stats_out=stats)
        luts = stats['slice_local_luts']
        actual = np.stack([luts.lut_for(z)[np.asarray(store[z])] for z in range(len(source))])
        assert count == expected_count
        np.testing.assert_array_equal(_first_occurrence_ids(actual), _first_occurrence_ids(expected))
        np.testing.assert_array_equal(actual != 0, source != 0)
        np.testing.assert_array_equal(source, original)
        assert stats['topology_slab_count'] == 3
        timing = stats['topology_phase_seconds']['boundary_merge']
        assert np.isfinite(timing) and timing > 0
        assert int(stats['root_areas'][stats['unique_roots']].sum()) == int(source.sum())
    finally:
        if isinstance(store, np.memmap):
            runtime.close_memmap_array_without_flush(store, unlink_path=paths[0] if paths else None)
        store = None
        for path in paths:
            runtime.wait_for_retired_memmap_unlinks(path=Path(path))
            Path(path).unlink(missing_ok=True)

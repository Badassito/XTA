"""Endpoint seeds follow the labeler's 26-connected cross-slice adjacency."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from XTA import interpolation, topology


def _seeds(labels, *, wrap=False, cache=True, slice_luts=None):
    component_cache = (
        interpolation.SliceComponentTableCache(labels, slice_luts=slice_luts)
        if cache else None
    )
    try:
        seeds, count = topology.build_slice_endpoint_seeds_from_label_volume(
            labels, workers=1, wrap_axis=bool(wrap), component_cache=component_cache,
        )
    finally:
        if component_cache is not None:
            component_cache.clear()
    assert count == len(seeds)
    return seeds


def _simple(seeds):
    return [
        (int(seed.label), tuple(int(v) for v in seed.point), int(seed.direction_sign))
        for seed in seeds
    ]


def _z_signs(seeds, label=1):
    return {
        (int(seed.point[0]), int(seed.direction_sign))
        for seed in seeds if int(seed.label) == int(label)
    }


def _oblique_line():
    labels = np.zeros((20, 32, 32), dtype=np.uint16)
    for z in range(16):
        labels[z, 2 + z, 2 + z] = 1
    return labels


def _wrap_seam():
    # Slice 11 continues into slice 0 at u -> width-1-u; x=5 mirrors to x=18,
    # one pixel diagonal from x=19.
    labels = np.zeros((12, 24, 24), dtype=np.uint16)
    labels[6:12, 10, 5] = 1
    labels[0:4, 10, 19] = 1
    return labels


def _two_pixel_offset():
    labels = np.zeros((3, 16, 16), dtype=np.uint16)
    labels[0, 5, 5] = 1
    labels[1, 5, 7] = 1
    labels[2, 6, 8] = 1
    return labels


def _trunk(shape, offshoot=None):
    labels = np.zeros(shape, dtype=np.uint16)
    labels[0:10, 10:22, 10:22] = 1
    if offshoot == 'face':
        labels[10:13, 10, 10] = 1
    elif offshoot == 'diagonal':
        for k in range(3):
            labels[10 + k, 9 - k, 9 - k] = 1
    return labels


def _trunk_with_diagonal_offshoot_beside_vessel():
    labels = np.zeros((32, 48, 48), dtype=np.uint16)
    labels[0:10, 10:22, 10:22] = 1
    for k in range(4):
        labels[10 + k, 9 - k, 9 - k] = 1
    labels[:, 10:22, 23:31] = 2
    return labels


def _offshoot_parity(offshoot):
    labels = _trunk((32, 48, 48), offshoot)
    labels[16:32, 10:22, 10:22] = 2
    return labels


def _candidate_total(labels, *, max_slice_distance, search_angle_deg, label=1):
    cache = interpolation.SliceComponentTableCache(labels, slice_luts=None)
    try:
        seeds, _count = topology.build_slice_endpoint_seeds_from_label_volume(
            labels, workers=1, component_cache=cache,
        )
        return sum(
            len(interpolation._find_slice_projection_candidates(
                labels, seed, max_slice_distance=int(max_slice_distance),
                search_angle_deg=float(search_angle_deg), max_candidates=1,
                component_cache=cache,
            ))
            for seed in seeds if int(seed.label) == int(label)
        )
    finally:
        cache.clear()


class EndpointConnectivityTests(unittest.TestCase):
    def test_oblique_one_pixel_line_has_only_its_two_ends(self):
        self.assertEqual(
            _simple(_seeds(_oblique_line())),
            [(1, (0, 2, 2), -1), (1, (15, 17, 17), 1)],
        )

    def test_sparse_local_label_store_matches_the_labeler(self):
        mask = (_oblique_line() > 0).astype(np.uint8)
        stats: dict[str, object] = {}
        with (
            tempfile.TemporaryDirectory() as tmp,
            mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False),
        ):
            labels, count, _paths = topology.label_foreground_volume_streaming(
                mask, Path(tmp) / 'lbl', prefer_memory=True, reserve_bytes=0, workers=1,
                compact_relabel=False, component_stats_out=stats, sparse_local_labels=True,
            )
            self.assertIsInstance(labels, topology.SparseSliceLabelStore)
            self.assertEqual(count, 1)
            seeds = _seeds(labels, slice_luts=stats['slice_local_luts'])
        self.assertEqual(_z_signs(seeds), {(0, -1), (15, 1)})
        self.assertEqual(len(seeds), 2)

    def test_wrap_seam_continuation_is_diagonal_after_mirroring(self):
        self.assertEqual(_z_signs(_seeds(_wrap_seam(), wrap=True)), {(3, 1), (6, -1)})

    def test_two_pixel_offset_is_not_continuation(self):
        self.assertEqual(
            [(int(s.point[0]), int(s.direction_sign)) for s in _seeds(_two_pixel_offset())],
            [(0, -1), (0, 1), (1, -1), (2, 1)],
        )

    def test_interior_slices_of_a_thin_object_find_no_lateral_candidates(self):
        labels = _oblique_line()
        labels[:, 12:15, 20:23] = 2
        self.assertEqual(
            _candidate_total(labels, max_slice_distance=8, search_angle_deg=45.0), 0,
        )

    def test_trunk_with_diagonal_offshoot_is_not_seeded_at_its_end(self):
        # Accepted trade-off: the offshoot tip, not the trunk end, is the endpoint,
        # so the trunk no longer bridges to the parallel vessel at the defaults.
        labels = _trunk_with_diagonal_offshoot_beside_vessel()
        self.assertEqual(_z_signs(_seeds(labels)), {(0, -1), (13, 1)})
        self.assertEqual(
            _candidate_total(labels, max_slice_distance=8, search_angle_deg=15.0), 0,
        )

    def test_diagonal_and_face_connected_offshoots_hide_the_trunk_end_alike(self):
        for offshoot, expected in (
            (None, {(0, -1), (9, 1)}),
            ('face', {(0, -1), (12, 1)}),
            ('diagonal', {(0, -1), (12, 1)}),
        ):
            with self.subTest(offshoot=offshoot):
                self.assertEqual(_z_signs(_seeds(_offshoot_parity(offshoot))), expected)

    def test_uncached_scan_matches_the_cached_scan(self):
        fixtures = (
            ('oblique', _oblique_line(), False),
            ('wrap', _wrap_seam(), True),
            ('offset', _two_pixel_offset(), False),
            ('trunk_vessel', _trunk_with_diagonal_offshoot_beside_vessel(), False),
            ('parity_none', _offshoot_parity(None), False),
            ('parity_face', _offshoot_parity('face'), False),
            ('parity_diagonal', _offshoot_parity('diagonal'), False),
        )
        for name, labels, wrap in fixtures:
            with self.subTest(fixture=name):
                self.assertEqual(
                    _simple(_seeds(labels, wrap=wrap, cache=False)),
                    _simple(_seeds(labels, wrap=wrap, cache=True)),
                )


if __name__ == '__main__':
    unittest.main()

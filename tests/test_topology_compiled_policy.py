"""Compiled topology remains authoritative, including after partial kernel failure."""

from __future__ import annotations

from unittest import mock

import numpy as np
import pytest

from XTA import topology
from tests.reference_backends.topology import union_pair_codes_python


def _new_uf(count: int) -> topology._UnionFind:
    uf = topology._UnionFind()
    uf.new_ids(count)
    uf.touches_boundary[[1, count // 2, count]] = True
    return uf


def _canonical_result(uf: topology._UnionFind, count: int) -> np.ndarray:
    roots = uf.root_map().astype(np.int64, copy=False)
    representative = np.full(uf.parent.size, count + 1, dtype=np.int64)
    np.minimum.at(representative, roots, np.arange(count + 1, dtype=np.int64))
    return np.column_stack((representative[roots], uf.touches_boundary[roots]))


def test_compiled_bulk_union_matches_scalar_oracle_with_boundary_flags() -> None:
    rng = np.random.default_rng(9052)
    count = 513
    a_ids = rng.integers(1, count + 1, size=4096, dtype=np.int64)
    b_ids = rng.integers(1, count + 1, size=4096, dtype=np.int64)
    codes = (a_ids.astype(np.uint64) << np.uint64(32)) | b_ids.astype(np.uint64)
    reference, compiled = _new_uf(count), _new_uf(count)

    union_pair_codes_python(reference, a_ids, b_ids)
    compiled.union_pair_codes(codes)

    np.testing.assert_array_equal(
        _canonical_result(compiled, count), _canonical_result(reference, count),
    )


def test_partial_compiled_failure_aborts_without_scalar_replay() -> None:
    uf = _new_uf(4)
    codes = np.array([(1 << 32) | 2, (3 << 32) | 4], dtype=np.uint64)

    def fail_after_first_link(parent, _rank, _touches, _a_ids, _b_ids):
        parent[2] = 1
        raise ValueError('injected mid-batch failure')

    with mock.patch.object(topology, '_numba_union_find_batch_kernel', side_effect=fail_after_first_link) as kernel:
        with pytest.raises(RuntimeError, match='batch state is partial'):
            uf.union_pair_codes(codes)

    kernel.assert_called_once()
    assert int(uf.parent[2]) == 1
    assert int(uf.parent[4]) == 4


def test_missing_compiled_bulk_kernel_does_not_use_scalar_reference() -> None:
    uf = _new_uf(2)
    with mock.patch.object(topology, '_numba_union_find_batch_kernel', None):
        with pytest.raises(RuntimeError, match='kernel is unavailable'):
            uf.union_pair_codes(np.array([(1 << 32) | 2], dtype=np.uint64))
    np.testing.assert_array_equal(uf.root_map(), np.array([0, 1, 2], dtype=np.uint32))


def test_compact_relabel_matches_local_id_lut_without_reference_rewrite(tmp_path) -> None:
    mask = np.zeros((3, 8, 9), dtype=np.uint8)
    mask[0, 2:4, 2:4] = 1
    mask[1, 2:4, 2:4] = 1
    mask[2, 5:7, 6:8] = 1
    local_stats: dict[str, object] = {}
    with mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False):
        local, local_count, _ = topology.label_foreground_volume_streaming(
            mask, tmp_path / 'local', reserve_bytes=0, workers=1,
            compact_relabel=False, component_stats_out=local_stats,
        )
        compact, compact_count, _ = topology.label_foreground_volume_streaming(
            mask, tmp_path / 'compact', reserve_bytes=0, workers=1,
            compact_relabel=True,
        )
    luts = local_stats['slice_local_luts']
    canonical = np.stack([
        luts.lut_for(z)[np.asarray(local[z])]
        for z in range(mask.shape[0])
    ])
    assert local_count == compact_count == 2
    np.testing.assert_array_equal(np.asarray(compact), canonical)


def test_compact_relabel_kernel_failure_does_not_replay_python(tmp_path) -> None:
    mask = np.zeros((2, 5, 5), dtype=np.uint8)
    mask[0, 2, 2] = 1
    mask[1, 2, 2] = 1

    def fail_after_write(labels, *_args):
        labels[0, 2, 2] = 999
        raise ValueError('injected relabel failure')

    with mock.patch.object(topology, 'gpu_slice_labeling_enabled', return_value=False), \
            mock.patch.object(topology, '_numba_compact_relabel_kernel', side_effect=fail_after_write) as kernel:
        with pytest.raises(RuntimeError, match='Compiled topology compact relabel failed'):
            topology.label_foreground_volume_streaming(
                mask, tmp_path / 'failure', reserve_bytes=0, workers=1,
                compact_relabel=True,
            )
    kernel.assert_called_once()

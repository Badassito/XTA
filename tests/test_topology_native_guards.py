"""Exercise both merge count guards by changing keys between the two passes."""

import numpy as np
import pytest

from XTA.native_cpu import load_native_cpu


@pytest.mark.parametrize(
    ("initial", "replacement", "count"),
    [
        ([1, 3], [2, 4], 2),
        ([2, 4], [1, 3], 4),
    ],
    ids=("union-grows", "union-shrinks"),
)
def test_adjacency_merge_rejects_changed_union_count(monkeypatch, initial, replacement, count):
    merge = load_native_cpu().topology_adjacency_merge
    a = np.array([1, 3], dtype=np.uint64)
    b = np.array(initial, dtype=np.uint64)
    sentinel = np.uint64(0xAABBCCDDEEFF1122)
    storage = np.full(a.size + b.size + 1, sentinel, dtype=np.uint64)
    allocations = []

    def allocate(shape, dtype):
        assert shape == (count,)
        assert np.dtype(dtype) == storage.dtype
        # The GIL is held during this allocation, after the first pass and
        # before the second. Extra physical storage protects the test process
        # and lets the guard check observe any write beyond the logical view.
        b[:] = replacement
        allocations.append(shape)
        return storage[:count]

    with monkeypatch.context() as patch:
        patch.setattr(np, "empty", allocate)
        with pytest.raises(ValueError, match="adjacency key count changed"):
            merge(a, b)

    assert allocations == [(count,)]
    assert np.all(storage[count:] == sentinel)


def _pair_codes(first, second):
    return np.array([(int(a) << 32) | int(b) for a, b in zip(first, second)], dtype=np.uint64)


def _forest(size):
    return np.arange(size, dtype=np.int64), np.zeros(size, dtype=np.int32), np.zeros(size, dtype=bool)


@pytest.mark.parametrize("base", [0, 2**32 - 19, (-1, -3)])
def test_packed_union_preserves_exact_forest_with_signed_strides(base):
    from types import SimpleNamespace

    from tests.reference_backends.topology import union_pair_codes_python

    a_ids = np.array([5, 4, 7, 5, 8, 9, 6, 7, 9], dtype=np.int64)
    b_ids = np.array([4, 6, 8, 7, 9, 6, 8, 10, 11], dtype=np.int64)
    left_base, right_base = base if isinstance(base, tuple) else (base, base)
    codes = _pair_codes(a_ids + left_base, b_ids + right_base)
    carrier = np.empty(codes.size * 2, dtype=np.uint64)
    reversed_codes = carrier[::2][::-1]
    reversed_codes[:] = codes
    reversed_codes.flags.writeable = False
    initial = _forest(16)
    initial[2][[4, 8, 11]] = True
    expected = tuple(value.copy() for value in initial)
    reference = SimpleNamespace(parent=expected[0], rank=expected[1], touches_boundary=expected[2])
    union_pair_codes_python(reference, a_ids, b_ids)
    merge = load_native_cpu().topology_union_find_batch
    if base == 0:
        merge(*initial, reversed_codes)
    else:
        merge(*initial, reversed_codes, base)
    for actual, wanted in zip(initial, expected):
        np.testing.assert_array_equal(actual, wanted)


def test_packed_union_snapshots_codes_that_alias_a_later_rank_write():
    from types import SimpleNamespace

    from tests.reference_backends.topology import union_pair_codes_python

    parent, rank, touches = _forest(8)
    codes = rank[:4].view(np.uint64)
    codes[:] = _pair_codes([2, 4], [4, 3])
    rank[4] = rank[2]
    reference = SimpleNamespace(parent=parent.copy(), rank=rank.copy(), touches_boundary=touches.copy())
    union_pair_codes_python(reference, np.array([2, 4]), np.array([4, 3]))

    load_native_cpu().topology_union_find_batch(parent, rank, touches, codes)
    np.testing.assert_array_equal(parent, reference.parent)
    np.testing.assert_array_equal(rank, reference.rank)
    np.testing.assert_array_equal(touches, reference.touches_boundary)
    assert parent[3] == 2


def test_packed_union_keeps_first_find_halving_and_prior_links_on_second_member_error():
    parent, rank, touches = _forest(10)
    parent[1:3] = [2, 3]
    codes = _pair_codes([104, 101], [105, 99])
    with pytest.raises(ValueError, match="invalid parent or member"):
        load_native_cpu().topology_union_find_batch(parent, rank, touches, codes, 100)
    assert parent[5] == 4
    assert rank[4] == 1
    assert parent[1] == 3


def test_shard_validation_reports_high_before_low_and_preserves_whole_batch():
    from XTA.topology import _UnionFind

    uf = _UnionFind()
    uf.new_ids(6)
    initial_parent = uf.parent.copy()
    codes = _pair_codes([1, 2, 4], [2, 0, 3])
    with pytest.raises(ValueError, match=r"cross-shard pair 0->1 high ids must be in 1..3"):
        uf.union_pair_codes(codes, gid_base=(0, -3), member_counts=(3, 3),
                            context="cross-shard pair 0->1")
    np.testing.assert_array_equal(uf.parent, initial_parent)
    assert not uf.rank.any()

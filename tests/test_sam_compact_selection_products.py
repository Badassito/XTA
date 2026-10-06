"""Bounded compact filter reuse preserves exact authenticated selection products."""
from concurrent.futures import ThreadPoolExecutor
import threading

import numpy as np
import pytest

from XTA import sam_filtering
from tests.test_sam_branch_support import bundle_fixture


@pytest.mark.parametrize('enabled,threshold', [(False, 3.), (True, 0.), (True, 2.), (True, 3.)])
def test_compact_products_preserve_exact_mask_diagnostics_and_immutability(tmp_path, enabled, threshold):
    bundle, _ = bundle_fixture(tmp_path, grown=True)
    spec = sam_filtering.build_mask_filter(bundle, enabled=enabled, min_radius=threshold)
    with bundle.reader(max_cache_bytes=1024**2) as reader:
        snapshot = reader.filter_snapshot(spec)
        for frame in range(5):
            expected, diagnostics = sam_filtering.measure_effective_raw_mask(bundle, 'f', frame, spec)
            actual, measured = reader.measure_effective_raw_mask('f', frame, snapshot)
            np.testing.assert_array_equal(actual, expected)
            assert measured == diagnostics
            with pytest.raises(ValueError):
                actual.setflags(write=True)
            measured['components'].clear()
            assert reader.measure_effective_raw_mask('f', frame, snapshot)[1] == diagnostics
        assert reader.stats['filter_computations'] == 5
        assert reader.stats['compact_filter_expansions'] == 10
        assert reader.stats['peak_cache_bytes'] <= reader.max_cache_bytes
        compact = [value for key, (value, _charge) in reader._compact_cache.items()]
        assert compact and all(isinstance(packet, bytes) and isinstance(diagnostic, bytes)
            for packet, diagnostic in compact)
        assert all(len(packet) == 16+(13*15+7)//8 for packet, _diagnostic in compact)


def test_completed_intrinsic_lane_shares_filter_work_with_ordered_selection(tmp_path):
    bundle, _ = bundle_fixture(tmp_path, grown=True)
    with bundle.reader(max_cache_bytes=1024**2) as parent:
        snapshot = parent.filter_snapshot(sam_filtering.build_mask_filter(bundle, min_radius=2.))
        with parent.fork(max_cache_bytes=1024**2) as child:
            borrowed = child.borrowed_filter_snapshot(snapshot)
            expected = child.measure_effective_raw_mask('f', 2, borrowed)
            assert child.stats['filter_computations'] == 1
            assert child.stats['compact_filter_parent_exports'] == 1
        actual = parent.measure_effective_raw_mask('f', 2, snapshot)
        np.testing.assert_array_equal(actual[0], expected[0])
        assert actual[1] == expected[1]
        assert parent.stats['filter_computations'] == 0
        assert parent.stats['peak_cache_bytes'] <= parent.max_cache_bytes
        with parent.fork(max_cache_bytes=1024**2) as another:
            np.testing.assert_array_equal(another.effective_raw_mask('f', 2,
                another.borrowed_filter_snapshot(snapshot)), expected[0])
            assert another.stats['compact_filter_parent_hits'] == 1
            assert another.stats['filter_computations'] == 0
    assert parent.stats['cache_bytes'] == 0 and parent.stats['integrity_checks'] == 2


def test_private_lane_filters_remain_concurrent_and_parent_close_is_refused(tmp_path, monkeypatch):
    bundle, _ = bundle_fixture(tmp_path, grown=True)
    barrier = threading.Barrier(2)
    original = sam_filtering.measure_effective_raw_mask
    def filtered(*args, **kwargs):
        barrier.wait(timeout=10)
        return original(*args, **kwargs)
    monkeypatch.setattr(sam_filtering, 'measure_effective_raw_mask', filtered)
    with bundle.reader(max_cache_bytes=1024**2) as parent:
        snapshot = parent.filter_snapshot(sam_filtering.build_mask_filter(bundle, min_radius=2.))
        def measure(run_id):
            with parent.fork(max_cache_bytes=1024**2) as child:
                with pytest.raises(RuntimeError, match='active reader lanes'):
                    parent.close()
                return child.measure_effective_raw_mask(run_id, 2, child.borrowed_filter_snapshot(snapshot))
        with ThreadPoolExecutor(max_workers=2) as pool:
            expected = dict(zip(('f', 'r'), pool.map(measure, ('f', 'r'))))
        for run_id, value in expected.items():
            actual = parent.measure_effective_raw_mask(run_id, 2, snapshot)
            np.testing.assert_array_equal(actual[0], value[0])
            assert actual[1] == value[1]
        assert parent.stats['filter_computations'] == 0


@pytest.mark.parametrize('cache_bytes', [0, 32, 4096])
def test_cache_pressure_keeps_exact_products_and_existing_byte_bound(tmp_path, cache_bytes):
    bundle, _ = bundle_fixture(tmp_path, partial=True)
    spec = sam_filtering.build_mask_filter(bundle, min_radius=2.)
    with bundle.reader(max_cache_bytes=cache_bytes) as parent:
        snapshot = parent.filter_snapshot(spec)
        for run_id in ('f', 'r'):
            for frame in range(5):
                with parent.fork(max_cache_bytes=cache_bytes) as child:
                    value = child.measure_effective_raw_mask(run_id, frame, child.borrowed_filter_snapshot(snapshot))
                expected = sam_filtering.measure_effective_raw_mask(bundle, run_id, frame, spec)
                actual = parent.measure_effective_raw_mask(run_id, frame, snapshot)
                np.testing.assert_array_equal(value[0], expected[0])
                np.testing.assert_array_equal(actual[0], expected[0])
                assert value[1] == actual[1] == expected[1]
                assert parent.stats['peak_cache_bytes'] <= cache_bytes


def test_filter_change_and_late_evidence_corruption_cannot_use_shared_product_as_proof(tmp_path):
    bundle, _ = bundle_fixture(tmp_path, grown=True)
    path = bundle.directory/'masks.bin'
    original = path.read_bytes()
    try:
        with pytest.raises(ValueError, match='changed'):
            with bundle.reader(max_cache_bytes=1024**2) as parent:
                before = parent.filter_snapshot(sam_filtering.build_mask_filter(bundle, min_radius=2.))
                with parent.fork(max_cache_bytes=1024**2) as child:
                    child.effective_raw_mask('f', 2, child.borrowed_filter_snapshot(before))
                after = parent.filter_snapshot(sam_filtering.build_mask_filter(bundle, min_radius=4.))
                assert not parent.effective_raw_mask('f', 2, after).any()
                assert parent.stats['filter_computations'] == 1
                path.write_bytes(bytes([original[0]^1])+original[1:])
                assert parent.effective_raw_mask('f', 2, before).any()
    finally:
        path.write_bytes(original)


def test_compact_shape_and_valid_bit_bounds_are_checked_before_expansion(tmp_path):
    bundle, _ = bundle_fixture(tmp_path)
    with bundle.reader(max_cache_bytes=1024**2) as reader:
        snapshot = reader.filter_snapshot(sam_filtering.build_mask_filter(bundle, enabled=False))
        reader.effective_raw_mask('f', 2, snapshot)
        key, (value, charge) = next(iter(reader._compact_cache.items()))
        packet, diagnostic = value
        reader._compact_cache[key] = (bytes(16)+packet[16:], diagnostic), charge
        with pytest.raises(ValueError, match='authenticated raw shape'):
            reader.effective_raw_mask('f', 2, snapshot)
        reader._compact_cache[key] = (packet[:-1], diagnostic), charge
        with pytest.raises(ValueError, match='malformed packed bounds'):
            reader.effective_raw_mask('f', 2, snapshot)


def test_ordinary_products_borrow_unused_compact_quota_without_evicting_filters(tmp_path):
    bundle, _ = bundle_fixture(tmp_path)
    with bundle.reader(max_cache_bytes=8192) as reader:
        ordinary = bytes(6000)  # Charge exceeds half the allowance, but fits total.
        key = ('ordinary-test', bundle.evidence_fingerprint)
        assert reader._product(key, lambda: ordinary) is ordinary
        assert reader._product(key, lambda: pytest.fail('unused compact quota must remain borrowable')) is ordinary
        assert reader.stats['cache_bytes'] == 6512
        compact_key = ('compact_effective_raw', 'shape-and-filter-identity')
        compact = bytes(2800), b'{}'
        assert reader._remember_product(compact_key, compact)
        assert key not in reader._cache
        assert reader.stats['cache_bytes'] == reader.stats['compact_cache_bytes'] <= 4096
        assert reader._remember_product(('other-ordinary',), bytes(3500))
        assert not reader._remember_product(('oversized-ordinary',), bytes(6000))
        assert compact_key in reader._compact_cache
        assert reader.stats['cache_bytes'] <= reader.max_cache_bytes
        assert reader.stats['peak_cache_bytes'] <= reader.max_cache_bytes
        assert reader.stats['peak_compact_cache_bytes'] <= reader._compact_cache_limit

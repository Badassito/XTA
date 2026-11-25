"""Exact edited-Global support for fractional 3D navigation."""
import numpy as np
import pytest

from XTA import native_review_edited_context as context
from XTA import native_review_overview
from XTA.native_review_overview import NativeOverviewCounts
from XTA.native_review_session import NativeReviewSession, StaleReviewWindow
from tests.test_native_review_edits import VolumeFixture


def _original_counter(monkeypatch, store, cutoff):
    counts = NativeOverviewCounts(store.shape, step=4, bounds_zyx=((5, 10), (5, 23), (8, 25)))
    for block in store.iter_blocks(store.global_layer_id, cutoff):
        low = tuple(s.start for s in store.block_slices(block))
        counts.add(low, store.read_block(store.global_layer_id, cutoff, block))
    calls = []

    def copy_original(source, selected):
        assert source is store and selected == cutoff
        calls.append(selected)
        result = NativeOverviewCounts(store.shape, step=counts.step, bounds_zyx=counts.bounds_zyx)
        result.data[:] = counts.data
        return result

    monkeypatch.setattr(context, "original_overview_counts", copy_original)
    return calls


def _write_global(session, cutoff, block, value=True):
    with session.transaction("small Global edit"):
        mask = np.full(session.block_extent(block), value, dtype=bool)
        session.write_block(session.global_layer_id, cutoff, block, mask)


def _assert_matches_full_pool(reader, session, cutoff):
    actual = reader._cached.counts
    expected = NativeOverviewCounts(
        session.shape, step=actual.step, bounds_zyx=actual.bounds_zyx,
    )
    for block in session.iter_blocks(session.global_layer_id, cutoff):
        low, _ = session.block_bounds(block)
        expected.add(low, session.read_block(session.global_layer_id, cutoff, block))
    assert np.array_equal(actual.data, expected.data)
    assert np.array_equal(reader.read(cutoff)[0], expected.scalar_field())


def test_edited_context_reuses_single_journal_steps_and_rebuilds_missed_history(tmp_path, monkeypatch):
    store = VolumeFixture(tmp_path / "source.sqlite")
    store.masks["global"][80][0, 0, 0] = False
    calls = _original_counter(monkeypatch, store, 80)
    with NativeReviewSession(store) as session:
        reader = context.EditedGlobalOverview(session)
        original = reader.read(80)
        assert original[2] == 0
        assert reader.read(80)[0] is original[0]
        block = (1, 1, 1)
        _write_global(session, 80, block, False)
        changed = reader.read(80)
        assert changed[2] == 1
        assert np.any(changed[0] != original[0])
        _assert_matches_full_pool(reader, session, 80)
        assert calls == [80]
        assert session.undo()
        assert np.array_equal(reader.read(80)[0], original[0])
        assert session.redo()
        assert np.array_equal(reader.read(80)[0], changed[0])
        assert len(calls) == 1

        # A contributor-only edit advances the journal and must not derive
        # a replacement scalar field or remesh the unchanged Global.
        cached = reader.read(80)
        with session.transaction("edit only a derived contributor"):
            session.write_block("derived", 80, block, np.ones(session.block_extent(block), dtype=bool))
        newer = reader.read(80)
        assert newer[2] == cached[2] + 1
        assert newer[0] is cached[0] and newer[1] is cached[1]
        assert session.undo()
        assert reader.read(80)[0] is cached[0]

        # Observe the new branch only after undo and a different action.
        assert session.undo()
        _write_global(session, 80, (1, 2, 2))
        _assert_matches_full_pool_after_change(reader, session, 80)
        assert len(calls) == 2
        before = session.revision
        _write_global(session, 80, (1, 3, 3), False)
        with pytest.raises(StaleReviewWindow):
            reader.read(80, expected_revision=before - 1)


def _assert_matches_full_pool_after_change(reader, session, cutoff):
    reader.read(cutoff)
    _assert_matches_full_pool(reader, session, cutoff)


def test_context_covers_a_painted_edge_when_original_cutoff_empty(tmp_path, monkeypatch):
    store = VolumeFixture(tmp_path / "source.sqlite")
    store.masks["global"][100][:] = False
    _original_counter(monkeypatch, store, 100)
    with NativeReviewSession(store) as session:
        reader = context.EditedGlobalOverview(session)
        assert not reader.read(100)[0].any()
        block = (3, 3, 4)
        _write_global(session, 100, block)
        scalar, _, _ = reader.read(100)
        assert scalar.any()
        actual = reader._cached.counts
        # X's last native block and navigation cell are both one voxel wide.
        assert actual.bounds_zyx == ((15, 19), (21, 28), (32, 33))
        assert actual.data.sum() == 28
        _assert_matches_full_pool(reader, session, 100)
        _write_global(session, 100, block, False)
        assert not reader.read(100)[0].any()
        assert session.undo()
        assert np.array_equal(reader.read(100)[0], scalar)


def test_outside_paint_can_require_exact_coarsening_to_a_bounded_grid(tmp_path, monkeypatch):
    store = VolumeFixture(tmp_path / "source.sqlite")
    store.masks["global"][80][0, 0, 0] = False
    # Force an adaptive step increase without allocating a large test volume.
    monkeypatch.setattr(native_review_overview, "MAX_SURFACE_SAMPLES", 80)
    _original_counter(monkeypatch, store, 80)
    with NativeReviewSession(store) as session:
        reader = context.EditedGlobalOverview(session)
        reader.read(80)
        count_before = reader._cached.counts.data.sum()
        block = (3, 3, 4)
        _write_global(session, 80, block)
        _assert_matches_full_pool_after_change(reader, session, 80)
        bounded = reader._cached.counts
        assert bounded.step > 4
        assert bounded.data.size <= 80
        assert bounded.data.sum() == count_before + 28


def test_local_scalar_patch_matches_global_conditioner_at_cropped_and_partial_faces():
    rng = np.random.default_rng(187)
    native_mask = rng.random((29, 41, 47)) > 0.77
    counts = NativeOverviewCounts(native_mask.shape, step=3, bounds_zyx=((3, 29), (6, 41), (3, 47)))
    counts.add((0, 0, 0), native_mask)
    original = counts.scalar_field()
    first, stop = (27, 35, 41), (29, 41, 47)
    erased = native_mask[tuple(slice(lo, hi) for lo, hi in zip(first, stop))]
    counts.add(first, erased, subtract=True)
    local = context._scalar_patch(counts, original, tuple(zip(first, stop)))
    assert np.array_equal(local, counts.scalar_field())
    assert not np.array_equal(local, original)


def test_recent_cutoffs_share_a_byte_bounded_context_cache(tmp_path, monkeypatch):
    store = VolumeFixture(tmp_path / "source.sqlite")
    calls = []

    def original(source, cutoff):
        calls.append(cutoff)
        counts = NativeOverviewCounts(source.shape, step=4)
        for block in source.iter_blocks(source.global_layer_id, cutoff):
            low = tuple(s.start for s in source.block_slices(block))
            counts.add(low, source.read_block(source.global_layer_id, cutoff, block))
        return counts

    monkeypatch.setattr(context, "original_overview_counts", original)
    with NativeReviewSession(store) as session:
        reader = context.EditedGlobalOverview(session, cache_bytes=2500)
        first = reader.read(80)
        second = reader.read(90)
        assert reader.read(80)[0] is first[0]
        assert calls == [80, 90]
        reader.read(100)
        assert reader.read(80)[0] is first[0]
        assert reader.read(90)[0] is not second[0]
        assert calls == [80, 90, 100, 90]
        assert sum(field.nbytes for field in reader._fields.values()) <= 2500

        uncached = context.EditedGlobalOverview(session, cache_bytes=0)
        assert uncached.read(80)[0].any()
        assert not uncached._fields and uncached._cached is None

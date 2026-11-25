"""Localized Update Both must preserve the cold solver's exact results."""
import numpy as np

from XTA.native_review_edits import NativeReviewEdits
from XTA.native_review_incremental import try_local_update_both
from XTA.native_review_session import NativeReviewSession
from tests.test_native_review_edits import VolumeFixture


def _store(tmp_path):
    base = VolumeFixture(tmp_path / "native.sqlite")
    for masks in base.masks.values():
        for mask in masks.values():
            mask.fill(False)
    for cutoff in base.cutoffs:
        base.masks["trunk"][cutoff][6:8, 12:14, 2:31] = True
        base.masks["trunk"][cutoff][17, 26, 32] = True
        base.masks["global"][cutoff][:] = base.masks["trunk"][cutoff]
    return base


def _cold(session):
    return NativeReviewEdits(session)._update(
        "Update Both", rebuild=True, clip=True, expected_revision=None, progress=None,
    )


def _compare(first, second):
    assert first.global_stale == second.global_stale
    for layer in first.layers:
        for cutoff in first.cutoffs:
            np.testing.assert_array_equal(
                first.read_region(layer, cutoff, (0, 0, 0), first.shape),
                second.read_region(layer, cutoff, (0, 0, 0), first.shape),
            )


def test_local_cleanup_same_masks_and_history_as_full_update(tmp_path):
    base = _store(tmp_path)
    with (NativeReviewSession(base, tmp_path / "local.sqlite") as local,
          NativeReviewSession(base, tmp_path / "full.sqlite") as full):
        for session in (local, full):
            _cold(session)
            # Connected deletion, new attached support, and a new island.
            # Existing hidden contributors count without becoming visible.
            with session.transaction("localized user changes"):
                origin = (6, 11, 17)
                before = session.read_region("trunk", 90, origin, (7, 14, 19))
                after = before.copy()
                after[0, 1, 0] = False
                after[0, 0, 1] = True
                assert session.apply_region_diff("trunk", 90, origin, before, after)
                origin = (6, 8, 18)
                before = session.read_region("hidden", 90, origin, (7, 9, 19))
                assert session.apply_region_diff("hidden", 90, origin, before, np.ones_like(before))
                # Global-only changes must also be undone by Update Both.
                origin = (7, 12, 19)
                before = session.read_region("global", 100, origin, (8, 13, 20))
                assert session.apply_region_diff("global", 100, origin, before, np.zeros_like(before))
        result = NativeReviewEdits(local).update_both()
        assert result["update_source"] == "localized"
        assert result["global_voxels"] == _cold(full)["global_voxels"]
        _compare(local, full)
        assert local.undo() and full.undo()
        _compare(local, full)
        assert local.redo() and full.redo()
        _compare(local, full)
        assert NativeReviewEdits(local).update_both()["update_source"] == "settled"


def test_split_falls_back_and_rolls_back_an_earlier_local_cutoff(tmp_path):
    base = _store(tmp_path)
    with NativeReviewSession(base, tmp_path / "session.sqlite") as session:
        _cold(session)
        with session.transaction("two edits"):
            block = (1, 1, 2)
            mask = session.read_block("trunk", 80, block)
            mask[1, 5, 1] = False
            session.write_block("trunk", 80, block, mask)
            mask = session.read_block("trunk", 90, block)
            mask[:, :, 1] = False
            session.write_block("trunk", 90, block, mask)
            session.set_global_stale(True)
        revision = session.revision
        before = session.read_block("global", 80, block)
        assert try_local_update_both(session) is None
        np.testing.assert_array_equal(session.read_block("global", 80, block), before)
        assert session.global_stale and session.revision == revision
        result = NativeReviewEdits(session).update_both()
        assert result["update_source"] == "settled_union"
        assert result["global_voxels"][90] == 60  # The x=2..16 component wins.
        assert session.revision == revision + 1


def test_exterior_tie_uses_full_raster_order_and_empty_state_updates(tmp_path):
    base = _store(tmp_path)
    for cutoff in base.cutoffs:
        source = base.masks["trunk"][cutoff]
        source.fill(False)
        source[13, 22, 1:3] = True
        base.masks["global"][cutoff][:] = source
    with NativeReviewSession(base, tmp_path / "session.sqlite") as session:
        _cold(session)
        with session.transaction("paint a tied island"):
            block = (0, 0, 3)
            mask = session.read_block("trunk", 90, block)
            mask[0, 0, :2] = True
            session.write_block("trunk", 90, block, mask)
        assert try_local_update_both(session) is None
        assert NativeReviewEdits(session).update_both()["global_voxels"][90] == 2
        assert session.read_block("global", 90, block)[0, 0, 0]
        with session.transaction("erase settled island"):
            mask = session.read_block("trunk", 90, block)
            mask.fill(False)
            session.write_block("trunk", 90, block, mask)
        result = NativeReviewEdits(session).update_both()
        assert result["update_source"] == "localized" and result["global_voxels"][90] == 0


def test_wide_fallback_reads_unchanged_union_from_global_only(tmp_path, monkeypatch):
    from XTA import native_review_incremental

    base = _store(tmp_path)
    with (NativeReviewSession(base, tmp_path / "reuse.sqlite") as reused,
          NativeReviewSession(base, tmp_path / "full.sqlite") as full):
        dirty_left, dirty_right = (1, 1, 0), (1, 1, 3)
        for session in (reused, full):
            _cold(session)
            with session.transaction("two regions far apart"):
                mask = session.read_block("trunk", 90, dirty_left)
                assert mask[1, 5, 4]
                mask[1, 5, 4] = False
                session.write_block("trunk", 90, dirty_left, mask)
                mask = session.read_block("global", 90, dirty_right)
                assert mask[1, 5, 4]
                mask[1, 5, 4] = False
                session.write_block("global", 90, dirty_right, mask)
                session.set_global_stale(True)
        # The production cap has the same outcome for large native views.
        # Force it below the envelope of these distant test changes.
        monkeypatch.setattr(native_review_incremental, "_MAX_REGION_VOXELS", 512)
        read = reused.read_packed_block
        reads = []

        def log_read(layer, cutoff, block):
            reads.append((layer, cutoff, block))
            return read(layer, cutoff, block)

        monkeypatch.setattr(reused, "read_packed_block", log_read)
        result = NativeReviewEdits(reused).update_both()
        assert result["update_source"] == "settled_union"
        assert result["global_voxels"] == _cold(full)["global_voxels"]
        assert {cutoff for _, cutoff, _ in reads} == {90}
        assert {block for layer, _, block in reads if layer != "global"} == {dirty_left, dirty_right}
        assert any(layer == "global" and block not in (dirty_left, dirty_right)
                   for layer, _, block in reads)
        _compare(reused, full)
        assert reused.undo() and full.undo()
        _compare(reused, full)

from concurrent.futures import ThreadPoolExecutor
import gzip
import itertools
import sqlite3
from threading import Event

import numpy as np
import pytest

from XTA.native_review_session import NativeReviewSession, StaleReviewWindow
from XTA.native_review_initial import prepare_initial_update


class ArrayStore:
    shape = (7, 9, 11)
    block_shape = chunk_shape = (3, 4, 5)
    cutoffs = (80, 90, 100)
    geometry = {
        "space": "left-posterior-superior",
        "directions_xyz": ((0.0, 1.5, 0.0), (-2.0, 0.0, 0.0), (0.0, 0.0, 2.5)),
        "origin_xyz": (11.0, -9.0, 5.0),
    }
    layers = {"radial": {"additive": True, "role": "direct"},
              "cartesian": {"additive": True, "role": "direct"},
              "global": {"additive": False, "role": "global"}}
    global_layer_id = "global"

    def __init__(self, path, identity="source-one"):
        self.path = path
        self.identity = identity
        self.path.touch()
        self.masks = {layer: {cutoff: np.zeros(self.shape, bool) for cutoff in self.cutoffs} for layer in self.layers}
        for cutoff in (80, 90):
            self.masks["radial"][cutoff][2:6, 3:8, 4:9] = True
            self.masks["cartesian"][cutoff][3, 5, 5] = True
            self.masks["global"][cutoff] = self.masks["radial"][cutoff].copy()

    def read_block(self, layer, cutoff, block):
        sl = tuple(slice(i * n, min((i + 1) * n, s)) for i, n, s in zip(block, self.chunk_shape, self.shape))
        return self.masks[layer][cutoff][sl]

    def read_packed_block(self, layer, cutoff, block):
        return np.packbits(self.read_block(layer, cutoff, block), axis=-1, bitorder="little")

    def iter_blocks(self, layer, cutoff):
        ranges = [range((s + n - 1) // n) for s, n in zip(self.shape, self.chunk_shape)]
        yield from (block for block in itertools.product(*ranges) if self.read_block(layer, cutoff, block).any())

    def layers_at(self, block, cutoff=None):
        cutoffs = self.cutoffs if cutoff is None else (cutoff,)
        return tuple(layer for layer in self.layers if any(self.read_block(layer, c, block).any() for c in cutoffs))


def test_native_brush_scopes_and_reopening_preserve_unloaded_foreground(tmp_path):
    source = ArrayStore(tmp_path / "study.native")
    with NativeReviewSession(source) as review:
        region = (2, 4, 4), (5, 7, 8)
        visible = review.read_region("radial", 90, *region)
        edited = visible.copy()
        edited[1, 1, :3] = False
        # This higher-cutoff-only paint adds to the lower cutoffs as well.
        blank = review.read_region("radial", 100, (6, 8, 10), (7, 9, 11))
        with review.transaction("paint and erase") as edit:
            assert edit.apply_region_diff("radial", 90, region[0], visible, edited) == 3
            assert edit.apply_region_diff("radial", 100, (6, 8, 10), blank, np.ones_like(blank)) == 1
            assert edit.layers_at((2, 2, 2), 100) == ("radial",)
            assert not edit.read_before_block("radial", 100, (2, 2, 2)).item()
            assert edit.read_block("radial", 100, (2, 2, 2)).item()
        assert review.global_stale
        np.testing.assert_array_equal(review.read_region("radial", 90, *region), edited)
        assert review.read_region("radial", 80, (5, 7, 8), (6, 8, 9)).item()
        assert review.read_region("radial", 100, (6, 8, 10), (7, 9, 11)).item()
        assert review.read_region("radial", 80, (6, 8, 10), (7, 9, 11)).item()
        assert review.read_painted_block("radial", (2, 2, 2)).item()
        global_row = review.read_region("global", 90, (3, 5, 4), (4, 6, 7))
        np.testing.assert_array_equal(global_row.ravel(), [False, True, False])
        # The next requested window neither clears hidden portions nor prevents undo.
        review.read_region("radial", 80, (0, 0, 0), (1, 1, 1))
    with NativeReviewSession(source) as review:
        assert review.undo()
        assert not review.global_stale
        np.testing.assert_array_equal(review.read_region("radial", 90, (0, 0, 0), source.shape), source.masks["radial"][90])
        assert not review.read_painted_block("radial", (2, 2, 2)).item()
        assert not review.undo()
    with NativeReviewSession(source) as review:
        assert review.redo()
        assert review.global_stale
        np.testing.assert_array_equal(review.read_region("radial", 90, *region), edited)
        assert not review.redo()


def test_transaction_rollback_and_noop_keep_redo(tmp_path):
    source = ArrayStore(tmp_path / "study.native")
    with NativeReviewSession(source) as review:
        block = (0, 0, 0)
        prior = review.read_block("global", 90, block)
        with review.transaction("erase") as edit:
            edit.write_block("global", 90, block, np.zeros_like(prior))
        assert review.undo()
        with pytest.raises(RuntimeError):
            with review.transaction("interrupted") as edit:
                edit.write_block("global", 90, block, np.zeros_like(prior))
                assert not edit.read_block("global", 90, block).any()
                raise RuntimeError("interrupted")
        with review.transaction("scratch edit") as edit:
            edit.write_block("global", 90, block, np.zeros_like(prior))
            edit.write_block("global", 90, block, prior)
        assert review.redo()
        assert not review.read_block("global", 90, block).any()
        with review.transaction("different edit") as edit:
            edit.write_block("global", 80, block, np.zeros_like(prior))
        assert review.undo()
        assert review.undo()
        with review.transaction("alternative edit") as edit:
            edit.write_block("global", 100, block, prior)
        assert not review.redo()


def test_edit_journal_rejects_other_source_and_bounds(tmp_path):
    source = ArrayStore(tmp_path / "study.native")
    with NativeReviewSession(source, max_region_bytes=10) as review:
        with pytest.raises(MemoryError):
            review.read_region("radial", 80, (0, 0, 0), source.shape)
        with pytest.raises(ValueError):
            review.read_region("radial", 80, (-1, 0, 0), (1, 1, 1))
        with pytest.raises(RuntimeError):
            review.write_block("global", 80, (0, 0, 0), np.zeros(source.chunk_shape, bool))
        with pytest.raises(RuntimeError, match="changed"):
            with review.transaction("stale crop") as edit:
                edit.apply_region_diff("radial", 90, (3, 5, 5), np.zeros((1, 1, 1), bool), np.ones((1, 1, 1), bool))
    with pytest.raises(ValueError, match="different"):
        NativeReviewSession(ArrayStore(tmp_path / "different.native", "two"), tmp_path / "study.edits.sqlite")
    with pytest.raises(ValueError, match="separate"):
        NativeReviewSession(source, source.path)


def test_native_export_preserves_rotated_geometry_and_empty_mask(tmp_path):
    source = ArrayStore(tmp_path / "study.native")
    with NativeReviewSession(source) as review:
        before = review.read_region("global", 90, (3, 5, 4), (4, 6, 7))
        with review.transaction("mark native deletion") as edit:
            edit.apply_region_diff("global", 90, (3, 5, 4), before, np.zeros_like(before))
            empty = np.zeros((1, 1, 1), dtype=bool)
            edit.apply_region_diff("global", 90, (6, 8, 10), empty, ~empty)
        expected = source.masks["global"][90].copy()
        expected[3, 5, 4:7] = False
        expected[6, 8, 10] = True
        for cutoff, wanted in ((90, expected), (100, np.zeros(source.shape, bool))):
            native = tmp_path / f"cutoff{cutoff}.nrrd"
            assert review.export_global(native, cutoff) == native
            header, data = native.read_bytes().split(b"\n\n", 1)
            assert b"sizes: 11 9 7" in header
            assert b"space directions: (0,1.5,0) (-2,0,0) (0,0,2.5)" in header
            assert b"space origin: (11,-9,5)" in header
            np.testing.assert_array_equal(np.frombuffer(gzip.decompress(data), np.uint8).reshape(source.shape), wanted)
            with pytest.raises(FileExistsError):
                review.export_global(native, cutoff)


def test_open_sessions_never_observe_uncommitted_mask_changes(tmp_path):
    source = ArrayStore(tmp_path / "study.native")
    with NativeReviewSession(source) as writer, NativeReviewSession(source) as reader:
        block = (2, 2, 2)
        fresh = np.ones((1, 1, 1), dtype=bool)
        with writer.transaction("native change") as edit:
            edit.write_block("global", 90, block, fresh)
            assert edit.read_block("global", 90, block).item()
            assert not reader.read_block("global", 90, block).item()
        assert reader.read_block("global", 90, block).item()
        assert reader.undo()
        assert not writer.read_block("global", 90, block).item()


def test_viewport_snapshot_pins_layers_and_rejects_outdated_gesture(tmp_path):
    source = ArrayStore(tmp_path / "study.native")
    block = (2, 2, 2)
    opened, ready_to_write, staged = Event(), Event(), Event()

    def background():
        with NativeReviewSession(source) as writer:
            opened.set()
            assert ready_to_write.wait(5)
            with writer.transaction("other window edit") as edit:
                addition = np.ones((1, 1, 1), dtype=bool)
                edit.write_block("global", 90, block, addition)
                edit.write_block("radial", 90, block, addition)
                staged.set()

    with NativeReviewSession(source) as reader, ThreadPoolExecutor(max_workers=1) as pool:
        result = pool.submit(background)
        assert opened.wait(5)
        with reader.snapshot() as revision:
            assert not reader.layers_at(block, 90)
            ready_to_write.set()
            assert staged.wait(5)
            assert not reader.read_block("global", 90, block).any()
            assert not reader.read_block("radial", 90, block).any()
            assert not reader.layers_at(block, 90)
        result.result(timeout=5)
        assert reader.revision == revision + 1
        with pytest.raises(StaleReviewWindow), reader.transaction("stale point", expected_revision=revision):
            pytest.fail("Stale edit must fail before writes")
        with pytest.raises(StaleReviewWindow):
            reader.undo(expected_revision=revision)
        assert reader.undo(expected_revision=revision + 1)
        assert reader.revision == revision + 2
        assert reader.redo(expected_revision=revision + 2)
        assert reader.revision == revision + 3
        assert reader.layers_at(block, 90) == ("radial", "global")


@pytest.mark.parametrize("width", [16, 17])
def test_packed_global_clip_preserves_history_and_native_row_edges(tmp_path, width):
    class EdgeStore(ArrayStore):
        shape = block_shape = chunk_shape = (7, 9, width)

    source = EdgeStore(tmp_path / "clip.native")
    block = (0, 0, 0)
    original = source.masks["radial"][90]
    original[:] = np.arange(original.size).reshape(original.shape) % 4 == 0
    changed = original.copy()
    changed[:, -1, :] = True
    changed |= np.arange(original.size).reshape(original.shape) % 7 == 0
    keep = np.packbits(original, axis=-1, bitorder="little")
    with NativeReviewSession(source) as review:
        with review.transaction("paint") as edit:
            edit.write_block("radial", 90, block, changed)
        with review.transaction("clip") as edit:
            assert edit.clip_packed_block("radial", 90, block, keep)
            assert not edit.clip_packed_block("radial", 90, block, keep)
            np.testing.assert_array_equal(edit.read_block("radial", 90, block), original)
        assert review.undo()
        np.testing.assert_array_equal(review.read_block("radial", 90, block), changed)
        assert review.redo()
        np.testing.assert_array_equal(review.read_block("radial", 90, block), original)


@pytest.mark.parametrize("width", [16, 17])
def test_packed_blocks_convert_journal_edges_and_read_baseline_after_undo(tmp_path, width):
    class EdgeStore(ArrayStore):
        shape = (7, 9, 32 + width)
        chunk_shape = block_shape = (3, 4, 32)

    source = EdgeStore(tmp_path / "study.native")
    block = (1, 1, 1)
    with NativeReviewSession(source) as review:
        prior = review.read_block("radial", 90, block)
        edited = np.arange(prior.size).reshape(prior.shape) % 5 != 1
        packed = np.packbits(edited, axis=-1, bitorder="little")
        assert edited.shape == (3, 4, width)
        with review.transaction("edge edit") as tx:
            tx.write_block("radial", 90, block, edited)
            np.testing.assert_array_equal(tx.read_packed_block("radial", 90, block), packed)
        np.testing.assert_array_equal(review.read_packed_block("radial", 90, block), packed)
        assert review.undo()
        np.testing.assert_array_equal(review.read_packed_block("radial", 90, block),
                                      np.packbits(prior, axis=-1, bitorder="little"))


def test_prepared_update_is_one_action_repeat_and_redo_survive_reopen(tmp_path):
    source = ArrayStore(tmp_path / "study.xtareview")
    # One disconnected contributor voxel will be clipped by Update Both.
    block = (2, 2, 2)
    source.masks["radial"][80][6, 8, 10] = True
    source.masks["global"][80][6, 8, 10] = True
    computed = tmp_path / "computed.sqlite"
    with NativeReviewSession(source, computed, initial_update_path=False) as session:
        with session.transaction("Update Both") as edit:
            edit.write_block("radial", 80, block, np.zeros((1, 1, 1), bool))
            edit.write_block("global", 80, block, np.zeros((1, 1, 1), bool))
            edit.record_update_both({80: 100, 90: 100, 100: 0})
    prepared = prepare_initial_update(source, computed)
    with NativeReviewSession(source) as session:
        committed = session.try_update_both(expected_revision=0)
        assert committed["update_source"] == "prepared"
        assert committed["global_voxels"] == {80: 100, 90: 100, 100: 0}
        assert session.revision == 1
        for layer in ("radial", "global"):
            assert not session.read_block(layer, 80, block).any()
        assert session.try_update_both()["update_source"] == "settled"
        assert session.revision == 1
        assert session.undo()
        assert session.revision == 2
        assert session.read_block("radial", 80, block).any()
    prepared.unlink()
    with NativeReviewSession(source) as session:
        assert session.try_update_both()["update_source"] == "recorded"
        assert session.revision == 3
        with session.transaction("paint after initial update") as edit:
            pixel = edit.read_block("global", 100, (2, 2, 2))
            assert not pixel.any()
            edit.write_block("global", 100, (2, 2, 2), ~pixel)
        assert session.try_update_both() is None
        assert session.undo()
        assert session.try_update_both()["update_source"] == "settled"
        assert session.redo()
        assert session.try_update_both() is None


def test_prepared_update_cancellation_identity_and_integrity(tmp_path):
    source = ArrayStore(tmp_path / "study.xtareview")
    block = (2, 2, 2)
    source.masks["global"][80][6, 8, 10] = True
    computed = tmp_path / "computed.sqlite"
    with NativeReviewSession(source, computed, initial_update_path=False) as session:
        with session.transaction("Update Both") as edit:
            edit.write_block("global", 80, block, np.zeros((1, 1, 1), bool))
            edit.record_update_both({80: 100, 90: 100, 100: 0})
    prepared = prepare_initial_update(source, computed)

    def cancel_after_sql(phase, cutoff, rows):
        assert phase == "apply_prepared" and cutoff is None
        if rows:
            raise InterruptedError("Cancel")

    with NativeReviewSession(source) as session:
        with pytest.raises(InterruptedError):
            session.try_update_both(progress=cancel_after_sql)
        assert session.revision == 0
        assert session.read_block("global", 80, block).any()
        assert not session.undo()
        with pytest.raises(StaleReviewWindow):
            session.try_update_both(expected_revision=1)

    other = ArrayStore(tmp_path / "other.xtareview", "other-identity")
    with NativeReviewSession(other, initial_update_path=prepared) as session:
        with pytest.raises(ValueError, match="different native review"):
            session.try_update_both()
        assert session.revision == 0
    db = sqlite3.connect(prepared)
    try:
        db.execute("UPDATE blocks SET payload=x'00'")
        db.commit()
    finally:
        db.close()
    with NativeReviewSession(source, initial_update_path=prepared) as session:
        with pytest.raises(ValueError, match="integrity"):
            session.try_update_both()
        assert session.revision == 0


def test_prepared_noop_needs_asserted_receipt_and_has_no_undo(tmp_path):
    source = ArrayStore(tmp_path / "study.xtareview")
    computed = tmp_path / "computed.sqlite"
    counts = {80: 100, 90: 100, 100: 0}
    with NativeReviewSession(source, computed, initial_update_path=False) as session:
        with pytest.raises(ValueError, match="receipt"):
            prepare_initial_update(source, computed, global_voxels=counts)
        with session.transaction("Update Both") as edit:
            edit.record_update_both(counts)
        assert session.revision == 0
        assert not session.undo()
    prepare_initial_update(source, computed, global_voxels=counts)
    with NativeReviewSession(source) as session:
        assert session.try_update_both()["update_source"] == "prepared"
        assert session.revision == 0
        assert not session.undo()
        assert session.try_update_both()["update_source"] == "settled"


def test_settled_ancestor_reads_first_before_value_and_discards_undone_branch(tmp_path):
    source = ArrayStore(tmp_path / "study.xtareview")
    block = (2, 2, 2)
    on = np.ones((1, 1, 1), bool)
    off = ~on
    with NativeReviewSession(source) as session:
        with session.transaction("Update Both") as edit:
            edit.record_update_both({80: 100, 90: 100, 100: 0})
        with session.transaction("paint") as edit:
            edit.write_block("global", 80, block, on)
            edit.write_painted_block("radial", block, on)
        with session.transaction("erase") as edit:
            edit.write_block("global", 80, block, off)
            assert edit.settled_update_ancestor() == (0, {80: 100, 90: 100, 100: 0})
            assert list(edit.iter_changed_blocks_since(0)) == [("global", 80, block)]
            assert not edit.read_historical_block("global", 80, block, 0).item()
            edit.record_update_both({80: 100, 90: 100, 100: 0})
        assert session.undo()
        with session.transaction("alternate branch") as edit:
            edit.write_block("global", 90, block, on)
        with session.transaction("inspect") as edit:
            assert edit.settled_update_ancestor() == (0, {80: 100, 90: 100, 100: 0})
            assert list(edit.iter_changed_blocks_since(0)) == [("global", 80, block), ("global", 90, block)]
            with pytest.raises(ValueError, match="active journal branch"):
                edit.read_historical_block("global", 80, block, 2)
        assert not session.redo()


@pytest.mark.parametrize("mismatch", [None, "changed_payload", "missing_block", "extra_block", "unexpected_before"])
def test_prepared_update_recognizes_legacy_first_action_without_replaying_history(tmp_path, mismatch):
    source = ArrayStore(tmp_path / "study.xtareview")
    block = (2, 2, 2)
    source.masks["global"][80][6, 8, 10] = True
    source.masks["radial"][80][6, 8, 10] = True
    journal = tmp_path / "legacy.sqlite"
    cleared = np.zeros((1, 1, 1), bool)
    counts = {80: 100, 90: 100, 100: 0}
    with NativeReviewSession(source, journal, initial_update_path=False) as session:
        with session.transaction("Update Both") as edit:
            edit.write_block("global", 80, block, cleared)
            edit.write_block("radial", 80, block, cleared)
            edit.record_update_both(counts)
    prepare_initial_update(source, journal)
    with NativeReviewSession(source, journal, initial_update_path=False) as session:
        with session.transaction("later brush") as edit:
            assert edit.apply_region_diff("radial", 100, (6, 8, 10), cleared, ~cleared) == 1
        with session.transaction("redoable brush") as edit:
            assert edit.apply_region_diff("global", 100, (6, 8, 10), cleared, ~cleared) == 1
        assert session.undo()
        revision = session.revision
    db = sqlite3.connect(journal)
    try:
        db.execute("DROP TABLE update_both_states")
        db.execute("PRAGMA user_version=2")
        if mismatch == "changed_payload":
            db.execute("UPDATE changes SET after=x'00' WHERE position=1 AND layer='global'")
        elif mismatch == "missing_block":
            db.execute("DELETE FROM changes WHERE position=1 AND layer='radial'")
        elif mismatch == "extra_block":
            db.execute("INSERT INTO changes VALUES(1,'cartesian',80,2,2,2,NULL,x'00')")
        elif mismatch == "unexpected_before":
            db.execute("UPDATE changes SET before=after WHERE position=1 AND layer='global'")
        db.commit()
    finally:
        db.close()
    with NativeReviewSession(source, journal) as session:
        assert session.try_update_both(expected_revision=revision) is None
        assert session.revision == revision
        assert session.read_block("radial", 100, block).item()
        assert session.read_painted_block("radial", block).item()
        assert not session.read_block("global", 100, block).any()
        with session.transaction("inspect") as edit:
            assert edit.settled_update_ancestor() == ((1, counts) if mismatch is None else None)
        assert session.redo()
        assert session.read_block("global", 100, block).item()
        assert session.undo() and session.undo()
        if mismatch is None:
            assert session.try_update_both()["update_source"] == "settled"
        assert session.redo()
        assert session.read_block("radial", 100, block).item()


def test_prepared_update_rejects_legacy_active_masks_that_differ_from_first_action(tmp_path):
    source = ArrayStore(tmp_path / "study.xtareview")
    block = (2, 2, 2)
    source.masks["global"][80][6, 8, 10] = True
    journal = tmp_path / "legacy-active.sqlite"
    with NativeReviewSession(source, journal, initial_update_path=False) as session:
        with session.transaction("Update Both") as edit:
            edit.write_block("global", 80, block, np.zeros((1, 1, 1), bool))
            edit.record_update_both({80: 100, 90: 100, 100: 0})
    prepare_initial_update(source, journal)
    db = sqlite3.connect(journal)
    try:
        db.execute("DROP TABLE update_both_states")
        db.execute("PRAGMA user_version=2")
        db.execute("DELETE FROM masks")
        db.commit()
    finally:
        db.close()
    with NativeReviewSession(source, journal) as session:
        assert session.try_update_both(expected_revision=1) is None
        assert session.read_block("global", 80, block).item()
        assert session.revision == 1
        with session.transaction("inspect") as edit:
            assert edit.settled_update_ancestor() is None

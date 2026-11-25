"""Native edit semantics across the archived block boundaries."""
from collections import Counter
from itertools import product

import numpy as np
import pytest
from scipy import ndimage

from XTA import native_review_edits
from XTA.native_review_edits import ComponentIndex, NativeReviewEdits
from XTA.native_review_session import NativeReviewSession, StaleReviewWindow
from tests import slicer_cutoff_store_support as support


class VolumeFixture:
    shape = (19, 28, 33)
    block_shape = (5, 7, 8)
    cutoffs = (80, 90, 100)
    global_layer_id = "global"
    identity = "native-edit-fixture"
    geometry = {
        "space": "left-posterior-superior",
        "directions_xyz": ((1, 0, 0), (0, 1, 0), (0, 0, 1)),
        "origin_xyz": (0, 0, 0),
    }
    layers = {
        "radial": {"role": "direct", "additive": True, "visible": True},
        "trunk": {"role": "direct", "additive": True, "visible": True},
        "hidden": {"role": "direct", "additive": True, "visible": False},
        "derived": {"role": "derived", "additive": False, "visible": True},
        "global": {"role": "global", "additive": False, "visible": True},
    }

    def __init__(self, path):
        self.path = path
        self.path.touch()
        self.masks = {
            lid: {cut: np.zeros(self.shape, dtype=bool) for cut in self.cutoffs}
            for lid in self.layers
        }
        for cutoff in self.cutoffs:
            self.masks["radial"][cutoff][6, 5:20, 10:12] = True
            self.masks["radial"][cutoff][6, 5:20, 19:21] = True
            self.masks["radial"][cutoff][6, 19, 10:21] = True
            self.masks["trunk"][cutoff][6:10, 16:21, 9:22] = True
            self.masks["trunk"][cutoff][0, 0, 0] = True
            self.masks["hidden"][cutoff][6, 17:19, 21:24] = True
        self.masks["radial"][80][6, 20:23, 10:12] = True
        self.masks["derived"][90][6, 10, 13] = True
        self.masks["derived"][100][6, 10, 13:15] = True
        for cutoff in self.cutoffs:
            self.masks["global"][cutoff] = np.logical_or.reduce(
                [self.masks[lid][cutoff] for lid in ("radial", "trunk", "hidden")]
            )

    def iter_roi(self):
        return product(*(range((n + b - 1) // b) for n, b in zip(self.shape, self.block_shape)))

    def block_slices(self, block):
        return tuple(slice(i * b, min(n, (i + 1) * b)) for i, b, n in zip(block, self.block_shape, self.shape))

    def read_block(self, lid, cutoff, block):
        return self.masks[lid][cutoff][self.block_slices(block)].copy()

    def read_packed_block(self, lid, cutoff, block):
        return np.packbits(self.read_block(lid, cutoff, block), axis=-1, bitorder="little")

    def iter_blocks(self, lid, cutoff):
        yield from (block for block in self.iter_roi() if self.read_block(lid, cutoff, block).any())

    def layers_at(self, block, cutoff=None):
        cutoffs = self.cutoffs if cutoff is None else (cutoff,)
        return tuple(lid for lid in self.layers if any(self.read_block(lid, cut, block).any() for cut in cutoffs))


@pytest.fixture
def editor():
    module = support.load_store_module("native_review_eager_oracle")
    yield module
    support.unload_store_module(module)


def _whole(review, lid, cutoff):
    return review.read_region(lid, cutoff, (0, 0, 0), review.shape)


@pytest.mark.parametrize("face_cache_bytes", [0, 64 * 1024])
def test_native_component_graph_merges_every_block_side_and_raster_tie(tmp_path, monkeypatch, face_cache_bytes):
    monkeypatch.setattr(native_review_edits, "_COMPONENT_FACE_CACHE_BYTES", face_cache_bytes)
    base = VolumeFixture(tmp_path / "source.sqlite")
    rng = np.random.default_rng(15)
    mask = rng.random(base.shape) < 0.19
    mask[3:17, 7:22, 9] = True
    mask[11, 17, 9:31] = True
    eager, count = ndimage.label(mask)
    assert count > 1
    sizes = np.bincount(eager.ravel())
    sizes[0] = 0
    expected = eager == sizes.argmax()
    with NativeReviewSession(base) as session:
        masks = ((block, mask[base.block_slices(block)]) for block in base.iter_roi())
        with ComponentIndex(session, masks) as index:
            root, reported = index.largest()
            actual = np.zeros(base.shape, dtype=bool)
            for block, roots in index.iter_blocks_with_roots():
                actual[base.block_slices(block)] = roots == root
            assert reported == np.count_nonzero(expected)
            assert np.array_equal(actual, expected)
            for block in base.iter_roi():
                box = base.block_slices(block)
                retained, discarded = index.largest_mask(block, root)
                np.testing.assert_array_equal(retained, expected[box])
                assert discarded == bool(np.any(mask[box] & ~expected[box]))
            points = np.argwhere(expected)[[0, 10, -1]]
            assert np.all(index.roots_at(points) == root)
        tie = np.zeros(base.shape, dtype=bool)
        tie[1, 15, 25] = tie[1, 16, 25] = True
        tie[2, 1, 1] = tie[2, 2, 1] = True
        masks = ((block, tie[base.block_slices(block)]) for block in base.iter_roi())
        with ComponentIndex(session, masks) as index:
            root, size = index.largest()
            assert size == 2
            assert index.roots_at([[1, 15, 25], [2, 1, 1]]).tolist()[0] == root


def test_update_both_and_native_point_line_actions_match_existing_store(tmp_path, editor):
    pytest.importorskip("SimpleITK")
    base = VolumeFixture(tmp_path / "source.sqlite")
    reference = editor.XTACutoffStore(base.shape, len(base.cutoffs))
    kit = support.StoreKit(editor)
    for lid, props in base.layers.items():
        kit.add_layer_from_masks(
            reference, lid, props["role"], tuple(base.masks[lid].values()),
            visible=props["visible"], additive=props["additive"],
        )

    def compare(session):
        for lid in base.layers:
            for i, cutoff in enumerate(base.cutoffs):
                assert np.array_equal(_whole(session, lid, cutoff), kit.full_mask(reference, lid, i)), (lid, cutoff)
        assert session.global_stale == reference.global_stale

    try:
        with NativeReviewSession(base) as session:
            edits = NativeReviewEdits(session, editor=editor)
            compare(session)
            edits.update_both()
            editor._xta_store_update_both_entry(reference, tie_breaker="raster_first")
            compare(session)
            assert not _whole(session, "trunk", 80)[0, 0, 0]

            point = editor._xta_store_rasterise_markers(base.shape, points_ijk=[(10, 6, 6)])
            edits.remove_islands(90, point)
            editor._xta_store_remove_islands(reference, 1, point)
            compare(session)

            line = editor._xta_store_rasterise_markers(base.shape, lines_ijk=[((9, 17, 7), (22, 17, 7))])
            edits.remove_slices(90, line, scope="stricter")
            editor._xta_store_remove_slices(reference, 1, line, scope="stricter")
            compare(session)
            for _ in range(3):
                assert session.undo() == reference.undo()
                compare(session)
            for _ in range(3):
                assert session.redo() == reference.redo()
                compare(session)
    finally:
        reference.close()


def test_native_gesture_rejects_a_stale_revision_before_changing_blocks(tmp_path, editor):
    base = VolumeFixture(tmp_path / "source.sqlite")
    with NativeReviewSession(base) as first, NativeReviewSession(base) as second:
        edits = NativeReviewEdits(first, editor=editor)
        displayed_revision = first.revision
        NativeReviewEdits(second).update_both()
        points = np.array([[6, 6, 10]])
        with pytest.raises(StaleReviewWindow):
            edits.remove_islands(90, points, expected_revision=displayed_revision)
        with pytest.raises(StaleReviewWindow):
            edits.update_global(expected_revision=displayed_revision)
        assert _whole(first, "radial", 90)[tuple(points[0])]
        assert first.undo()
        assert _whole(first, "trunk", 80)[0, 0, 0]


def test_update_both_only_rereads_layers_on_blocks_with_a_losing_component(tmp_path, monkeypatch):
    base = VolumeFixture(tmp_path / "source.sqlite")
    with NativeReviewSession(base) as session:
        reads, packed_reads = Counter(), Counter()
        read_block = session.read_block
        read_packed_block = base.read_packed_block

        def spy(lid, cutoff, block):
            reads[lid, cutoff, block] += 1
            return read_block(lid, cutoff, block)

        def packed_spy(lid, cutoff, block):
            packed_reads[lid, cutoff, tuple(block)] += 1
            return read_packed_block(lid, cutoff, block)

        monkeypatch.setattr(session, "read_block", spy)
        monkeypatch.setattr(base, "read_packed_block", packed_spy)
        NativeReviewEdits(session).update_both()
        # The radial layer in this block intersects only the largest island.
        assert packed_reads["radial", 80, (1, 1, 1)] == 1
        assert reads["radial", 80, (1, 1, 1)] == 0
        # Both passes over this losing component stay in packed bytes.
        assert packed_reads["trunk", 80, (0, 0, 0)] == 2
        assert reads["trunk", 80, (0, 0, 0)] == 0
        assert not read_block("trunk", 80, (0, 0, 0)).any()
        assert session.undo()
        assert read_block("trunk", 80, (0, 0, 0))[0, 0, 0]


def test_native_update_cancels_earlier_cutoffs_and_retires_component_scratch(tmp_path):
    base = VolumeFixture(tmp_path / "source.sqlite")
    events = []

    def cancel(phase, cutoff, block_count):
        events.append((phase, cutoff, block_count))
        if phase == "label" and cutoff == 90:
            raise InterruptedError("user cancelled")

    with NativeReviewSession(base) as session:
        with pytest.raises(InterruptedError, match="user cancelled"):
            NativeReviewEdits(session).update_both(progress=cancel)
        assert ("label", 80, 0) in events
        assert any(phase == "write_global_and_clip" and cutoff == 80 and blocks > 0
                   for phase, cutoff, blocks in events)
        assert session.revision == 0
        assert not session.undo()
        assert _whole(session, "trunk", 80)[0, 0, 0]
        assert not list(tmp_path.glob("xta-native-components-*"))

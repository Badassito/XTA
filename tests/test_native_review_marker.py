import itertools

import numpy as np
import pytest

from XTA.native_review_flood import BasinTooLarge
from XTA.native_review_marker import stage_marker_removals
from XTA.native_review_session import NativeReviewSession
from tests import slicer_cutoff_store_support as support
from tests.test_native_review_session import ArrayStore


class BranchStore(ArrayStore):
    shape = (12, 18, 28)
    block_shape = chunk_shape = (4, 6, 6)
    layers = {
        "radial": {"additive": True, "role": "direct"},
        "cartesian": {"additive": True, "role": "direct"},
        "hidden": {"additive": True, "role": "direct"},
        "derived": {"additive": False, "role": "derived"},
        "global": {"additive": False, "role": "global"},
    }

    def __init__(self, path, *, rng=None):
        super().__init__(path)
        for layer in self.masks.values():
            for mask in layer.values():
                mask.fill(False)
        for cent in self.cutoffs:
            radial = self.masks["radial"][cent]
            radial[5:8, 3:15, 2:26] = True
            radial[5:8, 6:12, 5:23] = False
            radial[6, 6:12, 18] = True
            if cent == 80:
                radial[6, 5:7, 13] = True
            elif cent == 100:
                radial[6, 10:15, 17:19] = True
            self.masks["cartesian"][cent][6:9, 11:14, 17:20] = True
            self.masks["hidden"][cent][6, 11, 13:17] = True
            self.masks["derived"][cent][6, 10:12, 16:19] = True
            if rng is not None:
                for lid in ("radial", "cartesian", "derived"):
                    dense = self.masks[lid][cent]
                    dense[5:8, 10:14, 15:20] ^= rng.random((3, 4, 5)) < 0.13
            self.masks["global"][cent] = radial | self.masks["cartesian"][cent] | self.masks["hidden"][cent]


@pytest.fixture()
def editor():
    pytest.importorskip("SimpleITK")
    module = support.load_store_module()
    try:
        yield module, support.StoreKit(module)
    finally:
        support.unload_store_module(module)


def _baseline(module, kit, source, visible):
    eager = module.XTACutoffStore(source.shape, len(source.cutoffs))
    for lid, meta in source.layers.items():
        kit.add_layer_from_masks(
            eager, lid, meta["role"], [source.masks[lid][c] for c in source.cutoffs],
            visible=lid in visible, additive=meta["additive"],
        )
    return eager


@pytest.mark.parametrize("slices,scope,k0", list(itertools.product((False, True), ("all", "stricter"), (0, 1))))
def test_streamed_marker_crosses_blocks_with_legacy_exactness(tmp_path, editor, slices, scope, k0):
    module, kit = editor
    visible = ("radial", "cartesian", "derived")
    for seed in (None, 31, 61):
        source = BranchStore(tmp_path/f"review-{seed}.native", rng=None if seed is None else np.random.default_rng(seed))
        eager = _baseline(module, kit, source, visible)
        try:
            markers = module._xta_store_rasterise_markers(
                source.shape, lines_ijk=[((18, 8, 6), (18, 11, 6))],
            )
            operation = module._xta_store_remove_slices if slices else module._xta_store_remove_islands
            expected = operation(eager, k0, markers, scope=scope)
            with NativeReviewSession(source) as session:
                result = stage_marker_removals(
                    session, source.cutoffs[k0], markers, visible_layer_ids=visible,
                    scope=scope, editor=vars(module), remove_slices=slices,
                )
                assert result["status"] == expected["status"]
                assert result["hit_layers"] == expected["hit_layers"]
                assert session.global_stale == eager.global_stale
                for lid, mask in source.masks.items():
                    for k, cutoff in enumerate(source.cutoffs):
                        raster = session.read_region(lid, cutoff, (0, 0, 0), source.shape)
                        np.testing.assert_array_equal(raster, kit.full_mask(eager, lid, k), err_msg=f"{seed=} {lid=} {cutoff=}")
                assert session.undo()
                for lid, mask in source.masks.items():
                    for cutoff in source.cutoffs:
                        raster = session.read_region(lid, cutoff, (0, 0, 0), source.shape)
                        np.testing.assert_array_equal(raster, mask[cutoff])
                assert session.redo()
        finally:
            eager.close()


def test_basin_over_budget_aborts_without_writing_any_mask(tmp_path, editor):
    module, _ = editor
    source = BranchStore(tmp_path / "review.native")
    with NativeReviewSession(source) as session:
        with pytest.raises(BasinTooLarge, match="budget 1"):
            stage_marker_removals(
                session, 90, np.array([[6, 8, 18]]), visible_layer_ids=("radial",),
                scope="all", editor=vars(module), max_voxels=1,
            )
        assert not session.undo()
        assert not session.global_stale
        for lid in source.layers:
            for cutoff in source.cutoffs:
                raster = session.read_region(lid, cutoff, (0, 0, 0), source.shape)
                np.testing.assert_array_equal(raster, source.masks[lid][cutoff])


def test_all_cutoff_basin_preserves_hidden_manual_paint_seed(tmp_path, editor):
    module, kit = editor
    source = BranchStore(tmp_path / "review.native")
    for layer in source.masks.values():
        for mask in layer.values():
            mask.fill(False)
    source.masks["radial"][80][6, 8, 1:27] = True
    source.masks["radial"][90][6, 8, 1:3] = True
    source.masks["radial"][90][6, 8, 22:27] = True
    for cutoff in source.cutoffs:
        source.masks["global"][cutoff][:] = source.masks["radial"][cutoff]
    visible = ("radial",)
    eager = _baseline(module, kit, source, visible)
    painted = (6, 8, 7)
    point = np.array([[6, 8, 2]])
    try:
        module._xta_store_sync_paint(eager, "hidden", 1, np.array([painted]))
        expected = module._xta_store_remove_islands(eager, 1, point)
        assert expected["painted_kept"] == 1
        with NativeReviewSession(source) as session:
            with session.transaction("Paint hidden keep marker") as edit:
                empty = np.zeros((1, 1, 1), dtype=bool)
                edit.apply_region_diff("hidden", 90, painted, empty, ~empty)
            stage_marker_removals(session, 90, point, visible_layer_ids=visible, scope="all", editor=vars(module))
            for lid in source.layers:
                for k, cutoff in enumerate(source.cutoffs):
                    raster = session.read_region(lid, cutoff, (0, 0, 0), source.shape)
                    np.testing.assert_array_equal(raster, kit.full_mask(eager, lid, k))
    finally:
        eager.close()

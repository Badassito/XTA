from pathlib import Path
from contextlib import closing
import sqlite3

import numpy as np
import pytest

from XTA.native_review_store import NativeReviewStore
from XTA.native_review_overview import overview_volume
from XTA.native_review_session import NativeReviewSession
from XTA.review_archive import ReviewLayer, merge_review_archives, write_review_archive
from XTA.slicer_preview_review import preview_manifest, write_preview_edits
from tools import prepare_pta_input as prep


GEOMETRY = dict(space="left-posterior-superior",
                directions_xyz=((1., 0., 0.), (0., 1., 0.), (0., 0., 1.)),
                origin_xyz=(0., 0., 0.))


def make_layer(mask, lid, cutoff, role, bounds=None):
    row = dict(logical_id=lid, cutoff_cent=cutoff, role=role,
               segment_name="sample_Global_final_output" if role == "global" else lid)
    if bounds is None:
        read = lambda a, b: mask[a:b]
    else:
        (_, _), (y0, y1), (x0, x1) = bounds
        read = lambda a, b: mask[a:b, y0:y1, x0:x1]
    return ReviewLayer(row, read, bounds)


def publish(path, layers, shape, native, *, cutoffs=(80,), crop=None, source=None, **kwargs):
    return write_review_archive(
        path, review_id=path.stem, grid_id="low_quality/fixture",
        shape_tyx=shape, native_shape_tyx=native, geometry=GEOMETRY,
        cutoffs_cent=cutoffs, layers=layers, axial_crop=crop,
        source_record=source or {"source_stem": "sample"}, chunk_shape=(2, 3, 8), **kwargs,
    )


def test_preview_archive_exact_independent_layers_and_sealed_inventory(tmp_path):
    shape = (5, 7, 19)
    radial = np.zeros(shape, bool)
    radial[1:4, 2:5, 3:17] = True
    transverse = np.zeros(shape, np.uint8)
    transverse[2:4, 1:6, 10:19] = 1
    lower = radial | transverse
    higher = transverse.copy()
    higher[0, 0, 1] = 1
    layers = [
        make_layer(radial, "radial", 80, "direct", ((1, 4), (2, 5), (3, 17))),
        make_layer(transverse, "transverse", 80, "direct"),
        make_layer(higher, "transverse", 90, "direct"),
        make_layer(lower, "global", 80, "global"),
        make_layer(higher, "global", 90, "global"),
    ]
    source = tmp_path / "two-cutoffs.xtareview"
    expected = [(layer.metadata["logical_id"], layer.metadata["cutoff_cent"]) for layer in layers]
    assert publish(source, layers, shape, (10, 14, 38), cutoffs=(80, 90), expected_layer_keys=expected) == source
    bounds = tuple((0, n) for n in shape)
    with NativeReviewStore(source) as store:
        identity = store.identity
        assert store.grid_role == "preview"
        assert store.native_shape_tyx == (10, 14, 38)
        assert store.layers["radial"]["cutoffs"] == (80,)
        np.testing.assert_array_equal(store.read_region("radial", 80, bounds), radial)
        np.testing.assert_array_equal(store.read_region("transverse", 90, bounds), higher)
        np.testing.assert_array_equal(store.read_region("global", 80, bounds), lower)
        manifest = preview_manifest(store)
        assert manifest["grids"][0]["payload"] == "archive"
        assert len(manifest["grids"][0]["groups"][0]["segments"]) == 5
    with pytest.raises(FileExistsError):
        publish(source, layers, shape, (10, 14, 38), cutoffs=(80, 90))
    with closing(sqlite3.connect(source)) as db, db:
        db.execute("DELETE FROM blocks WHERE lid='radial' AND z=0")
    with pytest.raises(ValueError, match="block inventory"):
        NativeReviewStore(source, expected_identity=identity)


def test_cropped_edited_global_adapts_to_original_preparation_lattice(tmp_path):
    initial = (4, 5, 11)
    native = (8, 10, 22)
    shape, target = (4, 5, 8), (8, 10, 15)
    crop = dict(original_preview_shape_tyx=initial, original_full_quality_shape_tyx=native,
                first_removed_preview_i=8, first_removed_native_i=15,
                marker_preview_ras=(-7.0, -2.0, 1.0), marker_preview_ras_exact=("-7", "-2", "1"),
                source_edits_sha256="a" * 64)
    radial = np.zeros(shape, bool)
    radial[2, 3, 7] = True
    original = radial.copy()
    original[2, 3, 6] = True
    archive = publish(tmp_path / "study.xtareview", [
        make_layer(radial, "radial", 80, "direct"), make_layer(original, "global", 80, "global"),
    ], shape, target, crop=crop)
    with NativeReviewStore(archive) as source:
        identity = source.identity
    selected = tmp_path / "edited"
    write_preview_edits(selected, archive, identity, [make_layer(radial, "global", 80, "global")],
                        edit_revision={"log_records": 1})
    low, full, entries, provenance = prep.load_bundle(selected, "sample")
    assert low == shape and full == target
    assert provenance["axial_crop"]["original_preview_shape_tyx"] == list(initial)
    assert provenance["preview_edit"]["archive_identity"] == identity
    reconstructed_input = np.zeros(initial, np.uint8)
    prep.insert_snapshot(reconstructed_input, selected / entries[0]["file"], entries[0], 2**20)
    np.testing.assert_array_equal(reconstructed_input[:, :, :shape[2]], radial)
    assert not reconstructed_input[:, :, shape[2]:].any()
    with pytest.raises(ValueError, match="different volume stem"):
        prep.load_bundle(selected, "different")
    with NativeReviewStore(selected / "preview.xtareview") as edit_store:
        with pytest.raises(ValueError, match="PTA snapshot"):
            preview_manifest(edit_store)


def test_scalar_cutoff_retains_exact_source_value_and_family_copy(tmp_path):
    shape = (3, 4, 9)
    mask = np.ones(shape, bool)
    branches = {}
    for cutoff, exact in ((40, .405), (80, .8)):
        branches[cutoff] = publish(
            tmp_path / f"source-{cutoff}.xtareview", [make_layer(mask, "global", cutoff, "global")],
            shape, (6, 8, 18), cutoffs=(cutoff,),
            source={"source_stem": "sample", "confidence_threshold_exact": exact},
        )
    with NativeReviewStore(branches[40]) as store:
        assert store.source_record["confidence_threshold_exact"] == .405
    destination = tmp_path / "combined.xtareview"
    with pytest.raises(ValueError, match="does not match"):
        merge_review_archives(destination, branches, review_id="sweep")
    branches[40] = publish(tmp_path / "exact-forty.xtareview", [make_layer(mask, "global", 40, "global")],
                           shape, (6, 8, 18), cutoffs=(40,),
                           source={"source_stem": "sample", "confidence_threshold_exact": .4})
    merge_review_archives(destination, branches, review_id="sweep", expected_grid_id="low_quality/fixture",
                          expected_shape_tyx=shape, expected_native_shape_tyx=(6, 8, 18))
    with NativeReviewStore(destination) as store:
        assert store.cutoffs == (40, 80)
        for cutoff in store.cutoffs:
            np.testing.assert_array_equal(store.read_region("global", cutoff, tuple((0, n) for n in shape)), mask)
    with pytest.raises(ValueError, match="requested output grid"):
        merge_review_archives(tmp_path / "incorrect.xtareview", branches, review_id="sweep", expected_grid_id="native")


def test_invalid_or_incomplete_layers_never_publish(tmp_path):
    shape = (3, 4, 9)
    invalid = np.full(shape, 2, np.uint8)
    path = tmp_path / "bad.xtareview"
    with pytest.raises(ValueError, match="exact binary"):
        publish(path, [make_layer(invalid, "global", 80, "global")], shape, shape)
    with pytest.raises(ValueError, match="inventory"):
        publish(path, [make_layer(invalid == 2, "global", 80, "global")], shape, shape,
                expected_layer_keys=[("global", 80), ("missing", 80)])
    assert not path.exists()
    assert not list(Path(tmp_path).glob(".xta-review-build-*"))


def test_native_archive_keeps_a_navigation_overview_when_merged(tmp_path):
    shape = (4, 5, 9)
    mask = np.zeros(shape, np.uint8)
    mask[1:3, 2:4, 4:7] = 1
    branches = {}
    for cutoff in (80, 90):
        branches[cutoff] = write_review_archive(
            tmp_path / f"native-{cutoff}.xtareview", review_id="volume", grid_id="native",
            shape_tyx=shape, native_shape_tyx=shape, geometry=GEOMETRY, cutoffs_cent=(cutoff,),
            layers=(make_layer(mask, "global", cutoff, "global"),),
        )
    merged = merge_review_archives(tmp_path / "family.xtareview", branches, review_id="sweep")
    with NativeReviewSession(merged) as session:
        assert session.store.grid_role == "native"
        for cutoff in session.cutoffs:
            overview, _ = overview_volume(session.store, cutoff)
            np.testing.assert_array_equal(overview, mask)

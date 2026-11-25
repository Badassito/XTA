"""Parity and integrity for the native window review backing archive."""
import hashlib
import json
import sqlite3

import numpy as np
import pytest

from XTA.grouped_seg_nrrd import GroupedSegment, write_grouped_seg_nrrd
from XTA import native_review_store
from XTA.native_review_overview import original_overview_counts, overview_volume
from XTA.native_review_store import NativeReviewStore, prepare_native_review
from XTA.reconciliation_io import ReferenceGeometry


def _review_fixture(tmp_path):
    shape = (7, 11, 15)
    space = "left-posterior-superior"
    origin = (17., 29., 3.)
    directions = ((0., 1., 0.), (2., 0., 0.), (0., 0., 3.))
    geometry = ReferenceGeometry(shape, space=space, directions_xyz=directions, origin_xyz=origin)
    grids = tmp_path / "study" / "review"
    grids.mkdir(parents=True)
    records, originals = [], {}
    for cutoff in (80, 90):
        first = np.zeros(shape, dtype=np.uint8)
        second = np.zeros(shape, dtype=np.uint8)
        first[2:7, 1:4, 2:13] = 1
        first[5:7, 3:5, 7:15] = 1
        second[3:5, 6:9, 8:10] = 1
        if cutoff == 90:
            first[2, 2, 11] = 0
            first[3, 4, 8] = 1  # Exact nonnested cutoff membership.
            second.fill(0)
        global_mask = first | second
        masks = dict(vessel=first, sheet=second, global_mask=global_mask)
        originals[cutoff] = {lid: v.astype(bool) for lid, v in masks.items()}
        filename = f"conf-{cutoff}.seg.nrrd"
        group = write_grouped_seg_nrrd(
            grids / filename, shape_tyx=shape, index_segments=True, geometry=geometry,
            segments=[
                GroupedSegment(f"{lid}-{cutoff}", lid, (.3, .5, .7), lambda z0, z1, m=v: m[z0:z1])
                for lid, v in masks.items()
            ],
        )
        group.update(path=filename, cutoff_cent=cutoff)
        for lid, row in zip(masks, group["segments"]):
            role = "global" if lid == "global_mask" else "direct"
            row.update(logical_id=lid, role=role, segment_name=lid, cutoff_cent=cutoff,
                       layer_role="checkpoint" if role == "global" else "additive_component",
                       recomposition_op="select" if role == "global" else "union")
        records.append(group)
    doc = {
        "schema": "xta.sweep_review/1", "status": "complete", "sweep_id": "native-store-test",
        "cutoffs_cent": [80, 90], "default_cutoff_cent": 90,
        "grids": [{"grid_id": "native", "payload": "grouped", "shape_tyx": shape, "space": space,
                   "origin_xyz": origin, "directions_xyz": directions, "groups": records}],
    }
    manifest = grids / "manifest.json"
    manifest.write_text(json.dumps(doc), encoding="utf-8")
    return manifest, doc, originals


def test_exact_native_archive_reads_full_layers_boundaries_and_overview(tmp_path):
    manifest, doc, originals = _review_fixture(tmp_path)
    dst = tmp_path / "study.xtareview"
    notices = []
    assert prepare_native_review(manifest, dst, chunk_shape=(4, 4, 8),
                                 progress=lambda *args: notices.append(args)) == dst
    assert notices[0] == ("start", 0, 2)
    assert notices[-2:] == [("groups", 2, 2), ("index", 2, 2)]
    with NativeReviewStore(dst) as store:
        assert store.identity == store.source_identity
        assert store.review_manifest_sha256 == hashlib.sha256(manifest.read_bytes()).hexdigest()
        assert store.global_layer_id == "global_mask"
        assert store.cutoffs == (80, 90)
        assert store.default_cutoff == 90
        np.testing.assert_array_equal(store.affine_ijk_to_ras[:3, 3], (-17, -29, 3))
        for cutoff, layers in originals.items():
            for lid, expected in layers.items():
                bounds = tuple((0, n) for n in expected.shape)
                np.testing.assert_array_equal(store.read_region(lid, cutoff, bounds), expected)
                for index in store.iter_blocks(lid, cutoff):
                    assert lid in set(store.layers_at(index, cutoff))
                    crop = tuple(slice(*x) for x in store.block_bounds(index))
                    np.testing.assert_array_equal(store.read_block(lid, cutoff, index), expected[crop])
                    np.testing.assert_array_equal(
                        store.read_packed_block(lid, cutoff, index),
                        np.packbits(expected[crop], axis=-1, bitorder="little"),
                    )
        np.testing.assert_array_equal(
            store.read_region("vessel", 90, ((3, 7), (2, 5), (6, 15))),
            originals[90]["vessel"][3:7, 2:5, 6:15],
        )
        assert store.read_block("sheet", 90, (1, 2, 1)).shape == (3, 3, 7)
        assert not store.read_block("sheet", 90, (1, 2, 1)).any()
        assert list(store.iter_blocks("sheet", 90)) == []
        overview, affine = overview_volume(store, 90)
        np.testing.assert_array_equal(overview, originals[90]["global_mask"])
        np.testing.assert_array_equal(affine, store.affine_ijk_to_ras)
        for cutoff, layers in originals.items():
            field = original_overview_counts(store, cutoff)
            assert field.step == 1
            crop = tuple(slice(o, o + n) for o, n in zip(field.origin_zyx, field.data.shape))
            np.testing.assert_array_equal(field.data, layers["global_mask"][crop])
            assert int(field.data.sum()) == layers["global_mask"].sum()
        assert not dst.with_name(dst.name + ".overview.sqlite").exists()
    previous = dst.read_bytes()
    with pytest.raises(FileExistsError):
        prepare_native_review(manifest, dst)
    assert dst.read_bytes() == previous


def test_prepare_can_defer_finer_overview_with_tight_workspace_budget(tmp_path, monkeypatch):
    manifest, _, originals = _review_fixture(tmp_path)
    archive = tmp_path / "study.xtareview"
    monkeypatch.setattr(native_review_store, "_PREPARATION_BYTES", 1400)
    prepare_native_review(manifest, archive, chunk_shape=(4, 4, 8))
    with NativeReviewStore(archive) as store:
        with store.read_connection() as db:
            assert db.execute(
                "SELECT name FROM sqlite_master WHERE name='native_surface_counts_v1'"
            ).fetchone() is None
        field = original_overview_counts(store, 80)
        assert int(field.data.sum()) == originals[80]["global_mask"].sum()
    assert archive.with_name(archive.name + ".overview.sqlite").is_file()


def test_native_archive_preparation_refuses_corrupt_group(tmp_path):
    manifest, doc, _ = _review_fixture(tmp_path)
    first = doc["grids"][0]["groups"][0]
    path = manifest.parent / first["path"]
    damaged = bytearray(path.read_bytes())
    damaged[-8] ^= 0x40
    path.write_bytes(damaged)
    dst = tmp_path / "study.xtareview"
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_native_review(manifest, dst)
    assert not dst.exists()
    assert not list(tmp_path.glob(".xta-native-review-*"))


@pytest.mark.parametrize("small_band", [False, True])
def test_ordered_packed_union_keeps_selected_layers_and_edge_padding(tmp_path, monkeypatch, small_band):
    manifest, _, originals = _review_fixture(tmp_path)
    dst = tmp_path / "study.xtareview"
    prepare_native_review(manifest, dst, chunk_shape=(4, 4, 8))
    if small_band:
        monkeypatch.setattr(native_review_store, "_PACKED_UNION_BYTES", 16)
    with NativeReviewStore(dst) as store:
        for cutoff, layers in originals.items():
            for selected in (("vessel", "sheet"), ("vessel",), ("sheet",)):
                expected = np.logical_or.reduce([layers[lid] for lid in selected])
                keys = set()
                previous = (-1, -1, -1)
                for index, packed in store.iter_packed_union(selected, cutoff):
                    assert index > previous
                    previous = index
                    crop = tuple(slice(*x) for x in store.block_bounds(index))
                    np.testing.assert_array_equal(
                        packed, np.packbits(expected[crop], axis=-1, bitorder="little"),
                    )
                    assert np.any(expected[crop])
                    keys.add(index)
                assert keys == {ind for lid in selected for ind in store.iter_blocks(lid, cutoff)}


def test_native_archive_reports_tampered_block_before_decoding(tmp_path):
    manifest, *_ = _review_fixture(tmp_path)
    dst = tmp_path / "study.xtareview"
    prepare_native_review(manifest, dst, chunk_shape=(4, 4, 8))
    with NativeReviewStore(dst) as store:
        first = next(store.iter_blocks("vessel", 80))
    with sqlite3.connect(dst) as db:
        db.execute("UPDATE blocks SET packed=? WHERE lid=? AND cutoff=? AND z=? AND y=? AND x=?",
                   (b"broken", "vessel", 80, *first))
    with NativeReviewStore(dst) as store:
        with pytest.raises(ValueError, match="checksum"):
            store.read_block("vessel", 80, first)
        with pytest.raises(ValueError, match="checksum"):
            list(store.iter_packed_union(("vessel",), 80))


def test_optional_tables_share_the_store_snapshot(tmp_path):
    manifest, *_ = _review_fixture(tmp_path)
    dst = tmp_path / "study.xtareview"
    prepare_native_review(manifest, dst)
    with NativeReviewStore(dst) as store, store.read_connection() as reader:
        assert reader.in_transaction
        assert reader.execute("SELECT count(*) FROM native_overviews").fetchone() == (2,)
        with sqlite3.connect(dst, timeout=.01) as writer:
            with pytest.raises(sqlite3.OperationalError, match="locked"):
                writer.execute("BEGIN EXCLUSIVE")

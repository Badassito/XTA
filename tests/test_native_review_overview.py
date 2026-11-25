from contextlib import closing
import os
import sqlite3
from types import SimpleNamespace
from zipfile import ZipFile

import numpy as np
import pytest

from XTA.native_review_overview import (
    SURFACE_ISOVALUE, NativeOverviewAccumulator, NativeOverviewCounts, native_ijk_to_ras,
    original_overview_counts, overview_volume,
)
from tools.build_slicer_review_bundle import build


def test_overview_preserves_foreground_and_maps_to_native_cell_centers(tmp_path):
    shape = (8, 9, 389)
    mask = np.zeros(shape, bool)
    mask[1, 2, 4:15] = True
    mask[3, 3, 3] = mask[-1, -1, -1] = True
    overview = NativeOverviewAccumulator(shape)
    assert overview.step == 3
    # Every interior seam splits a navigation cell, including nonzero data.
    for z0, z1 in ((0, 2), (2, 8)):
        for y0, y1 in ((0, 4), (4, 9)):
            for x0, x1 in ((0, 5), (5, 389)):
                overview.add((z0, y0, x0), mask[z0:z1, y0:y1, x0:x1])
    path = tmp_path / "overview.db"
    with sqlite3.connect(path) as db:
        overview.save(db, 80)
    geometry = dict(space="left-posterior-superior", origin_xyz=(7, 11, 13),
                    directions_xyz=((0, 2, 0), (-3, 0, 0), (0, 0, 4)))
    store = SimpleNamespace(path=path, shape=shape, geometry=geometry,
                            layers={"g": {"role": "global", "cutoffs": [80]}},
                            read_connection=lambda: closing(sqlite3.connect(path)))
    data, affine = overview_volume(store, 80)
    expected = np.zeros(data.shape, np.uint8)
    expected[tuple(np.asarray(np.where(mask)) // 3)] = 1
    np.testing.assert_array_equal(data, expected)
    native = native_ijk_to_ras(geometry)
    np.testing.assert_array_equal(native[:, 3], [-7, -11, 13, 1])
    np.testing.assert_array_equal(affine @ [5, 1, 0, 1], native @ [16, 4, 1, 1])
    store.layers.clear()
    assert not overview_volume(store, 80)[0].any()


def test_portable_bundle_contains_reader_editor_and_toolbar(tmp_path):
    path = build(tmp_path / "workstation.zip")
    with ZipFile(path) as bundle:
        assert bundle.testzip() is None
        assert {"xta_toolbar_startup.py", "xta_slicerrc.py", "XTA/slicer_native_window.py",
                "XTA/native_review_store.py", "XTA/native_review_edits.py",
                "XTA/native_review_session.py", "XTA/native_review_overview.py"} <= set(bundle.namelist())
    with pytest.raises(FileExistsError):
        build(path)


def _prepare_counts(tmp_path, *, write_embedded=False, max_count_bytes=None):
    shape = (5, 7, 389)
    origin = (0, 0, 8)
    mask = np.zeros((5, 7, 381), dtype=bool)
    mask[:3, :4, :8] = True
    mask[4, 5, 375:] = True
    mask[4, 6, 380] = True
    extent = (8, 388, 0, 6, 0, 4)
    path = tmp_path / "study.xtareview"
    with closing(sqlite3.connect(path)) as db, db:
        db.execute("CREATE TABLE study(proof TEXT)")
        db.execute("INSERT INTO study VALUES ('retained')")
        if write_embedded:
            acc = NativeOverviewAccumulator(shape, extent_xyz=extent, max_count_bytes=max_count_bytes)
            for z in range(5):
                acc.add((z, *origin[1:]), mask[z:z + 1])
            acc.save(db, 80)

    calls = []

    def read_block(lid, cutoff, index):
        calls.append((lid, cutoff, index))
        return mask

    store = SimpleNamespace(
        identity="source-a", path=path, shape=shape, global_layer_id="global",
        layers={"global": {"extent_by_cutoff": {80: extent}}},
        read_connection=lambda: closing(sqlite3.connect(path)),
        iter_blocks=lambda *_: iter(((0, 0, 0),)),
        block_bounds=lambda _: ((0, 5), (0, 7), (8, 389)),
        read_block=read_block,
    )
    expected = NativeOverviewCounts(shape, bounds_zyx=((0, 5), (0, 7), (8, 389)))
    expected.add(origin, mask)
    return store, calls, expected


def test_fractional_counts_preserve_thin_partially_padded_cells():
    counts = NativeOverviewCounts((5, 7, 9), step=4, bounds_zyx=((0, 5), (0, 7), (2, 9)))
    counts.add((0, 1, 3), np.ones((4, 4, 1), dtype=bool))
    counts.add((4, 6, 8), np.ones((1, 1, 1), dtype=bool))
    scalar = counts.scalar_field()
    assert not scalar[counts.data == 0].any()
    assert SURFACE_ISOVALUE < scalar[0, 0, 0] < 255  # A visible thin sheet.
    # The last cell's true capacity is 1 × 3 × 1, with no dense neighbor.
    assert scalar[1, 1, 2] == round(255 * (0.52 + 0.48 * np.sqrt(1 / 3)))
    counts.add((4, 6, 8), np.ones((1, 1, 1), dtype=bool), subtract=True)
    assert counts.scalar_field()[1, 1, 2] == 0


def test_fractional_surface_keeps_an_isolated_thin_tube_and_rejects_dense_fringe():
    counts = NativeOverviewCounts((32, 32, 48), step=8)
    native = np.zeros(counts.native_shape, dtype=bool)
    native[:8, :8, :8] = True
    native[0, 0, 8] = True
    native[24, 24, :] = True
    counts.add((0, 0, 0), native)
    values = counts.scalar_field()
    assert values[0, 0, 0] == 255
    assert values[0, 0, 1] < SURFACE_ISOVALUE  # One grazing voxel beside dense support.
    assert (values[3, 3, :] > SURFACE_ISOVALUE).all()
    assert not values[counts.data == 0].any()


@pytest.mark.parametrize("cache_case", ["generated", "embedded", "budget", "foreign", "same-file"])
def test_read_fractional_counts_legacy_and_embedded_cache(tmp_path, cache_case):
    store, calls, expected = _prepare_counts(tmp_path, write_embedded=cache_case in ("embedded", "budget"),
                                             max_count_bytes=1 if cache_case == "budget" else None)
    cache = store.path.with_name(store.path.name + ".overview.sqlite")
    if cache_case == "foreign":
        with closing(sqlite3.connect(cache)) as db, db:
            db.execute("CREATE TABLE unrelated(datum INTEGER)")
        untouched = cache.read_bytes()
    elif cache_case == "same-file":
        try:
            os.link(store.path, cache)
        except PermissionError:
            pytest.skip("This sandbox disallows creating hardlinks")
        untouched = cache.read_bytes()

    first = original_overview_counts(store, 80)
    second = original_overview_counts(store, 80)
    np.testing.assert_array_equal(first.data, expected.data)
    np.testing.assert_array_equal(second.data, expected.data)
    np.testing.assert_array_equal(first.scalar_field(), expected.scalar_field())
    if cache_case == "embedded":
        assert not calls and not cache.exists()
    elif cache_case in ("foreign", "same-file"):
        assert len(calls) == 2
        assert cache.read_bytes() == untouched
    else:
        assert len(calls) == 1 and cache.exists()
        store.identity = "source-b"
        np.testing.assert_array_equal(original_overview_counts(store, 80).data, expected.data)
        assert len(calls) == 2

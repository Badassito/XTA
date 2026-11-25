import sqlite3

import numpy as np
import pytest

from XTA.native_review_session import NativeReviewSession
from XTA.native_review_store import NativeReviewStore, prepare_native_review
from tests.test_native_review_store import _review_fixture


def test_sparse_viewport_matches_native_masks_and_pending_overrides(tmp_path):
    manifest, _, originals = _review_fixture(tmp_path)
    archive = tmp_path / "review.xtareview"
    prepare_native_review(manifest, archive, chunk_shape=(4, 4, 8))
    lo, hi = (2, 2, 7), (7, 11, 15)
    crop = np.s_[2:7, 2:11, 7:15]
    original = originals[90]
    before_chunk = original["vessel"][4:7, 0:4, 8:15].copy()
    vessel = original["vessel"].copy()
    vessel[4:7, 0:4, 8:15] = False
    sheet = original["sheet"].copy()
    sheet[4, 10, 14] = True
    with NativeReviewSession(archive) as session:
        def expect(*layers):
            expected = {lid: mask[crop] for lid, mask in layers if mask[crop].any()}
            actual = dict(session.iter_regions(np.int64(90), lo, hi))
            assert actual.keys() == expected.keys()
            for lid, mask in actual.items():
                np.testing.assert_array_equal(mask, expected[lid])
                np.testing.assert_array_equal(session.read_region(lid, np.int64(90), lo, hi), mask)

        expect(*original.items())
        with session.transaction("Paint and erase"):
            session.write_block("vessel", 90, (1, 0, 1), vessel[4:7, 0:4, 8:15])
            session.write_block("sheet", 90, (1, 2, 1), sheet[4:7, 8:11, 8:15])
            expect(("vessel", vessel), ("sheet", sheet), ("global_mask", original["global_mask"]))
        with session.transaction("Restore archive's vessel"):
            session.write_block("vessel", 90, (1, 0, 1), before_chunk)
            # The committed overlay exists but a pending NULL discards it.
            expect(("vessel", original["vessel"]), ("sheet", sheet), ("global_mask", original["global_mask"]))
        assert session.undo()
        expect(("vessel", vessel), ("sheet", sheet), ("global_mask", original["global_mask"]))
        assert session.undo()
        expect(*original.items())


def test_sparse_viewport_checks_used_archive_and_journal_payloads(tmp_path):
    manifest, _, _ = _review_fixture(tmp_path)
    archive = tmp_path / "review.xtareview"
    journal = tmp_path / "edits.sqlite"
    prepare_native_review(manifest, archive, chunk_shape=(4, 4, 8))
    key = (1, 0, 1)
    with NativeReviewSession(archive, journal) as session:
        self_mask = session.read_block("vessel", 90, key)
        with session.transaction("Erase"):
            session.write_block("vessel", 90, key, np.zeros_like(self_mask))
    with sqlite3.connect(journal) as db:
        db.execute("UPDATE masks SET payload=? WHERE layer=?", (b"broken", "vessel"))
    with NativeReviewSession(archive, journal) as session:
        with pytest.raises(ValueError, match="corrupt block"):
            list(session.iter_regions(90, (4, 0, 8), (7, 4, 15)))
    with sqlite3.connect(archive) as db:
        db.execute("UPDATE blocks SET packed=? WHERE lid=? AND cutoff=? AND (z,y,x)=(?,?,?)",
                   (b"broken", "vessel", 90, *key))
    with NativeReviewStore(archive) as store, NativeReviewSession(store, tmp_path / "clean.sqlite") as session:
        with pytest.raises(ValueError, match="checksum"):
            list(session.iter_regions(90, (4, 0, 8), (7, 4, 15)))
    with NativeReviewSession(archive, tmp_path / "limited.sqlite", max_region_bytes=8) as session:
        with pytest.raises(MemoryError, match="decode budget"):
            list(session.iter_regions(90, (4, 0, 8), (7, 4, 15)))

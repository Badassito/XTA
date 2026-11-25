from contextlib import closing
import sqlite3

import numpy as np
import pytest

from XTA import native_review_edits
from XTA.native_review_initial import initial_update_path
from XTA.native_review_session import NativeReviewSession
from tests.test_native_review_store import _review_fixture
from tools import prepare_native_review as preparer


def _read(session):
    return {
        (lid, cutoff): session.read_region(lid, cutoff, (0, 0, 0), session.shape)
        for lid in session.layers
        for cutoff in session.cutoffs
    }


def _assert_masks_equal(before, after):
    assert before.keys() == after.keys()
    for key, mask in before.items():
        np.testing.assert_array_equal(mask, after[key], err_msg=str(key))


def test_prepare_initial_update_with_new_archive(tmp_path, capsys):
    manifest, _, originals = _review_fixture(tmp_path)
    archive = tmp_path / "fold.xtareview"
    preparer.main([str(manifest), str(archive), "--chunk-shape", "4", "4", "8", "--prepare-update-both"])
    sidecar = initial_update_path(archive)
    assert sidecar.is_file()
    output = capsys.readouterr()
    assert "Computing initial Update Both" in output.err
    assert str(sidecar) in output.out
    with NativeReviewSession(archive, tmp_path / "user.edits.sqlite") as user:
        assert user.revision == 0
        baseline = _read(user)
        _assert_masks_equal(baseline, {
            (lid, cutoff): mask for cutoff, native_masks in originals.items() for lid, mask in native_masks.items()
        })
        native_review_edits.NativeReviewEdits(user).update_both()
        assert user.revision == 1
        assert not user.read_block("sheet", 80, (0, 1, 1)).any()
        assert user.undo()
        _assert_masks_equal(_read(user), baseline)


def test_legacy_journal_import_preserves_its_archive_and_keeps_edit_undoable(tmp_path, monkeypatch):
    manifest, *_ = _review_fixture(tmp_path)
    archive, journal = tmp_path / "legacy.xtareview", tmp_path / "prior.edits.sqlite"
    preparer.main([str(manifest), str(archive), "--quiet"])
    with NativeReviewSession(archive, journal) as previous:
        originals = _read(previous)
        native_review_edits.NativeReviewEdits(previous).update_both()
        expected = _read(previous)
    # Version 2 has the same original first-gesture history without the
    # settled-operation receipts used by current journals.
    with closing(sqlite3.connect(journal)) as db, db:
        db.execute("DROP TABLE update_both_states")
        db.execute("PRAGMA user_version=2")
    old_archive, old_journal = archive.read_bytes(), journal.read_bytes()

    def reject_recomputation(*_args, **_kwargs):
        raise AssertionError("Import of the trusted prepared journal should not rerun the component solver")

    monkeypatch.setattr(native_review_edits, "_layer_block_union", reject_recomputation)
    flags = [str(archive), "--prepare-update-both", "--from-existing-journal", str(journal), "--quiet"]
    preparer.main(flags)
    assert archive.read_bytes() == old_archive
    assert journal.read_bytes() == old_journal
    with NativeReviewSession(archive, tmp_path / "user.edits.sqlite") as user:
        native_review_edits.NativeReviewEdits(user).update_both()
        _assert_masks_equal(_read(user), expected)
        assert user.undo()
        _assert_masks_equal(_read(user), originals)
        assert user.redo()
        _assert_masks_equal(_read(user), expected)

    with pytest.raises(SystemExit) as refused:
        preparer.main(flags)
    assert refused.value.code == 2
    assert archive.read_bytes() == old_archive
    assert journal.read_bytes() == old_journal

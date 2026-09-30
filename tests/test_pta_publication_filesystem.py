"""Filesystem integrity contracts for PTA's final image scan."""

from __future__ import annotations

import os
from pathlib import Path
import stat
import tempfile
import unittest
from unittest import mock

from XTA import pta_publication as publication


class PtaPublicationFilesystemTests(unittest.TestCase):
    def test_missing_root_only_succeeds_for_zero_expected_images(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.assertEqual(publication.verify_published_image_tree(
                root, expected_count=0, image_format="jpg"), {
                    "selected": True, "verified_image_count": 0,
                    "verified_total_bytes": 0, "suffix": ".jpg",
                })
            with self.assertRaisesRegex(RuntimeError, "missing its image root"):
                publication.verify_published_image_tree(
                    root, expected_count=1, image_format="jpg")

    def test_nested_scan_reports_exact_count_and_bytes_without_rglob(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train = root / "images" / "train"
            val = root / "images" / "val"
            train.mkdir(parents=True)
            val.mkdir()
            (train / "a.jpg").write_bytes(b"abcd")
            (val / "b.JPG").write_bytes(b"1234567")
            with mock.patch.object(Path, "rglob", side_effect=AssertionError("rglob used")):
                result = publication.verify_published_image_tree(
                    root, expected_count=2, image_format="jpg")
            self.assertEqual(result["verified_image_count"], 2)
            self.assertEqual(result["verified_total_bytes"], 11)
            self.assertTrue(result["all_files_nonempty"])

    def test_wrong_suffix_empty_file_and_symlink_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()
            good = images / "good.jpg"
            good.write_bytes(b"1234")
            bad = images / "bad.png"
            bad.write_bytes(b"1234")
            with self.assertRaisesRegex(RuntimeError, "unexpected="):
                publication.verify_published_image_tree(root, expected_count=1, image_format="jpg")
            bad.unlink()
            empty = images / "empty.jpg"
            empty.touch()
            with self.assertRaisesRegex(RuntimeError, "empty_or_irregular="):
                publication.verify_published_image_tree(root, expected_count=2, image_format="jpg")
            empty.unlink()
            link = images / "link.jpg"
            try:
                os.symlink(good, link)
            except OSError:
                self.skipTest("symlink creation unavailable")
            with self.assertRaisesRegex(RuntimeError, "unexpected_symlink="):
                publication.verify_published_image_tree(root, expected_count=1, image_format="jpg")

    def test_irregular_entry_and_failed_stat_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            images = root / "images"
            images.mkdir()

            class Entry:
                name = "pipe.jpg"
                path = str(images / name)

                def is_symlink(self):
                    return False

                def is_dir(self, *, follow_symlinks):
                    return False

                def stat(self, *, follow_symlinks):
                    return type("Stat", (), {"st_mode": stat.S_IFIFO, "st_size": 5})()

            class Scanner:
                def __enter__(self):
                    return iter((Entry(),))

                def __exit__(self, *_args):
                    return False

            class Symlink(Entry):
                def is_symlink(self):
                    return True

            class SymlinkScanner(Scanner):
                def __enter__(self):
                    return iter((Symlink(),))

            with mock.patch.object(publication.os, "scandir", return_value=SymlinkScanner()):
                with self.assertRaisesRegex(RuntimeError, "unexpected_symlink="):
                    publication.verify_published_image_tree(
                        root, expected_count=1, image_format="jpg")

            with mock.patch.object(publication.os, "scandir", return_value=Scanner()):
                with self.assertRaisesRegex(RuntimeError, "empty_or_irregular="):
                    publication.verify_published_image_tree(
                        root, expected_count=1, image_format="jpg")

            class Unreadable(Entry):
                def stat(self, *, follow_symlinks):
                    raise OSError("file vanished")

            class UnreadableScanner(Scanner):
                def __enter__(self):
                    return iter((Unreadable(),))

            with mock.patch.object(publication.os, "scandir", return_value=UnreadableScanner()):
                with self.assertRaisesRegex(RuntimeError, "unreadable="):
                    publication.verify_published_image_tree(
                        root, expected_count=1, image_format="jpg")


if __name__ == "__main__":
    unittest.main()

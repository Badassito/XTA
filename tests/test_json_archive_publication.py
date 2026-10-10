"""Archived receipts retain the ordinary JSON publication contract."""
import json
from pathlib import Path
import tempfile
import unittest

from XTA.artifact_archive import read_artifact, reference
from XTA.json_publication import write_json_atomic


class ArchivedJsonPublicationTests(unittest.TestCase):
    def test_receipt_replacement_is_visible_without_creating_logical_directories(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            archive = root / "sam-artifacts.tar"
            target = Path(reference(archive, "sam_interpolation/" + "scope/" * 60 + "selection.json"))
            write_json_atomic(target, {"z": 1, "a": 2}, sort_keys=True, trailing_newline=True)
            self.assertEqual(read_artifact(target), b'{\n  "a": 2,\n  "z": 1\n}\n')
            write_json_atomic(target, {"selected": True})
            self.assertEqual(json.loads(read_artifact(target)), {"selected": True})
            self.assertEqual(sorted(path.name for path in root.iterdir()),
                             ["sam-artifacts.tar", "sam-artifacts.tar.lock"])

    def test_invalid_json_preserves_committed_receipt(self):
        with tempfile.TemporaryDirectory() as temporary:
            target = reference(Path(temporary) / "sam-artifacts.tar", "selection.json")
            write_json_atomic(target, {"valid": True})
            original = read_artifact(target)
            with self.assertRaises(ValueError):
                write_json_atomic(target, {"invalid": float("nan")})
            self.assertEqual(read_artifact(target), original)


if __name__ == "__main__":
    unittest.main()

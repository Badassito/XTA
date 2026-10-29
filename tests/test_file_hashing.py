from __future__ import annotations

import hashlib
from pathlib import Path
import tempfile
import unittest

from XTA.file_hashing import sha256_file


class FileHashingTests(unittest.TestCase):
    def test_empty_and_multichunk_artifacts_match_content_digest(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'artifact.bin'
            for payload in (b'', bytes(range(256)) * 8193 + b'final partial chunk'):
                with self.subTest(size=len(payload)):
                    path.write_bytes(payload)
                    expected = hashlib.sha256(payload).hexdigest()
                    self.assertEqual(sha256_file(path), expected)
                    self.assertEqual(sha256_file(str(path)), expected)

    def test_missing_artifact_is_not_silently_hashed(self):
        with tempfile.TemporaryDirectory() as temporary:
            with self.assertRaises(FileNotFoundError):
                sha256_file(Path(temporary) / 'missing.bin')


if __name__ == '__main__':
    unittest.main()

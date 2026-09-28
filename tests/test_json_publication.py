"""Failure and serialization contracts for JSON publication."""

from __future__ import annotations

import errno
import json
import multiprocessing
import os
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from unittest import mock

from XTA import json_publication
from XTA.confidence_evidence import _write_json_atomic as write_confidence
from XTA.lta_outputs import write_json_atomically as write_lta
from XTA.outputs import _write_json_atomically as write_sidecar
from XTA.unification.manifest import write_json_manifest


def _publish_in_forked_child(destination: str) -> None:
    json_publication.write_json_atomic(destination, {"child": True})


class JsonPublicationTests(unittest.TestCase):
    @staticmethod
    def _writers():
        return (
            ("run", write_json_manifest, True, True),
            ("lta", write_lta, True, True),
            ("sidecar", write_sidecar, False, False),
            ("confidence", write_confidence, True, True),
        )

    def test_writers_keep_their_json_bytes_and_reject_nan(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            for name, writer, sorted_keys, newline in self._writers():
                with self.subTest(writer=name):
                    destination = Path(temp_dir) / f"{name}.json"
                    writer(destination, {"z": 1, "a": 2})
                    expected = json.dumps(
                        {"z": 1, "a": 2}, indent=2, sort_keys=sorted_keys
                    ) + ("\n" if newline else "")
                    self.assertEqual(destination.read_bytes(), expected.encode("utf-8"))
                    with self.assertRaises(ValueError):
                        writer(destination, {"bad": float("nan")})
                    self.assertEqual(destination.read_bytes(), expected.encode("utf-8"))
            self.assertEqual(list(Path(temp_dir).glob(".*")), [])

    def test_file_sync_failure_preserves_previous_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "manifest.json"
            destination.write_bytes(b"previous")
            with mock.patch.object(
                json_publication.os, "fsync", side_effect=OSError(errno.EIO, "sync failed")
            ):
                with self.assertRaises(OSError):
                    write_json_manifest(destination, {"new": True})
            self.assertEqual(destination.read_bytes(), b"previous")
            self.assertEqual(list(Path(temp_dir).glob(".*")), [])

    def test_directory_sync_failure_does_not_acknowledge_publication(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "manifest.json"
            with mock.patch.object(
                json_publication,
                "_fsync_parent_directory",
                side_effect=OSError(errno.EIO, "directory sync failed"),
            ):
                with self.assertRaises(OSError):
                    write_lta(destination, {"new": True})
            self.assertEqual(json.loads(destination.read_text(encoding="utf-8")), {"new": True})
            self.assertEqual(list(Path(temp_dir).glob(".*")), [])

    def test_replace_failure_preserves_previous_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "sidecar.json"
            destination.write_bytes(b"previous")
            with mock.patch.object(
                json_publication.os, "replace", side_effect=OSError(errno.EIO, "replace failed")
            ):
                with self.assertRaises(OSError):
                    write_sidecar(destination, {"new": True})
            self.assertEqual(destination.read_bytes(), b"previous")
            self.assertEqual(list(Path(temp_dir).glob(".*")), [])

    def test_concurrent_writers_use_independent_stages(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "manifest.json"
            seen_stages: list[Path] = []
            real_replace = json_publication.os.replace

            def record_replace(source: Path, target: Path) -> None:
                seen_stages.append(Path(source))
                real_replace(source, target)

            with mock.patch.object(json_publication.os, "replace", side_effect=record_replace):
                with ThreadPoolExecutor(max_workers=8) as pool:
                    list(pool.map(
                        lambda number: json_publication.write_json_atomic(
                            destination, {"number": number, "data": "x" * 100_000}
                        ),
                        range(16),
                    ))

            self.assertEqual(len(set(seen_stages)), 16)
            self.assertIn(json.loads(destination.read_text(encoding="utf-8"))["number"], range(16))
            self.assertEqual(list(Path(temp_dir).glob(".*")), [])
            self.assertEqual(json_publication._REPLACE_LOCKS, {})

    def test_different_destinations_can_sync_in_parallel(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            barrier = threading.Barrier(2, timeout=5)
            destinations = [Path(temp_dir) / f"manifest-{number}.json" for number in range(2)]

            def sync_together(_parent: Path) -> None:
                barrier.wait()

            with mock.patch.object(
                json_publication, "_fsync_parent_directory", side_effect=sync_together
            ):
                with ThreadPoolExecutor(max_workers=2) as pool:
                    list(pool.map(
                        lambda destination: json_publication.write_json_atomic(
                            destination, {"name": destination.name}
                        ),
                        destinations,
                    ))

            for destination in destinations:
                self.assertEqual(
                    json.loads(destination.read_text(encoding="utf-8")),
                    {"name": destination.name},
                )
            self.assertEqual(json_publication._REPLACE_LOCKS, {})

    @unittest.skipUnless(
        os.name == "posix" and hasattr(os, "register_at_fork"),
        "requires POSIX fork support",
    )
    def test_fork_discards_locks_held_by_parent_threads(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            destination = Path(temp_dir) / "manifest.json"
            held = threading.Event()
            release = threading.Event()

            def hold_parent_locks() -> None:
                with json_publication._destination_lock(destination):
                    with json_publication._LOCK_REGISTRY_GUARD:
                        held.set()
                        release.wait(timeout=10)

            holder = threading.Thread(target=hold_parent_locks)
            holder.start()
            child = None
            try:
                self.assertTrue(held.wait(timeout=5))
                child = multiprocessing.get_context("fork").Process(
                    target=_publish_in_forked_child, args=(str(destination),)
                )
                child.start()
                child.join(timeout=5)
                if child.is_alive():
                    child.terminate()
                    child.join(timeout=5)
                self.assertEqual(child.exitcode, 0)
                self.assertEqual(
                    json.loads(destination.read_text(encoding="utf-8")),
                    {"child": True},
                )
            finally:
                release.set()
                holder.join(timeout=5)
                if child is not None and child.is_alive():
                    child.terminate()
                    child.join(timeout=5)


if __name__ == "__main__":
    unittest.main()

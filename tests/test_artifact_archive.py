from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
import hashlib
import mmap
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import unittest
from unittest import mock

from XTA import artifact_archive as artifacts


class ArtifactArchiveTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.archive = self.root / "sam-artifacts.tar"

    def ref(self, name):
        return artifacts.reference(self.archive, name)

    def test_long_logical_paths_stream_exact_bytes_without_filesystem_directories(self):
        logical = "sam_interpolation/" + "/".join(["very_long_view" * 10] * 4) + "/evidence/masks.bin"
        source = self.root / "source.bin"
        data = bytes(range(256)) * 10000
        source.write_bytes(data)
        artifacts.append_members(self.archive, {logical: source, "receipt.json": b"{}"})
        self.assertEqual(artifacts.read_member(self.ref(logical)), data)
        self.assertEqual(source.read_bytes(), data)
        self.assertEqual(artifacts.member_info(self.ref(logical))["sha256"], hashlib.sha256(data).hexdigest())
        self.assertEqual(sorted(p.name for p in self.root.iterdir()),
                         ["sam-artifacts.tar", "sam-artifacts.tar.lock", "source.bin"])
        with tarfile.open(self.archive) as archive:
            self.assertTrue(any(item.name.endswith(logical) for item in archive.getmembers()))

    def test_member_cursors_are_independent_and_cannot_read_neighbors(self):
        artifacts.append_members(self.archive, {"a": b"abcdef", "b": b"SECRET"})
        with artifacts.open_member(self.ref("a")) as first, artifacts.open_member(self.ref("a")) as second:
            self.assertEqual(first.read(2), b"ab")
            self.assertEqual(second.read(3), b"abc")
            first.seek(-2, os.SEEK_END)
            self.assertEqual(first.read(1000), b"ef")
            self.assertEqual(first.read(), b"")
            with self.assertRaises(ValueError):
                first.seek(7)
        self.assertEqual(artifacts.read_member(self.ref("b")), b"SECRET")

    def test_prior_open_member_survives_unrelated_appends(self):
        artifacts.append_members(self.archive, {"old": b"unchanged"})
        before = artifacts.member_info(self.ref("old"))
        with artifacts.open_member(self.ref("old")) as stream:
            artifacts.append_members(self.archive, {"new": b"later"})
            self.assertEqual(stream.read(), b"unchanged")
        self.assertEqual(artifacts.member_info(self.ref("old")), before)

    def test_committed_lookup_during_local_append_does_not_wait_for_new_payload(self):
        artifacts.append_members(self.archive, {"old": b"committed"})
        entered, release, reader_started = threading.Event(), threading.Event(), threading.Event()
        original = artifacts._write_member
        def pause(stream, name, source):
            if name.endswith("/new"):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("append was not released")
            return original(stream, name, source)
        def read_new():
            reader_started.set()
            return artifacts.read_member(self.ref("new"))
        with mock.patch.object(artifacts, "_write_member", side_effect=pause), ThreadPoolExecutor(max_workers=3) as pool:
            writer = pool.submit(artifacts.append_members, self.archive, {"new": b"later"})
            try:
                self.assertTrue(entered.wait(10))
                old = pool.submit(artifacts.read_artifact, self.ref("old"), max_bytes=100)
                self.assertEqual(old.result(timeout=2), b"committed")
                new = pool.submit(read_new)
                self.assertTrue(reader_started.wait(2))
                with self.assertRaises(FutureTimeout):
                    new.result(timeout=.1)
            finally:
                release.set()
            writer.result(timeout=10)
            self.assertEqual(new.result(timeout=2), b"later")
        self.assertNotIn(str(self.archive), artifacts._APPEND_SNAPSHOTS)

    def test_replacement_snapshot_and_failed_append_keep_exact_old_payload(self):
        artifacts.append_members(self.archive, {"old": b"committed"})
        for fail in (True, False):
            entered, release = threading.Event(), threading.Event()
            original = artifacts._write_member
            def pause(stream, name, source):
                if name.endswith("/old"):
                    entered.set()
                    if not release.wait(10):
                        raise TimeoutError("append was not released")
                    if fail:
                        raise OSError("injected append failure")
                return original(stream, name, source)
            cursor = None
            with self.subTest(fail=fail), mock.patch.object(artifacts, "_write_member", side_effect=pause), ThreadPoolExecutor(max_workers=2) as pool:
                writer = pool.submit(artifacts.append_members, self.archive, {"old": b"replacement"}, replace=True)
                try:
                    self.assertTrue(entered.wait(10))
                    cursor = pool.submit(artifacts.open_member, self.ref("old")).result(timeout=2)
                finally:
                    release.set()
                try:
                    if fail:
                        with self.assertRaisesRegex(OSError, "injected"):
                            writer.result(timeout=10)
                    else:
                        writer.result(timeout=10)
                    self.assertEqual(cursor.read(), b"committed")
                finally:
                    if cursor is not None:
                        cursor.close()
            self.assertNotIn(str(self.archive), artifacts._APPEND_SNAPSHOTS)
            self.assertEqual(artifacts.read_member(self.ref("old")), b"committed" if fail else b"replacement")

    def test_existing_prefix_projection_lookup_bypasses_append_but_new_inventory_waits(self):
        artifacts.append_members(self.archive, {"existing/meta.json": b"{}", "existing/data.bin": b"data"})
        entered, release = threading.Event(), threading.Event()
        full_started, missing_started = threading.Event(), threading.Event()
        original = artifacts._write_member
        def pause(stream, name, source):
            if name.endswith("/pending/new"):
                entered.set()
                if not release.wait(10):
                    raise TimeoutError("append was not released")
            return original(stream, name, source)
        def projection_lookup():
            self.assertTrue(artifacts.artifact_exists(self.ref("existing")))
            self.assertTrue(artifacts.artifact_exists(self.ref("existing/meta.json")))
            return sum(artifacts.artifact_size(path) for path in
                       artifacts.iter_artifacts(self.ref("existing"), recursive=False))
        def full_lookup():
            full_started.set()
            return artifacts.list_members(self.archive)
        def missing_lookup():
            missing_started.set()
            return list(artifacts.iter_artifacts(self.ref("pending")))
        with mock.patch.object(artifacts, "_write_member", side_effect=pause), ThreadPoolExecutor(max_workers=4) as pool:
            writer = pool.submit(artifacts.append_members, self.archive, {"pending/new": b"later"})
            try:
                self.assertTrue(entered.wait(10))
                self.assertEqual(pool.submit(projection_lookup).result(timeout=2), 6)
                full, missing = pool.submit(full_lookup), pool.submit(missing_lookup)
                self.assertTrue(full_started.wait(2))
                self.assertTrue(missing_started.wait(2))
                for pending in (full, missing):
                    with self.assertRaises(FutureTimeout):
                        pending.result(timeout=.1)
            finally:
                release.set()
            writer.result(timeout=10)
            self.assertEqual(full.result(timeout=2), ["existing/data.bin", "existing/meta.json", "pending/new"])
            self.assertEqual(missing.result(timeout=2), [Path(self.ref("pending/new"))])

    def test_append_snapshot_rejects_replaced_file_or_shortened_prefix(self):
        artifacts.append_members(self.archive, {"old": b"committed"})
        replacement = self.root / "replacement.tar"
        replacement.write_bytes(self.archive.read_bytes())
        for mutation in ("replacement", "truncation"):
            entered, release = threading.Event(), threading.Event()
            original = artifacts._write_member
            def pause(stream, name, source):
                if name.endswith("/" + mutation):
                    entered.set()
                    if not release.wait(10):
                        raise TimeoutError("append was not released")
                return original(stream, name, source)
            with self.subTest(mutation=mutation), mock.patch.object(artifacts, "_write_member", side_effect=pause), ThreadPoolExecutor(max_workers=1) as pool:
                writer = pool.submit(artifacts.append_members, self.archive, {mutation: b"later"})
                try:
                    self.assertTrue(entered.wait(10))
                    if mutation == "replacement":
                        real_open = open
                        def replaced_open(path, mode):
                            return real_open(replacement if mode == "rb" else path, mode)
                        with mock.patch.object(artifacts, "open", side_effect=replaced_open, create=True):
                            with self.assertRaisesRegex(artifacts.ArchiveError, "prefix changed"):
                                artifacts.read_member(self.ref("old"))
                    else:
                        end = artifacts._APPEND_SNAPSHOTS[str(self.archive)][2]
                        with self.archive.open("r+b") as stream:
                            stream.truncate(end - 1)
                        with self.assertRaisesRegex(artifacts.ArchiveError, "prefix changed"):
                            artifacts.member_info(self.ref("old"))
                    self.assertNotIn(str(self.archive), artifacts._APPEND_SNAPSHOTS)
                finally:
                    release.set()
                writer.result(timeout=10)
            self.assertNotIn(str(self.archive), artifacts._APPEND_SNAPSHOTS)

    def test_fork_reset_does_not_inherit_append_snapshot_authority(self):
        with mock.patch.object(artifacts, "_APPEND_SNAPSHOTS", {"inherited": object()}):
            artifacts._after_fork()
            self.assertEqual(artifacts._APPEND_SNAPSHOTS, {})

    def test_bounded_read_checks_opened_payload_after_logical_replacement(self):
        artifacts.append_members(self.archive, {"old": b"tiny"})
        original = artifacts.open_artifact
        def replace_before_open(value):
            artifacts.append_members(self.archive, {"old": b"larger replacement"}, replace=True)
            return original(value)
        with mock.patch.object(artifacts, "open_artifact", side_effect=replace_before_open):
            with self.assertRaisesRegex(artifacts.ArchiveError, "bounded read budget"):
                artifacts.read_artifact(self.ref("old"), max_bytes=4)

    def test_live_readonly_mapping_survives_append_and_failed_append(self):
        artifacts.append_members(self.archive, {"old": b"mapped evidence"})
        info = artifacts.member_info(self.ref("old"))
        with self.archive.open("rb") as source, mmap.mmap(source.fileno(), 0, access=mmap.ACCESS_READ) as mapped:
            artifacts.append_members(self.archive, {"later": b"another scope"})
            original = artifacts._write_member
            def fail(stream, name, data):
                original(stream, name, data)
                raise OSError("failed append while mapped")
            with mock.patch.object(artifacts, "_write_member", side_effect=fail):
                with self.assertRaises(OSError):
                    artifacts.append_members(self.archive, {"failed": b"discard"})
            self.assertEqual(mapped[info["offset"]:info["offset"] + info["bytes"]], b"mapped evidence")
            self.assertEqual(artifacts.list_members(self.archive), ["later", "old"])
            artifacts.append_members(self.archive, {"recovered": b"safe"})
            self.assertEqual(artifacts.read_member(self.ref("recovered")), b"safe")

    def test_torn_tail_never_exposes_incomplete_members_and_append_recovers(self):
        artifacts.append_members(self.archive, {"old": b"previous complete evidence"})
        first = self.archive.read_bytes()
        artifacts.append_members(self.archive, {"new": b"new evidence", "new-index": b"{}"})
        complete = self.archive.read_bytes()
        broken = self.root / "broken.tar"
        for cut in range(len(first) - 1024, len(complete) - 1024):
            broken.write_bytes(complete[:cut])
            old_ref = artifacts.reference(broken, "old")
            self.assertEqual(artifacts.read_member(old_ref), b"previous complete evidence")
            names = artifacts.list_members(broken)
            self.assertEqual("new" in names, "new-index" in names)
            if "new" in names:
                self.assertEqual(artifacts.read_member(artifacts.reference(broken, "new")), b"new evidence")
        broken.write_bytes(complete[:len(first) - 1024 + 600])
        artifacts.append_members(broken, {"recovered": b"safe"})
        self.assertEqual(artifacts.list_members(broken), ["old", "recovered"])

    def test_first_torn_transaction_is_empty_and_recoverable(self):
        self.archive.write_bytes(b"DATA/unfinished")
        self.assertEqual(artifacts.list_members(self.archive), [])
        artifacts.append_members(self.archive, {"recovered": b"yes"})
        self.assertEqual(artifacts.read_member(self.ref("recovered")), b"yes")

    def test_committed_payload_and_commit_corruption_fail(self):
        artifacts.append_members(self.archive, {"mask": b"original"})
        info = artifacts.member_info(self.ref("mask"))
        original = self.archive.read_bytes()
        with self.archive.open("r+b") as stream:
            stream.seek(info["offset"])
            stream.write(b"X")
        with self.assertRaisesRegex(artifacts.ArchiveError, "checksum"):
            artifacts.read_member(self.ref("mask"))
        self.archive.write_bytes(original.replace(b'"schema":"xta.artifact-archive/1"',
                                                 b'"schema":"zta.artifact-archive/1"'))
        with self.assertRaises(artifacts.ArchiveError):
            artifacts.list_members(self.archive)

    def test_corrupt_committed_headers_are_never_treated_as_recoverable_tails(self):
        artifacts.append_members(self.archive, {"first": b"one"})
        artifacts.append_members(self.archive, {"second": b"two"})
        original = self.archive.read_bytes()
        with tarfile.open(self.archive) as archive:
            second_offset = next(item.offset for item in archive.getmembers()
                                 if item.name.endswith("/second"))
        for offset in (0, second_offset):
            broken = bytearray(original)
            broken[offset] ^= 1
            self.archive.write_bytes(broken)
            with self.subTest(offset=offset):
                with self.assertRaises(artifacts.ArchiveError):
                    artifacts.list_members(self.archive)
                with self.assertRaises(artifacts.ArchiveError):
                    artifacts.append_members(self.archive, {"must-not-write": b"no"})
                self.assertEqual(self.archive.read_bytes(), broken)

    def test_negative_pax_size_is_rejected(self):
        header = tarfile.TarInfo("pax")
        header.type, header.size = tarfile.XHDTYPE, -512
        self.archive.write_bytes(header.tobuf(format=tarfile.GNU_FORMAT))
        with self.assertRaisesRegex(artifacts.ArchiveError, "negative size"):
            artifacts.list_members(self.archive)

    def test_other_process_waits_until_publication_succeeds_or_rolls_back(self):
        artifacts.append_members(self.archive, {"old": b"committed"})
        ready, release = threading.Event(), threading.Event()
        real_fsync = os.fsync
        calls = 0
        def paused_fsync(fd):
            nonlocal calls
            real_fsync(fd)
            calls += 1
            if calls == 2:
                ready.set()
                if not release.wait(10):
                    raise TimeoutError("test publication was not released")
                raise OSError("injected final fsync failure")
        marker = self.root / "reader-started"
        code = ("import json,sys; from pathlib import Path; from XTA.artifact_archive import list_members; "
                "Path(sys.argv[2]).write_text('ready'); print(json.dumps(list_members(sys.argv[1])))")
        with mock.patch.object(artifacts.os, "fsync", side_effect=paused_fsync), ThreadPoolExecutor(max_workers=1) as pool:
            writer = pool.submit(artifacts.append_members, self.archive, {"new": b"provisional"})
            reader = None
            try:
                self.assertTrue(ready.wait(10))
                reader = subprocess.Popen([sys.executable, "-B", "-c", code, str(self.archive), str(marker)],
                                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                deadline = time.monotonic() + 10
                while not marker.exists() and time.monotonic() < deadline:
                    time.sleep(.01)
                self.assertTrue(marker.exists())
                self.assertIsNone(reader.poll())
            finally:
                release.set()
            with self.assertRaises(OSError):
                writer.result(timeout=10)
            stdout, stderr = reader.communicate(timeout=10)
            self.assertEqual(reader.returncode, 0, stderr)
            self.assertEqual(stdout.strip(), '["old"]')

    def test_duplicate_replacement_is_explicit_and_prefix_publication_is_fresh(self):
        artifacts.append_members(self.archive, {"scope/selection.json": b"one"})
        with self.assertRaises(artifacts.ArchiveError):
            artifacts.append_members(self.archive, {"scope/selection.json": b"two"})
        artifacts.append_members(self.archive, {"scope/selection.json": b"two"}, replace=True)
        self.assertEqual(artifacts.read_member(self.ref("scope/selection.json")), b"two")
        stage = self.root / "stage"
        stage.mkdir()
        (stage / "new").write_bytes(b"keep")
        with self.assertRaises(FileExistsError):
            artifacts.publish_directory(stage, self.ref("scope"))
        self.assertEqual((stage / "new").read_bytes(), b"keep")

    def test_concurrent_producers_publish_whole_transactions(self):
        def publish(index):
            artifacts.append_members(self.archive, {f"scope{index}/mask": str(index).encode(),
                                                    f"scope{index}/index": b"{}"})
        with ThreadPoolExecutor(max_workers=6) as pool:
            list(pool.map(publish, range(20)))
        self.assertEqual(len(artifacts.list_members(self.archive)), 40)
        for index in range(20):
            self.assertEqual(artifacts.read_member(self.ref(f"scope{index}/mask")), str(index).encode())

    def test_failed_copy_preserves_preceding_commit(self):
        artifacts.append_members(self.archive, {"old": b"safe"})
        original = artifacts._write_member
        def fail(stream, name, source):
            result = original(stream, name, source)
            if name.endswith("bad"):
                raise OSError("injected failure")
            return result
        with mock.patch.object(artifacts, "_write_member", side_effect=fail):
            with self.assertRaises(OSError):
                artifacts.append_members(self.archive, {"bad": b"no"})
        self.assertEqual(artifacts.list_members(self.archive), ["old"])
        self.assertEqual(artifacts.read_member(self.ref("old")), b"safe")

    def test_legacy_bridges_discovery_and_directory_context(self):
        legacy = self.root / "legacy"
        with artifacts.artifact_directory(legacy) as stage:
            self.assertEqual(stage, legacy)
            artifacts.write_artifact(stage / "file.json", b"{}")
        self.assertEqual(artifacts.read_artifact(legacy / "file.json"), b"{}")
        self.assertEqual(list(artifacts.iter_artifacts(legacy, "*.json")), [legacy / "file.json"])
        with artifacts.artifact_directory(self.ref("scope"), temp_root=self.root) as stage:
            actual_stage = stage
            (stage / "file.json").write_bytes(b"{}")
            (stage / "mask.bin").write_bytes(b"packed")
        self.assertFalse(actual_stage.exists())
        self.assertTrue(artifacts.artifact_exists(Path(self.ref("scope"))))
        self.assertEqual(artifacts.artifact_size(self.ref("scope/mask.bin")), 6)
        self.assertEqual(list(artifacts.iter_artifacts(self.ref("scope"), "*.json")),
                         [Path(self.ref("scope/file.json"))])
        self.assertEqual(artifacts.split_reference(str(Path(self.ref("scope/mask.bin"))))[1], "scope/mask.bin")
        with self.assertRaises(artifacts.ArchiveError):
            artifacts.read_artifact(self.ref("scope/mask.bin"), max_bytes=5)
        with self.assertRaisesRegex(artifacts.ArchiveError, "recovery files preserved"):
            with artifacts.artifact_directory(self.ref("failed"), temp_root=self.root) as stage:
                failed_stage = stage
                (stage / "mask").write_bytes(b"recover me")
                raise OSError("producer failed")
        self.assertEqual((failed_stage / "mask").read_bytes(), b"recover me")
        self.assertFalse(artifacts.artifact_exists(self.ref("failed")))

    def test_unsafe_and_file_directory_conflicting_names_are_refused(self):
        for name in ("../escape", "/absolute", "folder/../escape", "NUL", "C:/drive"):
            with self.subTest(name=name), self.assertRaises(ValueError):
                artifacts.append_members(self.archive, {name: b"no"})
        with self.assertRaises(artifacts.ArchiveError):
            artifacts.append_members(self.archive, {"a": b"file", "a/b": b"child"})
        with self.assertRaisesRegex(artifacts.ArchiveError, "source must be a regular file"):
            artifacts.append_members(self.archive, {"directory": self.root})


if __name__ == "__main__":
    unittest.main()

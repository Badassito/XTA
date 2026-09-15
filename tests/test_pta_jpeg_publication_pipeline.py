"""Host-only nvJPEG ownership, fencing and asynchronous file publication."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError
import pickle
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA import pta_publication as publication


def jpeg(tag=b"payload"):
    return b"\xff\xd8" + tag + b"\xff\xd9"


class CodeStream(bytearray):
    pass


class JpegPublicationPipelineTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="xta-pta-jpeg-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        quarantine = mock.patch.object(publication, "_NVJPEG_QUARANTINED_BATCHES", [])
        quarantine.start()
        self.addCleanup(quarantine.stop)

    def paths(self, count=2):
        return tuple(self.root / "images" / f"sample-{i}.jpg" for i in range(count))

    def encode(self, payloads, *, paths=None, encoder=None, stream=None, device=None, input_owners=()):
        return publication._encode_nvjpeg_batch(
            encoder=encoder or SimpleNamespace(encode=lambda *args, **kwargs: payloads),
            images=[object() for _ in payloads], final_paths=paths or self.paths(len(payloads)),
            params=object(), cuda_stream=stream or SimpleNamespace(cuda_stream=7, synchronize=lambda: None),
            synchronize_device=device, input_owners=input_owners)

    def assert_no_stages(self):
        self.assertEqual(list(self.root.rglob(".*.nvjpeg.*.jpg")), [])

    def test_encode_returns_immutable_owned_bytes_after_both_fences_without_files(self):
        order = []
        backing = CodeStream(jpeg(b"first"))
        backing.size = len(backing)
        encoder = SimpleNamespace(encode=lambda *args, **kwargs: (order.append("encode") or [backing]))
        stream = SimpleNamespace(cuda_stream=13, synchronize=lambda: order.append("stream"))
        batch = self.encode([backing], encoder=encoder, stream=stream, device=lambda: order.append("device"))
        self.assertEqual(order, ["encode", "stream", "device"])
        self.assertEqual(batch.payloads, (jpeg(b"first"),))
        self.assertIs(type(batch.payloads[0]), bytes)
        self.assertEqual(batch.nbytes, len(jpeg(b"first")))
        self.assertFalse((self.root / "images").exists())
        backing[:] = jpeg(b"later")
        self.assertEqual(batch.payloads, (jpeg(b"first"),))
        with self.assertRaises(FrozenInstanceError):
            batch.payloads = ()
        publication._publish_nvjpeg_batch_atomically(batch)
        self.assertEqual(batch.final_paths[0].read_bytes(), jpeg(b"first"))
        self.assert_no_stages()

    def test_encode_count_size_and_corrupt_payload_fail_before_any_publication(self):
        invalid_size = CodeStream(jpeg())
        invalid_size.size = len(invalid_size) + 1
        for payloads, pattern in (
                ([jpeg(), None], "failed JPEG batch"), ([jpeg(), b""], "empty/truncated"),
                ([jpeg(), b"xxxx"], "invalid JPEG markers"),
                ([jpeg(), object()], "non-buffer"), ([jpeg(), invalid_size], "invalid JPEG CodeStream size")):
            with self.subTest(pattern=pattern):
                with self.assertRaisesRegex(RuntimeError, pattern):
                    self.encode(payloads)
                self.assertFalse((self.root / "images").exists())
                self.assert_no_stages()
        with self.assertRaisesRegex(RuntimeError, "1 result.*2 JPEG"):
            self.encode([jpeg(), jpeg()], encoder=SimpleNamespace(encode=lambda *a, **k: [jpeg()]))

    def test_duplicate_resolved_paths_are_rejected_before_encoding(self):
        encoder = SimpleNamespace(encode=mock.Mock(side_effect=AssertionError("encoded")))
        paths = (self.root / "images" / "one.jpg", self.root / "images" / ".." / "images" / "one.jpg")
        with self.assertRaisesRegex(ValueError, "duplicate"):
            self.encode([jpeg(), jpeg()], paths=paths, encoder=encoder)
        encoder.encode.assert_not_called()
        self.assertFalse((self.root / "images").exists())

    def test_encode_exception_still_drains_stream_and_device_before_reraising(self):
        order = []
        failure = RuntimeError("encoder failed after enqueue")
        def encode(*args, **kwargs):
            order.append("encode")
            raise failure
        stream = SimpleNamespace(cuda_stream=7, synchronize=lambda: order.append("stream"))
        with self.assertRaises(RuntimeError) as caught:
            self.encode([jpeg()], encoder=SimpleNamespace(encode=encode), stream=stream,
                        device=lambda: order.append("device"))
        self.assertIs(caught.exception, failure)
        self.assertEqual(order, ["encode", "stream", "device"])
        self.assertEqual(publication._NVJPEG_QUARANTINED_BATCHES, [])
        self.assertFalse((self.root / "images").exists())

    def test_failed_fence_quarantines_gpu_owners_and_sends_only_picklable_error(self):
        for encode_fails in (False, True):
            for failed_fence in ("stream", "device"):
                with self.subTest(encode_fails=encode_fails, failed_fence=failed_fence), \
                     mock.patch.object(publication, "_NVJPEG_QUARANTINED_BATCHES", []):
                    order, images, owners = [], [object()], (object(),)
                    streams = [CodeStream(jpeg())]
                    def encode(*args, **kwargs):
                        order.append("encode")
                        if encode_fails:
                            raise RuntimeError("encoder failure")
                        return streams
                    def fence(name):
                        order.append(name)
                        if failed_fence == name:
                            raise RuntimeError("unfenced native work")
                    encoder = SimpleNamespace(encode=mock.Mock(side_effect=encode))
                    stream = SimpleNamespace(cuda_stream=7, synchronize=lambda: fence("stream"))
                    with self.assertRaises(publication.NvjpegCudaFenceError) as caught:
                        publication._encode_nvjpeg_batch(encoder=encoder, images=images,
                            final_paths=self.paths(1), params=None, cuda_stream=stream,
                            synchronize_device=lambda: fence("device"), input_owners=owners)
                    self.assertIsInstance(caught.exception, RuntimeError)
                    self.assertEqual(pickle.loads(pickle.dumps(caught.exception)).args, caught.exception.args)
                    self.assertEqual(caught.exception.__dict__, {})
                    self.assertEqual(order, ["encode", "stream", "device"])
                    retained = publication._NVJPEG_QUARANTINED_BATCHES[0]
                    self.assertIs(retained[0], encoder)
                    self.assertIs(retained[1][0], images[0])
                    self.assertIs(retained[2][0], owners[0])
                    self.assertIs(retained[4], stream)
                    self.assertIs(retained[6], None if encode_fails else streams)
                    with self.assertRaisesRegex(publication.NvjpegCudaFenceError, "previous"):
                        self.encode([jpeg()], encoder=encoder)
                    encoder.encode.assert_called_once()
                    self.assertFalse((self.root / "images").exists())

    def test_checked_parallel_writes_finish_before_any_rename(self):
        batch = self.encode([jpeg(b"one"), jpeg(b"two")])
        entered, release = threading.Event(), threading.Event()
        validate = publication._validate_jpeg_file
        rename = publication.os.replace
        def validate_stage(path, **kwargs):
            if path.name.startswith(".sample-1."):
                entered.set()
                self.assertTrue(release.wait(3))
            return validate(path, **kwargs)
        try:
            with ThreadPoolExecutor(max_workers=2) as files, ThreadPoolExecutor(max_workers=1) as batches, \
                 mock.patch.object(publication, "_validate_jpeg_file", side_effect=validate_stage), \
                 mock.patch.object(publication.os, "replace", wraps=rename) as renamed:
                future = batches.submit(publication._publish_nvjpeg_batch_atomically, batch, executor=files)
                self.assertTrue(entered.wait(3))
                renamed.assert_not_called()
                self.assertFalse(future.done())
                # Encoding the next batch can complete without waiting for this
                # batch's filesystem publication or retaining its CUDA inputs.
                next_batch = self.encode([jpeg(b"next")], paths=(self.root / "next.jpg",))
                self.assertEqual(next_batch.payloads, (jpeg(b"next"),))
                release.set()
                future.result(timeout=3)
            for path, payload in zip(batch.final_paths, batch.payloads):
                self.assertEqual(path.read_bytes(), payload)
            self.assert_no_stages()
        finally:
            release.set()

    def test_short_write_preserves_existing_final_and_removes_private_stage(self):
        batch = self.encode([jpeg()])
        final = batch.final_paths[0]
        final.parent.mkdir(parents=True)
        final.write_bytes(b"previous")
        original_open = Path.open
        class ShortWriter:
            def __init__(self, handle):
                self.handle = handle
            def __enter__(self):
                return self
            def __exit__(self, *args):
                self.handle.close()
            def write(self, payload):
                return self.handle.write(payload[:-1])
        def open_file(path, mode="r", *args, **kwargs):
            handle = original_open(path, mode, *args, **kwargs)
            return ShortWriter(handle) if mode == "wb" and ".nvjpeg." in path.name else handle
        with mock.patch.object(Path, "open", side_effect=open_file, autospec=True):
            with self.assertRaisesRegex(RuntimeError, "write was short"):
                publication._publish_nvjpeg_batch_atomically(batch)
        self.assertEqual(final.read_bytes(), b"previous")
        self.assert_no_stages()

    def test_post_write_corruption_publishes_nothing(self):
        batch = self.encode([jpeg(b"one"), jpeg(b"two")])
        original = publication._validate_jpeg_file
        def corrupt(path, **kwargs):
            if path.name.startswith(".sample-1."):
                path.write_bytes(b"broken")
            return original(path, **kwargs)
        with mock.patch.object(publication, "_validate_jpeg_file", side_effect=corrupt):
            with self.assertRaisesRegex(RuntimeError, "invalid JPEG marker"):
                publication._publish_nvjpeg_batch_atomically(batch)
        self.assertTrue(all(not path.exists() for path in batch.final_paths))
        self.assert_no_stages()

    def test_failed_rename_keeps_published_prefix_and_never_reports_success(self):
        batch = self.encode([jpeg(b"one"), jpeg(b"two")])
        for path in batch.final_paths:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"previous")
        original = publication.os.replace
        def rename(stage, final):
            if final == batch.final_paths[1]:
                raise OSError("injected rename failure")
            original(stage, final)
        with mock.patch.object(publication.os, "replace", side_effect=rename):
            with self.assertRaisesRegex(OSError, "injected rename"):
                publication._publish_nvjpeg_batch_atomically(batch)
        self.assertEqual(batch.final_paths[0].read_bytes(), batch.payloads[0])
        self.assertEqual(batch.final_paths[1].read_bytes(), b"previous")
        self.assert_no_stages()

    def test_submission_failure_drains_running_writes_before_cleanup(self):
        batch = self.encode([jpeg(b"one"), jpeg(b"two")])
        started, release, settled = threading.Event(), threading.Event(), threading.Event()
        validate = publication._validate_jpeg_file
        def validation(path, **kwargs):
            started.set()
            self.assertTrue(release.wait(3))
            try:
                return validate(path, **kwargs)
            finally:
                settled.set()
        class FailSecondSubmission:
            def __init__(self, pool):
                self.pool, self.count = pool, 0
            def submit(self, fn, *args):
                self.count += 1
                if self.count == 2:
                    if not started.wait(3):
                        raise AssertionError("first writer did not start")
                    raise RuntimeError("injected file executor submission failure")
                return self.pool.submit(fn, *args)
        try:
            with ThreadPoolExecutor(max_workers=1) as files, ThreadPoolExecutor(max_workers=1) as batches, \
                 mock.patch.object(publication, "_validate_jpeg_file", side_effect=validation):
                future = batches.submit(publication._publish_nvjpeg_batch_atomically, batch,
                                        executor=FailSecondSubmission(files))
                self.assertTrue(started.wait(3))
                self.assertFalse(future.done())
                release.set()
                with self.assertRaisesRegex(RuntimeError, "executor submission failure"):
                    future.result(timeout=3)
                self.assertTrue(settled.is_set())
            self.assertTrue(all(not path.exists() for path in batch.final_paths))
            self.assert_no_stages()
        finally:
            release.set()


if __name__ == "__main__":
    unittest.main()

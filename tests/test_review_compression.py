from concurrent.futures import ThreadPoolExecutor
import gzip
import sys
from threading import Event
from types import SimpleNamespace

import pytest

from XTA import review_compression
from XTA.gzip_spans import PARALLEL_GZIP_MEMBER_BYTES as MIB


@pytest.mark.parametrize(("affinity", "allocation", "requested", "expected"), [
    (160, "160", None, 32),
    (160, "9", "64", 9),
    (6, "160", "64", 6),
    (160, "160", "1", 1),
])
def test_review_worker_limit(monkeypatch, affinity, allocation, requested, expected):
    monkeypatch.setattr(review_compression, "_affinity_count", lambda: affinity)
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", allocation)
    if requested is None:
        monkeypatch.delenv("YOLO_TTA_NRRD_GZIP_WORKERS", raising=False)
    else:
        monkeypatch.setenv("YOLO_TTA_NRRD_GZIP_WORKERS", requested)
    assert review_compression._review_gzip_workers() == expected


@pytest.mark.parametrize(("headroom", "payload", "plan"), [
    (4 * MIB, 100 * MIB, (0, 0)),
    (5 * MIB, 64 * 1024, (0, 0)),
    (5 * MIB, MIB // 2, (1, 3 * MIB)),
    (5 * MIB, 100 * MIB, (1, 5 * MIB)),
    (100 * MIB, MIB + 1, (2, 8 * MIB)),
    (100 * MIB, 100 * MIB, (4, 14 * MIB)),
])
def test_review_pool_reserves_input_output_and_codec_headroom(headroom, payload, plan):
    pool = review_compression.ReviewGzipPool(None, None, "libdeflate", max_workers=4)
    assert pool.plan(headroom, payload) == plan


@pytest.mark.parametrize(("version", "broken", "backend"), [
    ("0.9.0", False, "libdeflate"),
    ("0.8.1", False, "stdlib"),
    ("local", False, "stdlib"),
    ("0.9.0", True, "stdlib"),
    (None, False, "stdlib"),
])
def test_review_pool_falls_back_at_admission_and_settles(monkeypatch, version, broken, backend):
    encoder = (
        (lambda payload, level: bytes(payload)) if broken else
        (lambda payload, level: gzip.compress(payload, compresslevel=level, mtime=0))
    )
    module = None if version is None else SimpleNamespace(__version__=version, gzip_compress=encoder)
    monkeypatch.setitem(sys.modules, "deflate", module)
    monkeypatch.setattr(review_compression, "_review_gzip_workers", lambda: 2)
    chunks = (b"first" * 31, b"last\0" * 17)
    with review_compression.review_gzip_pool() as pool:
        assert pool.backend == backend
        assert pool.max_workers == 2
        pending = [pool.submit(chunk, 3) for chunk in chunks]
    assert [gzip.decompress(member.result()) for member in pending] == list(chunks)
    with pytest.raises(RuntimeError, match="shutdown"):
        pool.submit(b"too late", 3)


def test_review_pool_propagates_failure_after_admission(monkeypatch):
    def encoder(payload, level):
        if payload == b"bad":
            raise ValueError("injected data error")
        return gzip.compress(payload, compresslevel=level, mtime=0)

    module = SimpleNamespace(__version__="0.9.0", gzip_compress=encoder)
    monkeypatch.setitem(sys.modules, "deflate", module)
    with review_compression.review_gzip_pool() as pool:
        assert pool.backend == "libdeflate"
        with pytest.raises(ValueError, match="injected data error"):
            pool.submit(b"bad", 3).result()


def test_review_pool_counts_synchronous_writers_toward_worker_limit():
    pooled_entered, release_pooled = Event(), Event()
    producer_started, producer_entered = Event(), Event()

    def encode(payload, level):
        if payload == b"pool":
            pooled_entered.set()
            assert release_pooled.wait(5)
        else:
            producer_entered.set()
        return payload

    def producer():
        producer_started.set()
        return pool.compress(b"sync", 3)

    with ThreadPoolExecutor(max_workers=1) as executor, ThreadPoolExecutor(max_workers=1) as cutoff:
        pool = review_compression.ReviewGzipPool(executor, encode, "libdeflate", max_workers=1)
        pooled = pool.submit(b"pool", 3)
        try:
            assert pooled_entered.wait(5)
            synchronous = cutoff.submit(producer)
            assert producer_started.wait(5)
            assert not producer_entered.wait(.1)
        finally:
            release_pooled.set()
        assert pooled.result(timeout=5) == b"pool"
        assert synchronous.result(timeout=5) == b"sync"

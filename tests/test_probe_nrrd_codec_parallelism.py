"""Check the CPU-only codec probe's bounds and gzip integrity."""
import gzip

import pytest

from tools import probe_nrrd_codec_parallelism as probe


@pytest.mark.parametrize('seconds,workers', [
    (0, 4), (-1, 4), (probe.MAX_SECONDS + .01, 4),
    (1, 0), (1, -1), (1, probe.MAX_PARALLEL_WORKERS + 1),
])
def test_rejects_unbounded_or_invalid_work(seconds, workers):
    with pytest.raises(ValueError):
        probe.validate_options(seconds, workers)


def test_parallelism_respects_allowed_cpus():
    assert probe.effective_parallel_workers(8, 2) == 2
    assert probe.effective_parallel_workers(4, 16) == 4
    assert probe.effective_parallel_workers(4, 0) == 1


def test_fixed_payload_and_zlib_gzip_roundtrip():
    payload = probe.make_payload()
    assert len(payload) == 2 * 1024 * 1024
    assert payload == probe.make_payload()
    encoded = probe.zlib_gzip_compress(payload[:64 * 1024])
    assert gzip.decompress(encoded) == payload[:64 * 1024]
    assert probe.verify_codecs({'zlib_gzip': probe.zlib_gzip_compress}, payload) == {
        'zlib_gzip': len(encoded),
    }

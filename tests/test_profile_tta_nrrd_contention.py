"""CPU-only contention wrapper safety and exact-output helpers."""
from __future__ import annotations

import gzip
import hashlib

import pytest

from tools.profile_tta_nrrd_contention import _decoded_nrrd_sha256, plan


def test_plan_is_large_only_when_explicitly_executed(tmp_path):
    result = plan(trace=tmp_path / 'trace.jsonl', output_root=tmp_path / 'output',
        deps_dir=tmp_path / 'deps', shape=(256, 2048, 2048),
        load_workers=8, sink_jobs=12, line_window_seconds=30,
        max_tasks=1000, pre_replay_warmup_seconds=5)
    assert result['logical_input_bytes'] == 1 << 30
    assert result['gzip_workers'] == 64
    assert result['fill_workers'] == 32
    assert result['sink_workers'] == 12
    assert result['load_workers'] == 8 and result['sink_jobs'] == 12
    assert result['max_tasks'] == 1000 and result['mirrors'] is False
    assert not (tmp_path / 'output').exists()


def test_plan_rejects_invalid_bounds(tmp_path):
    common = dict(trace=tmp_path / 'trace', output_root=tmp_path / 'out',
        deps_dir=tmp_path / 'deps', shape=(1, 8, 8), load_workers=1,
        sink_jobs=0, line_window_seconds=1)
    with pytest.raises(ValueError):
        plan(**{**common, 'shape': (0, 8, 8)})
    with pytest.raises(ValueError):
        plan(**{**common, 'pre_replay_warmup_seconds': -1})


def test_decoded_nrrd_hash_uses_gzip_payload_after_header(tmp_path):
    payload = bytes(range(256)) * 32
    path = tmp_path / 'fixture.seg.nrrd'
    path.write_bytes(b'NRRD0005\ntype: uint8\nencoding: gzip\n\n' + gzip.compress(payload))
    assert _decoded_nrrd_sha256(path) == hashlib.sha256(payload).hexdigest()

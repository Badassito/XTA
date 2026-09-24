"""Bounded CPU NRRD output profiler plans and exact small-path checks."""
import gzip
import json
import sys
import threading
from types import ModuleType
from unittest import mock

from tools import profile_nrrd_output as profiler


def test_plan_is_bounded_and_does_not_allocate_fixture(tmp_path):
    output = tmp_path / 'evidence'
    result = profiler.plan(output_dir=output, deps_dir=tmp_path / 'deps',
        shape=(256, 2048, 2048), jobs=12, sink_workers=12,
        gzip_workers=80, fill_workers=32)
    assert result['logical_bytes_per_file'] == 1 << 30
    assert result['total_jobs'] == 24 and result['sink_workers'] == 12
    assert result['gzip_workers'] == 80 and result['fill_workers'] == 32
    assert result['gpu_used'] is False
    assert not output.exists()


def test_thread_timings_group_wait_totals_and_maximum():
    metrics = profiler.ThreadTimings()
    metrics.current()['lock_acquire_ns'] = 3_000_000
    metrics.current()['lock_acquire_max_ns'] = 2_000_000
    thread = threading.Thread(name='nrrd-gzip-libdeflate-0',
        target=lambda: metrics.current().update(lock_acquire_ns=5_000_000,
                                                lock_acquire_max_ns=4_000_000))
    thread.start()
    thread.join()
    report = metrics.report()
    assert report['other']['lock_acquire_seconds'] == .003
    assert report['gzip']['lock_acquire_max_seconds'] == .004


def test_encoded_identity_reuses_exact_decoded_proof(tmp_path):
    first, second, different = [tmp_path / f'{name}.nrrd' for name in ('a', 'b', 'c')]
    payload = gzip.compress(b'correct' * 100, mtime=0)
    first.write_bytes(b'NRRD0005\ncontent: a\n\n' + payload)
    second.write_bytes(b'NRRD0005\ncontent: b\n\n' + payload)
    different.write_bytes(b'NRRD0005\ncontent: c\n\n' + gzip.compress(b'changed', mtime=0))
    decoded, encoded, count = profiler._verify_nrrd_files([first, second, different], {})
    assert count == 2
    assert encoded[first.name] == encoded[second.name] != encoded[different.name]
    assert decoded[first.name] == decoded[second.name] != decoded[different.name]


def test_small_real_sink_path_decodes_exactly_in_both_modes(tmp_path):
    fake = ModuleType('deflate')
    fake.gzip_compress = lambda payload, level: gzip.compress(bytes(payload),
        compresslevel=min(9, max(1, int(level))), mtime=0)
    with mock.patch.dict(sys.modules, {'deflate': fake}), \
            mock.patch.dict('os.environ', {
                'YOLO_TTA_NRRD_MEMBER_CODEC': 'libdeflate',
                'YOLO_TTA_NRRD_GZIP_WORKERS': '2',
                'YOLO_TTA_NRRD_FILL_WORKERS': '2',
            }):
        fixture = profiler.OutputLoadFixture(tmp_path, (4, 16, 16))
        receipt = tmp_path / 'fixture-receipt.json'
        receipt.write_text(json.dumps(dict(shape_tyx=list(fixture.ref.shape),
            decoded_sha256=fixture.expected,
            segment_extent_ijk=list(fixture.ref.segment_extent_ijk))))
        existing = profiler.ExistingCvolFixture(fixture.ref.path, receipt)
        assert existing.expected == fixture.expected
        assert existing.ref.shape == fixture.ref.shape
        specs, warnings = profiler.outputs.resolve_low_quality_downbin_specs(
            '0.5', True, fixture.ref.shape)
        assert not warnings and len(specs) == 1
        mirror_expected = profiler._expected_mirror_sha256(
            fixture.ref, fixture.ref.shape, specs[0].output_shape_t_y_x)
        for mode in ('off', 'on'):
            result = profiler._run_mode(mode=mode, root=tmp_path,
                evidence_root=tmp_path, load=fixture, jobs=1,
                sink_workers=1, deflate_module=fake,
                mirror_spec=specs[0], expected_mirror=mirror_expected)
            assert result['all_exact']
            assert result['mirror_sha256']
            assert result['timings_by_thread_group']['gzip']['member_calls'] >= 1
            assert result['timings_by_thread_group']['gzip']['native_calls'] >= 1
        assert (tmp_path / 'telemetry-on.jsonl').is_file()

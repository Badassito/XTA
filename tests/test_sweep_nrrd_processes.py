"""Sibling NRRD layer files written by worker processes equal the sink thread path.

The sink keeps names, colours, manifest rows and publication records; a worker
receives the pinned plans and returns the plans it wrote. Every file, manifest and
record must match the thread path byte for byte in both layouts, through an
extent-violation fallback, a dead worker and a codec the worker cannot match.
"""
from __future__ import annotations

import contextlib
import gzip
import hashlib
import io
import json
import os
from pathlib import Path
import signal
import time
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import nrrd_compression, outputs, sweep_nrrd_processes
from XTA.interpolation import (
    CVOL_FORMAT,
    INTERNAL_PACKED_CVOL_FORMAT,
    NrrdLayerRef,
    write_raw_bbox_mask_store,
)
from XTA.sweep_nrrd_processes import (
    PROCESSES_ENV,
    SweepNrrdProcessPool,
    sweep_nrrd_process_count,
    sweep_nrrd_worker_gzip_workers,
)

SHAPE = (13, 37, 41)
LOW_QUALITY = (
    outputs.LowQualityDownbinSpec('0.5', '0p5', .5, (6, 18, 20)),
    outputs.LowQualityDownbinSpec('0.2', '0p2', .2, (2, 8, 8)),
)


@pytest.fixture(autouse=True)
def software_codec(monkeypatch):
    # A real, named codec: the worker selects it itself (no in-process mock reaches it).
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_NRRD_GPU_MIRROR_TEE', '0')
    monkeypatch.setenv('YOLO_TTA_NRRD_MEMBER_CODEC', 'zlib')
    monkeypatch.setenv('YOLO_TTA_NRRD_GZIP_CHUNK_MIB', '1')
    monkeypatch.delenv('YOLO_TTA_NRRD_LAYER_ZSHARDS', raising=False)


@pytest.fixture(scope='module')
def pool():
    workers = SweepNrrdProcessPool(2, warm=False)
    try:
        yield workers
    finally:
        workers.close()


class _CountingPool:
    """Delegate to a real pool and count the layers its workers wrote."""

    def __init__(self, pool):
        self.pool, self.remote = pool, 0

    def accepts(self, ref, z_shards):
        return self.pool.accepts(ref, z_shards)

    def write_layer_files(self, write_args):
        plans = self.pool.write_layer_files(write_args)
        self.remote += plans is not None
        return plans


def _store(root, name, volume, fmt=INTERNAL_PACKED_CVOL_FORMAT, *, extent=None, **fields):
    path = Path(root) / f'{name}.cvol'
    with contextlib.redirect_stdout(io.StringIO()):
        stats = write_raw_bbox_mask_store(volume, path, format_name=fmt, desc='test', workers=1)
    return NrrdLayerRef(key=name, name=name, path=path, shape=volume.shape, storage_format=fmt,
                        segment_extent_ijk=tuple(extent or stats['segment_extent_ijk']),
                        segment_extent_shape_tyx=volume.shape, segment_extent_source='raw_bbox_cvol_index',
                        view_name=name, source=fields.pop('source', 'fullframe'),
                        mask_kind=fields.pop('mask_kind', 'yolo'), **fields)


def _random_layers(root, seed):
    """Projected-store-shaped layers: random sparse boxes, empty, corner, fallback, ineligible."""
    rng = np.random.default_rng(seed)
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    layers = []
    for index in range(5):
        volume = np.zeros(SHAPE, np.uint8)
        for _ in range(int(rng.integers(1, 4))):
            lo = [int(rng.integers(0, n - 1)) for n in SHAPE]
            hi = [int(rng.integers(a + 1, n + 1)) for a, n in zip(lo, SHAPE)]
            box = tuple(slice(a, b) for a, b in zip(lo, hi))
            volume[box] |= (rng.random(volume[box].shape) < rng.uniform(.05, .9)).astype(np.uint8)
        volume[int(rng.integers(0, SHAPE[0]))] = 0  # an empty slice inside the stack
        fmt = (CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT)[index % 2]
        layers.append((f'random{index}', _store(root, f'random{index}', volume, fmt)))
    layers.append(('empty', _store(root, 'empty', np.zeros(SHAPE, np.uint8))))
    corner = np.zeros(SHAPE, np.uint8)
    corner[SHAPE[0] - 1, SHAPE[1] - 1, SHAPE[2] - 1] = 1
    layers.append(('corner', _store(root, 'corner', corner)))
    # Foreground outside the recorded X/Y extent: cropped writers republish full rasters.
    wrong = np.zeros(SHAPE, np.uint8)
    wrong[2:11, 3:30, 4:36] = rng.random((9, 27, 32)) < .4
    wrong[5, 3, 4] = wrong[6, 29, 35] = 1
    layers.append(('wrong', _store(root, 'wrong', wrong, extent=(12, 20, 9, 18, 2, 10))))
    # Ineligible layers stay on the sink thread in both runs.
    live = np.zeros(SHAPE, np.uint8)
    live[1:9, 2:20, 3:25] = rng.random((8, 18, 22)) < .5
    layers.append(('live', NrrdLayerRef(key='live', name='live', path=root / 'live.live', shape=SHAPE,
                                        storage_format='live_u8', live_array=live, view_name='live',
                                        source='fullframe', mask_kind='yolo')))
    return layers


def _publish(root, layout, monkeypatch, layers, processes=None):
    root = Path(root)
    sink = outputs.NrrdLayerSink(nrrd_dir=root / 'nrrd', stem='case', output_shape_tyx=SHAPE, max_workers=3,
                                 low_quality_specs=list(LOW_QUALITY), low_quality_root=root / 'low_quality',
                                 run_id='run', layer_file_processes=processes, layer_layout=layout)
    records = {}
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            paths = [(name, ref, sink.submit_layer(ref, name)) for name, ref in layers]
            for name, ref, path in paths:
                record = sink.wait_published_layer(ref, name, output_path=path)
                records[name] = {key: value for key, value in record.items() if key != 'files'}
            sink.wait()
        sink.write_manifest()
    finally:
        sink.shutdown()
    files = {path.relative_to(root).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
             for path in sorted(root.rglob('*')) if path.is_file()}
    return files, records


@pytest.mark.parametrize('layout', ['cropped', 'full'])
@pytest.mark.parametrize('seed', [3, 11])
def test_worker_files_rows_and_records_equal_thread_path(tmp_path, monkeypatch, pool, layout, seed):
    layers = _random_layers(tmp_path / 'sources', seed)
    expected = _publish(tmp_path / 'threads', layout, monkeypatch, layers)
    counting = _CountingPool(pool)
    actual = _publish(tmp_path / 'processes', layout, monkeypatch, layers, counting)
    assert actual == expected
    assert counting.remote == len(layers) - 1 and not pool.broken  # every store layer, not the live one
    files, records = actual
    assert len(files) == len(layers) * 3 + 3
    if layout == 'cropped':
        assert records['wrong']['native']['stored_layout'] == 'full_reference'
        assert records['random0']['native']['stored_layout'] == 'segment_bounding_box'
        assert records['empty']['native']['stored_shape_tyx'] == [1, 1, 1]
        assert records['corner']['native']['stored_shape_tyx'] == [1, 1, 2]
    else:
        assert 'stored_layout' not in records['wrong']['native']


def test_unmatched_parent_codec_keeps_every_layer_on_threads(tmp_path, monkeypatch, pool):
    def substitute(payload):  # a compressor no worker can select (mtime pinned: deterministic)
        return gzip.compress(bytes(payload), mtime=0)

    monkeypatch.setattr(nrrd_compression, '_require_nrrd_member_codec', lambda **_: ('zlib', 1, substitute))
    assert sweep_nrrd_processes.nrrd_codec_identity() is None
    layers = _random_layers(tmp_path / 'sources', 5)[:3]
    counting = _CountingPool(pool)
    expected = _publish(tmp_path / 'threads', 'cropped', monkeypatch, layers)
    assert _publish(tmp_path / 'processes', 'cropped', monkeypatch, layers, counting) == expected
    assert counting.remote == 0 and not pool.broken


def test_worker_refusing_the_producer_codec_falls_back_with_identical_bytes(tmp_path, monkeypatch):
    layers = _random_layers(tmp_path / 'sources', 7)
    expected = _publish(tmp_path / 'threads', 'cropped', monkeypatch, layers)
    workers = SweepNrrdProcessPool(1, warm=False)
    try:
        real = sweep_nrrd_processes.nrrd_codec_identity()
        monkeypatch.setattr(sweep_nrrd_processes, 'nrrd_codec_identity', lambda: (real[0], real[1] + 1, real[2]))
        counting = _CountingPool(workers)
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            actual = _publish(tmp_path / 'processes', 'cropped', monkeypatch, layers, counting)
        assert actual == expected and counting.remote == 0
        assert workers.broken and 'NrrdWorkerMismatch' in stderr.getvalue()
    finally:
        workers.close()


def _wait_for_workers(workers, timeout=120.0):
    deadline = time.monotonic() + timeout
    while not workers.worker_pids:
        assert time.monotonic() < deadline, 'NRRD workers never reported ready'
        time.sleep(0.05)
    return list(workers.worker_pids)


def test_dead_worker_joins_pool_and_thread_rewrites_identical_files(tmp_path, monkeypatch):
    layers = _random_layers(tmp_path / 'sources', 9)
    expected = _publish(tmp_path / 'threads', 'cropped', monkeypatch, layers)
    expected_first = _publish(tmp_path / 'threads-first', 'cropped', monkeypatch, layers[:1])
    workers = SweepNrrdProcessPool(1, warm=False)
    try:
        pid = _wait_for_workers(workers)[0]
        os.kill(pid, signal.SIGTERM)
        # The first layer after the death meets the broken executor; a staging file the
        # dead worker left beside that layer's outputs is removed, other files stay.
        native = tmp_path / 'first' / 'nrrd'
        native.mkdir(parents=True)
        basename = outputs.nrrd_layer_file_basename('case', 'random0')
        leftover = native / f'.{basename}.{pid}.1.2.assembling'
        leftover.write_bytes(b'partial')
        unrelated = native / f'.{basename}.{os.getpid()}.1.2.assembling'
        unrelated.write_bytes(b'not a worker file')
        counting = _CountingPool(workers)
        with contextlib.redirect_stderr(io.StringIO()) as stderr:
            first = _publish(tmp_path / 'first', 'cropped', monkeypatch, layers[:1], counting)
        assert workers.broken and counting.remote == 0 and 'BrokenProcessPool' in stderr.getvalue()
        assert not leftover.exists() and unrelated.exists()
        unrelated.unlink()
        first_files = {k: v for k, v in first[0].items() if not k.endswith('.assembling')}
        assert (first_files, first[1]) == expected_first
        # Every later layer is written on the sink threads with identical bytes.
        assert _publish(tmp_path / 'processes', 'cropped', monkeypatch, layers, counting) == expected
        assert counting.remote == 0
    finally:
        workers.close()


def test_cancelled_pool_starts_no_new_layer(tmp_path, monkeypatch):
    workers = SweepNrrdProcessPool(1, warm=False)
    try:
        workers.cancel()
        assert workers.closed and not workers.broken
        layers = _random_layers(tmp_path / 'sources', 1)[:1]
        with pytest.raises(RuntimeError, match='NRRD worker processes are closed'):
            _publish(tmp_path / 'processes', 'cropped', monkeypatch, layers, workers)
        assert not (tmp_path / 'processes' / 'nrrd' / outputs.nrrd_layer_file_basename('case', 'random0')).exists()
    finally:
        workers.close()


def test_eligibility_keeps_live_sharded_and_broken_layers_on_threads(tmp_path):
    store = _store(tmp_path, 'store', np.ones((2, 3, 4), np.uint8))
    fake = SweepNrrdProcessPool.__new__(SweepNrrdProcessPool)
    fake._broken = None
    assert fake.accepts(store, 1)
    assert not fake.accepts(store, 2)
    assert not fake.accepts(NrrdLayerRef(**{**store.__dict__, 'live_array': np.ones((2, 3, 4), np.uint8)}), 1)
    raw = tmp_path / 'raw.u8'
    np.ones((2, 3, 4), np.uint8).tofile(raw)
    assert not fake.accepts(NrrdLayerRef(**{**store.__dict__, 'path': raw, 'storage_format': 'raw_u8'}), 1)
    fake._broken = RuntimeError('dead')
    assert not fake.accepts(store, 1)


def test_process_count_resolution(monkeypatch):
    monkeypatch.delenv(PROCESSES_ENV, raising=False)
    assert sweep_nrrd_process_count(16, 4, cpus=128) == 21
    assert sweep_nrrd_process_count(2, 4, cpus=16) == 4
    assert sweep_nrrd_process_count(1, 2, cpus=128) == 2
    monkeypatch.setenv(PROCESSES_ENV, '0')
    assert sweep_nrrd_process_count(16, 4, cpus=128) == 0
    monkeypatch.setenv(PROCESSES_ENV, '7')
    assert sweep_nrrd_process_count(1, 1, cpus=1) == 7
    monkeypatch.setenv(PROCESSES_ENV, '500')
    assert sweep_nrrd_process_count(1, 1) == (61 if os.name == 'nt' else 500)
    for bad in ('-1', 'many', '1.5'):
        monkeypatch.setenv(PROCESSES_ENV, bad)
        with pytest.raises(ValueError, match=PROCESSES_ENV):
            sweep_nrrd_process_count(1, 1)


def test_worker_gzip_share_splits_the_producer_pool(monkeypatch):
    monkeypatch.setattr(nrrd_compression, '_cpu_count', lambda: 128)
    monkeypatch.delenv('YOLO_TTA_NRRD_GZIP_WORKERS', raising=False)
    assert sweep_nrrd_worker_gzip_workers(21) == 3  # 64-thread producer default over 21 workers
    monkeypatch.setattr(nrrd_compression, '_cpu_count', lambda: 16)
    assert sweep_nrrd_worker_gzip_workers(4) == 2
    monkeypatch.setenv('YOLO_TTA_NRRD_GZIP_WORKERS', '64')
    assert sweep_nrrd_worker_gzip_workers(21) == 3  # an explicit total is split, not copied
    assert sweep_nrrd_worker_gzip_workers(2) == 32
    monkeypatch.setenv('YOLO_TTA_NRRD_GZIP_WORKERS', '3')
    assert sweep_nrrd_worker_gzip_workers(21) == 2
    monkeypatch.setenv('YOLO_TTA_NRRD_LAYER_ZSHARDS', '1')
    keys = dict(sweep_nrrd_processes._job_environment())
    assert 'YOLO_TTA_NRRD_GZIP_WORKERS' not in keys
    assert keys['YOLO_TTA_NRRD_LAYER_ZSHARDS'] == '1' and keys['YOLO_TTA_NRRD_MEMBER_CODEC'] == 'zlib'


def test_worker_keeps_its_gzip_share_over_the_producer_setting(tmp_path, monkeypatch):
    layers = _random_layers(tmp_path / 'sources', 13)[:3]
    expected = _publish(tmp_path / 'threads', 'cropped', monkeypatch, layers)
    # The producer's explicit pool size is inherited at spawn and was replayed by every job.
    monkeypatch.setenv('YOLO_TTA_NRRD_GZIP_WORKERS', '64')
    workers = SweepNrrdProcessPool(1, gzip_workers=3, warm=False)
    try:
        _wait_for_workers(workers)
        assert workers._executor.submit(os.getenv, 'YOLO_TTA_NRRD_GZIP_WORKERS').result() == '3'
        monkeypatch.setenv('YOLO_TTA_NRRD_REPLAY_PROBE', 'replayed')  # set after spawn: reaches the worker by job
        counting = _CountingPool(workers)
        assert _publish(tmp_path / 'processes', 'cropped', monkeypatch, layers, counting) == expected
        assert counting.remote == len(layers) and not workers.broken
        assert workers._executor.submit(os.getenv, 'YOLO_TTA_NRRD_GZIP_WORKERS').result() == '3'
        assert workers._executor.submit(nrrd_compression.nrrd_gzip_workers).result() == 3
        assert workers._executor.submit(os.getenv, 'YOLO_TTA_NRRD_REPLAY_PROBE').result() == 'replayed'
    finally:
        workers.close()


def test_view_major_producer_owns_and_closes_its_nrrd_pool(tmp_path, monkeypatch):
    from XTA import sweep_evidence_processes
    from XTA.tta_sweep_view_publication import ViewMajorPublication, branch_run_id
    created = []

    class FakePool:
        def __init__(self, processes):
            self.processes, self.closed, self.cancelled = processes, False, False
            created.append(self)

        def accepts(self, ref, z_shards):
            return False

        def cancel(self):
            self.cancelled = True

        def close(self):
            self.closed = True

    monkeypatch.setattr(sweep_nrrd_processes, 'SweepNrrdProcessPool', FakePool)
    monkeypatch.setenv(sweep_evidence_processes.PROCESSES_ENV, '0')
    monkeypatch.setenv('XTA_YOLO_SWEEP_PUBLISHERS', '2')
    shape = (3, 8, 8)

    def owner(name, role, processes):
        if role is None:
            monkeypatch.delenv('XTA_YOLO_SWEEP_NATIVE_ROLE', raising=False)
        else:
            monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', role)
        monkeypatch.setenv(PROCESSES_ENV, processes)
        family = tmp_path / name / 'family'
        branch = family / 'confidence_cutoffs' / 'conf_050'
        sink = outputs.NrrdLayerSink(nrrd_dir=branch / 'nrrd', stem='case', output_shape_tyx=shape, max_workers=1,
                                     run_id=branch_run_id('family-one', 50))
        return ViewMajorPublication(family_root=family, workspace=tmp_path / name / 'work', sweep_id='family-one',
            cutoffs_hundredths=(50, 70, 100), current_cutoff=50, scalar_sink=sink, source_shape_tyx=shape,
            model_name='detector', task='segment', min_conf=0., confidence_floor_by_backend={'gpu': .5})

    producer = owner('produce', 'produce', '3')
    assert [pool.processes for pool in created] == [3]
    assert all(sink.layer_file_processes is created[0] for sink in producer._layers.sinks_by_cutoff.values())
    producer.cancel()
    assert created[0].cancelled and not created[0].closed
    producer.close()
    assert created[0].closed
    disabled = owner('disabled', 'produce', '0')
    scalar = owner('scalar', None, '3')
    assert len(created) == 1
    for publication in (disabled, scalar):
        assert all(sink.layer_file_processes is None for sink in publication._layers.sinks_by_cutoff.values())
        publication.close()


def test_producer_falls_back_to_threads_when_worker_pools_cannot_start(tmp_path, monkeypatch, capsys):
    from XTA import sweep_evidence_processes
    from XTA.tta_sweep_view_publication import ViewMajorPublication, branch_run_id

    def refuse(*_args, **_kwargs):
        raise OSError(24, 'Too many open files')

    monkeypatch.setattr(sweep_nrrd_processes, 'SweepNrrdProcessPool', refuse)
    monkeypatch.setattr(sweep_evidence_processes, 'SweepEvidenceProcessPool', refuse)
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'produce')
    monkeypatch.setenv(PROCESSES_ENV, '2')
    monkeypatch.setenv(sweep_evidence_processes.PROCESSES_ENV, '2')
    monkeypatch.setenv('XTA_YOLO_SWEEP_PUBLISHERS', '2')
    shape = (3, 8, 8)
    family = tmp_path / 'family'
    branch = family / 'confidence_cutoffs' / 'conf_050'
    sink = outputs.NrrdLayerSink(nrrd_dir=branch / 'nrrd', stem='case', output_shape_tyx=shape, max_workers=1,
                                 run_id=branch_run_id('family-one', 50))
    publication = ViewMajorPublication(family_root=family, workspace=tmp_path / 'work', sweep_id='family-one',
        cutoffs_hundredths=(50, 70, 100), current_cutoff=50, scalar_sink=sink, source_shape_tyx=shape,
        model_name='detector', task='segment', min_conf=0., confidence_floor_by_backend={'gpu': .5})
    try:
        assert publication._nrrd_processes is None and publication._evidence_processes is None
        assert all(layers.layer_file_processes is None
                   for layers in publication._layers.sinks_by_cutoff.values())
        out = capsys.readouterr().out
        assert 'NRRD files worker pool could not start' in out
        assert 'confidence evidence worker pool could not start' in out
    finally:
        publication.close()
        sink.shutdown()


def test_pool_that_cannot_spawn_every_worker_stops_the_ones_it_started(monkeypatch):
    from concurrent.futures import ProcessPoolExecutor
    from XTA import sweep_evidence_processes
    shut = []
    original = ProcessPoolExecutor.submit

    def failing_submit(self, *args, **kwargs):
        if getattr(self, '_xta_submits', 0) >= 1:
            raise OSError(11, 'Resource temporarily unavailable')
        self._xta_submits = getattr(self, '_xta_submits', 0) + 1
        return original(self, *args, **kwargs)

    original_shutdown = ProcessPoolExecutor.shutdown

    def recording_shutdown(self, *args, **kwargs):
        shut.append(kwargs.get('cancel_futures'))
        return original_shutdown(self, *args, **kwargs)

    monkeypatch.setattr(ProcessPoolExecutor, 'submit', failing_submit)
    monkeypatch.setattr(ProcessPoolExecutor, 'shutdown', recording_shutdown)
    for start in (lambda: SweepNrrdProcessPool(2, gzip_workers=1, warm=False),
                  lambda: sweep_evidence_processes.SweepEvidenceProcessPool(2)):
        with pytest.raises(OSError, match='temporarily unavailable'):
            start()
    assert shut == [True, True]


def test_extent_inversion_matches_the_per_slice_restore_walk():
    rng = np.random.default_rng(23)

    def walk(in_t, out_t, z0, z1):
        lo, hi = max(0, min(z0, in_t - 1)), max(0, min(z1, in_t - 1))
        if hi < lo:
            return None
        hits = [z for z in range(out_t)
                if not (outputs._restore_source_indices_for_output_z(in_t, out_t, z)[-1] < lo
                        or outputs._restore_source_indices_for_output_z(in_t, out_t, z)[0] > hi)]
        return (hits[0], hits[-1]) if hits else None

    pairs = [(1, 1), (1, 9), (9, 1), (3, 100), (100, 3), (1929, 386), (386, 1929)]
    pairs += [tuple(int(v) for v in rng.integers(1, 160, 2)) for _ in range(60)]
    for in_t, out_t in pairs:
        for _ in range(12):
            z0, z1 = (int(v) for v in rng.integers(-2, in_t + 2, 2))
            assert outputs._nrrd_output_z_extent_for_source_extent(in_t, out_t, z0, z1) == walk(in_t, out_t, z0, z1)
        targets = [[] for _ in range(in_t)]
        for mz in range(out_t):
            for sz in outputs._restore_source_indices_for_output_z(in_t, out_t, mz):
                if 0 <= sz < in_t:
                    targets[sz].append(mz)
        assert outputs._nrrd_mirror_targets_by_source(in_t, out_t) == tuple(tuple(v) for v in targets)


def test_view_major_producer_writes_sibling_files_in_workers_identically(tmp_path, monkeypatch):
    """The real producer path: projected sibling stores, worker files, later adoption/retirement."""
    from XTA import sweep_evidence_processes
    from XTA.geometry import ViewInfo
    from XTA.tta_sweep_view_publication import ViewMajorPublication, branch_run_id
    # Count worker jobs here rather than through the process-wide telemetry,
    # which an earlier test may have shut down.
    recorded = []
    monkeypatch.setattr(sweep_nrrd_processes, '_record_nrrd_process',
                        lambda name, value: recorded.append((name, value)))
    for name, value in (('XTA_YOLO_SWEEP_NATIVE_ROLE', 'produce'), (sweep_evidence_processes.PROCESSES_ENV, '0'),
                        ('YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN', '0'),
                        ('XTA_YOLO_SWEEP_PUBLISHERS', '2')):
        monkeypatch.setenv(name, value)
    shape, cuts = (5, 12, 12), (50, 70, 100)
    view = ViewInfo(name='transverse__tta_a0', num_slices=5, src_h=12, src_w=12, pad_mode='pad',
                    family='orthogonal', physical_view_name='transverse', full_t=5, full_h=12, full_w=12,
                    tta_aug_id='a0')
    mask = np.zeros(shape, np.uint8)
    mask[1:4, 2:9, 3:10] = np.random.default_rng(4).random((3, 7, 7)) < .6

    def owner(root, cent):
        family = root / 'family'
        branch = family / 'confidence_cutoffs' / f'conf_{cent:03d}'
        sink = outputs.NrrdLayerSink(nrrd_dir=branch / 'nrrd', stem='case', output_shape_tyx=shape, max_workers=2,
            low_quality_specs=[outputs.LowQualityDownbinSpec('0.5', 'half', .5, (3, 6, 6))],
            low_quality_root=branch / 'low_quality', run_id=branch_run_id('family-one', cent))
        publication = ViewMajorPublication(family_root=family, workspace=root / 'work', sweep_id='family-one',
            cutoffs_hundredths=cuts, current_cutoff=cent, scalar_sink=sink, source_shape_tyx=shape,
            model_name='detector', task='segment', min_conf=0., confidence_floor_by_backend={'gpu': .5})
        return publication, sink

    def crops(cent, z):
        if cent == 70 and mask[z].any():
            yield (0, 12, 0, 12, mask[z], np.full((12, 12), 214, np.uint8))

    def mask_crops(cent, z):
        for y0, y1, x0, x1, plane, _scores in crops(cent, z):
            yield y0, y1, x0, x1, plane

    def no_dense(*_):
        raise AssertionError('sparse readers only')

    results = {}
    for processes in ('0', '2'):
        monkeypatch.setenv(PROCESSES_ENV, processes)
        root = tmp_path / f'processes-{processes}'
        recorded.clear()
        producer, floor = owner(root, 50)
        try:
            assert (producer._nrrd_processes is not None) == (processes != '0')
            with contextlib.redirect_stdout(io.StringIO()):
                refs = producer.publish_view_family(view, native_shape=shape, read_mask_plane=no_dense,
                    read_score_plane=no_dense, read_mask_crops=mask_crops, read_native_crops=crops,
                    counts_by_cutoff={50: 3, 70: 2, 100: 0})
        finally:
            producer.close()
            floor.shutdown()
        jobs = sum(value for name, value in recorded if name == 'jobs')
        assert not any(name == 'thread_fallback_layers' for name, _ in recorded)
        branches = root / 'family' / 'confidence_cutoffs'
        files = {path.relative_to(branches).as_posix(): path.read_bytes()
                 for path in sorted(branches.rglob('*'))
                 if path.is_file() and ('nrrd' in path.parts or 'low_quality' in path.parts)}
        # A later-cut scalar child adopts and retires the store no worker still maps.
        consumer, sink = owner(root, 70)
        try:
            adopted, count = consumer.adopt_view(view)
            assert adopted == refs[70] and count == 2
        finally:
            sink.shutdown()
            consumer.close()
        assert not refs[70].path.exists() and refs[100] is None
        results[processes] = (files, jobs)
    assert results['2'][0] == results['0'][0] and len(results['0'][0]) == 2  # native + LQ; tails write manifests
    assert results['0'][1] == 0 and results['2'][1] == 1

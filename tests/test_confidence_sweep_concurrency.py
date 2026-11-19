"""Concurrent consumer cutoffs of a native-fanout confidence sweep.

Real child processes implement the scalar receipt contract without model
execution. The tests cover only family orchestration: admission limits, the
sequential fallbacks, output forwarding, failure propagation and cancellation.
"""
from __future__ import annotations

import io
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import types

import pytest

from XTA import confidence_sweep as sweep


_POPEN = subprocess.Popen
_CHILD = r'''
import json, os, signal, sys, time
from pathlib import Path
from XTA.unification.tta_manifest import capture_tta_artifact_identities, TTA_RUN_MANIFEST_SCHEMA
if hasattr(signal, 'SIGBREAK'):
    signal.signal(signal.SIGBREAK, signal.default_int_handler)

arguments = json.loads(sys.argv[1])
def option(name):
    return arguments[arguments.index(name) + 1]

cent = int(os.environ['XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS'])
role = os.environ.get('XTA_YOLO_SWEEP_NATIVE_ROLE')
plan = json.loads(os.environ.get('SWEEP_CONCURRENCY_PLAN', '{}'))
step = plan.get(str(cent), {})
out = Path(option('--output'))
out.mkdir(parents=True, exist_ok=True)
active = out.parent.parent / 'fixture_active'
active.mkdir(exist_ok=True)
marker = active / str(cent)
marker.write_text(str(os.getpid()))
if role == 'produce' and not plan.get('no_shared_source'):
    # Stands in for the decoded source volume receipt that later cutoffs reuse.
    (Path(os.environ['XTA_YOLO_SWEEP_WORK_DIR']) / 'source.gray8.json').write_text('{}')
if role == 'produce' and plan.get('shared_cube'):
    (Path(os.environ['XTA_YOLO_SWEEP_WORK_DIR']) / 'cube.gray8.json').write_text('{}')
record = {'cent': cent, 'pid': os.getpid(), 'role': role, 'started': time.time(),
          'defer': os.environ.get('XTA_YOLO_SWEEP_DEFER_NATIVE_RETIREMENT')}
print('fixture-out-' + str(cent), flush=True)
print('fixture-err-' + str(cent), file=sys.stderr, flush=True)
if step.get('bulk'):
    line = 'b' * 99 + '\n'
    count = int(step['bulk']) // len(line)
    for stream in (sys.stdout, sys.stderr):
        stream.write(line * count)
        stream.flush()
    sys.stdout.flush()
    sys.stdout.buffer.write(b'bad-\x81-byte\n')
    sys.stdout.buffer.write(b'tail-without-newline')
    sys.stdout.buffer.flush()
for other in step.get('wait_for', ()):
    deadline = time.time() + 30
    while not (active / str(other)).exists() and time.time() < deadline:
        time.sleep(.02)
peak = 0
end = time.time() + float(step.get('sleep', 0))
while True:
    peak = max(peak, len(os.listdir(active)))
    if time.time() >= end:
        break
    time.sleep(.02)
record.update(finished=time.time(), peak_active=peak)
(out / 'fixture_observed.json').write_text(json.dumps(record))
if 'exit' in step:
    marker.unlink()
    raise SystemExit(int(step['exit']))
model = Path(option('--model').split(':', 1)[1])
inputs = capture_tta_artifact_identities(input_path=option('--input'), gpu_model_path=model, cpu_model_path=None)
inputs.update(source_shape_t_y_x=[2, 3, 4], processing_shape_t_y_x=plan.get('processing_shape', [2, 3, 4]),
              fps=1.)
identity = {'sweep_id': os.environ['XTA_YOLO_SWEEP_ID'], 'cutoff_hundredths': cent,
            'sam_checkpoint_identity_sha256': None, 'augmentation_policy_sha256': None,
            'reconciliation_policy_sha256': 'b' * 64 if step.get('policy') else None}
if role is not None:
    targets = os.environ['XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS'].split(',')
    scalar = int(plan.get('scalar_views', 0))
    identity['upstream_fanout'] = {'role': role, 'validated': True,
        'produced_views': 2 if role == 'produce' else 0,
        'consumed_views': 2 if role == 'consume' else 0, 'scalar_views': scalar,
        'target_count': len(targets), 'fallback_reasons': ['fixture fallback route'] if scalar else []}
version = os.environ['SWEEP_TEST_VERSION']
receipt = {'schema': TTA_RUN_MANIFEST_SCHEMA, 'status': 'complete',
    'launcher': {'mode': 'tta', 'version': version, 'pipeline_version': version},
    'confidence_sweep': identity, 'resolved_configuration': {
        'conf': .123 if step.get('wrong_conf') else float(option('--conf')),
        'min_conf': float(option('--min_conf')), 'input': option('--input'),
        'model': option('--model'), 'output': str(out), 'temp': option('--temp')},
    'inputs': inputs, 'geometry': {'physical_views': []},
    'sam_requested_work_complete': True, 'sam_planning_omissions': [],
    'outputs': {'requested': [], 'paths': {'run_manifest': str(out / 'manifest.json')}}}
if not step.get('no_receipt'):
    (out / 'manifest.json').write_text(json.dumps(receipt))
marker.unlink()
'''


@pytest.fixture
def harness(tmp_path, monkeypatch):
    child = tmp_path / 'scalar_fixture.py'
    child.write_text(_CHILD, encoding='utf-8')
    source = tmp_path / 'source video.mkv'
    source.write_bytes(b'video fixture')
    model = tmp_path / 'model artifact.engine'
    model.write_bytes(b'model fixture')
    monkeypatch.setenv('SWEEP_TEST_VERSION', sweep.__version__)
    for name in ('SWEEP_CONCURRENCY_PLAN', 'XTA_YOLO_SWEEP_VIEW_MAJOR', 'XTA_YOLO_SWEEP_CACHE_GIB',
                 'XTA_YOLO_SWEEP_CONSUMER_PARALLEL', 'XTA_YOLO_SWEEP_DEFER_NATIVE_RETIREMENT'):
        monkeypatch.delenv(name, raising=False)
    system_temp = tmp_path / 'system-temp'
    system_temp.mkdir()
    monkeypatch.setattr(sweep.tempfile, 'gettempdir', lambda: str(system_temp))
    monkeypatch.setattr(sweep, '_CHILD_SETTLE_SECONDS', .5)
    monkeypatch.setattr(sweep, '_CHILD_GRACE_SECONDS', 1.)
    monkeypatch.setattr(sweep, '_CHILD_KILL_SECONDS', 5.)
    state = types.SimpleNamespace(root=tmp_path, launches=[], publications=[], on_launch=None,
        arguments=['--input', str(source), '--model', f'gpu:{model}'])

    def launch(command, **options):
        assert command[:5] == [sys.executable, '-m', 'XTA', '--mode', 'tta']
        assert 'cwd' not in options
        piped = 'stdout' in options or 'stderr' in options
        if piped:
            assert options['stdout'] is subprocess.PIPE and options['stderr'] is subprocess.PIPE
        process = _POPEN([sys.executable, '-B', str(child), json.dumps(command[5:])], **options)
        cutoff = int(options['env']['XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS'])
        state.launches.append(types.SimpleNamespace(cutoff=cutoff, piped=piped, process=process))
        if state.on_launch is not None:
            state.on_launch(cutoff, process)
        return process

    monkeypatch.setattr(sweep.subprocess, 'Popen', launch)
    publisher = types.ModuleType('XTA.sweep_publication')

    def publish(root, rows, *, sweep_id, temp_dir):
        state.publications.append([cent for cent, _ in rows])
        return None

    publisher.publish_sweep_review = publish
    cache_module = types.ModuleType('XTA.forward_replay')
    cache_module.replay_cache_summary = lambda root: {'entries': 0}
    monkeypatch.setitem(sys.modules, 'XTA.sweep_publication', publisher)
    monkeypatch.setitem(sys.modules, 'XTA.forward_replay', cache_module)

    def run(name, cutoffs, *, plan=None, parallel=None):
        if plan is not None:
            monkeypatch.setenv('SWEEP_CONCURRENCY_PLAN', json.dumps(plan))
        if parallel is not None:
            monkeypatch.setenv('XTA_YOLO_SWEEP_CONSUMER_PARALLEL', str(parallel))
        state.output = tmp_path / name
        return sweep.run_confidence_sweep(state.arguments, cutoffs_hundredths=cutoffs, output_root=state.output)

    state.run = run
    yield state
    for item in state.launches:
        if item.process.poll() is None:
            item.process.kill()
            item.process.wait(timeout=10)


def _read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def _observed(root, cent):
    return _read(root / 'confidence_cutoffs' / f'conf_{cent:03d}' / 'fixture_observed.json')


def _lines(text):
    return text.splitlines()


def test_limit_order_prefixes_and_retirement_marker(harness, capfd):
    plan = {str(cent): {'sleep': 1.5} for cent in (60, 70, 80, 90)}
    plan['60'] = {'sleep': 3.}
    path = harness.run('limited', (50, 60, 70, 80, 90), plan=plan, parallel=2)
    family = _read(path)
    assert family['status'] == 'complete'
    assert [row['cutoff_hundredths'] for row in family['branches']] == [50, 60, 70, 80, 90]
    assert [row['status'] for row in family['branches']] == ['complete'] * 5
    assert family['consumer_execution'] == {'consumers': 4, 'requested': 2, 'limit': 2,
        'reason': 'XTA_YOLO_SWEEP_CONSUMER_PARALLEL=2', 'mode': 'concurrent'}
    assert family['upstream_fanout']['status'] == 'shared_native_products'
    assert family['upstream_fanout']['validated_arms'] == 5
    assert harness.publications == [[50, 60, 70, 80, 90]]
    observed = {cent: _observed(harness.output, cent) for cent in (50, 60, 70, 80, 90)}
    assert observed[50]['peak_active'] == 1
    assert max(observed[cent]['peak_active'] for cent in (60, 70, 80, 90)) == 2
    # The slow lower cutoff finished after a higher one; receipts stay in cutoff order.
    assert observed[70]['finished'] < observed[60]['finished']
    assert observed[50]['defer'] is None
    assert all(observed[cent]['defer'] == '1' for cent in (60, 70, 80, 90))
    assert [(item.cutoff, item.piped) for item in harness.launches] == [
        (50, False), (60, True), (70, True), (80, True), (90, True)]
    streams = capfd.readouterr()
    out, err = _lines(streams.out), _lines(streams.err)
    assert 'fixture-out-50' in out and 'fixture-err-50' in err
    for cent in (60, 70, 80, 90):
        prefix = f'[conf {cent / 100:.2f}] '
        assert prefix + f'fixture-out-{cent}' in out
        assert prefix + f'fixture-err-{cent}' in err
        assert all(line.startswith(prefix) for line in out + err if line.endswith(f'fixture-out-{cent}')
                   or line.endswith(f'fixture-err-{cent}'))
    assert sum('Confidence sweep consumers:' in line for line in out) == 1
    assert not (harness.output / '.confidence_sweep.lock').exists()
    assert not Path(family['work_directory']).exists()


def test_auto_limit_admits_consumers_by_source_volume_memory(harness, monkeypatch):
    per_consumer = sweep._CONSUMER_SOURCE_VOLUME_FACTOR * 2 * 3 * 4
    monkeypatch.setattr(sweep, '_available_memory', lambda: (2 * per_consumer + 5, ('fixture',)))
    plan = {str(cent): {'sleep': 1.5} for cent in (60, 70, 80)}
    family = _read(harness.run('auto', (50, 60, 70, 80), plan=plan))
    execution = family['consumer_execution']
    assert execution['requested'] == 'auto' and execution['mode'] == 'concurrent'
    assert execution['limit'] == 2
    assert execution['estimated_bytes_per_consumer'] == per_consumer
    assert execution['available_memory_bytes'] == 2 * per_consumer + 5
    assert execution['memory_sources'] == ['fixture']
    assert max(_observed(harness.output, cent)['peak_active'] for cent in (60, 70, 80)) == 2


def test_unknown_memory_keeps_auto_sequential(harness, monkeypatch, capfd):
    monkeypatch.setattr(sweep, '_available_memory', lambda: (None, ()))
    family = _read(harness.run('unknown-memory', (50, 60, 70)))
    assert family['consumer_execution']['mode'] == 'sequential'
    assert family['consumer_execution']['limit'] == 1
    assert not any(item.piped for item in harness.launches)
    assert all(_observed(harness.output, cent)['defer'] is None for cent in (50, 60, 70))
    out = _lines(capfd.readouterr().out)
    assert 'fixture-out-60' in out
    assert any('consumers run sequentially (auto: available physical memory is unknown)' in line for line in out)


@pytest.mark.parametrize('requested,scalar,available,expected', (
    (None, 0, None, 1),
    (None, 0, 10**15, 3),
    (None, 0, 1, 1),
    (8, 0, None, 3),
    (2, 0, None, 2),
    (1, 0, None, 1),
    (8, 1, 10**15, 1),
    (None, 4, 10**15, 1),
))
def test_consumer_limit_decision(monkeypatch, requested, scalar, available, expected):
    monkeypatch.setattr(sweep, '_available_memory', lambda: (available, ('fixture',) if available else ()))
    record = sweep._consumer_execution(requested, 3, {'scalar_views': scalar}, [10, 20, 30])
    assert record['limit'] == expected
    assert record['mode'] == ('concurrent' if expected > 1 else 'sequential')
    if scalar:
        assert 'scalar-fallback' in record['reason']
    private = sweep._consumer_execution(requested, 3, {'scalar_views': scalar}, [10, 20, 30],
                                        shared_source=False)
    assert private['limit'] == 1 and private['mode'] == 'sequential'
    private_cube = sweep._consumer_execution(requested, 3, {'scalar_views': scalar}, [10, 20, 30],
                                             shared_cube=False)
    assert private_cube['limit'] == 1 and private_cube['mode'] == 'sequential'
    if not scalar:
        assert private_cube['reason'] == 'producer published no shared processing cube'


@pytest.mark.parametrize('processing,receipt,missing', (
    ([2, 3, 4], False, False),   # no cube resize
    ([4, 3, 4], False, False),   # T-only cube may be lazy; parent cannot require a receipt
    ([4, 3, 4], True, False),
    ([2, 4, 4], False, True),    # Y resize is never lazy
    ([2, 3, 5], False, True),    # X resize is never lazy
    ([4, 4, 5], True, False),
))
def test_shared_cube_missing(tmp_path, processing, receipt, missing):
    if receipt:
        (tmp_path / 'cube.gray8.json').write_text('{}')
    inputs = {'source_shape_t_y_x': [2, 3, 4], 'processing_shape_t_y_x': processing}
    assert sweep._shared_cube_missing(tmp_path, inputs) is missing


@pytest.mark.parametrize('processing,shared_cube,mode', (
    ([2, 4, 5], False, 'sequential'),
    ([2, 4, 5], True, 'concurrent'),
    ([4, 3, 4], False, 'concurrent'),
))
def test_unshared_resized_processing_cube_keeps_consumers_sequential(harness, processing, shared_cube, mode):
    plan = {'processing_shape': processing, 'shared_cube': shared_cube,
            **{str(cent): {'sleep': .5} for cent in (60, 70)}}
    family = _read(harness.run('cube', (50, 60, 70), plan=plan, parallel=2))
    assert family['status'] == 'complete'
    execution = family['consumer_execution']
    assert execution['mode'] == mode
    piped = [item.piped for item in harness.launches]
    if mode == 'sequential':
        assert execution['reason'] == 'producer published no shared processing cube'
        assert piped == [False, False, False]
        assert max(_observed(harness.output, cent)['peak_active'] for cent in (60, 70)) == 1
        assert all(_observed(harness.output, cent)['defer'] is None for cent in (60, 70))
    else:
        assert piped == [False, True, True]
        assert all(_observed(harness.output, cent)['defer'] == '1' for cent in (60, 70))


@pytest.mark.parametrize('value', ('0', '-1', '01', 'two', '', ' 2', '2.0', 'AUTO', '1' * 10))
def test_invalid_parallel_values_fail_before_outputs(harness, monkeypatch, value):
    monkeypatch.setenv('XTA_YOLO_SWEEP_CONSUMER_PARALLEL', value)
    out = harness.root / 'invalid'
    with pytest.raises(ValueError, match='XTA_YOLO_SWEEP_CONSUMER_PARALLEL'):
        sweep.run_confidence_sweep(harness.arguments, cutoffs_hundredths=(50, 60), output_root=out)
    assert not out.exists()
    assert harness.launches == []


def test_detector_fallback_keeps_consumers_sequential(harness, monkeypatch, capfd):
    monkeypatch.setenv('XTA_YOLO_SWEEP_DEFER_NATIVE_RETIREMENT', '1')
    plan = {'scalar_views': 1, **{str(cent): {'sleep': .5} for cent in (60, 70, 80)}}
    family = _read(harness.run('fallback', (50, 60, 70, 80), plan=plan, parallel=4))
    assert family['status'] == 'complete'
    assert family['upstream_fanout']['status'] == 'partial_fallback'
    execution = family['consumer_execution']
    assert execution['mode'] == 'sequential' and execution['limit'] == 1
    assert execution['reason'] == 'producer recorded 1 detector scalar-fallback views'
    assert not any(item.piped for item in harness.launches)
    for cent in (50, 60, 70, 80):
        observed = _observed(harness.output, cent)
        assert observed['peak_active'] == 1
        assert observed['defer'] is None
    assert 'fixture-out-70' in _lines(capfd.readouterr().out)


def test_missing_shared_source_volume_keeps_consumers_sequential(harness):
    plan = {'no_shared_source': True, **{str(cent): {'sleep': .5} for cent in (60, 70)}}
    family = _read(harness.run('private-source', (50, 60, 70), plan=plan, parallel=2))
    assert family['status'] == 'complete'
    assert family['consumer_execution']['mode'] == 'sequential'
    assert family['consumer_execution']['reason'] == 'producer published no shared decoded source volume'
    assert not any(item.piped for item in harness.launches)
    assert max(_observed(harness.output, cent)['peak_active'] for cent in (60, 70)) == 1


def test_reference_mode_stays_sequential_without_execution_record(harness, monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_VIEW_MAJOR', '0')
    monkeypatch.setenv('XTA_YOLO_SWEEP_DEFER_NATIVE_RETIREMENT', '1')
    family = _read(harness.run('reference', (50, 60, 70), parallel=3))
    assert family['status'] == 'complete'
    assert 'consumer_execution' not in family
    assert not any(item.piped for item in harness.launches)
    assert all(_observed(harness.output, cent)['defer'] is None for cent in (50, 60, 70))


def test_failing_consumer_stops_running_siblings(harness):
    plan = {'60': {'wait_for': [70, 80], 'exit': 7}, '70': {'sleep': 60}, '80': {'sleep': 60},
            '90': {'sleep': 60}}
    started = time.monotonic()
    with pytest.raises(subprocess.CalledProcessError) as raised:
        harness.run('failing', (50, 60, 70, 80, 90), plan=plan, parallel=3)
    assert raised.value.returncode == 7
    assert time.monotonic() - started < 30
    family = _read(harness.output / 'manifest.json')
    assert family['status'] == 'failed'
    assert family['error']['type'] == 'CalledProcessError'
    rows = {row['cutoff_hundredths']: row for row in family['branches']}
    assert list(rows) == [50, 60, 70, 80]
    assert [rows[cent]['status'] for cent in rows] == ['complete', 'failed', 'stopped', 'stopped']
    assert rows[60]['returncode'] == 7
    for cent in (70, 80):
        assert rows[cent]['stopped_after_failed_cutoff_hundredths'] == 60
        assert isinstance(rows[cent]['returncode'], int)
    assert [item.cutoff for item in harness.launches] == [50, 60, 70, 80]
    assert all(item.process.poll() is not None for item in harness.launches)
    assert family['work_cleanup_complete'] is True
    assert not Path(family['work_directory']).exists()
    assert not (harness.output / '.confidence_sweep.lock').exists()
    assert harness.publications == []


@pytest.mark.parametrize('failure,error,match', (
    ('policy', ValueError, 'identity changed'),
    ('wrong_conf', ValueError, 'incorrect conf'),
    ('no_receipt', FileNotFoundError, None),
))
def test_invalid_consumer_receipt_stops_siblings(harness, failure, error, match):
    plan = {'60': {'sleep': 60}, '70': {'wait_for': [60, 80], failure: True}, '80': {'sleep': 60}}
    with pytest.raises(error, match=match):
        harness.run(f'receipt-{failure}', (50, 60, 70, 80), plan=plan, parallel=3)
    family = _read(harness.output / 'manifest.json')
    assert family['status'] == 'failed'
    assert [row['status'] for row in family['branches']] == ['complete', 'stopped', 'failed', 'stopped']
    assert family['branches'][2]['returncode'] == 0
    assert all(item.process.poll() is not None for item in harness.launches)
    assert family['work_cleanup_complete'] is True
    assert harness.publications == []


def test_interrupt_cancels_every_running_consumer(harness):
    plan = {str(cent): {'sleep': 60} for cent in (60, 70, 80)}
    armed = [True]

    def interrupt_once_all_run(cutoff, process):
        if cutoff == 50:
            return
        active = harness.output / 'fixture_active'
        poll = process.poll

        def interrupting_poll():
            if armed[0] and all((active / str(cent)).exists() for cent in (60, 70, 80)):
                armed[0] = False
                raise KeyboardInterrupt('fixture cancellation')
            return poll()

        process.poll = interrupting_poll

    harness.on_launch = interrupt_once_all_run
    with pytest.raises(KeyboardInterrupt):
        harness.run('cancelled', (50, 60, 70, 80), plan=plan, parallel=3)
    family = _read(harness.output / 'manifest.json')
    assert family['status'] == 'cancelled'
    assert [row['status'] for row in family['branches']] == ['complete', 'cancelled', 'cancelled', 'cancelled']
    assert all(item.process.poll() is not None for item in harness.launches)
    assert family['work_cleanup_complete'] is True
    assert harness.publications == []


def test_unstoppable_consumers_retain_owned_work(harness, monkeypatch):
    monkeypatch.setattr(sweep, '_await_quiescence', lambda children, timeout: list(children))
    monkeypatch.setattr(sweep, '_OUTPUT_JOIN_SECONDS', .2)
    plan = {'60': {'wait_for': [70], 'exit': 3}, '70': {'sleep': 60}}
    with pytest.raises(sweep._ChildTreeUnsettled):
        harness.run('unsettled', (50, 60, 70), plan=plan, parallel=2)
    family = _read(harness.output / 'manifest.json')
    assert family['status'] == 'failed'
    assert family['work_cleanup_complete'] is False
    assert Path(family['work_directory']).is_dir()
    assert [row['status'] for row in family['branches']] == ['complete', 'failed', 'stopped']
    assert harness.publications == []


def test_large_unterminated_and_undecodable_output_is_drained(harness, capfd):
    plan = {str(cent): {'bulk': 2 * 1024 * 1024} for cent in (60, 70)}
    family = _read(harness.run('bulk', (50, 60, 70), plan=plan, parallel=2))
    assert family['status'] == 'complete'
    streams = capfd.readouterr()
    out, err = _lines(streams.out), _lines(streams.err)
    expected = (2 * 1024 * 1024) // 100
    for cent in (60, 70):
        prefix = f'[conf {cent / 100:.2f}] '
        assert out.count(prefix + 'b' * 99) == expected
        assert err.count(prefix + 'b' * 99) == expected
        assert prefix + 'tail-without-newline' in out
        assert any(line.startswith(prefix + 'bad-') and line.endswith('-byte') for line in out)


class _Chunks(io.RawIOBase):
    def __init__(self, chunks):
        self.chunks = list(chunks)

    def readable(self):
        return True

    def read1(self, size=-1):
        return self.chunks.pop(0) if self.chunks else b''


def test_prefixed_forwarding_splits_lines_across_chunks(capsys, monkeypatch):
    monkeypatch.setattr(sweep, '_OUTPUT_LINE_LIMIT', 8)
    stream = _Chunks([b'one\r\ntw', b'o\nthr', b'ee\n', b'0123456789', b'\nlast'])
    sweep._forward_prefixed_output(stream, 'stdout', '[p] ', 'utf-8')
    assert stream.closed
    assert capsys.readouterr().out.splitlines() == [
        '[p] one', '[p] two', '[p] three', '[p] 0123456789', '[p] last']


def test_broken_parent_stream_does_not_stop_draining(monkeypatch):
    class Broken:
        encoding = 'ascii'

        def write(self, text):
            raise OSError('parent output closed')

        def flush(self):
            pass

    monkeypatch.setattr(sys, 'stderr', Broken())
    stream = _Chunks([b'a\n'] * 100)
    sweep._forward_prefixed_output(stream, 'stderr', '[p] ', 'utf-8')
    assert stream.chunks == [] and stream.closed


def test_producer_early_publication_record_reaches_the_family_summary():
    from XTA import confidence_sweep as sweep
    receipt = {'confidence_sweep': {'upstream_fanout': {
        'role': 'produce', 'validated': True, 'produced_views': 3, 'consumed_views': 0,
        'scalar_views': 0, 'target_count': 2, 'fallback_reasons': [],
        'early_publication': {'publishers': 8, 'backlog_limit': 32, 'backlog_peak': 5,
                              'scheduler_wait_seconds': 0.0, 'scheduler_waits': 0}}}}
    evidence = sweep._fanout_evidence(receipt, role='produce', target_count=2)
    manifest = {'cutoffs_hundredths': [50, 60], 'upstream_fanout': {}, 'branches': [{'upstream_fanout': evidence}]}
    sweep._update_fanout_summary(manifest)
    assert manifest['upstream_fanout']['early_publication']['scheduler_wait_seconds'] == 0.0
    receipt['confidence_sweep']['upstream_fanout']['early_publication'] = {'backlog_peak': -1}
    with pytest.raises(ValueError, match='early_publication'):
        sweep._fanout_evidence(receipt, role='produce', target_count=2)


def test_sweep_work_storage_records_whether_temp_is_memory_backed(monkeypatch, tmp_path):
    from XTA import confidence_sweep as sweep, runtime
    shm = tmp_path / 'shm'
    monkeypatch.setattr(runtime, 'path_is_memory_backed', lambda path: sweep.Path(path) == shm)
    assert sweep._work_storage(shm) == {'root': str(shm), 'memory_backed': True}
    assert sweep._work_storage(tmp_path) == {'root': str(tmp_path), 'memory_backed': False}

    def unknown(path):
        raise OSError('no mount table')

    monkeypatch.setattr(runtime, 'path_is_memory_backed', unknown)
    assert sweep._work_storage(tmp_path)['memory_backed'] is None

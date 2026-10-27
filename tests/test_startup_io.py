"""Startup discovery overlaps independent reads without crossing execution gates."""

import ast
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import startup_io


def _source(tmp_path):
    path = tmp_path / 'source.mkv'
    path.write_bytes(b'original video')
    return path


def _threads():
    return [thread for thread in threading.enumerate()
            if thread.name.startswith('tta-startup-identity')]


def test_probe_overlaps_identity_but_execution_waits_for_join(tmp_path):
    source = _source(tmp_path)
    capture_started = threading.Event()
    probe_completed = threading.Event()
    release_capture = threading.Event()
    events = []
    info = dict(width=3064, height=3023, fps=30., num_frames=1930)

    def capture():
        capture_started.set()
        assert release_capture.wait(5)
        events.append('identity_done')
        return dict(source=startup_io._file_identity(source), models={})

    def probe(path):
        assert path == source
        assert events == ['in_progress']
        assert capture_started.wait(5)
        events.append('probe_done')
        probe_completed.set()
        return info

    def start_processing():
        result = startup_io.capture_and_probe_tta_startup(
            input_path=source, capture_identities=capture, probe=probe,
            invalidate_manifest=lambda: events.append('in_progress'))
        events.append('processing')
        return result

    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(start_processing)
        try:
            assert probe_completed.wait(5)
            assert not future.done()
            assert events == ['in_progress', 'probe_done']
        finally:
            release_capture.set()
        identities, result = future.result(timeout=5)
    assert result is info
    assert identities['source'] == startup_io._file_identity(source)
    assert events == ['in_progress', 'probe_done', 'identity_done', 'processing']
    assert not _threads()


@pytest.mark.parametrize('identity_fails', [False, True])
def test_probe_failure_joins_pending_identity_and_preserves_probe_error(tmp_path, identity_fails):
    source = _source(tmp_path)
    capture_started = threading.Event()
    probe_failed = threading.Event()
    release_capture = threading.Event()
    capture_finished = threading.Event()
    failure = OSError('packet count failed')
    invalidated = []

    def capture():
        capture_started.set()
        try:
            assert release_capture.wait(5)
            if identity_fails:
                raise ValueError('model digest failed too')
            return dict(source=startup_io._file_identity(source), models={})
        finally:
            capture_finished.set()

    def probe(path):
        assert invalidated == [True]
        assert capture_started.wait(5)
        probe_failed.set()
        raise failure

    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(startup_io.capture_and_probe_tta_startup,
            input_path=source, capture_identities=capture, probe=probe,
            invalidate_manifest=lambda: invalidated.append(True))
        try:
            assert probe_failed.wait(5)
            assert not future.done()
            assert not capture_finished.is_set()
        finally:
            release_capture.set()
        with pytest.raises(OSError) as caught:
            future.result(timeout=5)
    assert caught.value is failure
    assert capture_finished.is_set()
    assert not _threads()


def test_identity_failure_propagates_after_probe_and_leaves_no_thread(tmp_path):
    source = _source(tmp_path)
    failure = ValueError('model unreadable')
    events = []

    def capture():
        raise failure

    with pytest.raises(ValueError) as caught:
        startup_io.capture_and_probe_tta_startup(
            input_path=source, capture_identities=capture,
            probe=lambda path: events.append('probe') or {},
            invalidate_manifest=lambda: events.append('in_progress'))
    assert caught.value is failure
    assert events == ['in_progress', 'probe']
    assert not _threads()


@pytest.mark.parametrize('change', ['probe', 'captured_source'])
def test_changed_source_rejects_mixed_probe_and_identity(tmp_path, change):
    source = _source(tmp_path)
    initial = startup_io._file_identity(source)

    def capture():
        record = dict(initial)
        if change == 'captured_source':
            record['file_id'] += 1
        return dict(source=record, models={})

    def probe(path):
        if change == 'probe':
            path.write_bytes(b'changed video with a different size')
        return dict(width=3064, height=3023, fps=30., num_frames=1930)

    with pytest.raises(RuntimeError, match='changed during startup discovery'):
        startup_io.capture_and_probe_tta_startup(input_path=source,
            capture_identities=capture, probe=probe, invalidate_manifest=lambda: None)
    assert not _threads()


def test_manifest_invalidation_failure_starts_no_reader(tmp_path):
    source = _source(tmp_path)
    capture, probe = mock.Mock(), mock.Mock()
    failure = OSError('cannot invalidate previous complete manifest')
    with pytest.raises(OSError) as caught:
        startup_io.capture_and_probe_tta_startup(input_path=source,
            capture_identities=capture, probe=probe,
            invalidate_manifest=mock.Mock(side_effect=failure))
    assert caught.value is failure
    capture.assert_not_called()
    probe.assert_not_called()
    assert not _threads()


def test_source_stat_failure_still_invalidates_previous_complete_manifest(tmp_path):
    capture, probe = mock.Mock(), mock.Mock()
    invalidated = []
    with pytest.raises(FileNotFoundError):
        startup_io.capture_and_probe_tta_startup(input_path=tmp_path / 'missing.mkv',
            capture_identities=capture, probe=probe,
            invalidate_manifest=lambda: invalidated.append(True))
    assert invalidated == [True]
    capture.assert_not_called()
    probe.assert_not_called()
    assert not _threads()


@pytest.mark.parametrize('probe_fails', [False, True])
def test_pipeline_invalidates_manifest_and_joins_before_processing(tmp_path, probe_fails):
    from XTA import pipeline

    tree = ast.parse(Path(pipeline.__file__).read_text(encoding='utf-8'))
    main = next(node for node in tree.body
                if isinstance(node, ast.FunctionDef) and node.name == '_main_impl')
    start = next(index for index, node in enumerate(main.body)
                 if isinstance(node, ast.Assign)
                 and any(isinstance(target, ast.Name) and target.id == 'unified_launch_at_start'
                         for target in node.targets))
    stop = next(index for index, node in enumerate(main.body)
                if isinstance(node, ast.AnnAssign)
                and isinstance(node.target, ast.Name)
                and node.target.id == 'canonical_images_stage_root')
    code = compile(ast.Module(body=main.body[start:stop], type_ignores=[]),
                   pipeline.__file__, 'exec')
    source = _source(tmp_path)
    capture_started = threading.Event()
    probe_completed = threading.Event()
    release_capture = threading.Event()
    events, manifests = [], []
    probe_error = OSError('packet probe failed')
    info = dict(width=3064, height=3023, fps=30., num_frames=1930)

    def capture(**kwargs):
        capture_started.set()
        assert release_capture.wait(5)
        events.append('identity_done')
        return dict(source=startup_io._file_identity(source), models={})

    def probe(path):
        assert capture_started.wait(5)
        assert len(manifests) == 1
        assert manifests[0]['status'] == 'in_progress'
        assert manifests[0]['inputs'] is None
        probe_completed.set()
        if probe_fails:
            raise probe_error
        return info

    scope = dict(vars(pipeline))
    scope.update(
        current_unified_launch=lambda: SimpleNamespace(
            mode='tta', launcher='xta', version='development', command=('tta',)),
        input_path=source, gpu_model_path=None, cpu_model_path=None, out_dir=tmp_path,
        capture_tta_artifact_identities=capture, ffprobe_info=probe,
        runtime_trace_event=lambda *args, **kwargs: None,
        runtime_telemetry=lambda: SimpleNamespace(span=lambda *args: nullcontext()),
        write_json_manifest=lambda path, manifest: manifests.append(dict(manifest)),
    )

    def execute_startup_then_process():
        exec(code, scope)
        events.append('processing')

    with ThreadPoolExecutor(max_workers=1) as caller:
        future = caller.submit(execute_startup_then_process)
        try:
            assert probe_completed.wait(5)
            assert not future.done()
            assert events == []
        finally:
            release_capture.set()
        if probe_fails:
            with pytest.raises(OSError) as caught:
                future.result(timeout=5)
            assert caught.value is probe_error
        else:
            future.result(timeout=5)
    if probe_fails:
        assert events == ['identity_done']
        assert len(manifests) == 1
    else:
        assert events == ['identity_done', 'processing']
        assert len(manifests) == 2
        assert manifests[1]['inputs'] == scope['tta_artifact_identities']
        assert scope['startup_source_info'] is info
    assert all(manifest['status'] == 'in_progress' for manifest in manifests)
    assert not _threads()

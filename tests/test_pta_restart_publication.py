"""Restart failures must never retain certification of removed PTA output."""
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import json

import numpy as np
import pytest

from XTA import pta, json_publication
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    topology = SimpleNamespace(cuda_device_ids=(), gpu_cpu_sets=(),
        worker_cpu_order=(), allowed_cpus=(0,), summary='CPU publication test')
    monkeypatch.setattr(pta, 'discover_topology', lambda **kwargs: topology)


def fixture(root, *, owned=True):
    inputs, output = root / 'input', root / 'output'
    inputs.mkdir()
    output.mkdir()
    assert pta.cv2.imwrite(str(inputs / 'sample_0001.png'), np.full((8, 8), 127, np.uint8))
    (inputs / 'sample_0001.txt').write_text('0 0.1 0.1 0.8 0.1 0.8 0.8 0.1 0.8\n')
    if owned:
        pta.write_v18_output_sentinel(output)
    for name in ('images', 'labels'):
        (output / name).mkdir()
        (output / name / 'old.txt').write_text('previous output')
    (output / 'manifest.json').write_text(json.dumps({'schema': 'pta-tta.v21.manifest.1', 'status': 'complete'}))
    return inputs, output


def run(inputs, output):
    arguments = ['--input', str(inputs), '--output', str(output), '--imgsz', '8',
        '--output_format', 'png', '--enable_cartesian', 'transverse',
        '--save', 'images', 'labels', '--worker_backend', 'thread',
        '--pipeline_depth', '1', '--workers', '1', '--frame_workers', '1']
    options = build_runtime_options(parse_pta_args(arguments))
    pta.main(args=options, argv=arguments)


@pytest.mark.parametrize('failure', [OSError, KeyboardInterrupt])
def test_main_cleanup_interruption_retains_in_progress_then_can_restart(tmp_path, monkeypatch, failure):
    inputs, output = fixture(tmp_path)
    input_bytes = {p.name: p.read_bytes() for p in inputs.iterdir()}
    real_rmtree = pta.shutil.rmtree
    deleted = []
    def interrupt(path, *args, **kwargs):
        target = Path(path).resolve()
        target.relative_to(output.resolve())
        assert json.loads((output / 'manifest.json').read_text())['status'] == 'in_progress'
        deleted.append(target.name)
        if target == output / 'labels':
            raise failure('interrupted restart cleanup')
        return real_rmtree(path, *args, **kwargs)
    with monkeypatch.context() as patched:
        patched.setattr(pta.shutil, 'rmtree', interrupt)
        with pytest.raises(failure, match='interrupted restart cleanup'):
            run(inputs, output)
    assert deleted == ['images', 'labels']
    assert not (output / 'images' / 'old.txt').exists()
    assert (output / 'labels' / 'old.txt').exists()
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'in_progress'
    assert input_bytes == {p.name: p.read_bytes() for p in inputs.iterdir()}
    run(inputs, output)
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'
    assert any((output / 'images').rglob('*.png'))
    assert any((output / 'labels').rglob('*.txt'))
    assert not (output / '.v18_work').exists()


@pytest.mark.parametrize('stage', ['file_sync', 'replace', 'directory_sync'])
def test_main_failed_atomic_invalidation_never_deletes_output(tmp_path, monkeypatch, stage):
    inputs, output = fixture(tmp_path)
    prior = (output / 'manifest.json').read_bytes()
    real_writer = pta.write_json_manifest
    failure = OSError('invalidation publication failed')
    def writer(path, payload):
        if Path(path) != output / 'manifest.json':
            return real_writer(path, payload)
        function = {'file_sync': 'fsync', 'replace': 'replace'}
        if stage == 'directory_sync':
            patch = mock.patch.object(json_publication, '_fsync_parent_directory', side_effect=failure)
        else:
            patch = mock.patch.object(json_publication.os, function[stage], side_effect=failure)
        with patch:
            return real_writer(path, payload)
    monkeypatch.setattr(pta, 'write_json_manifest', writer)
    with mock.patch.object(pta.shutil, 'rmtree', side_effect=AssertionError('cleanup before invalidation')) as cleanup:
        with pytest.raises(OSError, match='invalidation publication failed'):
            run(inputs, output)
    cleanup.assert_not_called()
    assert (output / 'images' / 'old.txt').read_text() == 'previous output'
    assert (output / 'labels' / 'old.txt').read_text() == 'previous output'
    if stage == 'directory_sync':
        assert json.loads((output / 'manifest.json').read_text())['status'] == 'in_progress'
    else:
        assert (output / 'manifest.json').read_bytes() == prior


def test_main_safety_refusal_precedes_marker_and_cleanup(tmp_path, monkeypatch):
    inputs, output = fixture(tmp_path, owned=False)
    prior = (output / 'manifest.json').read_bytes()
    with mock.patch.object(pta, 'invalidate_v18_pta_completion') as invalidate, \
         mock.patch.object(pta, 'clean_generated_output_dirs') as cleanup:
        with pytest.raises(ValueError, match='ownership sentinel'):
            run(inputs, output)
    invalidate.assert_not_called()
    cleanup.assert_not_called()
    assert (output / 'manifest.json').read_bytes() == prior


def test_main_live_marker_survives_cleanup_until_final_publication(tmp_path, monkeypatch):
    inputs, output = fixture(tmp_path)
    real_cleanup = pta.clean_generated_output_dirs
    real_final = pta.write_v18_pta_manifest
    events = []
    def cleanup(path, **kwargs):
        real_cleanup(path, **kwargs)
        assert json.loads((output / 'manifest.json').read_text())['status'] == 'in_progress'
        assert (output / pta._V18_OUTPUT_SENTINEL_NAME).is_file()
        events.append('cleanup')
    def final(path, **kwargs):
        assert json.loads((output / 'manifest.json').read_text())['status'] == 'in_progress'
        assert not (output / '.v18_work').exists()
        events.append('complete')
        return real_final(path, **kwargs)
    monkeypatch.setattr(pta, 'clean_generated_output_dirs', cleanup)
    monkeypatch.setattr(pta, 'write_v18_pta_manifest', final)
    run(inputs, output)
    assert events == ['cleanup', 'complete']
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'


def test_interrupted_first_run_keeps_ownership_and_can_restart(tmp_path, monkeypatch):
    inputs, _previous = fixture(tmp_path)
    output = tmp_path / 'first_run'
    with monkeypatch.context() as patched:
        def failed_load(*args, **kwargs):
            raise OSError('controlled first-run failure')
        patched.setattr(pta, 'load_source_volume_from_spec', failed_load)
        with pytest.raises(OSError, match='controlled first-run failure'):
            run(inputs, output)
    assert (output / pta._V18_OUTPUT_SENTINEL_NAME).is_file()
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'in_progress'
    run(inputs, output)
    assert json.loads((output / 'manifest.json').read_text())['status'] == 'complete'

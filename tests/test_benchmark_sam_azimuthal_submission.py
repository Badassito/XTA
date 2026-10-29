"""The renderer stress CLI validates requests before loading or reserving CUDA."""
import builtins
from contextlib import contextmanager
from types import ModuleType

import pytest

from tools import benchmark_sam_azimuthal_submission as tool


def test_help_does_not_import_gpu_or_numerical_runtime(monkeypatch, capsys):
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'numpy', 'cupy', 'XTA'} or name == 'tools.qualify_release':
            pytest.fail('CLI help must not load GPU/runtime dependencies: '+name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(SystemExit) as stopped:
        tool.main(['--help'])
    assert stopped.value.code == 0
    assert '--output-dir' in capsys.readouterr().out


def test_default_request_is_bounded_and_uses_sibling_scratch(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, 'ROOT', tmp_path/'repo')
    calls = []
    monkeypatch.setattr(tool, 'run', calls.append)
    destination = tmp_path/'Scratch'/'renderer-stress'
    tool.main(['--output-dir', str(destination)])
    assert len(calls) == 1
    request = calls[0]
    assert request.output_dir == destination.resolve()
    assert request.blocks == ['32', 'default'] and request.consumers == [0, 12]
    assert request.warmup_seconds >= 30 and request.repeats >= 1
    assert not destination.exists()


def test_dependency_outputs_are_redirected_before_gpu_reservation(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, 'ROOT', tmp_path/'repo')
    monkeypatch.setattr(tool.os, 'environ', dict(tool.os.environ))
    monkeypatch.setattr(tool.sys, 'path', list(tool.sys.path))
    monkeypatch.setattr(tool.sys, 'dont_write_bytecode', tool.sys.dont_write_bytecode)
    output = tmp_path/'Scratch'/'renderer-stress'
    helper = ModuleType('tools.gpu_reservation')
    class ReservationIntercepted(Exception):
        pass
    @contextmanager
    def intercepted(*args, **kwargs):
        assert tool.os.environ['YOLO_CONFIG_DIR'] == str(output/'yolo-config')
        assert (output/'yolo-config').is_dir()
        assert tool.os.environ['TEMP'] == str(output/'temp')
        raise ReservationIntercepted
        yield
    helper.gpu_reservation = intercepted
    monkeypatch.setitem(tool.sys.modules, 'tools.gpu_reservation', helper)
    original = builtins.__import__
    def guarded(name, *args, **kwargs):
        if name.split('.')[0] in {'torch', 'numpy', 'cupy', 'XTA'}:
            pytest.fail('GPU/runtime imported before reservation: '+name)
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', guarded)
    with pytest.raises(ReservationIntercepted):
        tool.main(['--output-dir', str(output)])


@pytest.mark.parametrize('extra', [
    ['--warmup-seconds', '29'], ['--warmup-seconds', 'nan'], ['--warmup-seconds', 'inf'],
    ['--consumers', '-1'], ['--consumers', '13'], ['--repeats', '0'],
    ['--blocks', '0'], ['--blocks', 'unknown'],
])
def test_invalid_arguments_never_start_work(monkeypatch, tmp_path, extra):
    monkeypatch.setattr(tool, 'ROOT', tmp_path/'repo')
    monkeypatch.setattr(tool, 'run', lambda *_: pytest.fail('Invalid request started GPU work'))
    destination = tmp_path/'Scratch'/'renderer-stress'
    with pytest.raises(SystemExit) as stopped:
        tool.main(['--output-dir', str(destination), *extra])
    assert stopped.value.code == 2 and not destination.exists()


def test_repository_output_is_rejected_and_existing_evidence_is_preserved(monkeypatch, tmp_path):
    monkeypatch.setattr(tool, 'ROOT', tmp_path/'repo')
    monkeypatch.setattr(tool, 'run', lambda *_: pytest.fail('Invalid output started GPU work'))
    with pytest.raises(SystemExit) as stopped:
        tool.main(['--output-dir', str(tool.ROOT/'results')])
    assert stopped.value.code == 2
    output = tmp_path/'Scratch'/'already-qualified'
    output.mkdir(parents=True)
    receipt = output/'benchmark-results.json'
    receipt.write_text('preserve')
    with pytest.raises(SystemExit) as stopped:
        tool.main(['--output-dir', str(output)])
    assert stopped.value.code == 2 and receipt.read_text() == 'preserve'

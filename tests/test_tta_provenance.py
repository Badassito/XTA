"""TTA source receipts record actual bytes without loading optional runtimes."""
import hashlib
import sys

from XTA.tta_provenance import _source_record
from XTA import tta_provenance


def test_source_fingerprint_preserves_bytes_and_does_not_import(tmp_path, monkeypatch):
    source = tmp_path / 'optional' / 'runtime.py'
    source.parent.mkdir()
    content = b'raise RuntimeError("must not import")\r\n'
    source.write_bytes(content)
    monkeypatch.delitem(sys.modules, 'XTA.optional.runtime', raising=False)
    assert _source_record(tmp_path, 'optional.runtime') == {
        'path': str(source), 'loaded_in_parent': False,
        'sha256': hashlib.sha256(content).hexdigest(),
    }
    assert 'XTA.optional.runtime' not in sys.modules
    monkeypatch.setitem(sys.modules, 'XTA.optional.runtime', object())
    assert _source_record(tmp_path, 'optional.runtime')['loaded_in_parent'] is True


def test_unreadable_source_preserves_receipt_failure(tmp_path):
    assert _source_record(tmp_path, 'missing') == {
        'path': str(tmp_path / 'missing.py'), 'loaded_in_parent': False,
        'sha256': None, 'read_error': 'FileNotFoundError',
    }


def test_execution_receipt_tracks_sam_construction_without_importing_it(tmp_path, monkeypatch):
    names = ('lta_worker_adapter', 'lta_sam', 'sam_sdk_compat')
    expected = {}
    for name in names:
        data = f'raise RuntimeError("must not import {name}")\n'.encode()
        (tmp_path / f'{name}.py').write_bytes(data)
        expected[name] = hashlib.sha256(data).hexdigest()
        monkeypatch.delitem(sys.modules, f'XTA.{name}', raising=False)
    monkeypatch.setattr(tta_provenance, '__file__', str(tmp_path / 'tta_provenance.py'))
    before = tta_provenance._execution_runtime_provenance()['modules']
    for name in names:
        assert before[name]['sha256'] == expected[name]
        assert before[name]['loaded_in_parent'] is False
        assert f'XTA.{name}' not in sys.modules
    replacement = b'raise RuntimeError("changed construction helper")\n'
    (tmp_path / 'sam_sdk_compat.py').write_bytes(replacement)
    after = tta_provenance._execution_runtime_provenance()['modules']
    assert after['sam_sdk_compat']['sha256'] == hashlib.sha256(replacement).hexdigest()
    assert after['sam_sdk_compat'] != before['sam_sdk_compat']
    for name in names[:2]:
        assert after[name] == before[name]
    assert all(f'XTA.{name}' not in sys.modules for name in names)


def _fixture_package(tmp_path, monkeypatch, files):
    for relative, data in files.items():
        path = tmp_path.joinpath(*relative.split('/'))
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    monkeypatch.setattr(tta_provenance, '__file__', str(tmp_path / 'tta_provenance.py'))


def test_execution_receipt_fingerprints_every_package_source_automatically(tmp_path, monkeypatch):
    _fixture_package(tmp_path, monkeypatch, {'alpha.py': b'A = 1\n', 'nested/__init__.py': b'',
                                             'nested/beta.py': b'B = 2\r\n'})
    before = tta_provenance._execution_runtime_provenance()
    assert {'alpha', 'nested.__init__', 'nested.beta'} <= set(before['modules'])
    assert before['modules']['nested.beta']['sha256'] == hashlib.sha256(b'B = 2\r\n').hexdigest()
    assert before['package_python_sources']['file_count'] == 3
    (tmp_path / 'nested' / 'gamma.py').write_bytes(b'G = 3\n')
    after = tta_provenance._execution_runtime_provenance()
    assert after['modules']['nested.gamma']['sha256'] == hashlib.sha256(b'G = 3\n').hexdigest()
    assert after['package_python_sources']['file_count'] == 4
    assert after['package_python_sources']['sha256'] != before['package_python_sources']['sha256']
    assert 'spherical_sources' not in after


def test_execution_receipt_aggregate_equals_lta_package_fingerprint(tmp_path, monkeypatch):
    from XTA.lta_telemetry import lta_source_fingerprint
    _fixture_package(tmp_path, monkeypatch, {'alpha.py': b'A = 1\n', 'nested/__init__.py': b'',
                                             'nested/beta.py': b'B = 2\r\n'})
    receipt = tta_provenance._execution_runtime_provenance()['package_python_sources']
    expected = lta_source_fingerprint(package_root=tmp_path)
    assert receipt == {'scope': expected['scope'], 'sha256': expected['sha256'], 'file_count': len(expected['files'])}


def test_unreadable_package_source_is_recorded_without_failing_the_receipt(tmp_path, monkeypatch):
    _fixture_package(tmp_path, monkeypatch, {'alpha.py': b'A = 1\n', 'broken.py': b'B = 2\n'})
    original = tta_provenance.sha256_file

    def sha256_file(path):
        if path.name == 'broken.py':
            raise PermissionError(path)
        return original(path)
    monkeypatch.setattr(tta_provenance, 'sha256_file', sha256_file)
    receipt = tta_provenance._execution_runtime_provenance()
    assert receipt['modules']['broken']['sha256'] is None
    assert receipt['modules']['broken']['read_error'] == 'PermissionError'
    assert receipt['modules']['alpha']['sha256'] == hashlib.sha256(b'A = 1\n').hexdigest()
    assert receipt['package_python_sources']['sha256'] is None
    assert receipt['package_python_sources']['file_count'] == 2


def test_package_initializers_report_their_package_load_state(tmp_path, monkeypatch):
    (tmp_path / 'nested').mkdir()
    (tmp_path / '__init__.py').write_bytes(b'')
    (tmp_path / 'nested' / '__init__.py').write_bytes(b'')
    monkeypatch.setitem(sys.modules, 'XTA.nested', object())
    assert _source_record(tmp_path, '__init__')['loaded_in_parent'] is ('XTA' in sys.modules)
    assert _source_record(tmp_path, 'nested.__init__')['loaded_in_parent'] is True

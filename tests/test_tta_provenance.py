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

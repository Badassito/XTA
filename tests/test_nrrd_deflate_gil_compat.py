"""Known old python-deflate releases cannot silently serialize NRRD compression."""
from __future__ import annotations

from pathlib import Path
import sys
import tomllib
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import outputs


@pytest.mark.parametrize(('version', 'old'), [
    ('0.8.1', '0.8.1'), ('0.7.9', '0.7.9'),
    ('0.9.0', None), ('0.10.0', None), ('custom', None),
])
def test_imported_binding_version_gate_is_conservative(version, old):
    module = SimpleNamespace(__version__=version, __file__='/custom/deflate/__init__.py')
    assert outputs._old_nrrd_libdeflate_binding_version(module) == old


def test_metadata_fallback_requires_selected_module_in_distribution_record():
    module = SimpleNamespace(__file__='/task/deps/deflate/__init__.py')
    distribution = SimpleNamespace(
        version='0.8.1', files=[Path('deflate/__init__.py')],
        locate_file=lambda _file: '/task/deps/deflate/__init__.py',
    )
    with mock.patch.object(outputs.importlib_metadata, 'distribution', return_value=distribution):
        assert outputs._old_nrrd_libdeflate_binding_version(module) == '0.8.1'

    custom = SimpleNamespace(__file__='/custom/deflate/__init__.py')
    with mock.patch.object(outputs.importlib_metadata, 'distribution', return_value=distribution):
        assert outputs._old_nrrd_libdeflate_binding_version(custom) is None


def test_missing_metadata_never_claims_old_binding():
    custom = SimpleNamespace(__file__='/custom/deflate/__init__.py')
    with mock.patch.object(outputs.importlib_metadata, 'distribution', side_effect=LookupError):
        assert outputs._old_nrrd_libdeflate_binding_version(custom) is None


def _old_module():
    return SimpleNamespace(
        __version__='0.8.1', __file__='/task/deps/deflate/__init__.py',
        gzip_compress=lambda payload, level: bytes(payload),
    )


def _telemetry():
    return SimpleNamespace(gauge=lambda *_args: None, fallback=lambda *_args: None)


@pytest.mark.parametrize(('policy', 'isal_available'), [
    ('auto', True), ('cpu', True), ('cpu', False),
])
def test_policy_chain_skips_old_binding_for_next_validated_codec(policy, isal_available, capsys):
    actual_spec = outputs._nrrd_member_codec_spec
    attempted = []

    def spec(name):
        attempted.append(name)
        if name == 'qat':
            raise ImportError('optional hardware codec unavailable')
        if name == 'libdeflate':
            return actual_spec(name)
        if name == 'isal':
            if not isal_available:
                raise ImportError('optional ISA-L codec unavailable')
            return ('isal', 2, lambda payload: bytes(payload))
        if name == 'zlib':
            return ('zlib', 3, lambda payload: bytes(payload))
        raise AssertionError('unexpected codec tier')

    with mock.patch.dict(sys.modules, {'deflate': _old_module()}), \
            mock.patch.object(outputs, 'nrrd_member_codec_requested', return_value=policy), \
            mock.patch.object(outputs, '_nrrd_member_codec_spec', side_effect=spec), \
            mock.patch.object(outputs, '_nrrd_member_codec_self_test', return_value=True), \
            mock.patch.object(outputs, 'runtime_telemetry', return_value=_telemetry()), \
            mock.patch.object(outputs, '_NRRD_MEMBER_CODEC_FAILURES_ANNOUNCED', set()):
        selected = outputs._select_nrrd_member_codec()
    assert selected[0] == ('isal' if isal_available else 'zlib')
    assert attempted[-2:] == (['libdeflate', 'isal'] if isal_available else ['isal', 'zlib'])
    assert 'deflate>=0.9.0' in capsys.readouterr().out


def test_explicit_old_libdeflate_fails_with_upgrade_path():
    with mock.patch.dict(sys.modules, {'deflate': _old_module()}), \
            mock.patch.object(outputs, 'nrrd_member_codec_requested', return_value='libdeflate'), \
            mock.patch.object(outputs, 'runtime_telemetry', return_value=_telemetry()):
        with pytest.raises(RuntimeError, match=r'Explicit.*deflate>=0\.9\.0'):
            outputs._require_nrrd_member_codec()


def test_unknown_custom_binding_is_not_rejected_by_version_check():
    custom = SimpleNamespace(
        __file__='/custom/deflate/__init__.py',
        gzip_compress=lambda payload, level: bytes(payload),
    )
    with mock.patch.dict(sys.modules, {'deflate': custom}), \
            mock.patch.object(outputs.importlib_metadata, 'distribution', side_effect=LookupError):
        spec = outputs._nrrd_member_codec_spec('libdeflate')
    assert spec[0] == 'libdeflate'
    assert spec[2](b'abc') == b'abc'


def test_acceleration_extra_pins_gil_releasing_binding():
    manifest = tomllib.loads((Path(__file__).resolve().parents[1] / 'pyproject.toml').read_text())
    assert 'deflate>=0.9.0' in manifest['project']['optional-dependencies']['acceleration']

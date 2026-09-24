"""Selected NRRD codec logging is best-effort and does not change codec policy."""
from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

from XTA import outputs


def test_deflate_and_isal_distribution_identity():
    modules = {
        'deflate': SimpleNamespace(__file__='/task/deps/deflate/__init__.py', __version__='0.9.0'),
        'isal': SimpleNamespace(__file__='/task/deps/isal/__init__.py', __version__='1.7.2'),
    }
    versions = {'deflate': '0.9.0', 'isal': '1.7.2'}
    with mock.patch.object(outputs.importlib, 'import_module', side_effect=modules.__getitem__), \
            mock.patch.object(outputs.importlib_metadata, 'version', side_effect=versions.__getitem__):
        deflate = outputs._nrrd_codec_runtime_provenance('libdeflate')
        isal = outputs._nrrd_codec_runtime_provenance('isal')
    assert deflate['python_version'] == sys.version.split()[0]
    assert deflate['codec_distribution'] == 'deflate'
    assert deflate['codec_distribution_version'] == '0.9.0'
    assert deflate['codec_module_path'] == '/task/deps/deflate/__init__.py'
    assert deflate['codec_runtime_version'] == '0.9.0'
    assert isal['codec_distribution'] == 'isal'
    assert isal['codec_distribution_version'] == '1.7.2'
    assert isal['codec_module_path'] == '/task/deps/isal/__init__.py'


def test_zlib_records_runtime_version_without_distribution_lookup():
    module = SimpleNamespace(__file__='/python/zlib.so', ZLIB_RUNTIME_VERSION='1.3.1')
    with mock.patch.object(outputs.importlib, 'import_module', return_value=module), \
            mock.patch.object(outputs.importlib_metadata, 'version') as version:
        provenance = outputs._nrrd_codec_runtime_provenance('zlib')
    assert provenance['codec_runtime_version'] == '1.3.1'
    assert provenance['codec_module_path'] == '/python/zlib.so'
    assert provenance['codec_distribution_version'] is None
    version.assert_not_called()


def test_missing_optional_metadata_does_not_fail_or_warn():
    with mock.patch.object(outputs.importlib, 'import_module', side_effect=ImportError), \
            mock.patch.object(outputs.importlib_metadata, 'version', side_effect=LookupError):
        provenance = outputs._nrrd_codec_runtime_provenance('libdeflate')
    assert provenance['codec_module_path'] is None
    assert provenance['codec_distribution_version'] is None
    assert provenance['codec_runtime_version'] is None


def test_announcement_prints_and_records_provenance_once(capsys):
    gauges = []
    telemetry = SimpleNamespace(gauge=lambda name, value: gauges.append((name, value)))
    provenance = {'python_version': '3.12.14', 'codec_distribution_version': '0.9.0'}
    with mock.patch.object(outputs, '_NRRD_CPU_DEFLATE_BACKEND_ANNOUNCED', False), \
            mock.patch.object(outputs, '_nrrd_codec_runtime_provenance', return_value=provenance), \
            mock.patch.object(outputs, 'runtime_telemetry', return_value=telemetry), \
            mock.patch.object(outputs, 'nrrd_member_codec_requested', return_value='libdeflate'):
        codec = ('libdeflate', 3, lambda payload: payload)
        outputs._announce_nrrd_cpu_deflate_backend('member-parallel gzip', codec_spec=codec)
        outputs._announce_nrrd_cpu_deflate_backend('member-parallel gzip', codec_spec=codec)
    printed = capsys.readouterr().out
    assert printed.count('NRRD DEFLATE runtime provenance:') == 1
    assert ('nrrd.compression.runtime_provenance', provenance) in gauges
    assert sum(name == 'nrrd.compression.runtime_provenance' for name, _ in gauges) == 1

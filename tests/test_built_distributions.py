"""Exercise the distribution boundaries using real sdist and wheel artifacts."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tarfile
import tempfile
import zipfile

import pytest

from tools.verify_package_inventory import read_source_files


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def distributions():
    # Retain artifacts and build logs outside the checkout for failure diagnosis.
    destination = Path(os.environ.get(
        'XTA_PACKAGING_TEST_ROOT', str(ROOT.parent / 'Scratch' / 'Temp' / 'packaging-tests')))
    destination.mkdir(parents=True, exist_ok=True)
    workspace = Path(tempfile.mkdtemp(prefix='distributions-', dir=destination))
    source = workspace / 'source'
    payloads = read_source_files(ROOT)
    for name, data in payloads.items():
        path = source / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    artifacts = workspace / 'dist'
    artifacts.mkdir()
    temporary = workspace / 'temp'
    temporary.mkdir()
    environment = dict(os.environ, XTA_BUILD_INTEL='none', PYTHONDONTWRITEBYTECODE='1',
                       TEMP=str(temporary), TMP=str(temporary), TMPDIR=str(temporary))
    environment.pop('PYTHONPATH', None)

    def run(command, *, cwd, log):
        result = subprocess.run(command, cwd=cwd, env=environment,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                text=True, encoding='utf-8', errors='replace')
        (workspace / log).write_text(result.stdout, encoding='utf-8')
        assert result.returncode == 0, f'{log}: {result.stdout}'
        return result

    run([sys.executable, '-B', '-c',
         'from setuptools.build_meta import build_sdist; '
         f'build_sdist({str(artifacts)!r})'], cwd=source, log='sdist-build.log')
    archive = next(artifacts.glob('*.tar.gz'))
    with tarfile.open(archive) as bundle:
        members = {'/'.join(member.name.split('/')[1:]): bundle.extractfile(member).read()
                   for member in bundle.getmembers() if member.isfile()}
    # Build the wheel from the sdist to exercise its actual source completeness.
    extracted = workspace / 'sdist-source'
    for name, data in members.items():
        path = extracted / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    run([sys.executable, '-B', '-c',
         'from setuptools.build_meta import build_wheel; '
         f'build_wheel({str(artifacts)!r})'], cwd=extracted, log='wheel-build.log')
    wheel = next(artifacts.glob('*.whl'))
    installed = workspace / 'installed'
    run([sys.executable, '-B', '-m', 'pip', 'install', '--no-deps', '--no-index',
         '--no-compile', '--target', str(installed), str(wheel)],
        cwd=workspace, log='wheel-install.log')
    return payloads, members, wheel, installed, workspace, run


def test_sdist_contains_maintained_test_helpers_fixtures_and_root_rules(distributions):
    payloads, members, *_ = distributions
    expected = {name for name in payloads
                if name.startswith('tests/') and name.endswith(('.py', '.json'))}
    expected.update(('.gitattributes', '.gitignore', 'AGENTS.md', 'README.md'))
    assert expected <= members.keys(), sorted(expected - members.keys())
    for name in expected:
        assert members[name] == payloads[name], name
    assert not any('__pycache__' in name or name.endswith('.pyc') for name in members)


def test_wheel_keeps_documentation_and_all_tool_siblings_without_test_packages(distributions):
    payloads, _, wheel, *_ = distributions
    with zipfile.ZipFile(wheel) as archive:
        names = archive.namelist()
        data_prefix = next(name.split('/share/xta/')[0] + '/share/xta/'
                           for name in names if '/share/xta/' in name)
        expected = {name for name in payloads
                    if ((name.startswith('docs/') and name.endswith('.md'))
                        or (name.startswith('tools/') and name.endswith(('.py', '.md'))))}
        expected.update(('README.md', 'ARCHITECTURE.md'))
        for name in expected:
            assert archive.read(data_prefix + name) == payloads[name], name
        assert not any(name.startswith(('tests/', 'tools/')) for name in names)
        assert not any('/tests/' in name for name in names)


def test_installed_native_filter_loads_its_sparse_helper_outside_checkout(distributions):
    *_, installed, workspace, run = distributions
    tool_dir = installed / 'share' / 'xta' / 'tools'
    # Script execution uses its installed directory, with no checkout PYTHONPATH.
    probe = tool_dir / '_installed_helper_probe.py'
    probe.write_text(
        'from pathlib import Path\n'
        'import sys\n'
        'import xta_native_filter, xta_native_sparse, benchmark_cpu_backends\n'
        'from tools.cpu_backend_benchmarks import contracts\n'
        'assert Path(xta_native_sparse.__file__).parent == Path(__file__).parent\n'
        f'assert {str(ROOT)!r}.casefold() not in [str(Path(p).resolve()).casefold() for p in sys.path]\n'
        'assert callable(xta_native_sparse.sparse_label_stream)\n'
        'assert xta_native_filter.parser().prog\n'
        'assert "torch" not in sys.modules and "cupy" not in sys.modules\n'
        'print("installed native filter and sparse helper imported")\n', encoding='utf-8')
    run([sys.executable, '-B', str(probe)], cwd=workspace, log='installed-helper.log')
    run([sys.executable, '-B', str(tool_dir / 'xta_native_filter.py'), '--help'],
        cwd=workspace, log='installed-native-filter-help.log')
    run([sys.executable, '-B', str(tool_dir / 'benchmark_cpu_backends.py'), '--help'],
        cwd=workspace, log='installed-cpu-benchmark-help.log')

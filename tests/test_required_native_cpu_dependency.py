"""CPU kernels load a matching compiled artifact without a runtime compiler."""

from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import textwrap

from XTA import __version__


ROOT = Path(__file__).resolve().parents[1]
_REJECT_RUNTIME_COMPILATION = textwrap.dedent("""
    import sys

    class RejectCompilerImports:
        def find_spec(self, name, *args):
            if name.partition('.')[0] in {'numba', 'llvmlite'}:
                raise AssertionError(f'Unexpected runtime compiler import: {name}')

    def reject_child_process(event, args):
        if event in {'subprocess.Popen', 'os.system', 'os.spawn', 'os.posix_spawn', 'os.exec'}:
            raise AssertionError(f'Unexpected child process during a CPU kernel call: {event}')

    assert not {'numba', 'llvmlite'} & set(sys.modules)
    sys.meta_path.insert(0, RejectCompilerImports())
    sys.addaudithook(reject_child_process)
""")


def _run_probe(program, *, env=None):
    return subprocess.run(
        [sys.executable, '-X', 'utf8', '-B', '-c', _REJECT_RUNTIME_COMPILATION + textwrap.dedent(program)],
        cwd=ROOT, env=env, text=True, encoding='utf-8', errors='replace',
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=60, check=False,
    )


def test_missing_cpu_artifact_has_actionable_error_and_keeps_cli_version_light(tmp_path):
    artifacts = tmp_path / 'missing-artifacts'
    env = {**os.environ, 'XTA_NATIVE_CPU_ARTIFACTS': str(artifacts)}
    env.pop('XTA_NATIVE_CPU_BINARY', None)
    runtime = _run_probe("""
        from XTA.cli import run
        from XTA.native_cpu import load_native_cpu
        from XTA.native_cpu_build import MODULE, NativeCpuBuildError

        try:
            run(['--version'])
        except SystemExit as exit_signal:
            assert exit_signal.code == 0
        else:
            raise AssertionError('The version flag did not exit')
        assert MODULE not in sys.modules
        sys.modules[MODULE] = None   # Rule out an installed extension on sys.path as well.

        try:
            load_native_cpu()
        except NativeCpuBuildError as error:
            assert isinstance(error.__cause__, ImportError), repr(error.__cause__)
            assert 'binary was not found' in str(error.__cause__)
            assert 'Native CPU kernels unavailable' in str(error)
            assert 'matching platform XTA wheel' in str(error)
            assert 'build_native_cpu.py' in str(error)
            assert 'XTA_NATIVE_CPU_ARTIFACTS' in str(error)
        else:
            raise AssertionError('The loader accepted a missing extension')
    """, env=env)
    assert runtime.returncode == 0, runtime.stdout
    assert __version__ in runtime.stdout
    assert not artifacts.exists()


def test_cpu_kernel_first_call_needs_no_runtime_compiler():
    runtime = _run_probe("""
        import numpy as np
        from importlib.machinery import ExtensionFileLoader
        from XTA.confidence_capture_cpu import zero_unmasked_scores_flat
        from XTA.native_cpu_build import MODULE

        assert MODULE not in sys.modules
        mask = np.array([True, False, True, False])
        mask.setflags(write=False)
        scores = np.array([100, 175, 0, 210], dtype=np.uint8)
        zero_unmasked_scores_flat(mask, scores, 0, len(scores))
        assert scores.tolist() == [100, 0, 0, 0]
        assert isinstance(sys.modules[MODULE].__loader__, ExtensionFileLoader)
    """)
    assert runtime.returncode == 0, runtime.stdout

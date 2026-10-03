"""Qualify the complete checkout before building a release or development snapshot.

Run from any directory with the intended Python environment. Evidence and build
artifacts must live outside the repository. A release requires a clean Git HEAD;
--snapshot explicitly qualifies an uncommitted development tree instead.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]


def _git(*args: str) -> str:
    return subprocess.check_output(
        ['git', *args], cwd=ROOT, encoding='utf-8', errors='strict',
    )


def source_identity() -> dict[str, object]:
    """Identify every tracked or unignored source, including pending deletions."""
    paths = set(_git('ls-files', '-z').split('\0'))
    paths.update(_git('ls-files', '--others', '--exclude-standard', '-z').split('\0'))
    files = {}
    for name in sorted(paths - {''}):
        path = ROOT / name
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest() if path.is_file() else None
    return {'commit': _git('rev-parse', 'HEAD').strip(),
            'status': _git('status', '--porcelain'), 'files': files}


@contextmanager
def gpu_reservation(path: Path, timeout: float, *, task_name: str = 'full repository release qualification'):
    """Reserve this workspace's GPU without replacing another task's lock."""
    path.parent.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    record = {'task': str(task_name), 'pid': os.getpid(),
              'start_time': datetime.now(timezone.utc).isoformat(), 'token': token}
    deadline = time.monotonic() + timeout
    announced = False
    while True:
        try:
            with path.open('x', encoding='utf-8') as handle:
                json.dump(record, handle)
            break
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f'GPU remains reserved by another task: {path}')
            if not announced:
                print(f'Waiting for the existing GPU reservation: {path}', flush=True)
                announced = True
            time.sleep(min(5.0, max(0.01, deadline - time.monotonic())))
    try:
        yield
    finally:
        try:
            current = json.loads(path.read_text(encoding='utf-8'))
            if current.get('token') == token:
                path.unlink()
        except FileNotFoundError:
            pass


def _probe_cuda(env: dict[str, str]) -> dict[str, object]:
    command = [sys.executable, '-B', '-c',
               "import json,torch; available=torch.cuda.is_available(); "
               "print(json.dumps({'available':available,'torch':torch.__version__,"
               "'device':torch.cuda.get_device_name(0) if available else None}))"]
    result = subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True,
                            encoding='utf-8', errors='strict')
    probe = json.loads(result.stdout)
    if not probe['available']:
        raise RuntimeError('CUDA qualification requires an available device; use --snapshot --cpu-only for reduced coverage')
    return probe


def _verify_cuda_tests(junit: Path) -> None:
    """Require the numerical checks that originally revealed the TF32 leak."""
    cases = list(ET.parse(junit).iter('testcase'))
    required = {'test_cuda_elastic_conservative_inverse_matches_reference': 2,
                'test_shipped_policy_cuda_source_inverse_and_channels': 4}
    for prefix, minimum in required.items():
        selected = [case for case in cases
                    if case.get('classname') == 'tests.test_tta_augmentation_cuda'
                    and case.get('name', '').startswith(prefix)]
        if len(selected) < minimum or any(
                any(case.find(tag) is not None for tag in ('skipped', 'failure', 'error'))
                for case in selected):
            raise RuntimeError(f'Required CUDA numerical tests did not all run successfully: {prefix}')


def _qualification_environment(output: Path, *, cpu_only: bool) -> dict[str, str]:
    """Construct the subprocess environment before any CUDA runtime imports."""
    env = os.environ.copy()
    env.pop('PYTHONOPTIMIZE', None)
    generated_dirs = {
        'runtime-temp': output / 'runtime-temp',
        'cupy-cache': output / 'cupy-cache',
        'numba-cache': output / 'numba-cache',
        'cuda-cache': output / 'cuda-cache',
    }
    for directory in generated_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)
    runtime_temp = str(generated_dirs['runtime-temp'])
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONUTF8='1', PYTHONIOENCODING='utf-8',
               PYTHONPATH=str(ROOT), YOLO_CONFIG_DIR=str(output / 'ultralytics-config'),
               XTA_TEST_REPO=str(ROOT), TEMP=runtime_temp, TMP=runtime_temp,
               TMPDIR=runtime_temp, CUPY_CACHE_DIR=str(generated_dirs['cupy-cache']),
               NUMBA_CACHE_DIR=str(generated_dirs['numba-cache']),
               CUDA_CACHE_PATH=str(generated_dirs['cuda-cache']))
    Path(env['YOLO_CONFIG_DIR']).mkdir(parents=True, exist_ok=True)
    if cpu_only:
        # An empty string can leave the device visible on Windows. The explicit
        # invalid ordinal hides all devices in CUDA and PyTorch subprocesses.
        env['CUDA_VISIBLE_DEVICES'] = '-1'
    scratch = ROOT.parent / 'Scratch'
    workspace_tools = scratch / 'Environment' / 'tools'
    if workspace_tools.is_dir():
        env['PATH'] = str(workspace_tools) + os.pathsep + env.get('PATH', '')
    return env


def qualify(output: Path, *, snapshot: bool, gpu_lock_timeout: float,
            cpu_only: bool = False) -> dict[str, object]:
    output = output.resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError('Qualification artifacts must be outside the repository')
    before = source_identity()
    if before['status'] and not snapshot:
        raise ValueError('Release qualification requires a clean checkout; use --snapshot for development')
    if cpu_only and not snapshot:
        raise ValueError('Reduced CPU-only qualification is allowed only with --snapshot')
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'qualification.json'
    if receipt_path.exists():
        raise FileExistsError(f'Use a new output directory; qualification already exists: {receipt_path}')
    env = _qualification_environment(output, cpu_only=cpu_only)
    scratch = ROOT.parent / 'Scratch'
    steps = [
        ('full-tests', [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider',
                        '--maxfail=1',
                        '--basetemp', str(output / 'pytest-tmp'),
                        '--junitxml', str(output / 'junit.xml'), 'tests']),
        ('package-inventory', [sys.executable, '-B', 'tools/verify_package_inventory.py']),
        ('source-bundle', [sys.executable, '-B', 'tools/build_source_release.py',
                          '--output-dir', str(output / 'source'), *(['--snapshot'] if snapshot else [])]),
    ]
    receipt: dict[str, object] = {'kind': 'development-snapshot' if snapshot else 'release',
                                'source_before': before, 'started_utc': datetime.now(timezone.utc).isoformat(),
                                'coverage': 'cpu-only' if cpu_only else 'cpu-and-cuda',
                                'steps': [], 'success': False}
    try:
        reservation = (nullcontext() if cpu_only else
                       gpu_reservation(scratch / 'Temp' / 'GPU_LOCK', gpu_lock_timeout))
        with reservation:
            if not cpu_only:
                receipt['cuda'] = _probe_cuda(env)
            for name, command in steps:
                log = output / f'{name}.log'
                print(f'Running {name}; log: {log}', flush=True)
                with log.open('w', encoding='utf-8') as handle:
                    result = subprocess.run(command, cwd=ROOT, env=env,
                                            stdout=handle, stderr=subprocess.STDOUT)
                receipt['steps'].append({'name': name, 'command': command,
                                         'returncode': result.returncode, 'log': log.name})
                if result.returncode:
                    raise RuntimeError(f'{name} failed with exit {result.returncode}; see {log}')
                if source_identity() != before:
                    raise RuntimeError(f'{name} changed the source tree; no release is qualified')
                if name == 'full-tests' and not cpu_only:
                    _verify_cuda_tests(output / 'junit.xml')
        receipt['success'] = True
        return receipt
    finally:
        receipt['finished_utc'] = datetime.now(timezone.utc).isoformat()
        receipt['source_after'] = source_identity()
        receipt_path.write_text(json.dumps(receipt, indent=2) + '\n', encoding='utf-8')


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--snapshot', action='store_true')
    parser.add_argument('--cpu-only', action='store_true',
                        help='Explicit reduced coverage; requires --snapshot and does not qualify a release')
    parser.add_argument('--gpu-lock-timeout', type=float, default=3600)
    args = parser.parse_args()
    if args.gpu_lock_timeout < 0:
        parser.error('--gpu-lock-timeout must be nonnegative')
    qualify(args.output_dir, snapshot=args.snapshot, gpu_lock_timeout=args.gpu_lock_timeout,
            cpu_only=args.cpu_only)
    print('CPU-only development qualification passed.' if args.cpu_only else
          'Complete CPU and CUDA repository qualification passed.', flush=True)


if __name__ == '__main__':
    main()

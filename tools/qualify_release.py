"""Qualify the complete checkout before building a release or development snapshot.

Run from any directory with the intended Python environment. Evidence and build
artifacts must live outside the repository. A release requires a clean Git HEAD;
--snapshot explicitly qualifies an uncommitted development tree instead.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET

if __package__:
    from .check_release_version import check_release_version
    from .verify_package_inventory import capture_source_identity
else:
    from check_release_version import check_release_version
    from verify_package_inventory import capture_source_identity

ROOT = Path(__file__).resolve().parents[1]

# Full qualification owns GPU_LOCK before these explicit CUDA tests may run.
# Keep independent native-grid oracles mandatory: device availability and a
# passing augmentation suite alone do not establish projection coverage.
CUDA_COVERAGE_ENVIRONMENT = {
    'YOLO_TTA_TEST_D1_ORTHOGONAL_CUDA': '1',
    'XTA_TEST_NATIVE_AZIMUTHAL_CUDA': '1',
    'YOLO_TTA_TEST_RADIAL_COVERAGE_CUDA': '1',
    'XTA_TEST_SPHERICAL_CUDA': '1',
}
REQUIRED_CUDA_TESTS = (
    ('tests.test_tta_augmentation_cuda',
     'test_cuda_elastic_conservative_inverse_matches_reference', 5),
    ('tests.test_tta_augmentation_cuda',
     'test_shipped_policy_cuda_source_inverse_and_channels', 4),
    ('tests.test_d1_orthogonal_oracle_cuda',
     'test_cuda_native_fov_and_sharp_gaps_match_symbolic_oracle', 12),
    ('tests.test_d1_orthogonal_coverage',
     'test_cuda_packed_coverage_matches_native_reference', 4),
    ('tests.test_native_azimuthal_cuda_coverage',
     'test_real_upright_azimuthal_cuda_native_cells', 8),
    ('tests.test_native_azimuthal_cuda_coverage',
     'test_real_upright_azimuthal_cuda_degenerate_disk', 2),
    ('tests.test_native_azimuthal_cuda_coverage',
     'test_real_upright_azimuthal_cuda_rational_circle', 4),
    ('tests.test_radial_native_coverage_cuda',
     'test_cuda_reduced_model_covers_anisotropic_native_caps_and_preserves_annulus', 3),
    ('tests.test_radial_native_coverage_cuda',
     'test_cuda_tilted_radial_native_height_footprint_matches_independent_roi', 12),
    ('tests.test_radial_native_coverage_cuda',
     'test_cuda_rational_restoration_preserves_closed_radius_and_black_middle_shell', 3),
    ('tests.test_radial_native_coverage_cuda',
     'test_actual_radial_owner_bitset_covers_the_same_native_roi_across_shell_chunks', 5),
    ('tests.test_spherical_native_coverage_cuda', 'test_cuda_exact_rational_native_annulus', 6),
    ('tests.test_spherical_native_coverage_cuda', 'test_cuda_localized_face_keeps_negative_space', 2),
    ('tests.test_spherical_native_coverage_cuda',
     'test_cuda_radial_gap_and_exact_midpoints_survive_packed_roi', 1),
    ('tests.test_spherical_native_coverage_cuda', 'test_cuda_exact_inner_boundary_is_closed', 1),
    ('tests.test_spherical_native_coverage_cuda', 'test_cuda_beyond_roundoff_limits_remain_excluded', 1),
    ('tests.test_projection_sampling_cuda.ProjectionSamplingCudaTests',
     'test_all_one_native_masks_project_to_exact_declared_domains', 1),
    ('tests.test_projection_sampling_cuda.ProjectionSamplingCudaTests',
     'test_random_native_masks_match_cpu_for_coarse_shells_and_rescaled_outputs', 1),
)


def source_identity() -> dict[str, object]:
    """Identify every tracked or unignored source, including pending deletions."""
    return capture_source_identity(ROOT)


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


def _verify_cuda_tests(junit: Path) -> list[dict[str, object]]:
    """Reject skipped/missing numerical oracles and record executed case names."""
    cases = list(ET.parse(junit).iter('testcase'))
    records = []
    for classname, name, minimum in REQUIRED_CUDA_TESTS:
        selected = [case for case in cases
                    if case.get('classname') == classname
                    and (case.get('name') == name
                         or case.get('name', '').startswith(name + '['))]
        names = [case.get('name') for case in selected]
        if len(set(names)) != len(names) or len(selected) < minimum or any(
                any(case.find(tag) is not None for tag in ('skipped', 'failure', 'error'))
                for case in selected):
            raise RuntimeError(f'Required CUDA numerical tests did not all run successfully: {classname}::{name}')
        records.append({'classname': classname, 'test': name,
                        'minimum_cases': minimum, 'executed_cases': sorted(names)})
    return records


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
        # Inherited opt-ins must not turn reduced CPU qualification into a
        # requested CUDA oracle run that fails for its intentionally hidden GPU.
        env.update({name: '0' for name in CUDA_COVERAGE_ENVIRONMENT})
    scratch = ROOT.parent / 'Scratch'
    workspace_tools = scratch / 'Environment' / 'tools'
    if workspace_tools.is_dir():
        env['PATH'] = str(workspace_tools) + os.pathsep + env.get('PATH', '')
    return env


def qualify(output: Path, *, snapshot: bool, gpu_lock_timeout: float,
            cpu_only: bool = False, acknowledge_gap: str | None = None,
            reason: str | None = None) -> dict[str, object]:
    output = output.resolve()
    if output == ROOT or output.is_relative_to(ROOT):
        raise ValueError('Qualification artifacts must be outside the repository')
    before = source_identity()
    if before['status'] and not snapshot:
        raise ValueError('Release qualification requires a clean checkout; use --snapshot for development')
    if cpu_only and not snapshot:
        raise ValueError('Reduced CPU-only qualification is allowed only with --snapshot')
    version_check = check_release_version(ROOT, snapshot=snapshot,
                                         acknowledge_gap=acknowledge_gap, reason=reason)
    if version_check['head_commit'] != before['commit']:
        raise ValueError('Source commit changed during the release-version check; retry from a stable checkout')
    for warning in version_check['warnings']:
        print(f'WARNING: {warning}', file=sys.stderr)
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output / 'qualification.json'
    if receipt_path.exists():
        raise FileExistsError(f'Use a new output directory; qualification already exists: {receipt_path}')
    env = _qualification_environment(output, cpu_only=cpu_only)
    if not cpu_only:
        env.update(CUDA_COVERAGE_ENVIRONMENT,
                   XTA_SPHERICAL_TEST_ROOT=str(output))
    scratch = ROOT.parent / 'Scratch'
    gap_arguments = ([] if acknowledge_gap is None else
                     [f'--acknowledge-gap={acknowledge_gap}', f'--reason={reason}'])
    steps = [
        ('full-tests', [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider',
                        '--maxfail=1',
                        '--basetemp', str(output / 'pytest-tmp'),
                        '--junitxml', str(output / 'junit.xml'), 'tests']),
        ('package-inventory', [sys.executable, '-B', 'tools/verify_package_inventory.py']),
        ('source-bundle', [sys.executable, '-B', 'tools/build_source_release.py',
                          '--output-dir', str(output / 'source'), *(['--snapshot'] if snapshot else []),
                          *gap_arguments]),
    ]
    receipt: dict[str, object] = {'kind': 'development-snapshot' if snapshot else 'release',
                                'source_before': before, 'started_utc': datetime.now(timezone.utc).isoformat(),
                                'coverage': 'cpu-only' if cpu_only else 'cpu-and-cuda',
                                'release_version_check': version_check,
                                'required_cuda_tests': [] if cpu_only else [
                                    {'classname': classname, 'test': name, 'minimum_cases': minimum}
                                    for classname, name, minimum in REQUIRED_CUDA_TESTS],
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
                    receipt['cuda_numerical_tests'] = _verify_cuda_tests(output / 'junit.xml')
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
    parser.add_argument('--acknowledge-gap', metavar='FROM:TO',
                        help='Acknowledge this exact version gap only, e.g. v24.1.0:v26.0.0; requires --reason')
    parser.add_argument('--reason', help='Record the user-approved reason for --acknowledge-gap')
    args = parser.parse_args()
    if args.gpu_lock_timeout < 0:
        parser.error('--gpu-lock-timeout must be nonnegative')
    try:
        qualify(args.output_dir, snapshot=args.snapshot, gpu_lock_timeout=args.gpu_lock_timeout,
                cpu_only=args.cpu_only, acknowledge_gap=args.acknowledge_gap, reason=args.reason)
    except ValueError as exc:
        parser.error(str(exc))
    print('CPU-only development qualification passed.' if args.cpu_only else
          'Complete CPU and CUDA repository qualification passed.', flush=True)


if __name__ == '__main__':
    main()

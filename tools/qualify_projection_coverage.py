"""Run selected projection coverage checks under one atomic GPU reservation.

This is a numerical coverage/routing qualification, never a full release gate
or throughput benchmark. A suite names exact test functions and explicitly
labels supported CUDA routes versus deliberately guarded CPU routes. Artifacts
must live in Scratch. No GPU work occurs with --plan-only.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.qualify_release import gpu_reservation, _qualification_environment

FAMILIES = ('cartesian', 'tilted', 'azimuthal', 'tilted_azimuthal', 'radial', 'spherical')
SOURCE_PATTERNS = (
    'XTA/geometry.py', 'XTA/backprojection.py', 'XTA/assembly.py', 'XTA/outputs.py',
    'XTA/interpolation.py', 'XTA/pipeline.py', 'XTA/cuda_d1.py', 'XTA/d1_*coverage*.py',
    'XTA/projection_coverage*.py', 'XTA/cylindrical*.py', 'XTA/spherical*.py',
    'XTA/tilted_azimuthal*.py', 'XTA/unification/sampling.py',
    'XTA/unification/contracts.py', 'XTA/unification/tta_manifest.py',
    'tests/reference_backends/*.py', 'tools/qualify_projection_coverage.py',
    'tools/qualify_release.py',
)
GPU_FLAGS = dict(YOLO_TTA_TEST_D1_ORTHOGONAL_CUDA='1',
    XTA_TEST_TILTED_AZIMUTHAL_CUDA='1', XTA_RUN_CUDA_RADIAL_PROJECTION='1',
    XTA_TEST_SPHERICAL_CUDA='1')


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def read_suite(path, selected):
    suite = json.loads(Path(path).read_text(encoding='utf-8'))
    if suite.get('schema') != 'xta.projection-cuda-suite/1' or not isinstance(suite.get('groups'), dict):
        raise ValueError('Suite must declare schema xta.projection-cuda-suite/1 and groups')
    groups = {}
    for family in selected:
        spec = suite['groups'].get(family)
        if not isinstance(spec, dict) or spec.get('route') not in ('actual_cuda', 'guarded_cpu_route'):
            raise ValueError(f'{family} needs explicit actual_cuda or guarded_cpu_route routing')
        nodes = spec.get('node_ids')
        if not isinstance(nodes, list) or not nodes or len(nodes) != len(set(nodes)):
            raise ValueError(f'{family} requires unique exact test function node IDs')
        for node in nodes:
            pieces = str(node).split('::')
            if len(pieces) not in (2, 3) or not pieces[-1].startswith('test_'):
                raise ValueError(f'Use an exact file::test_function or file::Class::test_method: {node}')
            relative = Path(pieces[0])
            target = (ROOT/relative).resolve()
            if relative.is_absolute() or not target.is_relative_to(ROOT/'tests') or not target.is_file():
                raise ValueError(f'Test node must reference a repository tests file: {node}')
        environment = spec.get('environment', {})
        if not isinstance(environment, dict) or any(not isinstance(k, str) or not isinstance(v, str)
                                                   for k, v in environment.items()):
            raise ValueError(f'{family} environment must contain string values')
        if any(not key.startswith(('XTA_', 'YOLO_TTA_')) for key in environment):
            raise ValueError('Suite may override only XTA_/YOLO_TTA_ test controls')
        if any(key.endswith(('_ROOT', '_DIR', '_PATH', '_OUTPUT')) or '_TEMP' in key
               for key in environment):
            raise ValueError('Artifact paths are fixed by the Scratch-only runner, not suite overrides')
        groups[family] = {**spec, 'environment': environment}
    return suite, groups


def guarded_source_paths(groups, suite):
    paths = {path.resolve() for pattern in SOURCE_PATTERNS for path in ROOT.glob(pattern) if path.is_file()}
    for spec in groups.values():
        paths.update((ROOT/node.split('::')[0]).resolve() for node in spec['node_ids'])
    for name in suite.get('guard_files', ()):
        path = (ROOT/str(name)).resolve()
        if not path.is_relative_to(ROOT) or not path.is_file():
            raise ValueError(f'Additional guard must name an existing repository file: {name}')
        paths.add(path)
    return tuple(sorted(paths))


def source_hashes(paths):
    return {path.relative_to(ROOT).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
            if path.is_file() else None for path in paths}


def hardware_probe(env):
    command = [sys.executable, '-B', '-c',
        'import json,platform,torch,cupy as cp; '
        'available=torch.cuda.is_available(); count=torch.cuda.device_count() if available else 0; '
        'print(json.dumps({"platform":platform.platform(),"torch":torch.__version__, '
        '"cupy":cp.__version__,"cuda_available":available,"cupy_devices":cp.cuda.runtime.getDeviceCount(), '
        '"driver_version":cp.cuda.runtime.driverGetVersion(),"runtime_version":cp.cuda.runtime.runtimeGetVersion(), '
        '"devices":[{"index":i,"name":torch.cuda.get_device_name(i), '
        '"capability":list(torch.cuda.get_device_capability(i)), '
        '"total_bytes":torch.cuda.get_device_properties(i).total_memory} for i in range(count)]}))']
    completed = subprocess.run(command, cwd=ROOT, env=env, check=True, capture_output=True,
                               encoding='utf-8', errors='strict')
    probe = json.loads(completed.stdout)
    if not probe['cuda_available'] or not probe['devices'] or probe['cupy_devices'] < 1:
        raise RuntimeError('Actual CUDA qualification requires Torch CUDA and CuPy; no skip is accepted')
    return probe


def junit_result(path, nodes, minimum):
    cases = list(ET.parse(path).iter('testcase'))
    records = []
    for case in cases:
        bad = next((tag for tag in ('failure', 'error', 'skipped') if case.find(tag) is not None), None)
        records.append(dict(classname=case.get('classname'), name=case.get('name'),
                            status='passed' if bad is None else bad,
                            detail='' if bad is None else case.find(bad).get('message', '')))
    missing = []
    for node in nodes:
        parts = node.split('::')
        module = Path(parts[0]).with_suffix('').as_posix().replace('/', '.')
        classname = module + ('.'+parts[1] if len(parts) == 3 else '')
        leaf = parts[-1]
        if not any(row['classname'] == classname and
                   (row['name'] == leaf or ('[' not in leaf and row['name'].startswith(leaf+'[')))
                   for row in records):
            missing.append(node)
    success = len(cases) >= int(minimum) and not missing and all(row['status'] == 'passed' for row in records)
    return dict(success=success, test_cases=len(cases), missing_nodes=missing, cases=records)


def qualify(suite_path, output, scratch, selected, *, plan_only=False, lock_timeout=3600.):
    scratch, output = Path(scratch).resolve(), Path(output).resolve()
    if scratch == ROOT or scratch.is_relative_to(ROOT) or not output.is_relative_to(scratch):
        raise ValueError('Qualification output must live under Scratch, outside the repository')
    suite, groups = read_suite(suite_path, selected)
    paths = guarded_source_paths(groups, suite)
    before = source_hashes(paths)
    suite_bytes = Path(suite_path).read_bytes()
    output.mkdir(parents=True, exist_ok=True)
    receipt_path = output/'projection_coverage.json'
    if receipt_path.exists():
        raise FileExistsError('Use a fresh output directory for each qualification attempt')
    receipt = dict(schema='xta.projection-coverage-qualification/1', coverage_only=True,
        release_qualified=False, started_utc=utc_now(), python=sys.version, platform=platform.platform(),
        suite_path=str(Path(suite_path).resolve()), suite_sha256=hashlib.sha256(suite_bytes).hexdigest(),
        required_families=list(FAMILIES), selected_families=list(selected),
        source_guard_scope='executed projection modules, selected tests/oracles and helper; excludes docs/release pins',
        source_before=before, source_after=None, groups=[], success=False, plan_only=bool(plan_only))
    try:
        if plan_only:
            receipt['planned_groups'] = groups
            return receipt
        env = _qualification_environment(output, cpu_only=False)
        env.update(GPU_FLAGS, XTA_SPHERICAL_TEST_ROOT=str(output),
                   XTA_PROJECTION_TEST_ROOT=str(output))
        with gpu_reservation(scratch/'Temp'/'GPU_LOCK', lock_timeout,
                task_name='v25.0.1 bounded projection coverage CUDA qualification'):
            receipt['hardware_before'] = hardware_probe(env)
            for family, spec in groups.items():
                current_env = {**env, **spec['environment']}
                junit, log = output/(family+'.junit.xml'), output/(family+'.log')
                command = [sys.executable, '-B', '-m', 'pytest', '-p', 'no:cacheprovider',
                    '--maxfail=1', '-q', '--basetemp', str(output/(family+'-tmp')),
                    '--junitxml', str(junit), *spec['node_ids']]
                print(f'Projection {family} ({spec["route"]}); log {log}', flush=True)
                with log.open('w', encoding='utf-8') as handle:
                    completed = subprocess.run(command, cwd=ROOT, env=current_env,
                        stdout=handle, stderr=subprocess.STDOUT)
                parsed = (junit_result(junit, spec['node_ids'], spec.get('minimum_test_cases', len(spec['node_ids'])))
                    if junit.is_file() else dict(success=False, test_cases=0,
                        missing_nodes=spec['node_ids'], cases=[]))
                receipt['groups'].append(dict(family=family, route=spec['route'], command=command,
                    environment=spec['environment'], returncode=completed.returncode,
                    log=log.name, junit=junit.name, **parsed))
                if completed.returncode or not parsed['success']:
                    raise RuntimeError(f'{family} checks failed/missing/skipped; see {log}')
                if source_hashes(paths) != before or Path(suite_path).read_bytes() != suite_bytes:
                    raise RuntimeError('Guarded projection source/test suite changed during qualification')
            receipt['hardware_after'] = hardware_probe(env)
            if receipt['hardware_before'] != receipt['hardware_after']:
                raise RuntimeError('CUDA hardware/runtime identity changed during qualification')
        receipt['success'] = True
        receipt['complete_required_matrix'] = set(selected) == set(FAMILIES)
        receipt['actual_cuda_families'] = [row['family'] for row in receipt['groups'] if row['route'] == 'actual_cuda']
        receipt['guarded_cpu_families'] = [row['family'] for row in receipt['groups'] if row['route'] == 'guarded_cpu_route']
        return receipt
    except BaseException as error:
        receipt['error'] = f'{type(error).__name__}: {error}'
        raise
    finally:
        original_error = sys.exc_info()[1]
        receipt['finished_utc'] = utc_now()
        receipt['source_after'] = source_hashes(paths)
        try:
            suite_changed = Path(suite_path).read_bytes() != suite_bytes
        except OSError as error:
            suite_changed = True
            receipt['suite_after_error'] = str(error)
        changed = receipt['source_after'] != before or suite_changed
        if changed:
            receipt['success'] = False
            receipt['source_changed'] = True
        receipt_path.write_text(json.dumps(receipt, indent=2)+'\n', encoding='utf-8')
        if changed and not plan_only and original_error is None:
            raise RuntimeError('Guarded projection source/test suite changed at completion; no coverage was qualified')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--suite', type=Path, required=True, help='Exact coverage test node/route manifest')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--scratch-root', type=Path, default=ROOT.parent/'Scratch')
    parser.add_argument('--families', default=','.join(FAMILIES), help='Explicit subset remains partial coverage')
    parser.add_argument('--plan-only', action='store_true', help='Write source/selection inventory without GPU work')
    parser.add_argument('--gpu-lock-timeout', type=float, default=3600.)
    args = parser.parse_args()
    selected = tuple(args.families.split(','))
    if len(selected) != len(set(selected)) or any(family not in FAMILIES for family in selected):
        parser.error('families must be unique known family names')
    if args.gpu_lock_timeout < 0:
        parser.error('GPU lock timeout must be nonnegative')
    qualify(args.suite, args.output_dir, args.scratch_root, selected,
        plan_only=args.plan_only, lock_timeout=args.gpu_lock_timeout)
    print('Projection plan written; no GPU run.' if args.plan_only else
          'Selected projection CUDA/routing checks passed; this is not release qualification.')


if __name__ == '__main__':
    main()

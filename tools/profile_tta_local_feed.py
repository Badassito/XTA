"""Repeat a saved local TTA invocation under an exclusive GPU reservation.

This profiles the real scheduler, worker and output pipeline on one local GPU.
It does not predict multi-H100 throughput. All evidence stays in sibling Scratch.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent / 'Scratch'


def replace_option(argv, option, value):
    index = argv.index(option)
    argv[index + 1] = str(value)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--invocation', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--runs', type=int, default=2)
    parser.add_argument('--heat-seconds', type=float, default=60)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--cache-dir', type=Path,
                        help='Reuse a warmed compilation cache beneath Scratch')
    parser.add_argument('--source-root', type=Path, default=ROOT,
                        help='Source checkout/snapshot containing XTA for this arm')
    parser.add_argument('--cprofile', action='store_true',
                        help='Record parent Python call timings; use a separate diagnostic run')
    parser.add_argument('--line-profile', action='store_true',
                        help='Record bounded source-line gaps in scheduler dispatch')
    args = parser.parse_args()
    output = args.output.resolve()
    cache = (args.cache_dir or output).resolve()
    source_root = args.source_root.resolve()
    if (not output.is_relative_to(SCRATCH.resolve()) or output.exists()
            or not cache.is_relative_to(SCRATCH.resolve())
            or not (source_root / 'XTA/__init__.py').is_file()
            or (args.cprofile and args.line_profile)
            or args.runs < 1 or args.workers < 1 or args.heat_seconds < 0):
        parser.error('Use a new output beneath Scratch, positive runs/workers and nonnegative heat time')
    recipe = json.loads(args.invocation.read_text(encoding='utf-8'))
    argv = [sys.executable, *recipe['argv'][1:]]
    if '--device' not in argv or argv[argv.index('--device') + 1] != '0':
        parser.error('This local profiler requires a single device 0 invocation')
    output.mkdir(parents=True)
    environment = dict(os.environ)
    environment.update(recipe.get('environment', {}))
    environment.update(PYTHONDONTWRITEBYTECODE='1', PYTHONIOENCODING='utf-8',
                       SLURM_CPUS_PER_TASK=str(args.workers),
                       YOLO_TTA_TASK_TRACE='1', YOLO_TTA_TELEMETRY='1',
                       YOLO_TTA_TAIL_WORKER_BUDGET_EXPAND='0',
                       YOLO_AUTOINSTALL='false')
    environment['PATH'] = str(SCRATCH / 'Environment/tools') + os.pathsep + environment['PATH']
    environment['PYTHONPATH'] = os.pathsep.join(
        [str(source_root), str(ROOT), *environment.get('PYTHONPATH', '').split(os.pathsep)])
    environment.pop('YOLO_TTA_TELEMETRY_PATH', None)
    environment['CUPY_CACHE_DIR'] = str(cache / 'cupy-cache')
    environment['NUMBA_CACHE_DIR'] = str(cache / 'numba-cache')
    lock = SCRATCH / 'Temp/GPU_LOCK'
    lock.parent.mkdir(parents=True, exist_ok=True)
    claim = dict(task='148334 local feed profile', pid=os.getpid(),
                 start_time=datetime.now(timezone.utc).isoformat(), token=uuid.uuid4().hex)
    while True:
        try:
            with lock.open('x', encoding='utf-8') as handle:
                json.dump(claim, handle)
            break
        except FileExistsError:
            print('Waiting for GPU_LOCK', flush=True)
            time.sleep(15)
    report = dict(scope=__doc__, claim=claim, source=str(source_root), runs=[], status='running')
    receipt = output / 'profile.json'
    def save():
        receipt.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    try:
        os.environ.update(environment)
        sys.path.insert(0, str(ROOT))
        for entry in environment['PYTHONPATH'].split(os.pathsep):
            if entry and entry not in sys.path:
                sys.path.append(entry)
        import torch
        from tools.benchmark_radial_setup import heatsoak
        if not torch.cuda.is_available():
            raise RuntimeError('Local CUDA is unavailable')
        report['gpu'] = torch.cuda.get_device_name(0)
        save()
        print(f'Heatsoaking {report["gpu"]} for {args.heat_seconds:g}s', flush=True)
        report['heatsoak_seconds'] = heatsoak(args.heat_seconds, 0)
        for number in range(1, args.runs + 1):
            case = output / f'run-{number:02d}'
            case.mkdir()
            command = list(argv)
            replace_option(command, '--output', case / 'outputs')
            replace_option(command, '--temp', case / 'runtime')
            if args.cprofile:
                module_index = command.index('-m')
                command[module_index:module_index] = ['-m', 'cProfile', '-o', str(case / 'parent.prof')]
            elif args.line_profile:
                module_index = command.index('-m')
                command = [*command[:module_index], '-m', 'tools.profile_tta_scheduler_lines',
                           '--output', str(case / 'scheduler-lines.json'),
                           '--script', str(ROOT / 'GPT-6-Astra-Ultra_v22.3.2_SLURM.py'),
                           '--', *command[module_index + 2:]]
            child_env = dict(environment)
            child_env.update(YOLO_TTA_TELEMETRY_DIR=str(case / 'telemetry'),
                             YOLO_CONFIG_DIR=str(case / 'ultralytics-config'),
                             YOLO_TTA_TRACE_RUN_ID=f'local-feed-{number:02d}')
            (case / 'invocation.json').write_text(json.dumps(dict(argv=command,
                environment={key: value for key, value in child_env.items()
                             if key.startswith(('YOLO_', 'NUMBA_', 'CUPY_', 'SLURM_'))
                             or key in ('PYTHONPATH', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS')}),
                indent=2) + '\n', encoding='utf-8')
            started = time.perf_counter()
            print(f'Starting local pipeline run {number}', flush=True)
            with (case / 'pipeline.log').open('w', encoding='utf-8') as handle:
                result = subprocess.run(command, cwd=source_root, env=child_env,
                                        stdout=handle, stderr=subprocess.STDOUT)
            row = dict(run=number, process_seconds=time.perf_counter()-started,
                       returncode=result.returncode, output=str(case))
            report['runs'].append(row)
            save()
            print(row, flush=True)
            if result.returncode:
                raise RuntimeError(f'Pipeline failed; see {case / "pipeline.log"}')
        report['status'] = 'complete'
    except BaseException as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        try:
            save()
        finally:
            if json.loads(lock.read_text(encoding='utf-8')).get('token') == claim['token']:
                lock.unlink()


if __name__ == '__main__':
    main()

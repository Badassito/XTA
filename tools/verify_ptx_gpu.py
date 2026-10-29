"""Reserve the GPU and run reproducible PTX-assessment verification experiments.

All generated evidence and caches go to the required Scratch output directory.
Experimental kernels live in companion tools; production XTA is unchanged.
"""
from __future__ import annotations

import argparse
import gc
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent / 'Scratch'

if __package__:
    from .gpu_reservation import gpu_reservation
else:
    from gpu_reservation import gpu_reservation


def save(path, value):
    path.write_text(json.dumps(value, indent=2, default=str) + '\n', encoding='utf-8')


def heatsoak(torch, device, seconds):
    started = time.perf_counter()
    if not seconds:
        return dict(requested_seconds=0., actual_seconds=0., performed=False)
    with torch.inference_mode():
        left = torch.randn((4096,4096),device=device,dtype=torch.float16)
        right = torch.randn_like(left)
        target = torch.empty_like(left)
        last = started
        while time.perf_counter()-started < seconds:
            for _ in range(16):
                torch.mm(left,right,out=target)
            torch.cuda.synchronize(device)
            if time.perf_counter()-last >= 15:
                print(f'GPU heatsoak: {time.perf_counter()-started:.0f}s',flush=True)
                last = time.perf_counter()
        del left,right,target
    torch.cuda.synchronize(device)
    gc.collect()
    torch.cuda.empty_cache()
    return dict(requested_seconds=seconds,actual_seconds=time.perf_counter()-started,performed=True)



def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--stage', choices=('baseline', 'prototypes', 'followup', 'final-check'), required=True)
    parser.add_argument('--heat-seconds', type=float, default=90)
    args = parser.parse_args()
    out = args.output.resolve()
    if not out.is_relative_to(SCRATCH.resolve()):
        parser.error('--output must be beneath the sibling Scratch directory')
    out.mkdir(parents=True, exist_ok=True)
    lock = SCRATCH / 'Temp' / 'GPU_LOCK'
    with gpu_reservation(lock, task_name='PTX report verification / ' + args.stage,
                         poll_interval=15.) as token:
        receipt = dict(stage=args.stage, lock=token, commands=[], status='running',
                       retired_scatter_qualification='unsupported; omitted from current baseline')
        record = out / (args.stage + '-receipt.json')
        try:
            (SCRATCH / 'ultralytics-config').mkdir(parents=True, exist_ok=True)
            env = os.environ
            env.update(PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1',
                       PYTHONIOENCODING='utf-8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
                       CUPY_CACHE_DIR=str(out / 'cupy-cache'),
                       CUPY_CACHE_SAVE_CUDA_SOURCE='1', NUMBA_CACHE_DIR=str(out / 'numba-cache'),
                       TORCHINDUCTOR_CACHE_DIR=str(out / 'torchinductor-cache'),
                       YOLO_CONFIG_DIR=str(SCRATCH / 'ultralytics-config'),
                       TEMP=str(out / 'temp'), TMP=str(out / 'temp'))
            (out / 'temp').mkdir(exist_ok=True)
            sys.path.insert(0, str(ROOT))
            import torch
            import cupy as cp
            import numpy as np
            torch.set_num_threads(2)
            if not torch.cuda.is_available():
                raise RuntimeError('CUDA is unavailable')
            torch.cuda.set_device(0)
            cp.cuda.Device(0).use()
            # Actually compile and run a tiny CuPy operation; availability alone is insufficient.
            assert int(cp.sum(cp.arange(10)).get()) == 45
            sm = torch.cuda.get_device_capability(0)
            versions = {d.metadata['Name']: d.version for d in importlib.metadata.distributions()
                        if any(s in d.metadata['Name'].lower() for s in
                               ('torch', 'cupy', 'numpy', 'tensorrt', 'cuda', 'scipy'))}
            receipt.update(python=sys.version, executable=sys.executable, packages=versions,
                           gpu=torch.cuda.get_device_name(0), sm=sm, torch_cuda=torch.version.cuda,
                           cupy_runtime=cp.cuda.runtime.runtimeGetVersion(),
                           driver=cp.cuda.runtime.driverGetVersion(), nvrtc=cp.cuda.nvrtc.getVersion(),
                           environment={k: v for k, v in env.items()
                                        if k.startswith(('YOLO_', 'XTA_', 'CUPY_', 'CUDA_'))},
                           source_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT,
                               text=True, encoding='utf-8', errors='replace').strip(),
                           source_status=subprocess.check_output(['git', 'status', '--short'], cwd=ROOT,
                               text=True, encoding='utf-8', errors='replace'))
            def telemetry():
                return subprocess.check_output(['nvidia-smi', '--query-gpu=name,driver_version,temperature.gpu,power.draw,clocks.sm,clocks.mem,memory.used,utilization.gpu', '--format=csv'],
                    text=True, encoding='utf-8', errors='replace').strip()
            receipt['telemetry_before'] = telemetry()
            receipt['heatsoak'] = heatsoak(torch, 'cuda:0', args.heat_seconds)
            receipt['telemetry_after_heatsoak'] = telemetry()
            print(json.dumps({k: receipt[k] for k in ('gpu','sm','packages','nvrtc','telemetry_after_heatsoak')}, default=str), flush=True)
            save(record, receipt)

            def run(name, command, flags=None):
                command = [sys.executable, '-B'] + list(map(str, command))
                child_env = dict(env)
                child_env.update(flags or {})
                row = dict(name=name, command=command, flags=flags or {}, started=datetime.now(timezone.utc).isoformat())
                row['script_sha256'] = {str(value): hashlib.sha256((ROOT / value).read_bytes()).hexdigest()
                                        for value in command[2:] if str(value).endswith('.py') and (ROOT / value).is_file()}
                receipt['commands'].append(row)
                row['telemetry_before'] = telemetry()
                print('RUN ' + name, flush=True)
                started = time.perf_counter()
                with (out / (name + '.log')).open('w', encoding='utf-8') as log:
                    result = subprocess.run(command, cwd=ROOT, env=child_env, stdout=log, stderr=subprocess.STDOUT)
                row.update(returncode=result.returncode, wall_seconds=time.perf_counter()-started)
                row['telemetry_after'] = telemetry()
                save(record, receipt)
                print('DONE ' + name + ': ' + str(result.returncode), flush=True)
                if result.returncode:
                    print((out / (name + '.log')).read_text(encoding='utf-8', errors='replace')[-6000:], flush=True)

            if args.stage == 'baseline':
                for module, flag in (
                    ('test_radial_column_geometry', 'XTA_TEST_RADIAL_COLUMNS_CUDA'),
                    ('test_spherical_fp32', 'XTA_TEST_SPHERICAL_FP32_CUDA'),
                    ('test_cuda_interpolation', 'XTA_TEST_CUDA')):
                    run(module, ['-m', 'unittest', 'tests.' + module, '-v'], {flag: '1'})
                run('fp32-union', ['tools/benchmark_fp32_union.py', '--output', out/'fp32-union.json', '--heat-seconds', '30'])
                run('radial-columns', ['tools/benchmark_radial_columns.py', '--output-dir', out/'radial-columns',
                                      '--heat-seconds','30','--patch-sizes','64','768','3072'])
            elif args.stage == 'prototypes':
                run('d1-prototype', ['tools/verify_ptx_d1.py', '--output', out/'d1-prototype.json'])
                run('sdf-prototype', ['tools/verify_ptx_sdf.py', '--output', out/'sdf-prototype.json'])
            elif args.stage == 'followup':
                run('d1-graph', ['tools/verify_ptx_d1.py', '--output', out/'d1-graph.json',
                                 '--graph-timing', '--rounds','9','--iterations','100'])
                run('sdf-qualified', ['tools/verify_ptx_sdf.py', '--output', out/'sdf-qualified.json'])
                run('direct-tiled-tests', ['-m','unittest','tests.test_direct_tiled_compaction','-v'],
                    {'XTA_TEST_DIRECT_TILED_CUDA':'1'})
                run('mask-edges', ['tools/verify_ptx_mask_edges.py','--output',out/'mask-edges.json'])
                run('spherical-large-address', ['-m','unittest',
                    'tests.test_spherical_fp32.SphericalFp32CudaTests.test_source_offsets_above_four_gib_remain_uint64','-v'],
                    {'XTA_TEST_SPHERICAL_FP32_CUDA':'1','XTA_TEST_SPHERICAL_LARGE_SOURCE':'1'})
                run('packed-prototype', ['tools/verify_ptx_packed.py','--output',out/'packed-prototype.json'])
            else:
                run('sdf-final', ['tools/verify_ptx_sdf.py', '--output', out/'sdf-final.json'])
                run('packed-graph', ['tools/verify_ptx_packed.py', '--output', out/'packed-graph.json', '--graph-timing'])
            receipt['telemetry_final'] = telemetry()
            receipt['status'] = 'passed' if all(r['returncode'] == 0 for r in receipt['commands']) else 'failures'
        except BaseException as exc:
            receipt.update(status='failed', error=repr(exc), traceback=traceback.format_exc())
            raise
        finally:
            receipt['finished'] = datetime.now(timezone.utc).isoformat()
            save(record, receipt)
            print('GPU verification receipt: ' + str(record), flush=True)


if __name__ == '__main__':
    main()

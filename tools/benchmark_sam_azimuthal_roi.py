"""Qualify exact SAM native ROI pixels and measure the local rendering mechanism."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import threading
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--warmup-seconds', type=float, default=45.)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--qualification-receipt', type=Path,
                        help='Reuse a passed receipt with matching production/test hashes for timing only')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    out = args.output_dir.resolve()
    scratch = (repo.parent/'Scratch').resolve()
    if out == scratch or not out.is_relative_to(scratch) or args.warmup_seconds < 30 or args.repeats < 1:
        parser.error('Use a task directory under Scratch, at least 30s warmup, and positive repeats')
    if (out/'results.json').exists():
        parser.error('Use a fresh output directory to preserve qualification evidence')
    out.mkdir(parents=True, exist_ok=True)
    for key, folder in [('TEMP','temp'),('TMP','temp'),('CUDA_CACHE_PATH','cuda-cache'),
                        ('NUMBA_CACHE_DIR','numba-cache'),('YOLO_CONFIG_DIR','yolo-config')]:
        path = out/folder; path.mkdir(exist_ok=True); os.environ[key] = str(path)
    os.environ.update(CUDA_VISIBLE_DEVICES='0',YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER='0',
        YOLO_TTA_TELEMETRY='0',YOLO_TTA_TASK_TRACE='0',PYTHONDONTWRITEBYTECODE='1',
        OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMBA_NUM_THREADS='1',
        XTA_RUN_CUDA_RENDER_INTEGRATION='1')
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(repo))
    from tools.qualify_release import gpu_reservation
    report = dict(status='started',warmup_seconds=args.warmup_seconds,cases=[],
        note='Local mechanism check on a bounded 64x2048x2048 source; not target-system utilization or throughput proof.')
    files = ('XTA/cuda_backend.py','XTA/sam_gpu_rendering.py','tests/test_sam_azimuthal_native_roi.py',
             'tests/test_sam_azimuthal_roi_oracle.py','tools/benchmark_sam_azimuthal_roi.py')
    hashes = {name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in files}
    report['source_sha256'] = hashes
    if args.qualification_receipt:
        receipt_bytes = args.qualification_receipt.read_bytes()
        qualified = json.loads(receipt_bytes)
        required = [name for name in files if not name.startswith('tools/')]
        if (qualified.get('status') != 'passed' or qualified.get('pytest_exit_code') != 0
                or qualified.get('changed_during_test')
                or any(qualified.get('source_sha256',{}).get(name) != hashes[name] for name in required)):
            parser.error('Timing-only qualification receipt must prove the current production and test hashes')
        report.update(qualification_reference=str(args.qualification_receipt.resolve()),
                      qualification_reference_sha256=hashlib.sha256(receipt_bytes).hexdigest())
    def save():
        (out/'results.json').write_text(json.dumps(report,indent=2)+'\n')
    try:
        with gpu_reservation(scratch/'Temp/GPU_LOCK',300.,task_name='Run157372 exact azimuthal native ROI qualification'):
            import numpy as np
            import torch
            import pytest
            from dataclasses import replace
            from types import SimpleNamespace
            from tests.test_sam_azimuthal_native_roi import engine_for,views_for
            from XTA.sam_gpu_rendering import SamGpuCropRenderer
            from XTA.sam_evidence import _encode_mask
            torch.set_num_threads(1)
            torch.set_num_interop_threads(1)
            assert torch.cuda.is_available()
            report.update(device=torch.cuda.get_device_name(0),torch=torch.__version__,cuda=torch.version.cuda)
            save()
            @contextmanager
            def consumers(count):
                stop = threading.Event()
                errors = []
                def work(index):
                    mask = np.random.default_rng(index).integers(0,5,(512,512),dtype=np.uint8)==0
                    try:
                        while not stop.is_set():
                            encoded = _encode_mask(mask)
                            json.dumps([dict(frame=n,size=len(encoded[0]),sha256=encoded[2]) for n in range(192)])
                    except BaseException as error:
                        errors.append(repr(error))
                threads = [threading.Thread(target=work,args=(i,),daemon=True) for i in range(count)]
                for thread in threads: thread.start()
                try:
                    yield
                finally:
                    stop.set()
                    for thread in threads: thread.join(30)
                    assert not errors and not any(thread.is_alive() for thread in threads),errors
            a = torch.randn((4096,4096),device='cuda:0',dtype=torch.float16)
            b = torch.randn_like(a)
            print('GPU lock acquired; CPU/GPU heatsoak started',flush=True)
            started = time.perf_counter()
            with consumers(4):
                while time.perf_counter()-started < args.warmup_seconds:
                    for _ in range(4): product = a@b
                    torch.cuda.synchronize()
            del a,b,product
            print('Heatsoak finished; using qualified source' if args.qualification_receipt
                  else 'Heatsoak finished; exact CUDA qualification started',flush=True)
            test_result = 0 if args.qualification_receipt else pytest.main(['-q','-s','-p','no:cacheprovider',
                '--basetemp='+str(scratch/'Temp/r157372-roi-gpu'),
                str(repo/'tests/test_sam_azimuthal_native_roi.py'),
                str(repo/'tests/test_sam_azimuthal_roi_oracle.py'),
                str(repo/'tests/test_sam_azimuthal_tap_gather.py')])
            report['pytest_exit_code'] = int(test_result)
            save()
            if test_result:
                raise RuntimeError('Exact CUDA qualification failed')
            source = np.random.default_rng(157372).integers(0,256,(64,2048,2048),dtype=np.uint8)
            engine = engine_for(source,2048,'cuda:0')
            matrix = np.array([[.73,.04,71.25],[-.03,.81,59.5]],np.float32)
            try:
                views = [replace(view,src_h=2048,src_w=2048) for view in views_for(source,2048)[1:]]
                for count in (0,12):
                    with consumers(count):
                        for view in views:
                            renderer = SamGpuCropRenderer(SimpleNamespace(_check_image_lifetime=lambda:None),
                                view,source,2048,SimpleNamespace(device_index=0),torch,1024**2)
                            renderer.engine = engine
                            times = {'full':[], 'roi':[]}
                            def measure(mode):
                                if mode == 'full':
                                    renderer._azimuthal_native_u8_crop = lambda *_args:None
                                elif '_azimuthal_native_u8_crop' in renderer.__dict__:
                                    del renderer._azimuthal_native_u8_crop
                                started = time.perf_counter()
                                for frame in (2,3,6):
                                    renderer.render(view,frame,matrix,output_origin_yx=(0,0),
                                        output_height=129,output_width=242)
                                return (time.perf_counter()-started)/3
                            for mode in ('full','roi'): measure(mode)
                            for repeat in range(args.repeats):
                                for mode in ('full','roi','roi','full'):
                                    times[mode].append(measure(mode))
                            for mode in ('full','roi'):
                                case = dict(consumers=count,view=view.name,mode=mode,
                                    order='ABBA',median_seconds_per_frame=statistics.median(times[mode]),samples=times[mode])
                                report['cases'].append(case)
                                print(json.dumps(case),flush=True)
                                save()
            finally:
                engine._stream.synchronize()
                engine._volume_gpu = None
                engine._native_t_map_cache.clear()
                engine._fold_cache.clear()
        after = {name:hashlib.sha256((repo/name).read_bytes()).hexdigest() for name in files}
        report['changed_during_test'] = [name for name in hashes if hashes[name] != after[name]]
        if report['changed_during_test']:
            raise RuntimeError('Qualified source changed during the run')
        report['status'] = 'passed'
        save()
    except BaseException as error:
        report.update(status='failed',error=repr(error))
        save()
        raise


if __name__ == '__main__':
    main()

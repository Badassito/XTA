"""Heatsoaked, bounded-z CUDA qualification of Spherical output batching.

Writes source identity, exact normalized output hashes, handshake counts and
observed/bounded memory to Scratch. It does not run model inference or claim
cluster end-to-end throughput from a local GPU microbenchmark.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import struct
import sys
import time


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--span', type=int, default=224)
    parser.add_argument('--repetitions', type=int, default=3,
                        help='Number of ABBA timing cycles after warmup')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    scratch = repo.parent/'Scratch'
    output = args.output_dir.resolve()
    if repo == output or repo in output.parents:
        raise ValueError(f'Qualification output must be outside the repository: {output} (repo {repo})')
    output.mkdir(parents=True, exist_ok=True)
    for key, folder in (('TEMP','temp'),('TMP','temp'),('CUPY_CACHE_DIR','cupy'),('NUMBA_CACHE_DIR','numba')):
        target = output/folder
        target.mkdir(exist_ok=True)
        os.environ[key] = str(target)
    import tempfile
    tempfile.tempdir = str(output/'temp')
    sys.path.insert(0, str(repo))
    lock = scratch/'Temp'/'GPU_LOCK'
    lock.parent.mkdir(exist_ok=True)
    descriptor = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    lock_identity = os.fstat(descriptor)
    os.write(descriptor, json.dumps(dict(task='Spherical block admission qualification',
        pid=os.getpid(), started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()))).encode())
    os.close(descriptor)
    result = dict(status='started', heat_seconds=args.heat_seconds,
                  output_shape=[1930,3064,3023], bounded_z_span=args.span)
    try:
        import numpy as np
        import cupy as cp
        import psutil
        from XTA.spherical_geometry import build_spherical_view_infos
        from XTA.spherical_projection_cuda import SphericalCudaProjector
        from XTA.spherical_projection import _ordered_spherical_cuda_blocks
        from XTA.interpolation import IncrementalRawBBoxMaskStoreWriter, CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT
        from types import SimpleNamespace

        cp.cuda.Device(0).use()
        result['device'] = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        print('GPU lock acquired; generating representative Spherical mask.', flush=True)
        work = (2911,3064,3023)
        candidates = [view for view in build_spherical_view_infos(*work, targets=('transverse',),
            min_radius=None, patch_size=2048, tilted_views=()) if view.spherical_face == 0]
        candidates.sort(key=lambda view:view.spherical_min_radius)
        view = candidates[len(candidates)//2]
        source = np.zeros((view.num_slices,view.src_h,view.src_w),np.uint8)
        yy,xx = np.indices((view.src_h,view.src_w))
        pattern = (((yy//23 + xx//31) % 7) == 0).astype(np.uint8)
        pattern[:64] = pattern[-64:] = 0
        pattern[:,:64] = pattern[:,-64:] = 0
        for z in range(source.shape[0]): source[z] = np.roll(pattern,z%17,axis=0)
        boxes = np.tile(np.asarray((0,view.src_h,0,view.src_w),np.int64),(source.shape[0],1))
        result.update(view=view.name, source_shape=list(source.shape), source_bytes=source.nbytes)
        del yy,xx,pattern
        print(f'Heatsoaking CUDA for {args.heat_seconds:.0f} seconds.', flush=True)
        hot_a = cp.ones((2048,2048),cp.float32)
        hot_b = cp.ones_like(hot_a)
        hot_c = cp.empty_like(hot_a)
        heat_start, next_report = time.perf_counter(),15.
        while time.perf_counter()-heat_start < args.heat_seconds:
            for _ in range(16): cp.matmul(hot_a,hot_b,out=hot_c)
            cp.cuda.get_current_stream().synchronize()
            elapsed = time.perf_counter()-heat_start
            if elapsed >= next_report:
                print(f'Heatsoak {elapsed:.1f}s.',flush=True)
                next_report += 15
        del hot_a,hot_b,hot_c
        cp.get_default_memory_pool().free_all_blocks()
        cp.get_default_pinned_memory_pool().free_all_blocks()
        result['heat_elapsed_seconds'] = time.perf_counter()-heat_start
        process = psutil.Process()

        def run(cap, packed, label):
            gc.collect()
            free_before, _ = cp.cuda.runtime.memGetInfo()
            rss_before = process.memory_info().rss
            started = time.perf_counter()
            projector = SphericalCudaProjector(source,view,tuple(result['output_shape']),boxes,
                                               block_bytes=cap)
            constructor = time.perf_counter()-started
            bounds = projector.output_bounds.block(0,result['output_shape'][0])
            middle = (bounds[0]+bounds[1])//2
            first = max(bounds[0], min(middle-args.span//2,bounds[1]-args.span))
            stop = min(bounds[1], first+args.span)
            if first < 0 or stop <= first:
                raise RuntimeError('Representative Spherical output span is empty')
            sink = IncrementalRawBBoxMaskStoreWriter(shape=tuple(result['output_shape']),
                store_dir=output/f'{label}.cvol',
                format_name=INTERNAL_PACKED_CVOL_FORMAT if packed else CVOL_FORMAT,
                desc=label)
            sink.consume_empty_range(0,first)
            free_min = cp.cuda.runtime.memGetInfo()[0]
            rss_max = process.memory_info().rss
            pool_peak = projector._pool.total_bytes()
            host_bound = (4*projector.output_buffer_bytes + 2*projector.metadata_buffer_bytes
                          + 2*projector.offset_buffer_bytes + 64*1024**2)
            assert host_bound < 2*1024**3
            assert projector.required_device_bytes+projector.reserve_bytes <= projector.admission_free_bytes
            stage = SimpleNamespace(max_block_depth=projector.max_block_depth,
                                    project_encoded=projector.project_encoded,project=projector.project)
            blocks = _ordered_spherical_cuda_blocks(stage,stop,packed,first_z=first)
            publication = 0.
            loop_started = time.perf_counter()
            try:
                for z,block in blocks:
                    then = time.perf_counter()
                    sink.consume_encoded_block(z,block.records,block.payload,packed=packed)
                    publication += time.perf_counter()-then
                    free_min = min(free_min,cp.cuda.runtime.memGetInfo()[0])
                    pool_peak = max(pool_peak,projector._pool.total_bytes())
                    rss_max = max(rss_max,process.memory_info().rss)
                elapsed = time.perf_counter()-loop_started
                sink.consume_empty_range(stop,result['output_shape'][0]-stop)
                store_stats = sink.finalize()
                assert pool_peak <= projector.required_device_bytes
                # Verification is outside the overlapped producer/consumer
                # timer. Subtracting consumer hashing time would incorrectly
                # subtract work already hidden behind the prefetched GPU block.
                hash_started = time.perf_counter()
                digest = hashlib.sha256()
                for path in (sink.index_path, sink.chunks_path):
                    with path.open('rb') as handle:
                        while chunk := handle.read(8*1024**2):
                            digest.update(chunk)
                hashing = time.perf_counter()-hash_started
                metrics = {key:getattr(projector,key) for key in (
                    'max_block_depth','output_buffer_bytes','required_device_bytes','admission_free_bytes',
                    'gpu_projection_blocks','projection_stream_fences','kernel_seconds','metadata_seconds',
                    'pack_seconds','d2h_seconds','payload_d2h_bytes','metadata_d2h_bytes','roi_skipped_blocks')}
                metrics.update(label=label,packed=packed,cap_bytes=cap,first_z=first,stop_z=stop,
                    sha256=digest.hexdigest(),constructor_seconds=constructor,loop_seconds=elapsed,
                    hashing_seconds=hashing,publication_seconds=publication,
                    foreground_voxels=int(store_stats['foreground_voxels']),
                    observed_free_vram_drop_bytes=max(0,int(free_before-free_min)),
                    private_device_pool_peak_bytes=int(pool_peak),
                    observed_rss_growth_bytes=max(0,int(rss_max-rss_before)),host_output_bound_bytes=host_bound)
                print(json.dumps(metrics),flush=True)
                return metrics
            finally:
                blocks.close()
                projector.close()
                sink.discard()

        # Warm the actual QSC and metadata/encoding kernels before recorded ABBA.
        run(64*1024**2,False,'warm-baseline')
        run(None,False,'warm-adaptive')
        recorded=[]
        for index,cap in enumerate((64*1024**2,None,None,64*1024**2)*max(1,args.repetitions)):
            recorded.append(run(cap,False,f'raw-{index}'))
        packed_runs=[run(64*1024**2,True,'packed-baseline'),run(None,True,'packed-adaptive')]
        assert len({row['sha256'] for row in recorded}) == 1
        assert len({row['sha256'] for row in packed_runs}) == 1
        baseline=statistics.median(row['loop_seconds'] for row in recorded if row['cap_bytes'])
        adaptive=statistics.median(row['loop_seconds'] for row in recorded if row['cap_bytes'] is None)
        result.update(status='passed',raw_runs=recorded,packed_runs=packed_runs,
                      raw_median_baseline_seconds=baseline,raw_median_adaptive_seconds=adaptive,
                      raw_speedup=baseline/adaptive)
        for name in ('spherical_projection_cuda.py','spherical_projection.py'):
            result.setdefault('source_sha256',{})[name]=hashlib.sha256((repo/'XTA'/name).read_bytes()).hexdigest()
    except BaseException as error:
        result.update(status='failed',error=repr(error))
        raise
    finally:
        (output/'result.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
        try:
            current=lock.stat()
            if (current.st_dev,current.st_ino)==(lock_identity.st_dev,lock_identity.st_ino): lock.unlink()
        except FileNotFoundError:
            pass


if __name__ == '__main__':
    main()

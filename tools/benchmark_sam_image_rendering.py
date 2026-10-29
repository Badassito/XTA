"""SAM CUDA image preparation sanity; no model or H100 throughput claim."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    parser.add_argument('--frames', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=2)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    scratch = (ROOT.parent/'Scratch').resolve()
    if not output.is_relative_to(scratch) or args.heat_seconds < 30 or min(args.frames, args.repeats) < 1:
        parser.error('Use a Scratch output, >=30s heatsoak and positive frame/repeat counts')
    output.mkdir(parents=True, exist_ok=True)
    from tools.gpu_reservation import gpu_reservation
    lock = scratch/'Temp/GPU_LOCK'
    with gpu_reservation(lock, 300., task_name='SAM155646 image renderer GPU probe'):
        import numpy as np
        import torch
        from XTA import geometry
        from XTA.config import resolve_tilted_view_groups
        from XTA.sam_integration import SamInterpolationContext
        from XTA.sam_gpu_rendering import SamGpuCropRenderer, crop_inverse
        from XTA.sam_canvas_rendering import render_canonical_crop, prepare_native_shell_crop
        from XTA.backprojection import _try_acquire_specific_main_process_gpu_stage
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA qualification cannot skip an unavailable GPU')
        os.environ['YOLO_TTA_GPU_RENDER_RESERVE_GIB'] = '1'
        os.environ['YOLO_TTA_FAST_GEOMETRY'] = '1'
        os.environ['YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER'] = '0'
        os.environ['YOLO_TTA_SAM_GRAY_CACHE'] = '0'
        a = torch.full((4096,4096), .001, dtype=torch.float16, device='cuda:0')
        b = torch.empty_like(a)
        deadline = time.perf_counter()+args.heat_seconds
        def warm_cpu(worker):
            values = np.full(512*1024, .3+worker*.01, np.float64)
            while time.perf_counter() < deadline:
                np.sin(values,out=values)
                np.add(values,.3,out=values)
        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=4) as pool:
            jobs = [pool.submit(warm_cpu, worker) for worker in range(4)]
            while time.perf_counter() < deadline:
                torch.matmul(a,a,out=b)
                torch.cuda.synchronize()
            for job in jobs:
                job.result()
        heat = time.perf_counter()-started
        del a,b
        torch.cuda.empty_cache()
        source = np.random.default_rng(155646).integers(0,256,(512,512,512),np.uint8)
        views = geometry.get_view_infos(*source.shape, cartesian_views=(),
            azimuthal_views=('transverse','tilted_transverse'), azimuthal_azimuth_angles=(30.,30.),
            radial_views=('transverse','tilted_transverse'), radial_min_radius=1., radial_patch_size=512,
            spherical_views=('transverse','tilted_transverse'), spherical_min_radius=1., spherical_patch_size=512,
            tilt_groups=resolve_tilted_view_groups(['transverse:25:both']))
        selected = {}
        for view in views:
            if view.family in {'radial','spherical','azimuthal'}:
                tilted = (bool(view.radial_tilted_source) if view.family=='radial' else
                    bool(view.tilt_angle_deg) if view.family=='azimuthal' else
                    view.spherical_rotation_xyz != (1.,0.,0.,0.,1.,0.,0.,0.,1.))
                selected.setdefault((view.family,tilted),view)
        identity = np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)
        rows = []
        report = dict(schema='xta.sam_gpu_image_component_probe/1',
            device=torch.cuda.get_device_name(0),torch=str(torch.__version__),cuda=torch.version.cuda,
            source_shape=list(source.shape),source_bytes=int(source.nbytes),heatsoak_seconds=heat,
            cpu_heatsoak_workers=4,frames_per_cohort=args.frames,repeats=args.repeats,
            timing_includes='fresh CUDA renderer, source upload, crop renders/D2H, stream-safe retirement',
            limitation='Synthetic local4090mobile/Oculink component sanity. No model, parent concurrency, H100, or downstream mask quality.',rows=rows)
        def save():
            (output/'benchmark.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
        for (family, tilted), view in selected.items():
            frames = tuple(int(value) for value in np.linspace(0,view.num_slices-1,args.frames))
            height,width = min(384,int(view.src_h)-16),min(384,int(view.src_w)-16)
            bbox=(8,8,8+height,8+width)
            row=dict(family=family,tilted=tilted,view=view.name,crop_bbox_yx=bbox,
                native_shape_yx=[view.src_h,view.src_w],cpu_seconds=[],gpu_seconds=[],
                cold_gpu_seconds=None,max_yolo_cuda_delta=0,max_cpu_delta=0,
                cpu_difference_pixels=0,compared_pixels=0,actual_sampler=None)
            context=SamInterpolationContext(model_path='unused',device_ids=(0,),
                temp_dir=output/view.name,evidence_root=output/(view.name+'-evidence'),
                source_volume=source,source_identity='synthetic128MiB')
            def gpu_cohort(compare=False):
                lease=_try_acquire_specific_main_process_gpu_stage(torch,0,'SAM image probe')
                if lease is None:
                    raise RuntimeError('GPU compute lease unexpectedly unavailable')
                renderer=SamGpuCropRenderer(context,view,source,512,lease,torch,256*1024**2)
                started=time.perf_counter()
                pixels=[]
                try:
                    for frame in frames:
                        image=renderer.render(view,frame,identity,output_origin_yx=(8,8),
                            output_height=height,output_width=width)
                        assert image.dtype==np.uint8 and image.shape==(height,width) and image.flags.c_contiguous
                        pixels.append(image)
                    elapsed=time.perf_counter()-started
                    engine=renderer.engine
                    row['actual_sampler']=getattr(engine,'_spherical_sampler_mode',
                        'radial_columns' if getattr(engine,'_radial_column_geometry_announced',False) else
                        'radial_native' if getattr(engine,'_radial_native_kernel_announced',False) else 'resident_torch')
                    if compare:
                        for frame,image in zip(frames,pixels):
                            with torch.cuda.stream(engine._stream):
                                native=engine._render_native_plane(view,frame).round().clamp_(0,255).to(torch.uint8)
                                oracle=engine.warp_native_uint8_frame(native,crop_inverse(identity,(8,8)),height,width).cpu().numpy()
                            delta=np.abs(image.astype(np.int16)-oracle.astype(np.int16))
                            row['max_yolo_cuda_delta']=max(row['max_yolo_cuda_delta'],int(delta.max(initial=0)))
                            assert row['max_yolo_cuda_delta']==0, 'SAM differs from same configured YOLO CUDA renderer'
                        native=None
                finally:
                    retired=time.perf_counter()
                    renderer.close()
                    retirement=time.perf_counter()-retired
                return elapsed+retirement,pixels
            def cpu_cohort():
                started=time.perf_counter()
                pixels=[]
                for frame in frames:
                    cache={}
                    if family in {'radial','spherical'}:
                        plane,origin=prepare_native_shell_crop(source,view,frame,affine=identity,
                            output_bbox_yx=bbox,max_workspace_bytes=256*1024**2)
                        cache=dict(plane=plane,origin_xy=origin)
                    pixels.append(render_canonical_crop(source,view,frame,affine=identity,inverse=identity,
                        output_origin_yx=(8,8),output_height=height,output_width=width,
                        output_canvas_width=view.src_w,native_frame_cache=cache,max_workspace_bytes=256*1024**2))
                return time.perf_counter()-started,pixels
            try:
                row['cold_gpu_seconds'],_=gpu_cohort(compare=True)
                for repeat in range(args.repeats):
                    # Reverse measurement order to avoid one consistent cold side.
                    if repeat%2:
                        gpu_time,gpu_pixels=gpu_cohort()
                        cpu_time,cpu_pixels=cpu_cohort()
                    else:
                        cpu_time,cpu_pixels=cpu_cohort()
                        gpu_time,gpu_pixels=gpu_cohort()
                    row['cpu_seconds'].append(cpu_time)
                    row['gpu_seconds'].append(gpu_time)
                    for cpu,gpu in zip(cpu_pixels,gpu_pixels):
                        delta=np.abs(cpu.astype(np.int16)-gpu.astype(np.int16))
                        row['max_cpu_delta']=max(row['max_cpu_delta'],int(delta.max(initial=0)))
                        row['cpu_difference_pixels']+=int(np.count_nonzero(delta))
                        row['compared_pixels']+=int(delta.size)
                assert any(np.any(image) for image in gpu_pixels), 'Probe returned only empty output'
            finally:
                context.close()
            rows.append(row)
            save()
            print(json.dumps(row),flush=True)
        env=dict(os.environ,XTA_RUN_CUDA_RENDER_INTEGRATION='1',XTA_TEST_SPHERICAL_FP32_CUDA='1')
        tests=[
            'tests/test_sam_gpu_rendering.py::test_real_fused_tilted_sam_crops_keep_cpu_geometry_and_small_pixel_delta',
            'tests/test_sam_gpu_rendering.py::test_real_source_pair_uses_one_upload_and_retires_before_sdk_compute',
            'tests/test_sam_radial_source_window.py::test_tiny_cuda_superset_handoff_keeps_planned_identity_and_actual_origin',
            'tests/test_sam_radial_source_window.py::test_tiny_cuda_source_windows_match_full_scalar_columns_and_torch',
            'tests/test_spherical_fp32.py::SphericalFp32CudaTests::test_gray8_error_padding_and_stream_eviction_across_faces_and_t_ratios']
        with (output/'cuda-tests.txt').open('w',encoding='utf-8') as log:
            result=subprocess.run([sys.executable,'-B','-m','pytest','-q','-s',*tests,
                '--basetemp',str(output/'pytest-temp'),'--junitxml',str(output/'cuda-tests.xml')],
                cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        report['cuda_test_returncode']=result.returncode
        save()
        if result.returncode:
            raise RuntimeError('Existing CUDA qualification tests failed; see cuda-tests.txt')


if __name__=='__main__':
    main()

"""Qualify derived confidence bounds on a bounded, real generic Radial GPU lease.

The identical retained device mask/scores feed cropped and dense capture, so this
isolates evidence publication from model variability. Local timings are not a
prediction of cluster pipeline walltime. Acquire Scratch/Temp/GPU_LOCK first.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import time
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('MKL_NUM_THREADS', '2')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '2')

import numpy as np
from tools.capture_native_proto_outputs import chosen_views, digest_file, load_video
from tools.qualify_native_trt_lease import array_digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--engine', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, default=3072)
    parser.add_argument('--start-frame', type=int, default=900)
    parser.add_argument('--source-frames', type=int, default=24)
    parser.add_argument('--lease-frames', type=int, default=16)
    parser.add_argument('--conf', type=float, default=.5)
    parser.add_argument('--heat-seconds', type=float, default=60.)
    args = parser.parse_args()
    args.input, args.engine = args.input.resolve(strict=True), args.engine.resolve(strict=True)
    args.output_dir = args.output_dir.resolve()
    if (args.output_dir.is_relative_to(ROOT) or not 1 <= args.source_frames <= 64
            or not 1 <= args.lease_frames <= 32 or args.start_frame < 0
            or args.imgsz <= 0 or args.imgsz % 64 or args.heat_seconds < 0
            or not 0 <= args.conf <= 1):
        parser.error('Use Scratch output, bounded frame counts, and valid dimensions/confidence')
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir/'ultralytics').mkdir()
    os.environ.update(YOLO_TTA_NATIVE_TRT_RING='0', YOLO_TTA_FAST_GEOMETRY='1',
                      YOLO_AUTOINSTALL='false', YOLO_CONFIG_DIR=str(args.output_dir/'ultralytics'),
                      YOLO_TTA_TELEMETRY_DIR=str(args.output_dir/'telemetry'))
    import torch
    from XTA import cuda_backend, cuda_d1, geometry, inference
    from XTA.config import resolve_channel_format
    from XTA.confidence_evidence import ConfidenceEvidenceRef
    from XTA.media import compute_cube_resize_shape
    from tests.test_cylindrical_cuda import resident_engine
    from tools.benchmark_radial_setup import heatsoak

    torch.set_num_threads(2)
    volume = load_video(args.input, args.source_frames, args.start_frame)
    logical_shape = tuple(map(int, compute_cube_resize_shape(*volume.shape)))
    if logical_shape[1:] != volume.shape[1:]:
        raise ValueError('Source block must require only logical T expansion')
    physical, middle = next(pair for pair in chosen_views(logical_shape, args.imgsz)
                            if pair[0].family == 'radial' and pair[0].radial_base_view == 'sagittal')
    view = geometry.expand_views_into_tta_variants((physical,), (0.,))[0]
    offset = min(max(0, middle-args.lease_frames//2), view.num_slices-args.lease_frames)
    if offset < 0:
        raise ValueError('Requested lease exceeds view length')
    renderer = resident_engine(volume, 'cuda:0', logical_t=logical_shape[0])
    renderer._stream = torch.cuda.Stream(device=renderer.device)
    renderer._stream.wait_stream(torch.cuda.current_stream())
    renderer._stream.synchronize()
    cfg = inference.PredictConfig(imgsz=args.imgsz, conf=args.conf, device='cuda:0',
        quantize='fp16', batch=1, input_channels=1, channel_token='gray')
    inference.set_retina_mask_processor('gpu')
    inference.set_angle_variant_gpu_fastpath(0., 0.)
    model = inference.load_ultralytics_model(str(args.engine), task='segment')
    inference.require_channel_aware_yolo_preprocess_patch('gray')
    if inference._ensure_predictor_for_direct_predict(model, cfg) is None:
        raise RuntimeError('Direct predictor unavailable')
    report = dict(scope=__doc__, engine=str(args.engine), engine_sha256=digest_file(args.engine),
        source=str(args.input), source_sha256=digest_file(args.input),
        source_shape=list(volume.shape), source_start_frame=args.start_frame,
        logical_shape=logical_shape, view=view.name, slice_offset=offset,
        source_frames_consumed=args.lease_frames, device=torch.cuda.get_device_name(),
        native_ring=False, runs=[], source_files={name:digest_file(ROOT/name) for name in
            ('XTA/cuda_d1.py', 'XTA/inference.py', 'tools/qualify_d1_confidence_bounds.py')})
    if args.heat_seconds:
        print(f'Heatsoaking for {args.heat_seconds:g}s', flush=True)
        report['heat_seconds_actual'] = heatsoak(args.heat_seconds, 0)
    job = geometry.build_aug_job_for_variant(view, args.imgsz, args.output_dir)
    native_h, native_w = geometry.view_processing_plane_shape(view, args.imgsz)
    matrix = geometry.output_to_view_processing_affine(view, job.aff.M_out_to_src, args.imgsz)
    target = np.zeros((args.lease_frames,native_h,native_w),np.uint8)
    source = cuda_backend.GpuRenderedYoloSource(renderer,view,job,slice_offset=offset,
        num_frames=args.lease_frames,batch_size=1,out_size=args.imgsz,fp16=True,
        name='generic-radial-confidence',channel_format=resolve_channel_format('gray'))

    def consume(accumulator):
        before = (array_digest(accumulator.union_dev.cpu().numpy()),
                  array_digest(accumulator.conf_dev.cpu().numpy()))
        if accumulator.compute_d1_slice_metadata(synchronize_device=False) is not None:
            raise AssertionError('Generic Radial unexpectedly emitted resident-ring bounds')
        baseline = None
        for ordinal, mode in enumerate(('derived', 'dense', 'dense', 'derived')):
            task = dict(view=view, slice_start=offset, slice_count=args.lease_frames,
                d1_store_dir=str(args.output_dir/f'{ordinal:02d}-{mode}'/'mask.cvol'),
                model_name='qualification',task_id=ordinal,d1_output_shape=logical_shape)
            torch.cuda.synchronize()
            started = time.perf_counter()
            with mock.patch.object(accumulator,'compute_slice_metadata',
                    **({'return_value':None} if mode == 'dense' else
                       {'wraps':accumulator.compute_slice_metadata})):
                shard = cuda_d1._d1_write_task_confidence(task, accumulator)
            elapsed = time.perf_counter()-started
            ref = ConfidenceEvidenceRef.open(shard['path'])
            hashes = {name:digest_file(ref.path/name) for name in ('scores.u8.zlib','index.bin')}
            baseline = baseline or hashes
            row=dict(mode=mode, host_seconds=elapsed, metrics=shard['capture_metrics'],
                hashes=hashes, exact_bytes=hashes == baseline, known_voxels=shard['known_voxels'])
            report['runs'].append(row)
            print(json.dumps(row),flush=True)
        after = (array_digest(accumulator.union_dev.cpu().numpy()),
                 array_digest(accumulator.conf_dev.cpu().numpy()))
        report.update(device_sources_unchanged=before==after, mask_sha256=before[0],
                      score_sha256=before[1])
        target[:] = accumulator.union_dev.cpu().numpy()
        return {}

    inference.predict_source_and_accumulate(model,source,source_label='generic-radial-confidence',
        num_frames=args.lease_frames,out_size=args.imgsz,cfg=cfg,view_union_mm=target,
        view_confmap_mm=None,M_out_to_native=matrix,native_h=native_h,native_w=native_w,
        postprocess_workers=2,streaming_cleanup_enabled=False,device_hole_fill=False,
        require_device_union=True,retain_confidence=True,device_union_consumer=consume)
    if source.count != args.lease_frames or source._direct_count != 0 or source.resident_ring_supported:
        raise AssertionError('Native TensorRT ring or wrong frame count used')
    source.close()
    report['all_exact'] = report['device_sources_unchanged'] and all(r['exact_bytes'] for r in report['runs'])
    report['foreground_voxels'] = int(np.count_nonzero(target))
    (args.output_dir/'qualification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    if not report['all_exact']:
        raise AssertionError('Confidence bounds changed publication bytes or source tensors')


if __name__ == '__main__':
    main()

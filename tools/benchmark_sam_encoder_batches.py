"""Local SAM image-encoder batching probe; excludes tracking and transfers."""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import statistics
import sys
import time

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.diagnose_sam_interpolation import gpu_lock, resource_monitor


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--frames', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--gpu-lock', type=Path, required=True)
    parser.add_argument('--profile', choices=('auto', 'egpu', 'h100'), default='auto')
    parser.add_argument('--attention-backend', choices=('auto', 'sdpa', 'fa3_fp8'), default='auto')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    report = dict(schema='xta.sam_encoder_batch_probe/1', invocation=sys.argv,
                  scope='Image backbone, both tracker necks and decoder feature projections; no temporal tracking',
                  source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  frames_path=str(args.frames.resolve()), rows=[], validation={})

    def save():
        (args.output / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')

    with gpu_lock(args.gpu_lock, 'sam_encoder_batch_probe'), resource_monitor(args.output / 'resources.json'):
        import numpy as np
        import torch
        import pynvml
        from PIL import Image
        from XTA.lta_sam import prepare_sam_video_frames
        from XTA.lta_worker_adapter import build_worker_predictor, close_worker_predictor

        context = None
        try:
            torch.cuda.set_device(0)
            torch.manual_seed(0)
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)

            def device_state():
                memory = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return dict(temperature_c=pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                            clock_mhz=pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_GRAPHICS),
                            power_w=pynvml.nvmlDeviceGetPowerUsage(handle) / 1000,
                            device_used_bytes=int(memory.used), device_free_bytes=int(memory.free))

            report.update(torch_version=torch.__version__, gpu=torch.cuda.get_device_name(0),
                          initial_device=device_state())
            print('Loading production SAM predictor', flush=True)
            context = build_worker_predictor(dict(model_path=str(args.model), profile=args.profile,
                attention_backend=args.attention_backend, conf=0., max_num_objects=16))
            model = context.predictor.model
            report.update(profile=context.profile, sam_runtime=context.sam_runtime,
                          model_allocated_bytes=torch.cuda.memory_allocated())
            frames = np.load(args.frames, mmap_mode='r', allow_pickle=False)
            assert frames.dtype == np.uint8 and frames.ndim == 3 and len(frames) >= 4
            indices = np.linspace(0, len(frames) - 1, 4, dtype=int).tolist()

            class Clip(list):
                pass

            clip = Clip(Image.fromarray(np.asarray(frames[i])).convert('RGB') for i in indices)
            report['input_indices'] = indices
            report['preparation'] = prepare_sam_video_frames(context.predictor, clip, torch_module=torch)
            assert report['preparation']['prepared']
            images = clip._xta_prepared_video_frames.images.to(device='cuda', dtype=torch.float32)
            assert tuple(images.shape) == (4, 3, int(model.image_size), int(model.image_size))
            report['input_shape'] = list(images.shape)
            report['normalized_input_sha256'] = hashlib.sha256(images.cpu().numpy().tobytes()).hexdigest()

            def encode(batch):
                features = model.detector.backbone.forward_image(batch, need_sam3_out=False,
                    need_interactive_out=True, need_propagation_out=True)
                output = {}
                for key, decoder in (('interactive', model.tracker.interactive_sam_mask_decoder),
                                     ('sam2_backbone_out', model.tracker.sam_mask_decoder)):
                    branch = features[key]
                    assert branch.get('vision_mask') is None
                    assert len(branch['backbone_fpn']) == len(branch['vision_pos_enc']) == 3
                    for level, (feature, position) in enumerate(zip(branch['backbone_fpn'], branch['vision_pos_enc'])):
                        value = feature.tensors.to(dtype=torch.bfloat16)
                        assert feature.mask is None and value.shape[0] == batch.shape[0]
                        if level < 2:
                            value = (decoder.conv_s0 if level == 0 else decoder.conv_s1)(value)
                        output[f'{key}/feature/{level}'] = value
                        output[f'{key}/position/{level}'] = position
                return output

            with torch.inference_mode(), torch.autocast('cuda', dtype=torch.bfloat16):
                references = []
                for i in range(4):
                    output = encode(images[i:i + 1])
                    references.append({key: value.float().cpu() for key, value in output.items()})
                    del output
                report['feature_shapes'] = {key: list(value.shape) for key, value in references[0].items()}
                valid = [1]
                for batch_size in (2, 4):
                    try:
                        differences = {}
                        for start in range(0, 4, batch_size):
                            output = encode(images[start:start + batch_size])
                            for key, tensor in output.items():
                                values = tensor.float().cpu()
                                assert bool(torch.isfinite(values).all())
                                for offset in range(batch_size):
                                    reference = references[start + offset][key]
                                    value = values[offset:offset + 1]
                                    assert value.shape == reference.shape
                                    delta = value - reference
                                    row = differences.setdefault(key, dict(max_abs=0., squared_error=0., squared_reference=0., elements=0))
                                    row['max_abs'] = max(row['max_abs'], float(delta.abs().max()))
                                    row['squared_error'] += float(delta.square().sum())
                                    row['squared_reference'] += float(reference.square().sum())
                                    row['elements'] += delta.numel()
                            del output, tensor, values, value, reference, delta
                        for row in differences.values():
                            row['relative_l2'] = (row['squared_error'] / max(row['squared_reference'], 1e-30)) ** .5
                        report['validation'][str(batch_size)] = differences
                        valid.append(batch_size)
                        print(f'Batch {batch_size} fits; max feature relative L2={max(r["relative_l2"] for r in differences.values()):.6g}', flush=True)
                    except torch.cuda.OutOfMemoryError as error:
                        report['validation'][str(batch_size)] = dict(out_of_memory=str(error))
                        print(f'Batch {batch_size}: CUDA out of memory', flush=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    save()
                del references
                print('Warming actual encoder for 20 seconds', flush=True)
                start = time.perf_counter()
                while time.perf_counter() - start < 20:
                    output = encode(images[:1])
                    del output
                    torch.cuda.synchronize()
                # Alternate order so cooling/boost drift cannot favor larger batches.
                for round_index, order in enumerate(((1, 2, 4), (4, 2, 1), (2, 1, 4))):
                    for batch_size in order:
                        if batch_size not in valid:
                            continue
                        gc.collect()
                        torch.cuda.empty_cache()
                        for _ in range(2):
                            output = encode(images[:batch_size])
                            del output
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        before = device_state()
                        event_start, event_end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        wall = time.perf_counter()
                        event_start.record()
                        for _ in range(4):
                            for index in range(0, 4, batch_size):
                                output = encode(images[index:index + batch_size])
                                del output
                        event_end.record()
                        event_end.synchronize()
                        wall = time.perf_counter() - wall
                        cuda_ms = event_start.elapsed_time(event_end)
                        row = dict(round=round_index, batch=batch_size, frames=16, wall_seconds=wall,
                                   cuda_ms=cuda_ms, fps=16 / wall,
                                   peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                                   peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                                   before=before, after=device_state())
                        report['rows'].append(row)
                        print(json.dumps(row), flush=True)
                        save()
            report['summary'] = {}
            baseline = statistics.median(row['fps'] for row in report['rows'] if row['batch'] == 1)
            for size in valid:
                rows = [row for row in report['rows'] if row['batch'] == size]
                fps = statistics.median(row['fps'] for row in rows)
                report['summary'][str(size)] = dict(median_fps=fps, speedup=fps / baseline,
                    min_fps=min(row['fps'] for row in rows), max_fps=max(row['fps'] for row in rows),
                    peak_allocated_bytes=max(row['peak_allocated_bytes'] for row in rows),
                    peak_reserved_bytes=max(row['peak_reserved_bytes'] for row in rows))
            report['status'] = 'complete'
            print(json.dumps(report['summary'], indent=2), flush=True)
        except Exception as error:
            report['status'] = 'failed'
            report['error'] = repr(error)
            raise
        finally:
            save()
            if context is not None:
                close_worker_predictor(context)


if __name__ == '__main__':
    main()

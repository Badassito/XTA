"""Bounded real-CUDA lifecycle/parity smoke for worker-lazy retirement lanes.

This is a functional ownership check, not a throughput benchmark. It acquires
the shared GPU reservation before importing torch and retires all CUDA owners
before releasing it. No model is loaded or source renderer state fabricated.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import sys
import weakref


ROOT = Path(__file__).resolve().parents[1]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--side', type=int, default=64)
    args = parser.parse_args(argv)
    if not 8 <= args.side <= 512:
        parser.error('side must be between 8 and 512 for this bounded smoke')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, subdir in (('YOLO_CONFIG_DIR','yolo'), ('TEMP','temp'), ('TMP','temp'),
                         ('XDG_CACHE_HOME','cache'), ('TORCH_HOME','torch'), ('HF_HOME','huggingface')):
        directory = args.output_dir / 'runtime' / subdir
        directory.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(directory)
    os.environ['YOLO_TTA_GPU_UNION_RETIREMENT_LANES'] = '3'
    os.environ['YOLO_TTA_GPU_UNION_RETIREMENT_CHUNK_SLICES'] = '16'
    sys.path.insert(0, str(ROOT))
    from tools.gpu_reservation import gpu_reservation
    lock = ROOT.parent / 'Scratch/Temp/GPU_LOCK'
    retired = False
    with gpu_reservation(lock, timeout=1, task_name='TTA lazy retirement CUDA lifecycle smoke',
                         release_when=lambda: retired) as reservation:
        import numpy as np
        import torch
        from XTA import inference
        if not torch.cuda.is_available():
            retired = True
            raise RuntimeError('CUDA unavailable')
        torch.cuda.set_device(0)
        device = torch.device('cuda:0')
        original_empty = torch.empty
        pin_allocations = []
        def observed_empty(*values, **kwargs):
            value = original_empty(*values, **kwargs)
            if kwargs.get('pin_memory'):
                pin_allocations.append(value.numel() * value.element_size())
            return value
        torch.empty = observed_empty
        outputs, cases = {}, []
        accumulator = producer = mask_tensor = conf_tensor = manager = lane = None
        try:
            rng = np.random.default_rng(7139)
            shape = (5, args.side - 3, args.side - 1)
            mask = (rng.random(shape) > .7).astype(np.uint8)
            mask[0] = 0
            confidence = rng.integers(0, 256, shape, dtype=np.uint8) * mask
            for mode in ('unarmed_fallback', 'eager', 'lazy'):
                inference._shutdown_gpu_union_retirement_manager()
                pin_allocations.clear()
                manager = None
                if mode == 'eager':
                    manager = inference._init_gpu_union_retirement_manager('cuda:0', args.side)
                elif mode == 'lazy':
                    inference._arm_worker_gpu_union_retirement_manager('cuda:0', args.side)
                    assert inference._gpu_union_retirement_manager() is None
                    assert not pin_allocations
                accumulator = inference._DeviceUnionAccumulator(torch, device, *shape, want_conf=True)
                producer = torch.cuda.Stream(device=device)
                with torch.cuda.stream(producer):
                    for index in range(shape[0]):
                        mask_tensor = torch.as_tensor(mask[index], device=device)
                        conf_tensor = torch.as_tensor(confidence[index], device=device)
                        accumulator.write_frame(index, mask_tensor, conf_tensor, producer_stream=producer)
                # D1-like device-only retirement needs no ordinary pinned lane.
                if mode == 'lazy':
                    accumulator.synchronize_for_retirement(None)
                    assert not pin_allocations
                    assert inference._gpu_union_retirement_manager() is None
                manager = inference._ensure_worker_gpu_union_retirement_manager()
                if mode == 'lazy':
                    assert manager is not None
                    expected = 3 * (2 * 16 * args.side ** 2 + 16)
                    assert sum(pin_allocations) == expected
                    assert manager is inference._ensure_worker_gpu_union_retirement_manager()
                lane = manager.acquire() if manager is not None else None
                destination = np.zeros(shape, np.uint8)
                destination_conf = np.zeros(shape, np.uint8)
                metadata = accumulator.flush_into(destination, destination_conf,
                    retirement_lane=lane, collect_slice_metadata=True)
                if manager is not None:
                    manager.release(lane)
                outputs[mode] = (destination, destination_conf, metadata)
                owners = ([] if manager is None else [weakref.ref(tensor)
                    for item in manager.lanes for tensor in (item.pinned_u8,item.pinned_stats)])
                cases.append(dict(mode=mode,pinned_allocation_bytes=sum(pin_allocations),
                                  lane_count=0 if manager is None else manager.capacity))
                torch.cuda.synchronize(device)
                del lane, accumulator, producer, mask_tensor, conf_tensor, manager
                inference._shutdown_gpu_union_retirement_manager()
                gc.collect()
                assert all(owner() is None for owner in owners)
                assert inference._ensure_worker_gpu_union_retirement_manager() is None
            reference = outputs['unarmed_fallback']
            for mode in ('eager','lazy'):
                for index in (0,1):
                    np.testing.assert_array_equal(outputs[mode][index],reference[index])
                assert outputs[mode][2].keys() == reference[2].keys()
                for key in reference[2]:
                    np.testing.assert_array_equal(outputs[mode][2][key],reference[2][key])
            result = dict(scope='functional CUDA lifecycle; no model/throughput claim',
                device=torch.cuda.get_device_name(device),shape=shape,
                exact_mask_confidence_metadata=True,cases=cases,
                pure_device_retirement_pinned_bytes=0,reservation=reservation)
            (args.output_dir/'report.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
            print(json.dumps(result,indent=2))
        finally:
            torch.empty = original_empty
            accumulator = producer = mask_tensor = conf_tensor = manager = lane = None
            inference._shutdown_gpu_union_retirement_manager()
            torch.cuda.synchronize(device)
            gc.collect()
            torch.cuda.empty_cache()
            retired = True
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

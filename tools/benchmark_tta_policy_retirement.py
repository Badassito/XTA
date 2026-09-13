"""Heat-soaked comparison of CPU and CUDA policy mask retirement; no model timing."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--height', type=int, default=2911)
    parser.add_argument('--width', type=int, default=3022)
    parser.add_argument('--samples', type=int, default=30)
    parser.add_argument('--heatsoak-seconds', type=float, default=90)
    args = parser.parse_args()
    if min(args.height, args.width, args.samples) < 1 or args.heatsoak_seconds < 0:
        parser.error('positive dimensions/samples and nonnegative heatsoak required')
    output = args.output.resolve()
    if output.is_relative_to(ROOT):
        parser.error('Generated measurements belong in task Scratch')
    output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('CUPY_CACHE_DIR', str(output / 'cupy-cache'))
    import numpy as np
    import torch
    from XTA.inference import _DeviceUnionAccumulator, _GpuUnionRetirementLane
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required; CPU emulation is not a benchmark')
    torch.cuda.set_device(0)
    with torch.inference_mode():
        warm = torch.randn((4096, 4096), device='cuda', dtype=torch.float16)
        result = torch.empty_like(warm)
        started = time.perf_counter()
        last_report = started
        while time.perf_counter() - started < args.heatsoak_seconds:
            for _ in range(32):
                torch.mm(warm, warm, out=result)
            torch.cuda.synchronize()
            if time.perf_counter() - last_report > 20:
                print(f'Heatsoak {time.perf_counter() - started:.0f}s', flush=True)
                last_report = time.perf_counter()
        del warm, result
        source = np.zeros((1, args.height, args.width), dtype=np.uint8)
        source[:, args.height // 4:3 * args.height // 4, args.width // 5:4 * args.width // 5] = 1
        device_source = torch.as_tensor(source, device='cuda')
        target = np.memmap(output / 'target.u8', mode='w+', dtype=np.uint8, shape=source.shape)
        lane = _GpuUnionRetirementLane(torch, torch.device('cuda:0'), 0, args.height * args.width)
        records = {'reference_cpu_scan_or': [], 'owned_gpu_scan_copy': []}
        expected = None
        for sample in range(args.samples + 3):
            # Alternate ordering to limit thermal/cache bias. Clear the target
            # and build the accumulator outside the retirement timing window.
            for owned in ((False, True) if sample % 2 == 0 else (True, False)):
                target[:] = 0
                accumulator = _DeviceUnionAccumulator(torch, torch.device('cuda:0'), 1, args.height, args.width, False)
                accumulator.union_dev.copy_(device_source)
                accumulator.written[:] = True
                torch.cuda.synchronize()
                cpu_start, start = time.process_time(), time.perf_counter()
                metadata = accumulator.flush_into(target, None, retirement_lane=lane,
                    collect_slice_metadata=True, owned_disjoint_output=owned)
                measurement = dict(wall_ms=(time.perf_counter() - start) * 1000,
                                   process_cpu_ms=(time.process_time() - cpu_start) * 1000)
                np.testing.assert_array_equal(target, source)
                if expected is None:
                    expected = metadata
                for name, value in expected.items():
                    np.testing.assert_array_equal(metadata[name], value)
                if sample >= 3:
                    records['owned_gpu_scan_copy' if owned else 'reference_cpu_scan_or'].append(measurement)
        target._mmap.close()
        summary = {name: {metric: statistics.median(row[metric] for row in rows)
                          for metric in ('wall_ms', 'process_cpu_ms')}
                   for name, rows in records.items()}
        for name, rows in records.items():
            summary[name]['mean_process_cpu_ms'] = statistics.mean(row['process_cpu_ms'] for row in rows)
        report = dict(scope='One policy mask D2H, metadata and host write; no model/render/augmentation/coordinator/projection/NRRD',
                      device=torch.cuda.get_device_name(0), shape=list(source.shape),
                      heatsoak_seconds=args.heatsoak_seconds, samples=args.samples,
                      cpu_clock_note='Windows process CPU clock is coarse; a zero per-sample median does not mean zero CPU work.',
                      mask_and_metadata_exact=True, summary=summary, records=records)
        (output / 'retirement.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()

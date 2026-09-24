"""Qualify an isolated SM80 D1 warp-OR experiment against the shipping helper.

The caller must own Scratch/Temp/GPU_LOCK and heatsoak before benchmarking.
Production dispatch is unchanged. Synthetic cases use an injective host-created
key-to-slot mapping, so high 64-bit keys can be checked with bounded allocations.
Production geometry cases compile the actual source with only the helper changed.
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import statistics
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def sources():
    path = ROOT / 'XTA' / 'cuda_d1.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    function = next(n for n in tree.body if isinstance(n, ast.FunctionDef)
                    and n.name == '_d1_backproject_kernels')
    source = next(ast.literal_eval(n.value) for n in ast.walk(function)
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name)
                  and t.id == 'source' for t in n.targets))
    match = re.search(r'    __device__ __forceinline__ void d1_warp_aggregated_atomic_or\(.*?\n    }', source, re.S)
    if match is None:
        raise RuntimeError('Shipping D1 helper no longer matches the expected shape')
    original = match.group()
    guard = '#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700'
    if original.count(guard) != 1 or 'unsigned long long word' not in original:
        raise RuntimeError('Shipping D1 architecture guard/key changed; audit extraction')
    candidate = original.replace(guard, '''#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
      unsigned int active = __activemask();
      unsigned int group = __match_any_sync(active, word);
      unsigned int combined = __reduce_or_sync(group, bit);
      int lane = (int)threadIdx.x & 31;
      if (lane == __ffs(group) - 1) atomicOr(output_bits + word, combined);
    #elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700''')
    production = {'baseline': source, 'redux': source.replace(original, candidate, 1)}
    wrapper = r'''
extern "C" __global__ void synthetic_d1(
    const unsigned long long* keys, const unsigned int* slots,
    const unsigned int* values, const unsigned char* enabled,
    unsigned int* output, unsigned long long count) {
    __shared__ unsigned int warp_bits[256];
    unsigned long long q = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= count || !enabled[q]) return;
    d1_warp_aggregated_atomic_or(output, keys[q], values[q], warp_bits, slots[q]);
}
'''
    result = {}
    for name, helper in (('baseline', original), ('redux', candidate)):
        mapped = helper.replace('unsigned int* warp_bits)',
                                'unsigned int* warp_bits, unsigned int physical)')
        mapped = mapped.replace('output_bits + word', 'output_bits + physical')
        result[name] = {'production': production[name], 'synthetic': mapped + wrapper}
    return path, result


def cpu_or(slots, values, enabled, count):
    output = np.zeros(count, np.uint32)
    np.bitwise_or.at(output, slots[enabled.astype(bool)], values[enabled.astype(bool)])
    return output


def paired_timing(cp, stream, runners, rounds, iterations, graph_timing=False):
    samples = {name: [] for name in runners}
    for runner in runners.values():
        for _ in range(3):
            runner()
    stream.synchronize()
    graphs = {}
    if graph_timing:
        for name, runner in runners.items():
            stream.begin_capture()
            for _ in range(iterations):
                runner()
            graphs[name] = stream.end_capture()
            graphs[name].launch(stream=stream)
        stream.synchronize()
    for round_id in range(rounds):
        order = list(runners) if round_id % 2 == 0 else list(reversed(runners))
        for name in order:
            start, end = cp.cuda.Event(), cp.cuda.Event()
            start.record(stream)
            if graph_timing:
                graphs[name].launch(stream=stream)
            else:
                for _ in range(iterations):
                    runners[name]()
            end.record(stream)
            end.synchronize()
            samples[name].append(float(cp.cuda.get_elapsed_time(start, end)) / iterations)
    medians = {name: statistics.median(values) for name, values in samples.items()}
    return {'median_ms': medians, 'samples_ms': samples, 'rounds': rounds,
            'iterations': iterations, 'graph_timing': graph_timing,
            'speedup': medians['baseline'] / medians['redux']}


def compile_variants(cp, source_map, folder, cache):
    kernels, attributes = {}, {}
    for variant, kinds in source_map.items():
        kernels[variant], attributes[variant] = {}, {}
        for kind, source in kinds.items():
            label = variant + '-' + kind
            (folder / (label + '.cu')).write_text(source, encoding='utf-8')
            before = set(cache.rglob('*.cubin'))
            name = 'synthetic_d1' if kind == 'synthetic' else 'd1_backproject_bboxes_to_bits'
            module = cp.RawModule(code=source, options=('--std=c++14',), name_expressions=(name,))
            fn = module.get_function(name)
            kernels[variant][kind] = fn
            attrs = dict(fn.attributes)
            attrs['source_sha256'] = hashlib.sha256(source.encode()).hexdigest()
            attrs['artifacts'] = []
            for index, item in enumerate(sorted(set(cache.rglob('*.cubin')) - before)):
                data = item.read_bytes()
                # CuPy prepends a cache checksum to its ELF CUBIN. Preserve the
                # cache file and extract ELF for standard NVIDIA disassemblers.
                (folder / f'{label}-{index}.cupy-cache').write_bytes(data)
                offset = data.find(b'\x7fELF', 0, 128)
                if offset >= 0:
                    target = folder / f'{label}-{index}.cubin'
                    target.write_bytes(data[offset:])
                    attrs['artifacts'].append(str(target))
            attributes[variant][kind] = attrs
    return kernels, attributes


def synthetic_case(cp, stream, kernels, rng, name, keys, enabled, args, timing):
    keys = np.asarray(keys, np.uint64)
    unique, inverse = np.unique(keys, return_inverse=True)
    # Qualification compresses arbitrary 64-bit keys into a bounded injective
    # mapping. Benchmarks preserve actual physical addresses so strided words
    # remain strided; compressing sorted keys would silently make them contiguous.
    if timing:
        if int(keys.max()) >= 2**32:
            raise ValueError('Timing keys must fit bounded physical word addressing')
        slots = keys.astype(np.uint32)
        output_words = int(keys.max()) + 1
    else:
        slots = inverse.astype(np.uint32)
        output_words = len(unique)
    values = rng.integers(0, 2**32, len(keys), dtype=np.uint32)
    expected = cpu_or(slots, values, enabled, output_words)
    inputs = tuple(cp.asarray(a) for a in (keys, slots, values, enabled))
    outputs = {variant: cp.zeros(output_words, cp.uint32) for variant in kernels}
    grid = ((len(keys) + 255) // 256,)
    runners = {}
    for variant in kernels:
        fn, output = kernels[variant]['synthetic'], outputs[variant]
        runners[variant] = lambda fn=fn, output=output: fn(
            grid, (256,), (*inputs, output, np.uint64(len(keys))), stream=stream)
        runners[variant]()
    stream.synchronize()
    for variant, output in outputs.items():
        np.testing.assert_array_equal(cp.asnumpy(output), expected, err_msg=name + '/' + variant)
    active_counts = np.bincount(inverse[enabled.astype(bool)], minlength=len(unique))
    row = {'name': name, 'count': len(keys), 'enabled_count': int(enabled.sum()),
           'distinct_keys': len(unique), 'output_words': output_words,
           'physical_mapping': 'key_as_word' if timing else 'injective_compressed',
           'max_updates_per_key': int(active_counts.max(initial=0)),
           'exact_cpu_or': True, 'high_key_bits': bool(np.any(keys >> np.uint64(32)))}
    if timing:
        row.update(paired_timing(cp, stream, runners, args.rounds, args.iterations, args.graph_timing))
    return row


def production_cases(cp, stream, kernels, rng, args):
    results = []
    depth, height, width = 8, args.geometry_size, args.geometry_size
    angles = np.arange(depth, dtype=np.float32) * np.float32(np.pi / depth)
    angle_cos, angle_sin = cp.asarray(np.cos(angles)), cp.asarray(np.sin(angles))
    for density in (0.02, 1.0):
        host_mask = (rng.random((depth, height, width)) < density).astype(np.uint8)
        mask = cp.asarray(host_mask)
        boxes = cp.asarray(np.asarray([[i, 0, 0, height, width] for i in range(depth)], np.int32))
        for family, base, tilt in ((0, 0, 0.0), (0, 1, 0.0), (0, 2, 0.0),
                                   (1, 0, 0.577350269), (2, 0, 0.0), (2, 0, -0.577350269)):
            # Cartesian views use consistent native and output orientation.
            shape = ((depth, height, width) if base == 0 else
                     ((height, depth, width) if base == 1 else (height, width, depth)))
            # Azimuthal has azimuth slices and stack rows; its stack axis is H.
            if family == 2:
                shape = (height, height, width)
            count = int(np.prod(shape))
            outputs = {variant: cp.zeros((count + 31) // 32, cp.uint32) for variant in kernels}
            runners = {}
            for variant in kernels:
                fn, output = kernels[variant]['production'], outputs[variant]
                call_args = (mask, np.int32(height), np.int32(width), np.int32(0),
                             boxes, np.int32(depth), np.int32(depth), np.int32(height), np.int32(width),
                             *(np.int32(x) for x in shape), *(np.int32(x) for x in shape),
                             np.int32(family), np.int32(base), np.int32(0), np.float32(tilt),
                             np.int32(shape[0]), np.float32((width-1)*0.5),
                             np.float32((height-1)*0.5), np.float32((width-1)*0.5),
                             angle_cos, angle_sin, output)
                runners[variant] = lambda fn=fn, call_args=call_args: fn(
                    ((height * width + 255) // 256, depth), (256,), call_args, stream=stream)
                runners[variant]()
            stream.synchronize()
            baseline, redux = (cp.asnumpy(outputs[name]) for name in ('baseline', 'redux'))
            np.testing.assert_array_equal(redux, baseline)
            cpu_exact = None
            if family == 0:
                order = (0, 1, 2) if base == 0 else ((1, 0, 2) if base == 1 else (1, 2, 0))
                voxels = host_mask.transpose(order).reshape(-1)
                indices = np.flatnonzero(voxels).astype(np.uint64)
                expected = cpu_or((indices >> 5).astype(np.uint32),
                                  np.left_shift(np.uint32(1), (indices & 31).astype(np.uint32)),
                                  np.ones(len(indices), np.uint8), len(baseline))
                np.testing.assert_array_equal(baseline, expected)
                cpu_exact = True
            row = {'family': family, 'base': base, 'tilt': tilt, 'density': density,
                   'mask_shape': list(host_mask.shape), 'output_shape': list(shape),
                   'exact_baseline': True, 'exact_cpu_or': cpu_exact,
                   'nonzero_words': int(np.count_nonzero(baseline))}
            row.update(paired_timing(cp, stream, runners, args.rounds, args.iterations, args.graph_timing))
            results.append(row)
            print(json.dumps({'production': row}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--rounds', type=int, default=7)
    parser.add_argument('--iterations', type=int, default=30)
    parser.add_argument('--count', type=int, default=1 << 20)
    parser.add_argument('--geometry-size', type=int, default=512)
    parser.add_argument('--seed', type=int, default=20260915)
    parser.add_argument('--graph-timing', action='store_true',
                        help='Time a captured sequence to avoid per-kernel Python submission gaps.')
    args = parser.parse_args()
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT) or min(args.rounds, args.iterations, args.count,
                                              args.geometry_size) < 1:
        parser.error('Use a Scratch output path and positive sizes/rounds')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    artifact_dir = args.output.parent / (args.output.stem + '-artifacts')
    artifact_dir.mkdir(parents=True, exist_ok=True)
    cache = artifact_dir / 'cupy-cache'
    cache.mkdir(exist_ok=True)
    os.environ['CUPY_CACHE_DIR'] = str(cache)
    os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '1'
    source_path, source_map = sources()
    report = {'status': 'running', 'scope': __doc__, 'seed': args.seed,
              'source_path': str(source_path),
              'source_sha256': hashlib.sha256(source_path.read_bytes()).hexdigest(),
              'timing_policy': 'Alternating A/B CUDA-event rounds; compile, allocation, copies and initial output clear excluded. Repeated OR into already-populated words; exactness checked separately from zero output. Caller owns heatsoak and GPU lock. Direct submission can include Python gaps; graph mode captures the repeated sequence before timing.',
              'synthetic_qualification': [], 'synthetic_benchmark': [], 'production': []}

    def save():
        args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')

    save()
    started = time.perf_counter()
    try:
        import cupy as cp
        with cp.cuda.Device(args.device):
            props = cp.cuda.runtime.getDeviceProperties(args.device)
            report['environment'] = {'device': props['name'].decode(), 'sm': [props['major'], props['minor']],
                                     'cupy': cp.__version__, 'runtime': cp.cuda.runtime.runtimeGetVersion(),
                                     'driver': cp.cuda.runtime.driverGetVersion(),
                                     'nvrtc': list(cp.cuda.nvrtc.getVersion()), 'options': ['--std=c++14']}
            if props['major'] < 8:
                raise RuntimeError('The hardware redux experiment requires SM80 or newer')
            rng = np.random.default_rng(args.seed)
            stream = cp.cuda.Stream(non_blocking=True)
            with stream:
                kernels, attrs = compile_variants(cp, source_map, artifact_dir, cache)
                report['kernel_attributes'] = attrs
                for count in (1, 17, 31, 32, 33, 255, 256, 257, 4099):
                    q = np.arange(count, dtype=np.uint64)
                    high_keys = (q % 7) | ((q % 5) << np.uint64(32))
                    for active in (0.0, 0.125, 0.5, 1.0):
                        enabled = (rng.random(count) < active).astype(np.uint8)
                        row = synthetic_case(cp, stream, kernels, rng,
                            f'partial-{count}-active-{active}', high_keys, enabled, args, False)
                        report['synthetic_qualification'].append(row)
                q = np.arange(args.count, dtype=np.uint64)
                distributions = {
                    'one_key': np.zeros(args.count, np.uint64),
                    'contiguous_high_collision': q // 32,
                    'contiguous_distinct': q,
                    'strided_distinct': q * 17,
                    'random_32_words': rng.integers(0, 32, args.count, dtype=np.uint64),
                    'random_65536_words': rng.integers(0, 65536, args.count, dtype=np.uint64),
                }
                for name, keys in distributions.items():
                    for active in (0.05, 1.0):
                        enabled = (rng.random(args.count) < active).astype(np.uint8)
                        row = synthetic_case(cp, stream, kernels, rng,
                            f'{name}-active-{active}', keys, enabled, args, True)
                        report['synthetic_benchmark'].append(row)
                        print(json.dumps({'synthetic': row}), flush=True)
                        save()
                report['production'] = production_cases(cp, stream, kernels, rng, args)
            report['status'] = 'passed'
    except BaseException as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['wall_seconds'] = time.perf_counter() - started
        save()


if __name__ == '__main__':
    main()

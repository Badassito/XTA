"""Experimental noncontracted SDF fusion; caller owns GPU lock/heat-soak.

Production code is imported for the baseline and is never monkey-patched. This
isolates pre-topology blending, not bridge rendering or application wall time.
The inspected CuPy compiler appends -ftz=true after requested flags. Separate
rounding and matching CuPy therefore do not imply IEEE gradual underflow.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path
import statistics
import sys
import time
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from XTA.cuda_interpolation import CudaInterpolationRenderer


SOURCE = r'''
extern "C" __global__ void blend_strict(
    const float* a, const float* b, float ca, float cb,
    unsigned char* output, unsigned long long n) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float left = __fmul_rn(a[i], ca);
  float right = __fmul_rn(b[i], cb);
  output[i] = __fadd_rn(left, right) >= 0.0f;
}
extern "C" __global__ void blend_diagnostic(
    const float* a, const float* b, float ca, float cb,
    float* lefts, float* rights, float* sums, float* fma_left, float* fma_right,
    unsigned long long n) {
  unsigned long long i = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= n) return;
  float left = __fmul_rn(a[i], ca);
  float right = __fmul_rn(b[i], cb);
  lefts[i] = left;
  rights[i] = right;
  sums[i] = __fadd_rn(left, right);
  fma_left[i] = __fmaf_rn(a[i], ca, right);
  fma_right[i] = __fmaf_rn(b[i], cb, left);
}
'''
OPTIONS = ('--std=c++14', '--fmad=false', '--ftz=false')


def coefficients(alpha: float) -> tuple[np.float32, np.float32]:
    return np.float32(1.0 - float(alpha)), np.float32(float(alpha))


def host_stages(a, b, alpha):
    ca, cb = coefficients(alpha)
    with np.errstate(all='ignore'):
        left = np.multiply(a, ca)
        right = np.multiply(b, cb)
        summed = np.add(left, right)
    return left, right, summed


def bits(value) -> str:
    return f'0x{int(np.asarray(value, dtype=np.float32).view(np.uint32)):08x}'


def nonnegative_bits(values):
    """Interpret GPU FP32 comparison without the host's denormal mode."""
    u = np.asarray(values).view(np.uint32)
    magnitude = u & np.uint32(0x7fffffff)
    return ((u >> np.uint32(31) == 0) | (magnitude == 0)) & (magnitude <= 0x7f800000)


def cpu_denormal_probe(label):
    raw = np.tile(np.array([1, 0x80000001, 0x00800000, 0x80800000], dtype=np.uint32), 16)
    value = raw.view(np.float32)
    with np.errstate(all='ignore'):
        one = np.multiply(value, np.float32(1.0)).view(np.uint32)
        half = np.multiply(value, np.float32(.5)).view(np.uint32)
        ge = value >= np.float32(0.0)
    return {'after': label, 'multiply_one_bits': [hex(int(v)) for v in one[:4]],
            'multiply_half_bits': [hex(int(v)) for v in half[:4]], 'ge_zero': ge[:4].tolist(),
            'subnormal_input_preserved': bool(one[0] == 1 and one[1] == 0x80000001),
            'subnormal_result_preserved': bool(half[2] == 0x00400000 and half[3] == 0x80400000)}


def _soft_unpack(u):
    sign, exponent, fraction = u >> 31, (u >> 23) & 255, u & 0x7fffff
    if exponent == 255:
        return sign, ('nan' if fraction else 'inf'), 0, 0
    if exponent == 0:
        return sign, 'finite', fraction, -149
    return sign, 'finite', (1 << 23) + fraction, exponent - 150


def _soft_pack(sign, mantissa, exponent):
    """Integer-only round-to-nearest-even of mantissa * 2**exponent."""
    if mantissa == 0:
        return sign << 31
    quantum = max(mantissa.bit_length()-1+exponent-23, -149)
    shift = quantum-exponent
    if shift > 0:
        rounded, residue = divmod(mantissa, 1 << shift)
        half = 1 << (shift-1)
        rounded += int(residue > half or (residue == half and rounded & 1))
    else:
        rounded = mantissa << -shift
    if rounded == 0:
        return sign << 31
    if rounded >= 1 << 24:
        rounded >>= 1
        quantum += 1
    if rounded < 1 << 23:
        return (sign << 31) | rounded
    biased = quantum+23+127
    if biased >= 255:
        return (sign << 31) | 0x7f800000
    return (sign << 31) | (biased << 23) | (rounded-(1 << 23))


def soft_f32(a, b, operation):
    sa, ka, ma, ea = _soft_unpack(int(a))
    sb, kb, mb, eb = _soft_unpack(int(b))
    if ka == 'nan' or kb == 'nan':
        return 0x7fc00000
    if operation == 'mul':
        if (ka == 'inf' and kb == 'finite' and mb == 0) or (kb == 'inf' and ka == 'finite' and ma == 0):
            return 0x7fc00000
        if ka == 'inf' or kb == 'inf':
            return ((sa ^ sb) << 31) | 0x7f800000
        return _soft_pack(sa ^ sb, ma*mb, ea+eb)
    if ka == 'inf' or kb == 'inf':
        if ka == kb == 'inf' and sa != sb:
            return 0x7fc00000
        return ((sa if ka == 'inf' else sb) << 31) | 0x7f800000
    common = min(ea, eb)
    total = (-1 if sa else 1)*(ma << (ea-common)) + (-1 if sb else 1)*(mb << (eb-common))
    if total == 0:
        return 0x80000000 if ma == mb == 0 and sa == sb == 1 else 0
    return _soft_pack(int(total < 0), abs(total), common)


def software_stages(a, b, ca, cb):
    au, bu = a.view(np.uint32).ravel(), b.view(np.uint32).ravel()
    cau, cbu = int(ca.view(np.uint32)), int(cb.view(np.uint32))
    left = np.array([soft_f32(v, cau, 'mul') for v in au], dtype=np.uint32)
    right = np.array([soft_f32(v, cbu, 'mul') for v in bu], dtype=np.uint32)
    summed = np.array([soft_f32(x, y, 'add') for x, y in zip(left, right)], dtype=np.uint32)
    return [v.reshape(a.shape).view(np.float32) for v in (left, right, summed)]


def make_cases():
    """Bounded, CPU-only input generation. No random device work in timings."""
    values = np.array([
        0x00000000, 0x80000000, 0x00000001, 0x80000001,
        0x007fffff, 0x807fffff, 0x00800000, 0x80800000,
        0x3f800000, 0xbf800000, 0x3f7fffff, 0xbf7fffff,
        0x3f800001, 0xbf800001, 0x7f7fffff, 0xff7fffff,
        0x7f800000, 0xff800000, 0x7fc00000, 0xffc00000,
        0x7fc12345, 0x7f800001,
    ], dtype=np.uint32).view(np.float32)
    a, b = np.meshgrid(values, values, indexing='ij')
    for alpha in (0.0, -0.0, 1.0, 0.5, 1/3, 2/3, 0.1, 0.9,
                  float(np.nextafter(0., 1.)), float(np.nextafter(1., 0.)),
                  -1.0, 2.0, float('nan'), float('inf'), -float('inf')):
        yield 'exceptional_cross_product', a.copy(), b.copy(), alpha

    rng = np.random.default_rng(20260915)
    raw_a = rng.integers(0, 2**32, 65536, dtype=np.uint32).view(np.float32)
    raw_b = rng.integers(0, 2**32, 65536, dtype=np.uint32).view(np.float32)
    for alpha in (0.0, 0.5, 1/3, 0.1, 1.0):
        yield 'random_fp32_bits', raw_a, raw_b, alpha

    a = rng.uniform(-2048, 2048, 65536).astype(np.float32)
    for alpha in (1/3, 2/3, 1/7, 0.1, 0.9):
        ca, cb = coefficients(alpha)
        # Opposite terms cancel near an FP32 rounding boundary; +/- one ULP
        # neighbors exercise transitions that uniform random pairs rarely hit.
        b = (-a.astype(np.float64) * float(ca) / float(cb)).astype(np.float32)
        for direction in (None, -np.inf, np.inf):
            near_b = b if direction is None else np.nextafter(b, np.float32(direction))
            yield 'adversarial_cancellation', a, near_b, alpha

    import cv2
    yy, xx = np.mgrid[:129, :131]
    masks = (((yy-62)**2 + (xx-56)**2 < 33**2),
             (((yy-66)/1.1)**2 + (xx-68)**2 < 28**2))
    # Match XTA.interpolation._signed_distance_2d's actual OpenCV operation.
    sdfs = []
    for mask in masks:
        mask_u8 = np.ascontiguousarray(mask, dtype=np.uint8)
        inside = cv2.distanceTransform(mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        outside = cv2.distanceTransform(np.uint8(1)-mask_u8, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
        sdfs.append(np.asarray(inside-outside, dtype=np.float32))
    for steps in (2, 3, 7, 17):
        for step in range(1, steps):
            yield 'realistic_edt_bridge', *sdfs, float(step)/float(steps)


def qualify(cp, fused, diagnostic, stream):
    owner = SimpleNamespace(xp=cp)
    results, witnesses = [], []
    denormal_checkpoints = [cpu_denormal_probe('qualification_start')]
    all_ok = True
    numpy_agrees = True
    software_agrees = True
    for name, host_a, host_b, alpha in make_cases():
        host_a, host_b = (np.ascontiguousarray(v, dtype=np.float32) for v in (host_a, host_b))
        a, b = cp.asarray(host_a), cp.asarray(host_b)
        ca, cb = coefficients(alpha)
        n = a.size
        output = cp.empty(a.shape, dtype=cp.bool_)
        stages = [cp.empty(a.shape, dtype=cp.float32) for _ in range(5)]
        baseline = CudaInterpolationRenderer._blend_section(owner, a, b, alpha)
        launch_args = ((n+255)//256,), (256,)
        fused(*launch_args, (a, b, ca, cb, output, np.uint64(n)), stream=stream)
        diagnostic(*launch_args, (a, b, ca, cb, *stages, np.uint64(n)), stream=stream)
        # These are separate CuPy ufuncs exactly as in the production helper.
        cp_left = cp.multiply(a, ca)
        cp_right = cp.multiply(b, cb)
        cp_sum = cp.add(cp_left, cp_right)
        stream.synchronize()
        got, ref = cp.asnumpy(output), cp.asnumpy(baseline)
        gpu_stages = list(map(cp.asnumpy, stages))
        baseline_stages = list(map(cp.asnumpy, (cp_left, cp_right, cp_sum)))
        expected_stages = host_stages(host_a, host_b, alpha)
        item = {'case': name, 'shape': list(host_a.shape), 'elements': n,
                'alpha': repr(alpha), 'coefficient_bits': [bits(ca), bits(cb)],
                'boolean_mismatches_vs_actual_helper': int(np.count_nonzero(got != ref)),
                'boolean_mismatches_vs_numpy': int(np.count_nonzero(got != (expected_stages[2] >= 0))),
                'stages': {}}
        for stage, value, base, expected in zip(('left', 'right', 'sum'), gpu_stages, baseline_stages, expected_stages):
            nan_value, nan_base, nan_expected = map(np.isnan, (value, base, expected))
            equal_bits_base = value.view(np.uint32) == base.view(np.uint32)
            equal_bits_numpy = value.view(np.uint32) == expected.view(np.uint32)
            stat = {
                'non_nan_bit_mismatches_vs_cupy': int(np.count_nonzero(~(nan_value | nan_base) & ~equal_bits_base)),
                'nan_classification_mismatches_vs_cupy': int(np.count_nonzero(nan_value != nan_base)),
                'non_nan_bit_mismatches_vs_numpy': int(np.count_nonzero(~(nan_value | nan_expected) & ~equal_bits_numpy)),
                'nan_classification_mismatches_vs_numpy': int(np.count_nonzero(nan_value != nan_expected)),
                'nan_payload_bit_differences_vs_cupy': int(np.count_nonzero(nan_value & nan_base & ~equal_bits_base)),
            }
            item['stages'][stage] = stat
            # Replacement qualification is against the actual shipping CuPy
            # path. CPU FP32 denormal modes can differ from GPU gradual
            # underflow; retain that independent comparison without falsely
            # attributing a pre-existing CPU/GPU difference to kernel fusion.
            all_ok &= (stat['non_nan_bit_mismatches_vs_cupy'] == 0 and
                       stat['nan_classification_mismatches_vs_cupy'] == 0)
            numpy_agrees &= (stat['non_nan_bit_mismatches_vs_numpy'] == 0 and
                             stat['nan_classification_mismatches_vs_numpy'] == 0)
        if name == 'exceptional_cross_product':
            exact = software_stages(host_a, host_b, ca, cb)
            item['software_ieee754'] = {}
            for label, got_stage, expected in zip(('left', 'right', 'sum'), gpu_stages, exact):
                gu, eu = got_stage.view(np.uint32), expected.view(np.uint32)
                gn = (gu & np.uint32(0x7fffffff)) > 0x7f800000
                en = (eu & np.uint32(0x7fffffff)) > 0x7f800000
                mismatches = int(np.count_nonzero((gn != en) | (~gn & ~en & (gu != eu))))
                item['software_ieee754'][label+'_non_nan_bit_or_nan_classification_mismatches'] = mismatches
                software_agrees &= mismatches == 0
            mismatches = int(np.count_nonzero(got != nonnegative_bits(exact[2])))
            item['software_ieee754']['boolean_mismatches'] = mismatches
            software_agrees &= mismatches == 0
        for label, variant in zip(('fma_left', 'fma_right'), gpu_stages[3:]):
            variant_boolean = nonnegative_bits(variant)
            mismatch = np.flatnonzero(variant_boolean.ravel() != ref.ravel())
            item[label + '_boolean_changes'] = int(mismatch.size)
            if mismatch.size and (len(witnesses) < 6 or
                                  (name == 'adversarial_cancellation' and len(witnesses) < 20)):
                i = int(mismatch[0])
                witnesses.append({'case': name, 'variant': label, 'alpha': repr(alpha),
                                  'a_bits': bits(host_a.ravel()[i]), 'b_bits': bits(host_b.ravel()[i]),
                                  'ca_bits': bits(ca), 'cb_bits': bits(cb),
                                  'separate_sum_bits': bits(gpu_stages[2].ravel()[i]),
                                  'fma_sum_bits': bits(variant.ravel()[i]),
                                  'separate_boolean': bool(ref.ravel()[i]),
                                  'fma_boolean': bool(variant_boolean.ravel()[i])})
        all_ok &= item['boolean_mismatches_vs_actual_helper'] == 0
        numpy_agrees &= item['boolean_mismatches_vs_numpy'] == 0
        results.append(item)
        if len(results) == 1:
            denormal_checkpoints.append(cpu_denormal_probe('after_first_gpu_case'))
    denormal_checkpoints.append(cpu_denormal_probe('qualification_end_including_opencv_sdf'))
    return {'passed': bool(all_ok), 'exact_replacement_of_shipping_cupy': bool(all_ok),
            'numpy_stage_and_boolean_agreement': bool(numpy_agrees),
            'software_ieee754_exceptional_cases_agreement': bool(software_agrees),
            'software_oracle_note': 'Integer-only IEEE-754 binary32 multiplication/addition with nearest-even '
                                    'rounding and gradual underflow; NaNs canonicalized. Applied to crossed '
                                    'exceptional patterns independently of CPU floating-point state.',
            'cpu_denormal_checkpoints': denormal_checkpoints,
            'gate_note': 'The replacement gate checks the actual CuPy helper and separate CuPy stages. '
                         'All NumPy discrepancies remain separately reported; this is not a universal CPU equivalence claim.',
            'cases': results, 'fma_boundary_witnesses': witnesses,
            'nan_policy': 'NaN classification and final Boolean exact; NaN payload bits reported separately.'}


def benchmark(cp, fused, stream, sizes, rounds, target_ms):
    owner = SimpleNamespace(xp=cp)
    results = []
    for side in sizes:
        # Analytic circle fields retain mixed signs; tests above separately use
        # actual EDT-generated inputs. Field generation/H2D are outside timing.
        yy, xx = np.mgrid[:side, :side].astype(np.float32)
        a = cp.asarray((side*.27 - np.hypot(yy-side*.46, xx-side*.42)).astype(np.float32))
        b = cp.asarray((side*.23 - np.hypot(yy-side*.51, xx-side*.56)).astype(np.float32))
        alpha = 1/3
        ca, cb = coefficients(alpha)
        output = cp.empty(a.shape, dtype=cp.bool_)
        left, right, summed = [cp.empty_like(a) for _ in range(3)]
        preallocated_output = cp.empty(a.shape, dtype=cp.bool_)
        n = int(a.size)

        def run(mode):
            if mode == 'production_allocating':
                return CudaInterpolationRenderer._blend_section(owner, a, b, alpha)
            if mode == 'ufunc_preallocated':
                cp.multiply(a, ca, out=left)
                cp.multiply(b, cb, out=right)
                cp.add(left, right, out=summed)
                cp.greater_equal(summed, np.float32(0.0), out=preallocated_output)
                return preallocated_output
            fused(((n+255)//256,), (256,), (a, b, ca, cb, output, np.uint64(n)), stream=stream)
            return output

        def measure(mode, repeats):
            start, end = cp.cuda.Event(), cp.cuda.Event()
            stream.synchronize()
            wall_start = time.perf_counter()
            start.record(stream)
            for _ in range(repeats):
                run(mode)
            end.record(stream)
            end.synchronize()
            return {'cuda_event_ms': float(cp.cuda.get_elapsed_time(start, end))/repeats,
                    'synchronized_wall_ms': (time.perf_counter()-wall_start)*1000/repeats}

        modes = ('production_allocating', 'ufunc_preallocated', 'fused_preallocated')
        for _ in range(8):
            for mode in modes:
                run(mode)
        stream.synchronize()
        ref = cp.asnumpy(run(modes[0]))
        for mode in modes[1:]:
            if not np.array_equal(ref, cp.asnumpy(run(mode))):
                raise AssertionError(f'Benchmark output mismatch: {side}, {mode}')
        pilot = [measure(mode, 10)['synchronized_wall_ms'] for mode in modes]
        repeats = max(10, min(2000, int(target_ms/max(pilot))))
        samples = {mode: [] for mode in modes}
        for round_index in range(rounds):
            # Alternating order counters systematic drift; the fused/production
            # pair reverses order on each round alongside the allocation control.
            for mode in (modes if round_index % 2 == 0 else modes[::-1]):
                samples[mode].append(measure(mode, repeats))
        medians = {mode: {key: statistics.median(sample[key] for sample in rows)
                          for key in ('cuda_event_ms', 'synchronized_wall_ms')}
                   for mode, rows in samples.items()}
        row = {'shape': [side, side], 'repeats_per_round': repeats, 'rounds': rounds,
               'medians': medians, 'samples': samples, 'exact_boolean': True,
               'full_fp32_temporary_plane_bytes_avoided': n*12,
               'nominal_global_bytes_per_call': {'four_ufuncs': n*33, 'fused': n*9},
               'production_vs_fused_speedup': {key: medians[modes[0]][key]/medians[modes[2]][key]
                                              for key in medians[modes[0]]},
               'preallocated_ufunc_vs_fused_speedup': {key: medians[modes[1]][key]/medians[modes[2]][key]
                                                      for key in medians[modes[0]]}}
        results.append(row)
        print(json.dumps({key: value for key, value in row.items() if key != 'samples'}), flush=True)
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sizes', default='32,128,512,2048')
    parser.add_argument('--rounds', type=int, default=9)
    parser.add_argument('--target-ms', type=float, default=30)
    args = parser.parse_args()
    sizes = [int(v) for v in args.sizes.split(',')]
    if not sizes or min(sizes) < 1 or max(sizes) > 8192 or args.rounds < 3 or args.target_ms <= 0:
        parser.error('Use sizes 1..8192, at least 3 rounds and a positive target-ms.')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    source_path = args.output.with_suffix('.cu')
    source_path.write_text(SOURCE, encoding='utf-8')
    cache_dir = args.output.parent / (args.output.stem + '_cupy_cache')
    cache_dir.mkdir(exist_ok=True)
    os.environ['CUPY_CACHE_DIR'] = str(cache_dir)
    os.environ['CUPY_CACHE_SAVE_CUDA_SOURCE'] = '1'
    denormal_checkpoints = [cpu_denormal_probe('before_cupy_import')]
    import cupy as cp
    denormal_checkpoints.append(cpu_denormal_probe('after_cupy_import'))

    module = cp.RawModule(code=SOURCE, options=OPTIONS, backend='nvrtc',
                          name_expressions=('blend_strict', 'blend_diagnostic'))
    fused, diagnostic = [module.get_function(name) for name in ('blend_strict', 'blend_diagnostic')]
    denormal_checkpoints.append(cpu_denormal_probe('after_nvrtc_compile'))
    device_id = cp.cuda.runtime.getDevice()
    props = cp.cuda.runtime.getDeviceProperties(device_id)
    device_name = props['name'].decode() if isinstance(props['name'], bytes) else props['name']
    stream = cp.cuda.Stream(non_blocking=True)
    config = __import__('io').StringIO()
    from contextlib import redirect_stdout
    with redirect_stdout(config):
        cp.show_config()
    from cupy.cuda import compiler as cupy_compiler
    compiler_path = Path(cupy_compiler.__file__)
    compiler_bytes = compiler_path.read_bytes()
    compiler_lines = compiler_bytes.decode('utf-8').splitlines()
    inherited_ftz_lines = [i+1 for i, line in enumerate(compiler_lines)
                           if "options += ('-ftz=true',)" in line]
    report = {'scope': 'Experimental pre-topology SDF blend; no production patch or application speedup claim.',
              'caller_owns_gpu_lock_and_heatsoak': True, 'device': device_name,
              'compute_capability': [props['major'], props['minor']], 'cupy': cp.__version__,
              'numpy': np.__version__, 'python': sys.version, 'cupy_config': config.getvalue(),
              'cpu_denormal_checkpoints': denormal_checkpoints,
              'driver': cp.cuda.runtime.driverGetVersion(), 'runtime': cp.cuda.runtime.runtimeGetVersion(),
              'nvrtc_version': list(cp.cuda.nvrtc.getVersion()),
              'compiler_backend': 'nvrtc', 'compiler_options_requested': list(OPTIONS),
              'cupy_compiler_source': str(compiler_path),
              'cupy_compiler_source_sha256': hashlib.sha256(compiler_bytes).hexdigest(),
              'cupy_compiler_appended_ftz_true_lines': inherited_ftz_lines,
              'effective_compiler_note': 'Inspected CuPy appends -ftz=true after user options; actual SASS '
                                         'must determine effective arithmetic. The verified run used FMUL.FTZ and '
                                         'FADD.FTZ even though --ftz=false was requested. The replacement gate '
                                         'checks shipping CuPy, not universal IEEE/NumPy equivalence.',
              'kernel_attributes': {'strict': fused.attributes, 'diagnostic': diagnostic.attributes},
              'source': str(source_path), 'source_sha256': hashlib.sha256(SOURCE.encode()).hexdigest(),
              'baseline_source': inspect.getsource(CudaInterpolationRenderer._blend_section),
              'allocation_note': 'Production baseline requests 3 FP32 intermediate planes and 1 Boolean output per call; '
                                 'ufunc control and fused kernel reuse outputs. CuPy pool is warm. Three intermediate '
                                 'planes total 12 bytes/pixel but are not necessarily simultaneously live at peak.',
              'timing_note': 'CUDA events bracket repeated stream operations and may include host submission gaps. '
                             'Synchronized wall time includes Python submission and end-event wait, excludes inputs, '
                             'compilation, warmup, topology, transfers and painting. Bytes are nominal, not measured DRAM traffic.'}
    with stream:
        report['qualification'] = qualify(cp, fused, diagnostic, stream)
        args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
        if not report['qualification']['passed']:
            raise AssertionError(f'Strict qualification failed; see {args.output}')
        report['benchmarks'] = benchmark(cp, fused, stream, sizes, args.rounds, args.target_ms)
    report['compiler_cache_files'] = [{'path': str(path), 'bytes': path.stat().st_size}
                                      for path in cache_dir.rglob('*') if path.is_file()]
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f'Saved {args.output}; strict qualification passed.', flush=True)


if __name__ == '__main__':
    main()

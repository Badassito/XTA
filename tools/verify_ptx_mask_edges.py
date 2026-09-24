"""Record scalar/tiled union behavior for finite and exceptional box fields.

Caller owns the GPU lock. This probe records discrepancies without changing
production code or treating intentionally exceptional inputs as benchmark data.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def differences(left, right):
    a, b = left.detach().cpu().numpy(), right.detach().cpu().numpy()
    equal = a.view(np.uint32) == b.view(np.uint32)
    indices = np.flatnonzero(~equal)
    result = {'shape': list(a.shape), 'bit_mismatches': int(indices.size),
              'scalar_positive': int(np.count_nonzero(a > 0)),
              'tiled_positive': int(np.count_nonzero(b > 0)),
              'boolean_mismatches_gt_zero': int(np.count_nonzero((a > 0) != (b > 0))),
              'scalar_nan_count': int(np.count_nonzero(np.isnan(a))),
              'tiled_nan_count': int(np.count_nonzero(np.isnan(b)))}
    if indices.size:
        i = int(indices[0])
        result['first_mismatch'] = {
            'index': list(np.unravel_index(i, a.shape)),
            'scalar_value': repr(float(a.ravel()[i])),
            'tiled_value': repr(float(b.ravel()[i])),
            'scalar_bits': f'0x{int(a.ravel().view(np.uint32)[i]):08x}',
            'tiled_bits': f'0x{int(b.ravel().view(np.uint32)[i]):08x}',
        }
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault('OMP_NUM_THREADS', '2')
    os.environ.setdefault('MKL_NUM_THREADS', '2')
    os.environ['CUPY_CACHE_DIR'] = str(args.output.parent / (args.output.stem+'_cupy_cache'))
    import torch
    from XTA import inference

    torch.set_num_threads(2)
    report = {'scope': 'Actual generic direct payload scalar/tiled parity on exceptional bboxes; no production change.',
              'caller_owns_gpu_lock': True, 'torch': torch.__version__,
              'device': torch.cuda.get_device_name(), 'cases': [], 'unexpected_errors': []}

    def save():
        args.output.write_text(json.dumps(report, indent=2, default=int)+'\n', encoding='utf-8')

    try:
        kernels = inference._resident_mask_kernels()
        if kernels is None:
            raise RuntimeError('Resident mask kernel setup returned None')
        cp = kernels.cp
        report.update(cupy=cp.__version__, nvrtc_version=list(cp.cuda.nvrtc.getVersion()),
                      cuda_runtime=cp.cuda.runtime.runtimeGetVersion(),
                      cuda_driver=cp.cuda.runtime.driverGetVersion(),
                      kernel_attributes={name: getattr(kernels, name).attributes for name in
                                         ('union_f16_f16', 'union_f16_f16_tiled',
                                          'union_f32_f32', 'union_f32_f32_tiled')})
        scenarios = [('finite_control', None, 0.0)]
        for field, index in (('cx', 0), ('cy', 1), ('width', 2), ('height', 3)):
            for tag, value in (('nan', float('nan')), ('positive_inf', float('inf')),
                               ('negative_inf', -float('inf'))):
                scenarios.append((f'{field}_{tag}', index, value))

        for dtype in (torch.float16, torch.float32):
            for name, field, value in scenarios:
                item = {'dtype': str(dtype), 'scenario': name}
                try:
                    # One retained detection, positive first-channel logit.
                    # Width remains even and C32 so both tiled paths really admit.
                    ph, pw, ih, iw = 16, 18, 32, 36
                    head = torch.zeros((37, 1), device='cuda', dtype=dtype)
                    head[:4, 0] = torch.tensor([18., 16., 18., 16.], device='cuda', dtype=dtype)
                    head[4, 0] = .75
                    head[5, 0] = 1.
                    if field is not None:
                        head[field, 0] = value
                    proto = torch.zeros((32, ph, pw), device='cuda', dtype=dtype)
                    proto[0].fill_(1.)
                    image = torch.empty((1, 1, ih, iw), device='cuda', dtype=dtype)
                    payloads = []
                    for enabled in (False, True):
                        with (mock.patch.dict(os.environ, {'YOLO_TTA_DIRECT_DEVICE_COMPACTION': '1',
                                                          'YOLO_TTA_DIRECT_TILED_PROTO_UNION': str(int(enabled))}),
                              mock.patch.object(inference, 'gpu_flatten_conf_tracking_enabled', return_value=True),
                              mock.patch.object(inference, 'angle_variant_gpu_fastpath', return_value=None)):
                            payload = inference._build_direct_device_compacted_payload(head, proto, image, .5)
                        if payload is None:
                            raise RuntimeError(f'Direct payload returned None, tiled={enabled}')
                        payloads.append(payload)
                    torch.cuda.synchronize()
                    scalar, tiled = payloads
                    expected_tiled = 'tiled_f16' if dtype == torch.float16 else 'tiled_f32'
                    selected = [scalar.compaction_kernel, tiled.compaction_kernel]
                    if selected != ['scalar', expected_tiled]:
                        raise RuntimeError(f'Unexpected dispatch {selected}')
                    item.update(selected_kernels=selected,
                                retained_counts=[int(p.instance_count_device.item()) for p in payloads],
                                logits=differences(scalar.device_refs[4], tiled.device_refs[4]),
                                prototype_confidence=differences(scalar.device_refs[5], tiled.device_refs[5]),
                                mask=differences(scalar.union_gpu, tiled.union_gpu),
                                image_confidence=differences(scalar.conf_gpu, tiled.conf_gpu))
                    item['all_outputs_bit_exact'] = all(item[key]['bit_mismatches'] == 0
                                                       for key in ('logits', 'prototype_confidence', 'mask', 'image_confidence'))
                except Exception as exc:
                    item['unexpected_error'] = f'{type(exc).__name__}: {exc}'
                    report['unexpected_errors'].append({'dtype': str(dtype), 'scenario': name,
                                                        'traceback': traceback.format_exc()})
                report['cases'].append(item)
                save()
                print(json.dumps(item, default=int), flush=True)
    except Exception:
        report['unexpected_errors'].append({'setup_traceback': traceback.format_exc()})
    report['summary'] = {'cases': len(report['cases']),
                         'mismatching_cases': sum(item.get('all_outputs_bit_exact') is False for item in report['cases']),
                         'unexpected_error_count': len(report['unexpected_errors'])}
    save()
    print(f'Saved {args.output}', flush=True)


if __name__ == '__main__':
    main()

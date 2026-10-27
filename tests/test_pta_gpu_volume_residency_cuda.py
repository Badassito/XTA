"""Opt-in, locked real-CUDA functional qualification of PTA volume turnover.

No fork/process pool or augmentation policy is needed. The root runner owns the
GPU lock and selects this test explicitly; ordinary CPU suites only skip it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest


pytestmark = pytest.mark.skipif(
    os.environ.get('XTA_RUN_PTA_VOLUME_RESIDENCY_CUDA') != '1',
    reason='explicit CUDA qualification with an owned GPU lock only',
)


def _locked_owner():
    path = Path(os.environ['XTA_GPU_LOCK_PATH'])
    owner = json.loads(path.read_text(encoding='utf-8'))
    assert owner['pid'] == os.getpid()
    assert owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    return dict(path=str(path), task=owner.get('task'), pid=owner['pid'],
                start=owner.get('start', owner.get('start_time')))


def _memory(torch, runtime, label):
    device = int(runtime['device_id'])
    torch.cuda.synchronize(device)
    free, total = torch.cuda.mem_get_info(device)
    renderer = runtime['azimuthal_renderer']
    owner = runtime.get('categorical_volume_owner')
    mask = getattr(owner, 'mask_gpu', None)
    coverage = getattr(owner, 'coverage_gpu', None)
    return dict(label=label, free_bytes=int(free), total_bytes=int(total),
        allocated_bytes=int(torch.cuda.memory_allocated(device)),
        reserved_bytes=int(torch.cuda.memory_reserved(device)),
        intensity_bytes=int(getattr(getattr(renderer, '_volume_gpu', None), 'nbytes', 0)),
        categorical_bytes=int(getattr(mask, 'nbytes', 0)) +
            (0 if coverage is mask else int(getattr(coverage, 'nbytes', 0))))


def _transverse_plan(root, shape):
    from XTA import pta
    from XTA.pta_config import parse_pta_args

    depth, height, width = shape
    config = parse_pta_args(['--input', 'synthetic', '--imgsz', str(width),
                             '--enable_cartesian', 'transverse'])
    views, _ = pta.compile_v18_pta_views(t_dim=depth, h=height, w=width,
                                        config=config, azimuthal_native_raster=width)
    view = next(v for v in views if v.family == 'transverse')
    aff = pta.build_affine(view.src_w, view.src_h, 0., view.pad_mode, width,
                           shared_view=view.shared_view)
    return pta.build_render_plan(view=view, aff=aff, tag=view.name,
        out_dir=root / 'unused-plan', stem='residency', tile_configs=(),
        save_overlay=False, imgsz=width, label_enabled=True,
        publish_images=False, publish_labels=False)


def _exact_generation(torch, runtime, source, mask, coverage, plan, identity):
    from XTA import pta_cuda_masks

    renderer = runtime['azimuthal_renderer']
    assert renderer.ensure_volume_array(source, identity=identity,
        require_azimuthal_texture=False) == 'resident'
    np.testing.assert_array_equal(renderer._volume_gpu.cpu().numpy(), source)
    for frame in (0, source.shape[0] - 1):
        with torch.cuda.stream(renderer._stream):
            image = renderer.render_cartesian_grid_resident(plan.view.shared_view,
                plan.aff.M_out_to_src, frame, plan.aff.out_h, plan.aff.out_w)
        renderer._stream.synchronize()
        np.testing.assert_array_equal(image.cpu().numpy(), source[frame])
        projected = pta_cuda_masks.render_gpu_categorical_item(runtime,
            mask, coverage, plan, frame, 'full', identity=identity + ':categorical')
        assert projected is not None
        projected_mask, projected_coverage, ready = projected
        ready.synchronize()
        np.testing.assert_array_equal(projected_mask.cpu().numpy(), mask[frame])
        np.testing.assert_array_equal(projected_coverage.cpu().numpy(), coverage[frame])
    owner = runtime['categorical_volume_owner']
    np.testing.assert_array_equal(owner.mask_gpu.cpu().numpy(), mask)
    np.testing.assert_array_equal(owner.coverage_gpu.cpu().numpy(), coverage)
    assert renderer._native_u8_plane_cache  # Exercise actual private frame owners.
    return dict(shape=list(source.shape), source_bytes=int(source.nbytes),
        source_sha256=hashlib.sha256(source.tobytes()).hexdigest(),
        mask_sha256=hashlib.sha256(mask.tobytes()).hexdigest(),
        coverage_sha256=hashlib.sha256(coverage.tobytes()).hexdigest(),
        weak_sources=(weakref.ref(renderer._volume_gpu), weakref.ref(owner.mask_gpu),
                      weakref.ref(owner.coverage_gpu)))


def test_real_cuda_repeated_volume_handoff_and_capacity_only_retry(tmp_path, monkeypatch):
    lock = _locked_owner()  # Check ownership before touching the CUDA runtime.
    import torch
    from XTA import cuda_backend, pta_cuda_masks, pta_workers

    assert torch.cuda.is_available(), 'Explicit CUDA qualification requested but CUDA is unavailable'
    device = int(os.environ.get('XTA_TEST_PTA_DEVICE', '0'))
    monkeypatch.setenv('YOLO_TTA_GPU_RENDER_RESIDENT', '1')
    monkeypatch.setenv('YOLO_TTA_GPU_NATIVE_PLANE_CACHE', '2')
    monkeypatch.setenv('YOLO_TTA_PTA_GPU_CATEGORICAL', '1')
    monkeypatch.setattr(cuda_backend, 'gpu_render_reserve_bytes', lambda: 0)
    monkeypatch.setattr(pta_cuda_masks, '_reserve_bytes', lambda: 0)
    clock = SimpleNamespace(now=time.monotonic())
    # Only shadow this module's admission clock; leave Python/worker clocks real.
    module_clock = SimpleNamespace(monotonic=lambda: clock.now,
        perf_counter=time.perf_counter, time=time.time, sleep=time.sleep)
    monkeypatch.setattr(cuda_backend, 'time', module_clock)
    renderer = cuda_backend._GpuWorkerRenderEngine(f'cuda:{device}')
    runtime = dict(torch=torch, device_id=device, azimuthal_renderer=renderer,
        azimuthal_render_lock=threading.RLock(), azimuthal_texture_required=False)
    rng = np.random.default_rng(160926)
    volumes = {label: rng.integers(0, 256, shape, dtype=np.uint8)
               for label, shape in [('A', (32, 512, 512)), ('B', (16, 1024, 1024))]}
    masks = {label: np.ascontiguousarray((value % 7) == 0, dtype=np.uint8)
             for label, value in volumes.items()}
    coverages = {label: np.ascontiguousarray((value % 3) != 0, dtype=np.uint8)
                 for label, value in volumes.items()}
    plans = {label: _transverse_plan(tmp_path / label, value.shape)
             for label, value in volumes.items()}
    receipt = dict(scope='Locked real-CUDA functional test; no throughput benchmark or fork pool',
        lock=lock, device=device, device_name=torch.cuda.get_device_name(device),
        torch_version=torch.__version__, cuda_version=torch.version.cuda,
        memory=[], generations=[], pressure={},
        admission_clock='Module-local controlled clock; +5.1s tests cooldown without a benchmark sleep')
    previous = None
    real_empty_cache = torch.cuda.empty_cache
    try:
        with mock.patch.object(torch.cuda, 'empty_cache', wraps=real_empty_cache) as trim:
            for generation, label in enumerate(('A', 'B', 'A')):
                identity = 'residency-' + label
                before = _memory(torch, runtime, f'{label}{generation}:before-handoff')
                receipt['memory'].append(before)
                pta_workers._prepare_gpu_volume_task(runtime,
                    identity=(identity, identity + ':mask', identity + ':coverage', None),
                    generation=generation)
                after = _memory(torch, runtime, f'{label}{generation}:after-handoff')
                receipt['memory'].append(after)
                assert after['intensity_bytes'] == after['categorical_bytes'] == 0
                if previous is not None:
                    assert all(reference() is None for reference in previous['weak_sources'])
                    assert before['allocated_bytes'] - after['allocated_bytes'] >= 3 * previous['source_bytes']
                if label == 'B':
                    # Reserve policy induces a real capacity rejection without
                    # allocating a large filler tensor or falsifying free bytes.
                    torch.cuda.empty_cache()
                    free, _ = torch.cuda.mem_get_info(device)
                    reserve_fraction = 1.05
                    reserve = int(free * reserve_fraction)
                    receipt['pressure'].update(driver_free_before=int(free),
                        reserve_fraction=reserve_fraction, requested_reserve_bytes=reserve)
                    monkeypatch.setattr(cuda_backend, 'gpu_render_reserve_bytes', lambda: reserve)
                    before_trim = trim.call_count
                    assert renderer.ensure_volume_array(volumes[label], identity=identity) == 'stream'
                    assert trim.call_count - before_trim == 1
                    assert renderer._source_residency_timings['admission_reclaim_attempted']
                    receipt['pressure']['declined_admission'] = dict(renderer._source_residency_timings)
                    assert renderer._volume_gpu is None and not renderer._resident_runtime_disabled
                    # Item-level repeats and an immediate task boundary cannot
                    # bypass the minimum retry interval or issue another trim.
                    assert renderer.ensure_volume_array(volumes[label], identity=identity) == 'stream'
                    pta_workers._prepare_gpu_volume_task(runtime,
                        identity=(identity, identity + ':mask', identity + ':coverage', None),
                        generation=generation)
                    assert renderer.ensure_volume_array(volumes[label], identity=identity) == 'stream'
                    assert trim.call_count - before_trim == 1
                    monkeypatch.setattr(cuda_backend, 'gpu_render_reserve_bytes', lambda: 0)
                    clock.now += 5.1
                    pta_workers._prepare_gpu_volume_task(runtime,
                        identity=(identity, identity + ':mask', identity + ':coverage', None),
                        generation=generation)
                checked = _exact_generation(torch, runtime, volumes[label], masks[label],
                                            coverages[label], plans[label], identity)
                if label == 'B':
                    assert renderer._source_residency_timings['admission_capacity_retry']
                    receipt['pressure']['recovered_admission'] = dict(renderer._source_residency_timings)
                receipt['generations'].append({k:v for k,v in checked.items() if k != 'weak_sources'} |
                                               dict(label=label, generation=generation))
                previous = checked
                receipt['memory'].append(_memory(torch, runtime, f'{label}{generation}:validated'))
            # A permanent runtime-disable is not a capacity rejection and must
            # remain sticky even after the retry interval or a task boundary.
            renderer.disable_resident_after_runtime_failure()
            clock.now += 6.
            assert not renderer.arm_residency_retry()
            assert renderer.ensure_volume_array(volumes['A'], identity='residency-A') == 'stream'
            assert renderer._resident_runtime_disabled and renderer._volume_gpu is None
            receipt['runtime_disable_was_not_rearmed'] = True
    finally:
        # Preserve a partial receipt even when a real device assertion fails.
        try:
            pta_cuda_masks.retire_gpu_categorical_volume(runtime)
            renderer.retire_volume_array()
            receipt['memory'].append(_memory(torch, runtime, 'final-sources-retired'))
        finally:
            (tmp_path / 'gpu-volume-residency-receipt.json').write_text(
                json.dumps(receipt, indent=2), encoding='utf-8')

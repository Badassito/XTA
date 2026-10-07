"""CPU-only admission and lifetime checks for TTA's persistent SAM context."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import ast
import os
from pathlib import Path
import sys
import threading
import types
from unittest import mock

import numpy as np
import pytest

from XTA import backprojection, geometry, interpolation, pipeline, sam_integration


def _context(tmp_path, *, devices=('0',)):
    # These legacy admission fakes have no model or allocator. Dual-context
    # startup and its measured gates are exercised in the resource suite.
    with mock.patch.dict(os.environ, {'YOLO_TTA_SAM_SESSIONS_PER_GPU': '1'}):
        return sam_integration.SamInterpolationContext(
            model_path='test-bundle', device_ids=devices,
            temp_dir=tmp_path / 'temporary', evidence_root=tmp_path / 'retained',
            source_volume=np.arange(60, dtype=np.uint8).reshape(3, 4, 5),
            source_identity='source-identity', detector_identity='detector-identity',
        )


def _fake_runtime(factory):
    module = types.ModuleType('XTA.sam_tracker_runtime')
    module.SamInterpolationTracker = factory
    # No CUDA operation belongs to these admission tests. Avoid importing Torch
    # inside patch.dict, which would remove its newly imported C-extension state
    # on restoration and make later imports unsafe.
    torch_module = types.ModuleType('torch')
    torch_module.cuda = types.SimpleNamespace(device_count=lambda: 3)
    return mock.patch.dict(sys.modules, {
        'XTA.sam_tracker_runtime': module, 'torch': torch_module,
    })


def _view():
    return geometry.ViewInfo(
        name='transverse__tta_a0', physical_view_name='transverse',
        num_slices=3, src_h=4, src_w=5, pad_mode='clamp', family='orthogonal',
        tta_angle_deg=0.0,
    )


def test_runtime_start_waits_for_retired_detector_then_loads_once(tmp_path):
    context = _context(tmp_path)
    lease = mock.Mock()
    runtime = mock.Mock()
    factory = mock.Mock(return_value=runtime)
    started = threading.Event()

    def begin():
        started.set()
        context._start()

    with _fake_runtime(factory), mock.patch.object(
        backprojection, '_try_acquire_specific_main_process_gpu_stage', return_value=lease,
    ) as acquire, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(begin)
        assert started.wait(2)
        try:
            assert not future.done()
            factory.assert_not_called()
            acquire.assert_not_called()
            context.detector_assets_retired()
            future.result(timeout=5)
            context._start()
            factory.assert_called_once()
            assert factory.call_args.kwargs['device_ids'] == (0,)
            runtime.start.assert_called_once()
            acquire.assert_called_once()
        finally:
            context.close()
    runtime.close.assert_called_once()
    lease.release.assert_called_once()
    assert context.source_volume is None


def test_cancellation_settles_readiness_without_loading_model(tmp_path):
    context = _context(tmp_path)
    factory = mock.Mock()
    started = threading.Event()

    def begin():
        started.set()
        context._start()

    with _fake_runtime(factory), ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(begin)
        assert started.wait(2)
        try:
            context.cancel('detector inference failed')
            with pytest.raises(RuntimeError, match='detector inference failed'):
                future.result(timeout=5)
            factory.assert_not_called()
            assert context._ready.is_set()
        finally:
            context.close()


def test_cancellation_during_device_admission_releases_prior_leases(tmp_path):
    context = _context(tmp_path, devices=('0', '2'))
    context.detector_assets_retired()
    lease = mock.Mock()
    factory = mock.Mock()

    def acquire(_torch, index, _purpose):
        if index == 0:
            return lease
        context.cancel('cancelled during SAM admission')
        return None

    with _fake_runtime(factory), mock.patch.object(
        backprojection, '_try_acquire_specific_main_process_gpu_stage', side_effect=acquire,
    ):
        with pytest.raises(RuntimeError, match='cancelled during SAM admission'):
            context._start()
    factory.assert_not_called()
    lease.release.assert_called_once()
    assert not context._leases
    context.close()


def test_failed_runtime_start_releases_device_lease(tmp_path):
    context = _context(tmp_path)
    context.detector_assets_retired()
    lease = mock.Mock()
    runtime = mock.Mock()
    runtime.start.side_effect = RuntimeError('SAM worker initialization failed')
    with _fake_runtime(mock.Mock(return_value=runtime)), mock.patch.object(
        backprojection, '_try_acquire_specific_main_process_gpu_stage', return_value=lease,
    ):
        with pytest.raises(RuntimeError, match='SAM worker initialization failed'):
            context._start()
    assert context._runtime is None
    assert not context._leases
    lease.release.assert_called_once()
    runtime.close.assert_called_once()
    context.close()


def test_cancelled_resident_context_admits_no_new_sam_jobs(tmp_path):
    context = _context(tmp_path)
    context._runtime = mock.Mock()
    context.cancel('upstream generation failed')
    try:
        with pytest.raises(RuntimeError, match='upstream generation failed'):
            context._start()
    finally:
        context.close()


def test_native_source_cache_is_exact_and_reused_until_close(tmp_path):
    context = _context(tmp_path)
    original = context.source_volume.copy()
    provider = context.image_provider(_view(), original.shape)
    assert context.image_provider(_view(), original.shape) is provider
    cache = provider.open()
    try:
        assert np.array_equal(cache, original)
    finally:
        cache._mmap.close()
    context.close()
    with pytest.raises(RuntimeError, match='lifetime has ended'):
        context.image_provider(_view(), original.shape)


def test_different_processing_shapes_keep_prior_source_cache_immutable(tmp_path):
    context = _context(tmp_path)
    native = context.image_provider(_view(), (3, 4, 5))
    resized = context.image_provider(_view(), (3, 6, 6))
    try:
        assert native.path != resized.path
        native.revalidate()
        native_cache = native.open()
        try:
            assert np.array_equal(native_cache, context.source_volume)
        finally:
            native_cache._mmap.close()
    finally:
        context.close()


def _drain_callback(context, records, *, complete=True, collected=2):
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding='utf-8'))
    function = next(node for node in ast.walk(tree)
                    if isinstance(node, ast.FunctionDef)
                    and node.name == '_announce_process_inference_drain_if_complete')
    request = mock.Mock(return_value=complete)
    namespace = dict(vars(pipeline))
    namespace.update({
        'scheduler_state': types.SimpleNamespace(
            gpu_worker_results_collected=collected, gpu_worker_total_tasks=2,
            gpu_inference_drain_announced=True,
            gpu_inference_asset_release_results_by_worker=records,
        ),
        'gpu_worker_process_active': True, 'sam_context': context,
        'scheduler': types.SimpleNamespace(request_gpu_inference_asset_release=request),
        '_restore_parent_post_inference_affinity': mock.Mock(),
        '_set_main_process_gpu_asset_retirement_pending': mock.Mock(),
        '_set_main_process_gpu_inference_priority_active': mock.Mock(),
    })
    exec(compile(ast.Module(body=[function], type_ignores=[]),
                 '<sam-pipeline-drain-callback>', 'exec'), namespace)
    return namespace[function.name], request


def test_pipeline_waits_for_complete_actual_detector_retirement():
    context = mock.Mock()
    records = {0: {'ok': True, 'stats': {'released': True, 'assets_intact': False}}}
    callback, request = _drain_callback(context, records, complete=False)
    callback()
    request.assert_called_once()
    context.detector_assets_retired.assert_not_called()
    context.cancel.assert_not_called()
    callback, _ = _drain_callback(context, records)
    callback()
    context.detector_assets_retired.assert_called_once()


@pytest.mark.parametrize('record', (
    {'ok': False, 'stats': {'released': False, 'assets_intact': True, 'phase': 'validate_drain'}},
    {'ok': True, 'stats': {'released': False, 'assets_intact': True}},
    {'ok': True, 'stats': {}},
))
def test_safe_refusal_or_missing_retirement_proof_cannot_admit_sam(record):
    context = mock.Mock()
    records = {0: {'ok': True, 'stats': {'released': True}}, 2: record}
    callback, _ = _drain_callback(context, records)
    with pytest.raises(RuntimeError, match='SAM GPU admission failed'):
        callback()
    context.cancel.assert_called_once()
    context.detector_assets_retired.assert_not_called()


def test_pipeline_does_not_settle_sam_before_detector_result_drain():
    context = mock.Mock()
    callback, request = _drain_callback(context, {}, collected=1)
    callback()
    request.assert_not_called()
    context.detector_assets_retired.assert_not_called()
    context.cancel.assert_not_called()


def test_gate_support_fingerprint_is_stable_across_policy_receipt_updates(tmp_path):
    support = np.zeros((2, 5, 7), dtype=np.uint8)
    support[1, 2, 3] = 1
    path = tmp_path / 'support.cvol'
    interpolation.write_raw_bbox_mask_store(
        support, path, format_name=interpolation.CVOL_FORMAT, workers=1,
    )
    store = interpolation.RawBBoxMaskStore.open(path)
    try:
        original = sam_integration.publish_sam_gate_identity(
            store, policy_identity='policy-a', evidence_path='evidence-a',
        )
        repeated = sam_integration.publish_sam_gate_identity(
            store, policy_identity='policy-a', evidence_path='evidence-a',
        )
        rescored = sam_integration.publish_sam_gate_identity(
            store, policy_identity='policy-b', evidence_path='evidence-b',
        )
        assert original == repeated == rescored
        assert store.meta['gate_support_identity'] == original
        assert store.meta['interpolation_policy_identity'] == 'policy-b'
    finally:
        store.close()


def test_cpu_detector_sam_pool_does_not_keep_detector_inference_priority(tmp_path):
    # Execute the actual coordinator configuration and immediately following
    # conditional initialization, without launching the rest of the pipeline.
    tree = ast.parse(Path(pipeline.__file__).read_text(encoding='utf-8'))
    statements = None
    for parent in ast.walk(tree):
        body = getattr(parent, 'body', ())
        if not isinstance(body, list):
            continue
        for index, node in enumerate(body):
            if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                    and isinstance(node.value.func, ast.Name)
                    and node.value.func.id == '_configure_main_process_gpu_stage_workers'):
                statements = [node]
                for following in body[index + 1:]:
                    if not isinstance(following, ast.If):
                        break
                    statements.append(following)
                break
        if statements is not None:
            break
    assert statements is not None
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    namespace = dict(vars(pipeline))
    namespace.update(
        gpu_logical_indices=[],
        interpolation_settings=types.SimpleNamespace(sam_devices=('0',)),
        gpu_worker_process_active=False,
        _configure_main_process_gpu_stage_workers=coordinator.configure_workers,
        _set_main_process_gpu_inference_priority_active=coordinator.set_inference_priority_active,
    )
    exec(compile(ast.Module(body=statements, type_ignores=[]),
                 '<sam-coordinator-configuration>', 'exec'), namespace)
    assert not coordinator.snapshot()['inference_priority_active']
    context = _context(tmp_path)
    context.detector_assets_retired()
    runtime = mock.Mock()
    fake_torch = types.SimpleNamespace(cuda=types.SimpleNamespace(device_count=lambda: 1))
    with _fake_runtime(mock.Mock(return_value=runtime)), mock.patch.object(
        backprojection, '_try_acquire_specific_main_process_gpu_stage',
        side_effect=lambda _torch, index, purpose:
            coordinator.try_acquire_specific_stage(fake_torch, index, purpose),
    ):
        try:
            context._start()
            runtime.start.assert_called_once()
            assert coordinator.snapshot()['stage_leases']
        finally:
            context.close()
    assert not coordinator.snapshot()['stage_leases']

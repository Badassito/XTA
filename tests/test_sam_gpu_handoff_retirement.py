"""A fenced source transfer skips broad cleanup only with real driver headroom."""
import gc
import time
import weakref
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA.cuda_backend import _GpuWorkerRenderEngine
from XTA import inference
from XTA.sam_gpu_rendering import SamGpuCropRenderer
from tests.test_sam_gpu_rendering import ready_handoff_pair


class _Owner:
    pass


@pytest.fixture
def owned_pair(tmp_path, monkeypatch):
    context, first, second, old_engine, lease = ready_handoff_pair(tmp_path)
    engine = object.__new__(_GpuWorkerRenderEngine)
    engine.__dict__.update(old_engine.__dict__)
    del engine.clear_native_plane_cache
    owners = [_Owner() for _ in range(9)]
    references = [weakref.ref(owner) for owner in owners]
    engine._native_plane_cache = {'plane': owners[0]}
    engine._native_u8_plane_cache = {'plane': owners[1]}
    engine._spherical_direction_cache = {'directions': owners[2]}
    engine._spherical_rotation_cache = {'rotation': owners[3]}
    engine._tilted_plans = {'plan': owners[4]}
    engine._fold_cache = {'fold': owners[5]}
    engine._fused_azimuthal_taps = {'taps': owners[6]}
    engine._standalone_render_meta = engine._standalone_render_meta_ref = owners[7]
    monkeypatch.setattr(inference, '_AFFINE_GRID_CACHE', {
        ('cuda:0', 'grid'): owners[8], ('cuda:1', 'grid'): _Owner()})
    source = _Owner()
    source_reference = weakref.ref(source)
    engine._volume_gpu = engine._volume_flat = engine._fused_volume_ref = source
    first.engine = engine
    del owners, source, old_engine
    try:
        yield context, first, second, engine, lease, references, source_reference
    finally:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            context.cancel('test cleanup')
            for renderer in (first, second, *tuple(context._gpu_image_waiters)):
                context._finish_gpu_image_wait(renderer)
                renderer.close()
        context._runtime = None
        context.close()


def test_owned_handoff_drops_workspace_aliases_without_gc_or_allocator_trim(owned_pair):
    context, first, second, engine, lease, references, source_reference = owned_pair
    enabled = gc.isenabled()
    gc.disable()
    try:
        def fence():
            assert all(reference() is not None for reference in references)
        engine._stream.synchronize.side_effect = fence
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device') as trim, \
                mock.patch.object(gc, 'collect') as collect:
            first.close()
        trim.assert_not_called()
        collect.assert_not_called()
        assert all(reference() is None for reference in references)
        assert tuple(inference._AFFINE_GRID_CACHE) == (('cuda:1', 'grid'),)
        assert source_reference() is engine._volume_gpu
        assert first.engine is first.lease is None
        assert second.engine is engine and second.lease is lease
        assert context._gpu_image_owners == {0: second}
        lease.release.assert_not_called()
        first.close()
        lease.release.assert_not_called()
    finally:
        if enabled:
            gc.enable()


@pytest.mark.parametrize('first_probe', [256, RuntimeError('memory probe unavailable')])
def test_low_or_unproven_free_memory_keeps_trim_and_rechecks_headroom(owned_pair, first_probe):
    _context, first, second, engine, lease, _references, _source = owned_pair
    probe = first.torch.cuda.mem_get_info
    probe.side_effect = [first_probe if isinstance(first_probe, Exception) else (first_probe, 2**40),
                         (1024, 2**40)]
    with mock.patch('XTA.backprojection._trim_main_process_cuda_device') as trim:
        first.close()
    trim.assert_called_once()
    assert trim.call_args.kwargs['desc'] == 'SAM source handoff'
    assert trim.call_args.kwargs['repeat_garbage_collection'] is False
    assert probe.call_count == 2
    assert second.engine is engine and second.lease is lease


def test_changed_fifo_head_cannot_use_prior_candidates_smaller_allowance(owned_pair):
    context, first, second, engine, lease, _references, _source = owned_pair
    third = SamGpuCropRenderer(context, None, context.source_volume, 7, None, first.torch,
        2**20, required_gpu=context.source_volume.nbytes+1024)
    def change_head(_device):
        context._queue_gpu_image(third)
        with context._gpu_lease_lock:
            context._gpu_image_waiters.rotate(1)
            third._gpu_wait_deadline = time.monotonic()+60
        return 512, 2**40
    first.torch.cuda.mem_get_info.side_effect = change_head
    with mock.patch('XTA.backprojection._trim_main_process_cuda_device') as trim:
        first.close()
    trim.assert_called_once()
    assert trim.call_args.kwargs['desc'] == 'SAM image preparation'
    engine.release_inference_assets.assert_called_once()
    lease.release.assert_called_once()
    assert second.engine is second.lease is third.engine is third.lease is None
    assert not context._gpu_image_owners


def test_cancellation_during_free_probe_retires_source_conservatively(owned_pair):
    context, first, second, engine, lease, _references, _source = owned_pair
    def cancel(_device):
        context.cancel('cancel during headroom probe')
        return 2**30, 2**40
    first.torch.cuda.mem_get_info.side_effect = cancel
    with mock.patch('XTA.backprojection._trim_main_process_cuda_device') as trim:
        first.close()
    trim.assert_called_once()
    assert trim.call_args.kwargs['desc'] == 'SAM image preparation'
    assert trim.call_args.kwargs['repeat_garbage_collection'] is True
    engine.release_inference_assets.assert_called_once()
    lease.release.assert_called_once()
    assert second.engine is second.lease is None
    assert not context._gpu_image_owners


@pytest.mark.parametrize('state', ['unknown_engine', 'unsettled', 'render_failed', 'cancelled', 'active_error'])
def test_unproven_state_keeps_conservative_cleanup(owned_pair, state):
    context, first, _second, engine, _lease, _references, _source = owned_pair
    if state == 'unknown_engine':
        first.engine = SimpleNamespace(**engine.__dict__)
        first.engine.clear_native_plane_cache = mock.Mock()
    elif state == 'unsettled':
        context._unsettled_image_renderers.append(first)
    elif state == 'render_failed':
        first._render_failed = True
    elif state == 'cancelled':
        context.cancel('cancel image')
    with mock.patch('XTA.backprojection._trim_main_process_cuda_device') as trim:
        if state == 'active_error':
            try:
                raise RuntimeError('active render error')
            except RuntimeError:
                first.close()
        else:
            first.close()
    trim.assert_called_once()
    assert trim.call_args.kwargs['repeat_garbage_collection'] is True

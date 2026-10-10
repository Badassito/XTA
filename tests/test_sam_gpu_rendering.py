"""GPU transaction routing, pixel provenance and retirement without CUDA."""
from dataclasses import replace
from contextlib import nullcontext
import os
from pathlib import Path
import threading
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import geometry
from XTA.lta_rendering import LtaPhysicalViewCacheRef
from XTA.sam_gpu_rendering import (SAM_GPU_IMAGE_CONTRACT, SamGpuCropRenderer,
    SamGpuRenderingUnavailable, crop_inverse, live_image_sampling,
    record_live_image, same_live_image_geometry, try_gpu_crop_renderer)
from tests.test_sam_view_image_cache import context_for, crop_from, demand
from tests.test_sam_view_orientations import SOURCE_SHAPE, VIEWS
from tests.test_sam_resident_lease_adversarial import admission


class FakeRenderer:
    def __init__(self, context, *, fail_after=None):
        self.context, self.fail_after = context, fail_after
        self.calls, self.closed = 0, False

    def sampling_identity(self):
        return dict(contract=SAM_GPU_IMAGE_CONTRACT, backend='cuda', implementation_sha256='test')

    def render(self, view, index, inverse, **kwargs):
        self.calls += 1
        if self.fail_after is not None and self.calls > self.fail_after:
            raise SamGpuRenderingUnavailable('simulated GPU capacity refusal')
        # Distinct bytes prove that caches do not combine CPU/GPU donors.
        return np.full((kwargs['output_height'], kwargs['output_width']), 201+index%7, np.uint8)

    def close(self):
        self.closed = True


@pytest.mark.parametrize('view', VIEWS, ids=lambda view: view.name)
def test_all_view_demands_use_gpu_renderer_without_cpu_projection(tmp_path, view):
    source=np.zeros(SOURCE_SHAPE,np.uint8)
    context=context_for(tmp_path,source)
    shape=(view.num_slices,8,8);frame=view.num_slices//2;bbox=(1,2,7,6)
    renderer=FakeRenderer(context)
    try:
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=renderer):
            reference=context.image_provider(view,shape,demand(shape,{frame:bbox}))
        np.testing.assert_array_equal(crop_from(reference,frame,bbox),201+frame%7)
        assert renderer.calls==1 and renderer.closed
        assert live_image_sampling(reference)['backend']=='cuda'
    finally:
        context.close()


def test_cpu_gpu_cache_identity_and_unknown_descriptors_remain_separate(tmp_path):
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    view=geometry.get_view_infos(*SOURCE_SHAPE,cartesian_views=('sagittal',))[0]
    shape=(view.num_slices,8,8);planned=demand(shape,{2:(1,1,7,7)})
    try:
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=None):
            cpu=context.image_provider(view,shape,planned)
        renderer=FakeRenderer(context)
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=renderer):
            gpu=context.image_provider(view,shape,planned)
        assert cpu.identity_sha256!=gpu.identity_sha256
        assert same_live_image_geometry(cpu.identity_sha256,gpu)
        assert not same_live_image_geometry(cpu.identity_sha256,LtaPhysicalViewCacheRef.from_payload(gpu.payload()))
        table={};record_live_image(table,cpu);record_live_image(table,gpu)
        assert set(table['image_sampling_sources'])=={cpu.identity_sha256,gpu.identity_sha256}
        assert table['image_sampling_backend']['backend']=='cpu'
        # Changed physical geometry can never inherit a sampler-switch proof.
        changed=replace(view,center_x=view.center_x+.125)
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=FakeRenderer(context)):
            other=context.image_provider(changed,shape,planned)
        assert not same_live_image_geometry(cpu.identity_sha256,other)
        context.close()
        assert not same_live_image_geometry(cpu.identity_sha256,gpu)
    finally:
        context.close()


def test_failed_gpu_transaction_retires_before_complete_cpu_rebuild(tmp_path):
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    view=geometry.get_view_infos(*SOURCE_SHAPE,cartesian_views=('sagittal',))[0]
    shape=(view.num_slices,8,8);frames={1:(1,1,7,7),3:(1,2,6,7)}
    renderer=FakeRenderer(context,fail_after=1)
    original=context._image_provider
    def build(*args,**kwargs):
        if renderer.calls:
            assert renderer.closed
        return original(*args,**kwargs)
    try:
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=renderer), \
                mock.patch.object(context,'_image_provider',side_effect=build):
            reference=context.image_provider(view,shape,demand(shape,frames))
        assert renderer.closed and live_image_sampling(reference)['backend']=='cpu'
        for frame,bbox in frames.items():
            np.testing.assert_array_equal(crop_from(reference,frame,bbox),0)
        assert len(context._cache_entries)==1
        assert len(list((context.temp_dir/'sam_image_cache').glob('*.dat')))==1
        assert not context._image_builds
    finally:
        context.close()


def test_crop_matrix_preserves_global_coordinates_for_rectangular_nonzero_origin():
    matrix=np.array([[.719, .013, -.453],[-.017,.811,.391]],np.float32)
    shifted=crop_inverse(matrix,(391,527))
    points=np.array([[0,0],[13,29],[51,7]],np.float64)
    expected=(points+np.array([527,391]))@matrix[:,:2].T+matrix[:,2]
    actual=points@shifted[:,:2].T+shifted[:,2]
    np.testing.assert_allclose(actual,expected,rtol=0,atol=2e-5)
    with pytest.raises(ValueError):crop_inverse(matrix,(-1,0))


@pytest.mark.parametrize('quantized', (False, True))
def test_tilted_sam_uses_public_fused_crop_entrypoint_on_owned_stream(tmp_path, quantized):
    import torch
    from contextlib import contextmanager
    view = next(view for view in VIEWS if view.family == 'tilted')
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    lease = SimpleNamespace(device_index=0, release=mock.Mock())
    renderer = SamGpuCropRenderer(context, view, context.source_volume, SOURCE_SHAPE[0],
        lease, torch, 2**20)
    active = []
    @contextmanager
    def enter(name):
        active.append(name)
        try:
            yield
        finally:
            active.pop()
    values = torch.linspace(-10, 270, 15).reshape(3, 5)
    expected = values.round().clamp(0, 255).to(torch.uint8)
    matrix = np.array([[.75, .031, -.25], [-.014, .83, .27]], np.float32)
    origin = (1, 2)
    frame = view.num_slices//2
    def crop(actual_view, actual_matrix, actual_frame, height, width):
        assert active == ['device', 'stream']
        assert actual_view is view and actual_frame == frame
        assert (height, width) == (3, 5)
        np.testing.assert_array_equal(actual_matrix, crop_inverse(matrix, origin))
        return expected if quantized else values
    engine = SimpleNamespace(device='cuda:0', _stream=object(), _volume_mm=None,
        render_tilted_grid_resident=mock.Mock(side_effect=crop),
        _render_tilted_frame=mock.Mock(side_effect=AssertionError('full native reconstruction')),
        clear_native_plane_cache=mock.Mock(), release_inference_assets=mock.Mock())
    renderer.engine = engine
    try:
        with mock.patch.object(torch.cuda, 'device', side_effect=lambda _device:enter('device')), \
                mock.patch.object(torch.cuda, 'stream', side_effect=lambda _stream:enter('stream')):
            pixels = renderer.render(view, frame, matrix, output_origin_yx=origin,
                output_height=3, output_width=5)
        assert pixels.shape == (3, 5) and pixels.dtype == np.uint8
        np.testing.assert_array_equal(pixels, expected.numpy())
        assert not active and engine.render_tilted_grid_resident.call_count == 1
        engine._render_tilted_frame.assert_not_called()
    finally:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            renderer.close()
        context.close()


@pytest.mark.parametrize('scenario', ('busy_then_free', 'insufficient', 'mixed', 'timeout', 'cancel'))
def test_gpu_image_busy_retry_is_bounded_and_waits_without_gpu_ownership(tmp_path, scenario):
    from XTA import sam_gpu_rendering as rendering
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    context.device_ids = ('cuda:0', 'cuda:1')
    clock, waits, held, probes = [0.], [], set(), []
    renderer = SamGpuCropRenderer(context, None, context.source_volume, SOURCE_SHAPE[0],
        None, SimpleNamespace(cuda=SimpleNamespace(), device=lambda token:token), 256,
        required_gpu=1024)
    def claim(_torch, device, _purpose):
        probes.append(device)
        busy = (scenario in ('timeout', 'cancel') or scenario == 'busy_then_free' and not waits
                or scenario == 'mixed' and device == 0 and not waits)
        if busy:
            return None
        assert device not in held
        held.add(device)
        return SimpleNamespace(device_index=device, release=mock.Mock(side_effect=lambda:held.remove(device)))
    def free(device):
        assert device in held
        return (0 if scenario == 'insufficient' or scenario == 'mixed' and device == 1 else 2048, 4096)
    renderer.torch.cuda.mem_get_info = mock.Mock(side_effect=free)
    class Cancel:
        cancelled = False
        def is_set(self):return self.cancelled
        def set(self):self.cancelled = True
        def wait(self, seconds):
            assert not held and renderer.lease is None and renderer.engine is None
            assert 0 < seconds <= .05
            waits.append(seconds)
            clock[0] += seconds
            if scenario == 'cancel':
                context.cancel('controlled image cancellation')
    context._cancel = Cancel()
    engine = SimpleNamespace(_volume_mm=None, ensure_volume_array=mock.Mock(return_value='resident'),
        release_inference_assets=mock.Mock())
    telemetry = SimpleNamespace(add=mock.Mock())
    try:
        with mock.patch.object(rendering, 'time', SimpleNamespace(monotonic=lambda:clock[0])), \
                mock.patch('XTA.sam_integration.time.monotonic',side_effect=lambda:clock[0]), \
                mock.patch('XTA.runtime.runtime_telemetry', return_value=telemetry), \
                mock.patch('XTA.sam_integration.runtime_telemetry', return_value=telemetry), \
                mock.patch('XTA.backprojection._try_acquire_specific_main_process_gpu_stage', side_effect=claim), \
                mock.patch('XTA.cuda_backend._GpuWorkerRenderEngine', return_value=engine) as create:
            if scenario in ('busy_then_free', 'mixed'):
                assert renderer._start() is engine
                assert waits == [.05] and len(probes) > 2
                assert held == {0} and renderer.torch.cuda.mem_get_info.call_count >= 1
            elif scenario == 'cancel':
                with pytest.raises(RuntimeError, match='controlled image cancellation'):
                    renderer._start()
                create.assert_not_called()
                assert waits == [.05] and not held
            else:
                with pytest.raises(SamGpuRenderingUnavailable, match='No idle GPU'):
                    renderer._start()
                create.assert_not_called()
                assert not held
                assert (not waits and len(probes) == 2) if scenario == 'insufficient' else clock[0] == 30.
            recorded = telemetry.add.call_args_list
            count = lambda key:sum(call.args[1] for call in recorded if call.args[0] == 'sam.gpu_images.'+key)
            assert count('admission_attempts') == len(probes)
            assert count('admission_wait_seconds') == pytest.approx(clock[0])
            assert count('admission_wait_timeouts') == int(scenario == 'timeout')
    finally:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            renderer.close()
        assert not held
        context.close()


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned CUDA qualification')
def test_real_fused_tilted_sam_crops_keep_cpu_geometry_and_small_pixel_delta(tmp_path, monkeypatch):
    import torch
    import cv2
    from pathlib import Path
    from XTA import cuda_backend
    from XTA.config import resolve_tilted_view_groups
    lock = Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file(), 'caller must hold GPU_LOCK for explicit CUDA qualification'
    assert torch.cuda.is_available()
    monkeypatch.setenv('YOLO_TTA_GPU_RENDER_RESERVE_GIB', '1')
    monkeypatch.setenv('YOLO_TTA_FUSED_DIRECT_RENDER', '1')
    monkeypatch.setenv('YOLO_TTA_FUSED_TILTED_RENDER', '1')
    source = np.memmap(tmp_path/'source.dat', mode='w+', dtype=np.uint8, shape=(17, 39, 47))
    source[:] = np.random.default_rng(571).integers(0, 256, source.shape, dtype=np.uint8)
    source.flush()
    kernels = cuda_backend._fused_direct_render_kernels()
    assert kernels is not None, cuda_backend._FUSED_DIRECT_RENDER_KERNELS_ERROR
    deltas, old_deltas, cpu_deltas = [], [], []
    for logical_t in (17, 25):
        cube = (np.asarray(source) if logical_t == source.shape[0] else
            cv2.resize(np.asarray(source).reshape(source.shape[0], -1),
                (source.shape[1]*source.shape[2], logical_t), interpolation=cv2.INTER_LINEAR)
                .reshape(logical_t, *source.shape[1:]))
        views = geometry.get_view_infos(logical_t, 39, 47, cartesian_views=(),
            tilt_groups=resolve_tilted_view_groups(['transverse,sagittal,coronal:20:both']),
            azimuthal_views=(), azimuthal_azimuth_angles=())
        context = context_for(tmp_path/str(logical_t), source)
        renderer = SamGpuCropRenderer(context, None, source, logical_t,
            SimpleNamespace(device_index=0, release=mock.Mock()), torch, 2**20)
        try:
            engine = renderer._start()
            for view in views:
                matrix = np.array([[.79, .023, -.41], [-.017, .83, .29]], np.float32)
                for frame in (0, view.num_slices//2, view.num_slices-1):
                    # Make fallback an explicit failure of this accelerated proof.
                    with mock.patch.object(engine, '_render_tilted_frame',
                            side_effect=AssertionError('fused path fell back')):
                        pixels = renderer.render(view, frame, matrix, output_origin_yx=(1, 2),
                            output_height=11, output_width=13)
                    assert pixels.shape == (11, 13) and pixels.dtype == np.uint8
                    # Keep both the quantized TTA oracle and the former SAM
                    # float-native-before-affine path as separate comparisons.
                    with torch.cuda.stream(engine._stream):
                        reference = engine._render_tilted_frame(view,
                            np.array([[1,0,0],[0,1,0]],np.float32),
                            view.src_h, view.src_w, frame).round().clamp(0,255).to(torch.uint8)
                        expected = engine.warp_native_uint8_frame(reference,
                            crop_inverse(matrix,(1,2)),11,13).cpu().numpy()
                        old = engine._render_tilted_frame(view,crop_inverse(matrix,(1,2)),
                            11,13,frame).round().clamp(0,255).to(torch.uint8).cpu().numpy()
                    delta = np.abs(pixels.astype(np.int16)-expected.astype(np.int16))
                    assert delta.max() <= 1
                    deltas.append(int(delta.max()))
                    old_delta = np.abs(pixels.astype(np.int16)-old.astype(np.int16))
                    assert old_delta.max() <= 1
                    old_deltas.append(int(old_delta.max()))
                    cpu = geometry.render_tilted_frame_on_grid(cube, view, frame,
                        crop_inverse(matrix,(1,2)),11,13)
                    cpu_delta = np.abs(pixels.astype(np.int16)-cpu.astype(np.int16))
                    # OpenCV quantizes affine fractions to 1/32; Torch samples
                    # continuous fractions. Two 255-level gradients plus
                    # native/final quantization have a <=9-level bound.
                    assert cpu_delta.max() <= 9
                    cpu_deltas.append(int(cpu_delta.max()))
            assert 'tilted' not in engine._fused_disabled_families
        finally:
            renderer.close()
            context.close()
    source._mmap.close()
    assert len(deltas) >= 36
    print({'fused_crop_cases':len(deltas), 'max_torch_gray8_delta':max(deltas),
           'max_former_sam_gpu_gray8_delta':max(old_deltas),
           'max_cpu_gray8_delta':max(cpu_deltas)})


def test_complete_cpu_cache_is_reused_without_claiming_or_uploading_gpu(tmp_path):
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    view=geometry.get_view_infos(*SOURCE_SHAPE,cartesian_views=('sagittal',))[0]
    shape=(view.num_slices,8,8);planned=demand(shape,{2:(1,1,7,7)})
    try:
        cpu=context.image_provider(view,shape,planned)
        context.detector_assets_retired();context._resource_local.profile=object()
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',
                side_effect=AssertionError('uploaded existing CPU pixels')):
            assert context.image_provider(view,shape,planned) is cpu
    finally:context.close()


def test_gpu_demand_origin_isolates_pixels_and_features_without_cross_demand_donors(tmp_path):
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    view=geometry.get_view_infos(*SOURCE_SHAPE,cartesian_views=('sagittal',))[0]
    shape=(view.num_slices,8,8);frame=2
    class OriginRenderer(FakeRenderer):
        def render(self,*args,**kwargs):
            pixels=super().render(*args,**kwargs)
            return pixels+np.uint8(kwargs['output_origin_yx'][1]%2)
    left=demand(shape,{frame:(0,0,6,6)})
    right=demand(shape,{frame:(0,1,6,7)})
    first_renderer,second_renderer=OriginRenderer(context),OriginRenderer(context)
    try:
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=first_renderer):
            first=context.image_provider(view,shape,left)
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=second_renderer):
            second=context.image_provider(view,shape,right)
        assert first.identity_sha256!=second.identity_sha256
        shared_bbox=(2,2,4,4)
        np.testing.assert_array_equal(crop_from(first,frame,shared_bbox),203)
        np.testing.assert_array_equal(crop_from(second,frame,shared_bbox),204)
        assert context.image_cache_reused_pixels==0
        # The production frame-feature key binds this actual image identity,
        # even when the physical frame and SDK session crop are identical.
        crop=(2,2,4,4)
        keys=[(ref.identity_sha256,ref.physical_view_id,crop,frame,'pinned_rgb_u8_loader_v1')
            for ref in (first,second)]
        assert keys[0]!=keys[1]
        assert same_live_image_geometry(first.identity_sha256,second)
        repeat=OriginRenderer(context)
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=repeat):
            assert context.image_provider(view,shape,right) is second
        assert repeat.calls==0 and repeat.closed
    finally:context.close()


def test_unfinished_decode_declines_before_cuda_or_lease_access(tmp_path):
    from XTA.media import VolumeReadiness, register_volume_readiness, _VOLUME_READINESS_BY_ARRAY_ID
    source=np.zeros((3,4,5),np.uint8);register_volume_readiness(source,VolumeReadiness(3))
    context=context_for(tmp_path,source);context.detector_assets_retired()
    view=geometry.get_view_infos(*source.shape,cartesian_views=('sagittal',))[0]
    context._resource_local.profile=object()
    try:
        with mock.patch('XTA.sam_resources.validate_live_sam_resource_profile',
                return_value={'base_non_cpu_allowance_bytes':2**30}), \
                mock.patch.object(context,'_try_sam_compute_lease',side_effect=AssertionError('claimed GPU')):
            assert try_gpu_crop_renderer(context,view,(4,8,8),demand((4,8,8),{1:(1,1,7,7)})) is None
    finally:
        _VOLUME_READINESS_BY_ARRAY_ID.pop(id(source),None)
        context.close()


def test_failed_render_retirement_keeps_lease_until_successful_retry(tmp_path):
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    lease=SimpleNamespace(device_index=0,release=mock.Mock())
    renderer=SamGpuCropRenderer(context,None,context.source_volume,7,lease,
        SimpleNamespace(device=lambda token:token),256)
    renderer.engine=SimpleNamespace(release_inference_assets=mock.Mock(
        side_effect=[RuntimeError('unsettled stream'),{}]),_volume_mm=None)
    with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
        with pytest.raises(RuntimeError,match='unsettled stream'):renderer.close()
        lease.release.assert_not_called()
        assert context._unsettled_image_renderers==[renderer]
        renderer.close()
        assert lease.release.call_count==1 and not context._unsettled_image_renderers
    context.close()


def ready_handoff_pair(tmp_path, *, free=2**30):
    import time
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    lease=SimpleNamespace(device_index=0,release=mock.Mock())
    torch=SimpleNamespace(device=lambda value:value,
        cuda=SimpleNamespace(mem_get_info=mock.Mock(return_value=(free,2**40))))
    first=SamGpuCropRenderer(context,None,context.source_volume,7,lease,torch,2**20)
    second=SamGpuCropRenderer(context,None,context.source_volume,7,None,torch,2**20,
        required_gpu=context.source_volume.nbytes+512)
    engine=SimpleNamespace(_stream=SimpleNamespace(synchronize=mock.Mock()),
        _volume_gpu=object(),_volume_flat=object(),_volume_mm=None,_azimuthal_texture_ref=None,
        clear_native_plane_cache=mock.Mock(),_tilted_plans={'old':object()},_fold_cache={'old':object()},
        _fused_azimuthal_taps={'old':object()},_native_t_map_cache={'source':object()},
        _fused_volume_ref=object(),_standalone_render_meta=object(),_standalone_render_meta_ref=object(),
        release_inference_assets=mock.Mock())
    first.engine=engine
    first._uploaded_source_key=first._source_handoff_key()
    context._queue_gpu_image(first)
    context._finish_gpu_image_wait(first,granted=True)
    second._gpu_wait_deadline=time.monotonic()+60
    context._queue_gpu_image(second)
    return context,first,second,engine,lease


def test_same_source_handoff_moves_single_owner_and_bounds_burst_to_two(tmp_path):
    context,first,second,engine,lease=ready_handoff_pair(tmp_path)
    third=SamGpuCropRenderer(context,None,context.source_volume,7,None,first.torch,2**20)
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'), \
                mock.patch('XTA.cuda_backend._GpuWorkerRenderEngine',
                    side_effect=AssertionError('reuploaded handed source')):
            first.close()
            assert first.engine is None and first.lease is None
            assert second.engine is engine and second.lease is lease and second._image_burst_count==2
            assert context._gpu_image_owners=={0:second} and not context._gpu_image_waiters
            lease.release.assert_not_called()
            engine.release_inference_assets.assert_not_called()
            engine._stream.synchronize.assert_called_once()
            assert engine._native_t_map_cache and not engine._tilted_plans and not engine._fold_cache
            assert not engine._fused_azimuthal_taps and engine._standalone_render_meta is None
            assert second._start() is engine
            first.close()  # Former owner cannot release the recipient's lease.
            lease.release.assert_not_called()
            context._queue_gpu_image(third)
            second.close()
            engine.release_inference_assets.assert_called_once()
            lease.release.assert_called_once()
            assert third.lease is None and third.engine is None and not context._gpu_image_owners
            context._finish_gpu_image_wait(third)
    finally:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close();second.close();third.close()
        context._runtime=None
        context.close()


def test_ready_sdk_allows_one_source_handoff_then_gets_its_owed_turn(admission, tmp_path, monkeypatch):
    import sys
    coordinator, _auxiliary, torch = admission
    monkeypatch.setitem(sys.modules, 'torch', torch)
    context, first, second, engine, _ = ready_handoff_pair(tmp_path)
    resident = coordinator.try_acquire_specific_stage(torch, 0, 'SAM startup').promote_residency()
    context._resident_leases[0] = resident
    first.lease = resident.try_acquire_compute(torch, 'SAM image preparation')
    first.torch = second.torch = torch
    context._runtime = SimpleNamespace(has_ready_work=lambda:True, close=lambda:None)
    third = SamGpuCropRenderer(context, None, context.source_volume, 7, None, torch, 2**20)
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
            assert second.engine is engine and second._image_burst_count == 2
            assert coordinator.snapshot()['stage_leases'] == {0:'SAM image preparation'}
            context._queue_gpu_image(third)
            second.close()
            engine.release_inference_assets.assert_called_once()
            assert not context._gpu_image_owners and not coordinator.snapshot()['stage_leases']
            assert third.engine is None and context._try_gpu_image_lease(third, torch) is None
            assert not context._sam_compute_should_yield(0)
            sdk = context._try_sam_compute_lease(0, 'SAM tracker compute owed turn')
            assert sdk is not None and context._sam_compute_should_yield(0)
            assert context._try_gpu_image_lease(third, torch) is None
            context._release_sam_compute_lease(sdk)
            assert context._try_gpu_image_lease(third, torch) is not None
            context._finish_gpu_image_wait(third, granted=True)
            third.close()
        assert not context._gpu_image_waiters and not context._gpu_image_owners
        assert not coordinator.snapshot()['stage_leases']
    finally:
        context._finish_gpu_image_wait(third)
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close();second.close();third.close()
        context.close()


@pytest.mark.parametrize('case', ('healthy','render_failed','active_error','cancelled','unsettled','unknown_engine'))
def test_only_healthy_owned_renderer_skips_second_garbage_collection(tmp_path, case):
    from XTA.cuda_backend import _GpuWorkerRenderEngine
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    lease = SimpleNamespace(device_index=0, release=mock.Mock())
    torch = SimpleNamespace(device=lambda value:value)
    renderer = SamGpuCropRenderer(context, None, context.source_volume, 7, lease, torch, 2**20)
    renderer._image_burst_count = 2
    engine = (SimpleNamespace() if case=='unknown_engine' else object.__new__(_GpuWorkerRenderEngine))
    engine.release_inference_assets = mock.Mock()
    engine._volume_mm = None
    renderer.engine = engine
    if case=='render_failed':renderer._render_failed=True
    if case=='cancelled':context.cancel('cancelled image')
    if case=='unsettled':context._unsettled_image_renderers.append(renderer)
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device') as trim:
            if case=='active_error':
                try:raise RuntimeError('image failed')
                except RuntimeError:renderer.close()
            else:
                renderer.close()
        assert trim.call_args.kwargs['repeat_garbage_collection'] == (case!='healthy')
        engine.release_inference_assets.assert_called_once()
        lease.release.assert_called_once()
        assert renderer.engine is renderer.lease is None
    finally:
        context.close()


@pytest.mark.parametrize('refusal', ('changed_source','logical_t','expired','prefetch_cancel','headroom','unrelated_context'))
def test_source_handoff_refusals_retire_original_before_returning_compute(tmp_path,refusal):
    import time
    context,first,second,engine,lease=ready_handoff_pair(tmp_path,free=256 if refusal=='headroom' else 2**30)
    if refusal=='changed_source':second.source=context.source_volume.copy()
    if refusal=='logical_t':second.logical_t+=1
    if refusal=='expired':second._gpu_wait_deadline=time.monotonic()-1
    if refusal=='prefetch_cancel':
        second._gpu_prefetch_cancel=threading.Event();second._gpu_prefetch_cancel.set()
    if refusal=='unrelated_context':second.context=SimpleNamespace(source_identity=context.source_identity)
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
        engine.release_inference_assets.assert_called_once()
        lease.release.assert_called_once()
        assert first.lease is None and first.engine is None and not context._gpu_image_owners
        assert second.lease is None and second.engine is None
        context._finish_gpu_image_wait(second)
    finally:
        context._runtime=None
        context.close()


def test_failed_handoff_fence_keeps_exact_source_owner_until_retry(tmp_path):
    context,first,second,engine,lease=ready_handoff_pair(tmp_path)
    engine._stream.synchronize.side_effect=RuntimeError('handoff stream is unsettled')
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            with pytest.raises(RuntimeError,match='handoff stream is unsettled'):
                first.close()
            assert first.engine is engine and first.lease is lease
            assert second.engine is None and second.lease is None
            lease.release.assert_not_called()
            assert context._gpu_image_owners=={0:first} and context._unsettled_image_renderers==[first]
            assert context._cancel.is_set()
            first.close()  # Canceled context cannot hand off; normal fenced retirement retries.
        engine.release_inference_assets.assert_called_once()
        lease.release.assert_called_once()
        assert not context._gpu_image_owners and not context._unsettled_image_renderers
        context._finish_gpu_image_wait(second)
    finally:context.close()


def test_rejected_fresh_lease_cannot_be_overwritten_before_atomic_rejection(tmp_path):
    context,first,second,engine,lease=ready_handoff_pair(tmp_path)
    context.device_ids=('cuda:0','cuda:1')
    returned=SimpleNamespace(device_index=1,release=mock.Mock())
    second.lease,second.device_index=returned,1
    try:
        returned.release()
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            # Physical GPU1 is free, but GPU0's old source must not take over
            # while the CPU producer is completing GPU1's rejection bookkeeping.
            first.close()
        assert second.lease is returned and second.engine is None
        assert context._reject_gpu_image_device(second,returned)
        assert second.lease is None and second._gpu_image_rejected_devices=={1}
        lease.release.assert_called_once()
        engine.release_inference_assets.assert_called_once()
        context._finish_gpu_image_wait(second)
    finally:context.close()


def test_source_backing_change_prevents_reuse_of_prechange_gpu_pixels(tmp_path):
    context,first,second,engine,lease=ready_handoff_pair(tmp_path)
    source=np.memmap(tmp_path/'source.dat',mode='w+',dtype=np.uint8,shape=SOURCE_SHAPE)
    source[:]=0;source.flush()
    first.source=second.source=source
    first._uploaded_source_key=first._source_handoff_key()
    stat=(tmp_path/'source.dat').stat()
    os.utime(tmp_path/'source.dat',ns=(stat.st_atime_ns,stat.st_mtime_ns+10**9))
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            first.close()
        assert second.engine is None and second.lease is None
        engine.release_inference_assets.assert_called_once()
        lease.release.assert_called_once()
        context._finish_gpu_image_wait(second)
    finally:
        context.close();source._mmap.close()


def test_diagnostic_failure_cannot_undo_completed_source_handoff(tmp_path):
    context,first,second,engine,lease=ready_handoff_pair(tmp_path)
    try:
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'), \
                mock.patch('XTA.runtime.runtime_telemetry',return_value=SimpleNamespace(
                    add=mock.Mock(side_effect=RuntimeError('diagnostics unavailable')))):
            first.close()
        assert second.lease is lease and second.engine is engine
        assert first.lease is None and not context._cancel.is_set()
        lease.release.assert_not_called()
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):second.close()
        lease.release.assert_called_once()
    finally:context.close()


@pytest.mark.parametrize('cancel_after_transfer',(False,True))
def test_waiting_start_consumes_handoff_or_retires_canceled_assigned_owner(tmp_path,cancel_after_transfer):
    context,first,second,engine,lease=ready_handoff_pair(tmp_path)
    context._finish_gpu_image_wait(second)
    waiting,received=threading.Event(),threading.Event()
    errors=[]
    original=context._queue_gpu_image
    def enqueue(renderer):
        original(renderer);waiting.set()
    class Gate:
        def is_set(self):return received.is_set() and cancel_after_transfer
        def wait(self,_timeout):
            assert received.wait(5)
        def set(self):received.set()
    context._cancel=Gate()
    def run():
        try:
            assert second._start() is engine
        except BaseException as error:
            errors.append(error)
        finally:second.close()
    thread=threading.Thread(target=run)
    try:
        with mock.patch.object(context,'_queue_gpu_image',side_effect=enqueue), \
                mock.patch('XTA.backprojection._try_acquire_specific_main_process_gpu_stage',return_value=None), \
                mock.patch('XTA.backprojection._trim_main_process_cuda_device'), \
                mock.patch('XTA.cuda_backend._GpuWorkerRenderEngine',
                    side_effect=AssertionError('second source upload')):
            thread.start();assert waiting.wait(5)
            first.close()
            assert first.lease is None and second.lease is lease
            received.set()
            thread.join(5)
            assert not thread.is_alive()
        assert bool(errors)==cancel_after_transfer
        assert not context._gpu_image_waiters and not context._gpu_image_owners
        assert second.lease is None and second.engine is None
        lease.release.assert_called_once()
        engine.release_inference_assets.assert_called_once()
    finally:
        if thread.is_alive():
            context._cancel=threading.Event();context._cancel.set()
        received.set();thread.join(5)
        context.close()


@pytest.mark.skipif(os.environ.get('XTA_RUN_CUDA_RENDER_INTEGRATION') != '1',
                   reason='explicit GPU_LOCK-owned source handoff qualification')
def test_real_source_pair_uses_one_upload_and_retires_before_sdk_compute(tmp_path,monkeypatch):
    import torch
    lock=Path(__file__).resolve().parents[2]/'Scratch/Temp/GPU_LOCK'
    assert lock.is_file() and torch.cuda.is_available()
    monkeypatch.setenv('YOLO_TTA_GPU_RENDER_RESERVE_GIB','1')
    source=(np.arange(17*39*47,dtype=np.uint32)*17%251).astype(np.uint8).reshape(17,39,47)
    context=context_for(tmp_path,source)
    views=geometry.get_view_infos(*source.shape,cartesian_views=('sagittal','coronal'))
    first=SamGpuCropRenderer(context,views[0],source,17,None,torch,2**20,required_gpu=2**20)
    second=SamGpuCropRenderer(context,views[1],source,17,None,torch,2**20,required_gpu=2**20)
    identity=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)
    telemetry=SimpleNamespace(add=mock.Mock())
    waiting=threading.Event();received=[];errors=[]
    enqueue=context._queue_gpu_image
    def queue(renderer):
        enqueue(renderer)
        if renderer is second:waiting.set()
    def render_second():
        try:
            received.append(second.render(views[1],11,identity,output_origin_yx=(1,4),
                output_height=8,output_width=10))
        except BaseException as error:errors.append(error)
    thread=threading.Thread(target=render_second)
    try:
        with mock.patch('XTA.runtime.runtime_telemetry',return_value=telemetry), \
                mock.patch('XTA.sam_integration.runtime_telemetry',return_value=telemetry), \
                mock.patch.object(context,'_queue_gpu_image',side_effect=queue):
            pixels=first.render(views[0],5,identity,output_origin_yx=(2,3),output_height=8,output_width=10)
            np.testing.assert_array_equal(pixels,geometry.get_view_frame_by_index(source,views[0],5)[2:10,3:13])
            engine=first.engine
            thread.start();assert waiting.wait(5)
            first.close()
            thread.join(10)
            assert not thread.is_alive() and not errors
            assert second.engine is engine and second.lease is not None and first.lease is None
            np.testing.assert_array_equal(received[0],geometry.get_view_frame_by_index(source,views[1],11)[1:9,4:14])
            calls=telemetry.add.call_args_list
            value=lambda key:sum(call.args[1] for call in calls if call.args[0]=='sam.gpu_images.'+key)
            assert value('source_uploads')==1 and value('source_handoffs')==1
            assert value('source_upload_bytes')==source.nbytes and value('source_upload_bytes_saved')==source.nbytes
            second.close()
            assert engine._volume_gpu is None and engine._volume_flat is None
            assert not context._gpu_image_waiters and not context._gpu_image_owners
            # A following SDK admission observes no live source/stage lease.
            from XTA.backprojection import _try_acquire_specific_main_process_gpu_stage
            lease=_try_acquire_specific_main_process_gpu_stage(torch,0,'SAM handoff following SDK')
            assert lease is not None
            lease.release()
        torch.cuda.synchronize(0)
    finally:
        if thread.ident is not None:
            if thread.is_alive():context.cancel('source handoff qualification stopped')
            thread.join(10)
        first.close();second.close();context.close()


@pytest.mark.parametrize('error', (MemoryError('probe allocation failed'), OSError('probe device failed')))
def test_gpu_probe_error_retires_adopted_lease_before_propagation(tmp_path, error):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    lease = SimpleNamespace(device_index=0, release=mock.Mock())
    renderer = SamGpuCropRenderer(context, None, context.source_volume, SOURCE_SHAPE[0], None,
        SimpleNamespace(cuda=SimpleNamespace(mem_get_info=mock.Mock(side_effect=error)),
                        device=lambda token:token), 256, required_gpu=1024)
    try:
        with mock.patch('XTA.backprojection._try_acquire_specific_main_process_gpu_stage', return_value=lease), \
                mock.patch('XTA.cuda_backend._GpuWorkerRenderEngine') as create:
            with pytest.raises(type(error)) as caught:
                renderer._start()
            assert caught.value is error and renderer.lease is None and renderer.engine is None
            lease.release.assert_called_once()
            create.assert_not_called()
    finally:
        renderer.close()
        context.close()


def test_rejected_probe_release_failure_keeps_owner_and_close_quarantines_until_retry(tmp_path):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    lease = SimpleNamespace(device_index=0, release=mock.Mock(side_effect=[
        RuntimeError('first release failure'), RuntimeError('second release failure'), None]))
    renderer = SamGpuCropRenderer(context, None, context.source_volume, SOURCE_SHAPE[0], None,
        SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda _device:(0,4096)),
                        device=lambda token:token), 256, required_gpu=1024)
    try:
        with mock.patch('XTA.backprojection._try_acquire_specific_main_process_gpu_stage', return_value=lease), \
                mock.patch('XTA.cuda_backend._GpuWorkerRenderEngine') as create:
            with pytest.raises(RuntimeError, match='first release failure'):
                renderer._start()
            assert renderer.lease is lease and renderer.engine is None
            create.assert_not_called()
        with mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            with pytest.raises(RuntimeError, match='second release failure'):
                renderer.close()
            assert renderer.lease is lease and context._unsettled_image_renderers == [renderer]
            assert context._cancel.is_set()
            renderer.close()
            assert renderer.lease is None and not context._unsettled_image_renderers
        assert lease.release.call_count == 3
    finally:
        renderer.close()
        context.close()


def test_cpu_credit_wait_and_duplicate_image_ticket_hold_no_extra_gpu_lease(tmp_path):
    import torch
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8));context.detector_assets_retired()
    view=geometry.get_view_infos(*SOURCE_SHAPE,cartesian_views=('sagittal',))[0]
    shape=(view.num_slices,8,8);planned=demand(shape,{2:(1,1,7,7)})
    credit_wait,credit_go,render_wait,render_go=[threading.Event() for _ in range(4)]
    lease=SimpleNamespace(device_index=0,release=mock.Mock())
    results=[];errors=[]
    original=context._admit_image_build
    def admit(*args):
        credit_wait.set();assert credit_go.wait(5)
        return original(*args)
    def native(*args):
        render_wait.set();assert render_go.wait(5)
        return torch.zeros((view.src_h,view.src_w),dtype=torch.float32)
    engine=SimpleNamespace(device=torch.device('cuda:0'),_stream=object(),_volume_mm=None,
        ensure_volume_array=mock.Mock(return_value='resident'),
        _render_native_plane=native,
        warp_native_uint8_frame=lambda _plane,_matrix,h,w:torch.zeros((h,w),dtype=torch.uint8),
        clear_native_plane_cache=mock.Mock(),release_inference_assets=mock.Mock())
    def run():
        context._resource_local.profile=object()
        try:results.append(context.image_provider(view,shape,planned))
        except BaseException as error:errors.append(error)
    threads=[threading.Thread(target=run)for _ in range(2)]
    assigned={'base_non_cpu_allowance_bytes':2**30,'lease_id':'same-parent'}
    try:
        with mock.patch('XTA.sam_resources.validate_live_sam_resource_profile',return_value=assigned), \
                mock.patch('XTA.sam_canvas_rendering.cartesian_source_view',return_value=None), \
                mock.patch.object(context,'_admit_image_build',side_effect=admit), \
                mock.patch('XTA.backprojection._try_acquire_specific_main_process_gpu_stage',return_value=lease)as claim, \
                mock.patch('XTA.cuda_backend._GpuWorkerRenderEngine',return_value=engine), \
                mock.patch('XTA.backprojection._trim_main_process_cuda_device'), \
                mock.patch.object(torch.cuda,'is_available',return_value=True), \
                mock.patch.object(torch.cuda,'mem_get_info',return_value=(2**50,2**50)), \
                mock.patch.object(torch.cuda,'device',side_effect=lambda _device:nullcontext()), \
                mock.patch.object(torch.cuda,'stream',side_effect=lambda _stream:nullcontext()):
            threads[0].start();assert credit_wait.wait(5)
            assert claim.call_count==0
            credit_go.set();assert render_wait.wait(5)
            threads[1].start()
            with context._idle:
                assert context._idle.wait_for(lambda:any(ticket['waiters']==1
                    for ticket in context._image_builds.values()),timeout=5)
            assert claim.call_count==1 and lease.release.call_count==0
            render_go.set()
            for thread in threads:thread.join(5);assert not thread.is_alive()
            assert not errors and len(results)==2 and results[0] is results[1]
            assert claim.call_count==1 and lease.release.call_count==1
    finally:
        credit_go.set();render_go.set()
        for thread in threads:
            if thread.ident is not None:thread.join(5)
        context.close()


def test_alias_addressing_change_cannot_inherit_cpu_gpu_sampling_permission(tmp_path):
    from XTA.sam_cyclic import build_cyclic_frame_addressing
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    view=next(view for view in VIEWS if view.family=='azimuthal' and not view.azimuthal_tilted_source)
    shape=(view.num_slices,8,8);box=(1,1,7,7)
    try:
        original=context.image_provider(view,shape,demand(shape,{0:box}))
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=FakeRenderer(context)):
            alias=context.image_provider(view,shape,demand(shape,{shape[0]:box},
                build_cyclic_frame_addressing(shape,1)))
        assert not same_live_image_geometry(original.identity_sha256,alias)
    finally:context.close()


@pytest.mark.parametrize('mode',('whole','tiled'))
def test_mixed_sampler_cohorts_preserve_portable_actual_input_identity_per_run(tmp_path,mode):
    from contextlib import contextmanager
    from XTA.sam_evidence import SamEvidenceBundle
    from XTA.sam_extrapolation import (prepare_sam_extrapolation_pass,
        plan_sam_extrapolation_image_cohorts,extrapolate_sam_view_volume_pass)
    from tests.test_sam_extrapolation_image_cohorts import _baseline,_payload,Tracker
    import XTA.sam_extrapolation as core
    baseline=_baseline()
    context=context_for(tmp_path,np.zeros(baseline.shape,np.uint8))
    view=geometry.get_view_infos(*baseline.shape,cartesian_views=('transverse',))[0]
    prepared=prepare_sam_extrapolation_pass(baseline,view=view,distance=3,walk_back=0,min_radius=3.,crop_mode=mode)
    cap=max(_payload(core._cohort_prepared(prepared,(group.group_id,)))for group in prepared.groups)
    cohorts=plan_sam_extrapolation_image_cohorts(prepared,cap)
    assert len(cohorts)>1
    references=[]
    @contextmanager
    def provider(subset):
        renderer=None if not references else FakeRenderer(context)
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=renderer):
            reference=context.image_provider(view,baseline.shape,subset)
        references.append(reference)
        yield reference
    try:
        _,stats,_parts=extrapolate_sam_view_volume_pass(baseline,view=view,
            work_dir=tmp_path/'mixed-evidence',runtime=Tracker(baseline),prepared_plan=prepared,
            distance=3,walk_back=0,min_radius=3.,crop_mode=mode,image_cohorts=cohorts,image_cohort_provider=provider)
        bundle=SamEvidenceBundle.open(stats['sam_evidence_path'])
        identities=[reference.identity_sha256 for reference in references]
        assert len(set(identities))==len(cohorts)
        assert bundle.scope['image_snapshot_sha256']==identities[0]
        assert bundle.scope['image_sampling_backend']['backend']=='cpu'
        assert set(bundle.scope['image_sampling_sources'])==set(identities)
        for cohort,reference,receipt in zip(cohorts,references,stats['image_cohort_receipts']):
            assert receipt['image_snapshot_sha256']==reference.identity_sha256
            assert receipt['image_sampling_backend']==live_image_sampling(reference)
            for run in bundle.runs.values():
                if run['group_id'] in cohort.group_ids:
                    assert run['runtime_receipt']['image_snapshot_sha256']==reference.identity_sha256
    finally:context.close()


def test_chained_warp_error_retires_all_tensor_aliases_before_lease_return_and_cpu_fallback(tmp_path):
    import torch
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    view=geometry.get_view_infos(*SOURCE_SHAPE,cartesian_views=('sagittal',))[0]
    shape=(view.num_slices,8,8);planned=demand(shape,{2:(1,1,7,7)})
    tensor_refs=[]
    def released():
        assert tensor_refs and all(reference() is None for reference in tensor_refs)
    lease=SimpleNamespace(device_index=0,release=mock.Mock(side_effect=released))
    renderer=SamGpuCropRenderer(context,view,context.source_volume,7,lease,torch,2**20)
    engine=SimpleNamespace(device=torch.device('cuda:0'),_stream=object(),_volume_mm=None,
        _volume_gpu=torch.zeros(SOURCE_SHAPE),
        _render_native_plane=lambda _view,_frame:torch.zeros((view.src_h,view.src_w)),
        clear_native_plane_cache=mock.Mock())
    def fail_warp(native,_matrix,_height,_width):
        source_alias=engine._volume_gpu
        tensor_refs.extend((weakref.ref(native),weakref.ref(source_alias)))
        raise RuntimeError('inner warp retained tensor aliases')
    engine.warp_native_uint8_frame=fail_warp
    engine.release_inference_assets=lambda:setattr(engine,'_volume_gpu',None)
    renderer.engine=engine
    original=context._image_provider
    def build(*args,**kwargs):
        if tensor_refs:
            assert lease.release.call_count==1
            assert all(reference() is None for reference in tensor_refs)
        return original(*args,**kwargs)
    try:
        with mock.patch('XTA.sam_gpu_rendering.try_gpu_crop_renderer',return_value=renderer), \
                mock.patch.object(context,'_image_provider',side_effect=build), \
                mock.patch('XTA.backprojection._trim_main_process_cuda_device'), \
                mock.patch.object(torch.cuda,'device',side_effect=lambda _device:nullcontext()), \
                mock.patch.object(torch.cuda,'stream',side_effect=lambda _stream:nullcontext()):
            reference=context.image_provider(view,shape,planned)
        assert live_image_sampling(reference)['backend']=='cpu'
        lease.release.assert_called_once()
    finally:context.close()


def test_worker_shutdown_ack_cannot_release_parent_gpu_image_compute(tmp_path):
    context=context_for(tmp_path,np.zeros(SOURCE_SHAPE,np.uint8))
    lease=SimpleNamespace(device_index=0,release=mock.Mock())
    resident=SimpleNamespace(try_acquire_compute=mock.Mock(return_value=lease),
        release=mock.Mock(),quarantine=mock.Mock())
    context._resident_leases[0]=resident
    torch=SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda _device:(2**40,2**40)),
        device=lambda token:token)
    renderer=SamGpuCropRenderer(context,None,context.source_volume,7,None,torch,256,required_gpu=1024)
    engine=SimpleNamespace(ensure_volume_array=mock.Mock(return_value='resident'),
        _volume_mm=None,release_inference_assets=mock.Mock())
    try:
        with mock.patch('XTA.cuda_backend._GpuWorkerRenderEngine',return_value=engine), \
                mock.patch('XTA.backprojection._trim_main_process_cuda_device'):
            assert renderer._start() is engine
            assert context._active_compute=={}
            context._after_sam_worker_shutdown()
            lease.release.assert_not_called()
            renderer.close()
            lease.release.assert_called_once()
    finally:context.close()


@pytest.mark.parametrize('materialized,streaming',((False,False),(True,True)))
def test_lazy_source_gpu_recipe_uses_ready_processing_or_nonstreaming_native_depth(tmp_path,materialized,streaming):
    import torch
    from XTA.media import LazyProcessingCube
    decoded=np.memmap(tmp_path/'decoded.dat',mode='w+',dtype=np.uint8,shape=(3,4,5))
    decoded[:]=np.arange(decoded.size,dtype=np.uint8).reshape(decoded.shape);decoded.flush()
    # A ready materialized cube also supports XY changes without recomputing
    # them. The unmaterialized path supports only T, as TTA's renderer does.
    out_shape=(7,6,8) if materialized else (7,4,5)
    lazy=LazyProcessingCube(decoded,out_shape,tmp_path/'working.dat',workers=1,
        request_path=tmp_path/'request',ready_path=tmp_path/'ready',failed_path=tmp_path/'failed',
        streaming_backend=streaming)
    if materialized:lazy.materialize()
    context=context_for(tmp_path,lazy);context.detector_assets_retired()
    context._resource_local.profile=object()
    view=geometry.get_view_infos(*out_shape,cartesian_views=('sagittal',))[0]
    shape=(view.num_slices,8,8);planned=demand(shape,{1:(1,1,7,7)})
    try:
        with mock.patch('XTA.sam_resources.validate_live_sam_resource_profile',
                return_value={'base_non_cpu_allowance_bytes':2**30}), \
                mock.patch.object(torch.cuda,'is_available',return_value=True) as cuda_available, \
                mock.patch('XTA.backprojection._try_acquire_specific_main_process_gpu_stage',
                    side_effect=AssertionError('claimed GPU before winning CPU/image ticket')):
            renderer=try_gpu_crop_renderer(context,view,shape,planned)
        if materialized:
            assert renderer is None and lazy.materialized
            cuda_available.assert_not_called()
            return
        assert renderer is not None and renderer.lease is None
        assert renderer.source is (lazy._array if materialized else decoded)
        assert renderer.logical_t==out_shape[0]
        identity=renderer.sampling_identity()
        assert identity['native_shape_tyx']==list(out_shape if materialized else decoded.shape)
        assert identity['temporal_sampling']==('existing_processing_t' if materialized else
            'tta_center_aligned_virtual_processing_t_gray8')
        assert lazy.materialized is materialized
        renderer.close()
    finally:
        context.close();lazy.close();decoded._mmap.close()


@pytest.mark.parametrize('unfinished_decode',(False,True))
def test_unmaterialized_streaming_source_never_claims_gpu_for_decode_or_xy_changes(tmp_path,unfinished_decode):
    from XTA.media import (LazyProcessingCube,VolumeReadiness,register_volume_readiness,
        _VOLUME_READINESS_BY_ARRAY_ID)
    decoded=np.memmap(tmp_path/'decoded.dat',mode='w+',dtype=np.uint8,shape=(3,4,5))
    decoded[:]=0;decoded.flush()
    out_shape=(7,4,5) if unfinished_decode else (7,6,8)
    if unfinished_decode:register_volume_readiness(decoded,VolumeReadiness(3))
    lazy=LazyProcessingCube(decoded,out_shape,tmp_path/'working.dat',workers=1,
        request_path=tmp_path/'request',ready_path=tmp_path/'ready',failed_path=tmp_path/'failed',
        streaming_backend=True)
    context=context_for(tmp_path,lazy);context.detector_assets_retired()
    context._resource_local.profile=object()
    view=geometry.get_view_infos(*out_shape,cartesian_views=('sagittal',))[0]
    shape=(view.num_slices,8,8)
    try:
        with mock.patch('XTA.sam_resources.validate_live_sam_resource_profile',
                return_value={'base_non_cpu_allowance_bytes':2**30}), \
                mock.patch('XTA.backprojection._try_acquire_specific_main_process_gpu_stage',
                    side_effect=AssertionError('claimed GPU for unavailable source')):
            assert try_gpu_crop_renderer(context,view,shape,demand(shape,{1:(1,1,7,7)})) is None
        assert not lazy.materialized
    finally:
        _VOLUME_READINESS_BY_ARRAY_ID.pop(id(decoded),None)
        context.close();lazy.close();decoded._mmap.close()


def test_unmaterialized_streaming_gpu_refusal_preserves_endpoint_grid_and_corner_frames(tmp_path):
    from XTA.media import LazyProcessingCube,_linear_source_index
    import XTA.runtime as runtime
    decoded=np.memmap(tmp_path/'decoded.dat',mode='w+',dtype=np.uint8,shape=(3,4,5))
    decoded[0]=0;decoded[1]=84;decoded[2]=240;decoded.flush()
    lazy=LazyProcessingCube(decoded,(7,4,5),tmp_path/'working.dat',workers=1,
        request_path=tmp_path/'request',ready_path=tmp_path/'ready',failed_path=tmp_path/'failed',
        streaming_backend=True)
    context=context_for(tmp_path,lazy);context.detector_assets_retired()
    context._resource_local.profile=object()
    view=geometry.get_view_infos(*lazy.shape,cartesian_views=('transverse',))[0]
    planned=demand(lazy.shape,{frame:(0,0,4,5)for frame in (0,1,6)})
    telemetry=SimpleNamespace(add=mock.Mock(),gauge=mock.Mock())
    try:
        with mock.patch('XTA.sam_resources.validate_live_sam_resource_profile',
                return_value={'base_non_cpu_allowance_bytes':2**30,'lease_id':'endpoint-grid'}), \
                mock.patch.object(runtime,'runtime_telemetry',return_value=telemetry), \
                mock.patch('XTA.backprojection._try_acquire_specific_main_process_gpu_stage',
                    side_effect=AssertionError('claimed GPU with another temporal grid')):
            assert try_gpu_crop_renderer(context,view,lazy.shape,planned) is None
            reference=context.image_provider(view,lazy.shape,planned)
        telemetry.gauge.assert_any_call('sam.gpu_images.last_cpu_reason',
            'streamed_endpoint_processing_cube_not_materialized')
        endpoint=_linear_source_index(1,7,3)
        center=(1+.5)*(3/7)-.5
        assert abs(endpoint-center)>.15  # A coordinate change, not a gray8 tie.
        np.testing.assert_array_equal(crop_from(reference,0,(0,0,4,5)),0)
        np.testing.assert_array_equal(crop_from(reference,1,(0,0,4,5)),28)
        np.testing.assert_array_equal(crop_from(reference,6,(0,0,4,5)),240)
        assert live_image_sampling(reference)['backend']=='cpu'
        assert not lazy.materialized
    finally:context.close();lazy.close();decoded._mmap.close()

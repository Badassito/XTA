"""GPU transaction routing, pixel provenance and retirement without CUDA."""
from dataclasses import replace
from contextlib import nullcontext
import os
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
                mock.patch('XTA.runtime.runtime_telemetry', return_value=telemetry), \
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
                mock.patch.object(torch.cuda,'is_available',return_value=True), \
                mock.patch('XTA.backprojection._try_acquire_specific_main_process_gpu_stage',
                    side_effect=AssertionError('claimed GPU before winning CPU/image ticket')):
            renderer=try_gpu_crop_renderer(context,view,shape,planned)
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

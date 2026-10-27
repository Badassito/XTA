"""Native-T GPU uploads keep PTA's logical masks and CPU fallback arrays."""
from contextlib import nullcontext
from dataclasses import replace
import json
import os
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import cuda_backend, pta_workers
from tests.test_pta_native_u8_cache import engine_fixture


@pytest.mark.parametrize('tilted', (False, True))
def test_native_t_azimuthal_requires_reference_pointer_and_preserves_cpu_retry(monkeypatch, tilted):
    from XTA import geometry
    from tests.test_pta_fused_curved_fallback import _views
    view = _views()[0] if tilted else next(v for v in geometry.get_view_infos(
        11,13,15,cartesian_views=(),azimuthal_views=('transverse',),
        azimuthal_azimuth_angles=(37.,)) if geometry.is_azimuthal_view(v))
    engine = object.__new__(cuda_backend._GpuWorkerRenderEngine)
    engine._pta_native_t_exact = True
    engine._render_tilted_azimuthal_native_linear_resident = mock.Mock(return_value='pointer')
    engine._render_azimuthal_native_texture = mock.Mock(side_effect=AssertionError('texture'))
    engine._render_azimuthal_native_resident_torch = mock.Mock(side_effect=AssertionError('native-T Torch'))
    assert engine._render_azimuthal_native_resident(view, 0) == 'pointer'
    engine._render_tilted_azimuthal_native_linear_resident.side_effect = RuntimeError('kernel unavailable')
    with pytest.raises(RuntimeError, match='kernel unavailable'):
        engine._render_azimuthal_native_resident(view, 0)
    engine._render_azimuthal_native_texture.assert_not_called()
    engine._render_azimuthal_native_resident_torch.assert_not_called()


def test_native_payload_requires_opt_in_shared_source_and_t_only_upscale(monkeypatch):
    native = np.zeros((3, 5, 7), np.uint8)
    prep = SimpleNamespace(src=SimpleNamespace(volume=native, volume_block=SimpleNamespace(name='native')),
                           volume_for_render=np.zeros((6, 5, 7), np.uint8))
    monkeypatch.delenv('YOLO_TTA_PTA_GPU_NATIVE_T', raising=False)
    assert not pta_workers._native_intensity_payload(prep)
    monkeypatch.setenv('YOLO_TTA_PTA_GPU_NATIVE_T', '1')
    assert pta_workers._native_intensity_payload(prep) == {
        'native_intensity_shm': 'native', 'native_intensity_shape': (3, 5, 7)}
    for source in (np.zeros((6, 5, 7), np.uint8), np.zeros((3, 6, 7), np.uint8),
                   native.astype(np.float32), native[:, :, ::-1]):
        prep.src.volume = source
        assert not pta_workers._native_intensity_payload(prep)
    prep.src.volume = native
    prep.src.volume_block = None
    assert not pta_workers._native_intensity_payload(prep)


def test_array_residency_key_includes_logical_depth_and_rejects_invalid_depth(monkeypatch):
    engine = engine_fixture()
    volume = np.zeros((3, 5, 7), np.uint8)
    monkeypatch.setattr(cuda_backend, 'gpu_worker_render_resident_enabled', lambda: False)
    assert engine.ensure_volume_array(volume, identity='same', resize_to_t=6) == 'stream'
    assert engine._logical_t == 6
    assert engine._volume_key == ('same', (3, 5, 7), 6)
    engine.ensure_volume_array(volume, identity='same', resize_to_t=9)
    assert engine._logical_t == 9
    engine._stream.synchronize.assert_called_once()
    with pytest.raises(ValueError, match='logical depth'):
        engine.ensure_volume_array(volume, resize_to_t=0)


def test_worker_native_attachment_preserves_logical_arrays_and_retires_generations(monkeypatch):
    sources = {}
    closed = []
    for gen in (1, 2, 3):
        for kind, size in (('volume', 6*5*7), ('mask', 6*5*7), ('native', 3*5*7)):
            name = f'{kind}-{gen}'
            sources[name] = SimpleNamespace(buf=bytearray([gen] * size), close=lambda n=name: closed.append(n))
    monkeypatch.setattr(pta_workers, '_attach_shm_untracked', sources.__getitem__)
    monkeypatch.setattr(pta_workers, '_WORKER_GEN_CACHE', {})
    monkeypatch.setattr(pta_workers, '_WORKER_VOLUME_IDENTITY_BY_POINTER', {})
    monkeypatch.setattr(pta_workers, '_WORKER_NATIVE_INTENSITY_BY_POINTER', {})
    pointers = []
    for gen in (1, 2, 3):
        payload = dict(gen=gen, volume_shm=f'volume-{gen}', mask_shm=f'mask-{gen}',
            volume_shape=(6,5,7), mask_shape=(6,5,7), native_intensity_shm=f'native-{gen}',
            native_intensity_shape=(3,5,7))
        volume, mask, coverage = pta_workers._worker_volume_arrays(payload)
        assert volume.shape == mask.shape == (6,5,7)
        assert coverage is None
        pointer = int(volume.__array_interface__['data'][0])
        pointers.append(pointer)
        native, logical_t = pta_workers._WORKER_NATIVE_INTENSITY_BY_POINTER[pointer]
        assert native.shape == (3,5,7) and logical_t == 6
        assert np.all(volume == gen) and np.all(mask == gen) and np.all(native == gen)
    assert pointers[0] not in pta_workers._WORKER_NATIVE_INTENSITY_BY_POINTER
    assert set(closed) == {'volume-1', 'mask-1', 'native-1'}
    assert len(pta_workers._WORKER_GEN_CACHE) == 2


def test_projected_image_uses_native_upload_but_keeps_logical_source_for_fallback(monkeypatch):
    torch = pytest.importorskip('torch')
    logical = np.zeros((6,5,7), np.uint8)
    native = np.zeros((3,5,7), np.uint8)
    monkeypatch.setitem(pta_workers._WORKER_NATIVE_INTENSITY_BY_POINTER,
                        int(logical.__array_interface__['data'][0]), (native, 6))
    projected = torch.zeros((8,8), dtype=torch.uint8)
    renderer = SimpleNamespace(ensure_volume_array=mock.Mock(return_value='resident'),
        render_cartesian_grid_resident=mock.Mock(return_value=projected), _stream=object())
    runtime = dict(torch=SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext(),
        Event=lambda: SimpleNamespace(record=lambda _: None))),
        azimuthal_renderer=renderer, azimuthal_render_lock=threading.Lock())
    view = SimpleNamespace(name='transverse', family='transverse', physical_view_name='transverse')
    matrix = np.asarray(((1,0,0),(0,1,0)), np.float32)
    plan = SimpleNamespace(view=SimpleNamespace(shared_view=view),
        channel_variant=SimpleNamespace(kind='gray'), source_encoded_indices=(),
        aff=SimpleNamespace(out_h=8,out_w=8,M_out_to_src=matrix), tile_layout=())
    monkeypatch.setattr(pta_workers, '_require_pta_canonical_plan', lambda *a, **k: None)
    telemetry = SimpleNamespace(span=lambda _: nullcontext(), gauge=mock.Mock())
    monkeypatch.setattr(pta_workers, 'runtime_telemetry', lambda: telemetry)
    assert pta_workers._gpu_projected_item_image(runtime, logical, plan, 0, 'full')[0] is projected
    args, kwargs = renderer.ensure_volume_array.call_args
    assert args[0] is native and kwargs['resize_to_t'] == 6
    assert pta_workers._gpu_projected_item_image(runtime, logical, plan, 0, 'full')[0] is projected
    telemetry.gauge.assert_called_once()
    assert telemetry.gauge.call_args.args[0] == 'pta.render.source_residency'
    assert telemetry.gauge.call_args.args[1]['saved_resident_bytes'] == 3*5*7
    assert telemetry.gauge.call_args.args[1]['native_t_active'] is True
    renderer.ensure_volume_array.return_value = 'stream'
    assert pta_workers._gpu_projected_item_image(runtime, logical, plan, 0, 'full') is None
    assert telemetry.gauge.call_args.args[1]['uploaded_bytes'] == 0
    assert logical.shape == (6,5,7)


@pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_NATIVE_T_CUDA') != '1',
                    reason='explicit CUDA qualification only')
@pytest.mark.parametrize('channels', ('gray', 'custom'))
def test_cuda_native_t_all_families_full_tiles_and_channels(tmp_path, monkeypatch, channels):
    owner = json.loads(Path(os.environ['XTA_GPU_LOCK_PATH']).read_text())
    assert owner['pid'] == os.getpid() and owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    import torch
    from XTA import media, pta, pta_cuda_masks
    from tools.qualify_pta_gpu_masks import build_cases, make_volumes

    monkeypatch.setenv('YOLO_TTA_GPU_RENDER_RESERVE_GIB', '1')
    monkeypatch.setenv('YOLO_TTA_FAST_GEOMETRY', '0')
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_FP32', '0')
    monkeypatch.setenv('YOLO_TTA_GPU_NATIVE_PLANE_CACHE', '2')
    native = make_volumes(24)[0][:13].copy()
    logical = media.resize_volume_to_processing_cube_gray8(native, (24,24,24),
        tmp_path/'logical.uint8', workers=2, prefer_memory=True, reserve_bytes=0)
    mask, coverage = make_volumes(24)[1:]
    runtimes = []
    receipts = []
    for label in ('logical', 'native'):
        renderer = cuda_backend._GpuWorkerRenderEngine('cuda:0')
        source = native if label == 'native' else logical
        extra = {'resize_to_t': 24} if label == 'native' else {}
        assert renderer.ensure_volume_array(source, identity=f'test-{label}',
            require_azimuthal_texture=False, **extra) == 'resident'
        runtimes.append(dict(torch=torch, device_id=0, azimuthal_renderer=renderer,
            azimuthal_render_lock=threading.RLock(), azimuthal_texture_required=False))
        receipts.append(dict(mode=label, source_bytes=renderer._volume_gpu.numel(),
            logical_shape=(24,24,24), source_shape=tuple(renderer._volume_gpu.shape),
            timings=renderer._source_residency_timings))
    texture_renderer = cuda_backend._GpuWorkerRenderEngine('cuda:0')
    texture_runtime = dict(torch=torch, device_id=0, azimuthal_renderer=texture_renderer,
        azimuthal_render_lock=threading.RLock(), azimuthal_texture_required=True)
    pointer = int(logical.__array_interface__['data'][0])
    for family, plan, frame, item in build_cases(tmp_path, 24, 32, 16):
        if channels == 'custom':
            plan = replace(plan, channel_variant=pta.ChannelVariant(
                'custom', 'custom', 3, 3, False, (-1,0,1)))
        images, masks, knowns = [], [], []
        for index, runtime in enumerate(runtimes):
            if index:
                monkeypatch.setitem(pta_workers._WORKER_NATIVE_INTENSITY_BY_POINTER, pointer, (native,24))
            else:
                monkeypatch.delitem(pta_workers._WORKER_NATIVE_INTENSITY_BY_POINTER, pointer, raising=False)
                # Match F3's reference reconstruction; texture composition is
                # measured separately because it omits intermediate gray8 rounding.
                runtime['azimuthal_renderer']._azimuthal_texture_admitted = False
            result = pta_workers._gpu_projected_item_image(runtime, logical, plan, frame, item)
            assert result is not None, (family, channels, item)
            image, ready = result
            ready.synchronize()
            images.append(image.cpu().numpy())
            categorical = pta_cuda_masks.render_gpu_categorical_item(runtime, mask, coverage,
                plan, frame, item, identity='same-logical-mask')
            assert categorical is not None
            projected_mask, projected_known, ready = categorical
            ready.synchronize()
            masks.append(projected_mask.cpu().numpy())
            knowns.append(projected_known.cpu().numpy())
        assert images[0].shape == images[1].shape
        maximum = int(np.abs(images[0].astype(np.int16)-images[1].astype(np.int16)).max())
        assert maximum <= 2, (family, channels, item, maximum)
        np.testing.assert_array_equal(masks[0], masks[1])
        np.testing.assert_array_equal(knowns[0], knowns[1])
        comparison = dict(family=family, channels=channels, item=item,
                          shape=images[0].shape, maximum_gray8_difference=maximum)
        if 'azimuthal' in family:
            monkeypatch.delitem(pta_workers._WORKER_NATIVE_INTENSITY_BY_POINTER, pointer, raising=False)
            texture_image, ready = pta_workers._gpu_projected_item_image(
                texture_runtime, logical, plan, frame, item)
            ready.synchronize()
            texture_delta = np.abs(texture_image.cpu().numpy().astype(np.int16)-images[0].astype(np.int16))
            comparison['logical_texture_reference_maximum_difference'] = int(texture_delta.max())
            comparison['logical_texture_reference_mean_difference'] = float(texture_delta.mean())
        assert runtimes[1]['azimuthal_renderer']._azimuthal_texture_ref is None
        receipts.append(comparison)
    assert receipts[0]['source_bytes']-receipts[1]['source_bytes'] == (24-13)*24*24
    (tmp_path/f'native-t-{channels}-receipt.json').write_text(json.dumps(receipts, indent=2)+'\n')
    for runtime in (*runtimes, texture_runtime):
        pta_cuda_masks.retire_gpu_categorical_volume(runtime)
        runtime['azimuthal_renderer'].release_inference_assets()

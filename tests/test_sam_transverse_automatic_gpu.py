"""Opt-in real GPU smoke of the restored SAM Transverse assembly route."""
import hashlib
import json
import os
from pathlib import Path
import threading

import numpy as np
import pytest

from XTA import assembly, backprojection, cylindrical_projection as radial, spherical_projection
from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT, write_raw_bbox_mask_store
from tests import test_sam_sparse_shell_gpu_parity as parity


pytestmark = pytest.mark.skipif(os.environ.get('XTA_TEST_SAM_TRANSVERSE_GPU') != '1',
    reason='requires explicitly authorized real GPU smoke and an owned GPU lock')
VIEWS = tuple(view for view in parity.VIEWS
              if view.family == 'radial' and view.radial_base_view == 'transverse')
OTHER_VIEWS = tuple(next(view for view in parity.CASES
    if view.family == 'radial' and view.radial_base_view == base and view.radial_tilted_source)
    for base in ('sagittal', 'coronal')) + tuple(view for view in parity.CASES
    if view.family == 'spherical' and view.spherical_tilted_source)[:2]


@pytest.fixture(autouse=True)
def gpu_owner(monkeypatch):
    owner = json.loads(Path(os.environ['XTA_GPU_LOCK_PATH']).read_text())
    assert owner['pid'] == os.getpid()
    assert owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    import torch
    assert torch.cuda.is_available(), 'Automatic GPU smoke cannot silently use CPU'
    for flag in ('YOLO_TTA_GPU_BACKPROJECT', 'YOLO_TTA_GPU_RADIAL_BACKPROJECT',
                 'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT'):
        monkeypatch.delenv(flag, raising=False)
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_inference_priority_active(False)
    monkeypatch.setattr(backprojection, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    def unexpected(*args, **kwargs):
        pytest.fail('Transverse source backing entered the unqualified encoded adapter')
    monkeypatch.setattr('XTA.sam_sparse_shell_projection.encoded_shell_has_unused_canvas', unexpected)
    monkeypatch.setattr('XTA.sam_sparse_shell_projection.project_shell_sparse_store', unexpected)
    old_shape = assembly.final_source_output_shape()
    yield coordinator
    assembly.set_final_source_output_shape(old_shape)
    assert not coordinator.snapshot()['stage_leases']


def backing(source, view, work, shape):
    work.mkdir()
    assembly.set_final_source_output_shape(shape)
    result = assembly._sam_directional_source_backing(source, view=view,
        stage='sam_selected_forward', has_foreground=True, model_name='automatic-gpu-smoke',
        source='fullframe', pass_index=0, tile_config_id='', workers=3, _work_root=work)
    assert result['coordinate_space'] == 'source_grid'
    assert tuple(result['shape']) == shape
    return parity.dense_store(result['path'])


def report(row):
    with Path(os.environ['XTA_GPU_PARITY_ROWS']).open('a') as stream:
        stream.write(json.dumps(row) + '\n')


@pytest.mark.parametrize('view', VIEWS, ids=lambda view: view.name)
@pytest.mark.parametrize('route,shape', (
    ('automatic', parity.SOURCE_SHAPE), ('explicit_cpu', parity.SOURCE_SHAPE),
    ('automatic', (9, 11, 13))), ids=('automatic-native', 'cpu-native', 'automatic-odd-resized'))
def test_full_sam_backing_automatic_gpu_and_explicit_cpu(tmp_path, monkeypatch, view, shape, route):
    data = parity.source_for(view)
    source = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, source, format_name=INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    calls = []
    original = radial._try_radial_cuda_stage
    def admit(*args, **kwargs):
        stage = original(*args, **kwargs)
        if stage is not None:
            encoded = stage.project_encoded
            def project(first, count, *args, **kwargs):
                block = encoded(first, count, *args, **kwargs)
                calls.append(dict(first=first, count=count, device=stage.device_index))
                return block
            stage.project_encoded = project
        return stage
    monkeypatch.setattr(radial, '_try_radial_cuda_stage', admit)
    if route == 'explicit_cpu':
        monkeypatch.setenv('YOLO_TTA_GPU_RADIAL_BACKPROJECT', '0')
        from XTA import cylindrical_cuda_projection
        monkeypatch.setattr(cylindrical_cuda_projection, 'RadialCudaProjector',
            lambda *args, **kwargs: pytest.fail('Explicit CPU flag constructed a GPU projector'))
    else:
        assert 'YOLO_TTA_GPU_RADIAL_BACKPROJECT' not in os.environ
        assert 'YOLO_TTA_GPU_BACKPROJECT' not in os.environ
    actual = backing(source, view, tmp_path / 'actual', shape)
    if route == 'automatic':
        assert calls, 'Default available-GPU route did not execute a CUDA block'
        assert sum(row['count'] for row in calls) == shape[0]
    else:
        assert not calls
    monkeypatch.setenv('YOLO_TTA_GPU_RADIAL_BACKPROJECT', '0')
    expected = backing(source, view, tmp_path / 'cpu-reference', shape)
    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero(actual) > 0
    np.testing.assert_array_equal(parity.dense_store(source), data)
    report(dict(test='full_sam_backing', view=view.name, route=route, shape=shape,
        tilted=bool(view.radial_tilted_source), tilt_degrees=view.tilt_angle_deg,
        tilt_direction=view.tilt_direction,
        exact=True, changed_voxels=0, foreground=int(np.count_nonzero(actual)),
        decoded_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), gpu_encoded_calls=calls,
        flags_default_unset=route == 'automatic', source_unchanged=True))


@pytest.mark.parametrize('view', VIEWS, ids=lambda view: view.name)
def test_full_sam_backing_compiled_cpu_prefix_promotes_to_real_gpu(tmp_path, monkeypatch, view):
    shape = parity.SOURCE_SHAPE
    data = parity.source_for(view)
    source = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, source, format_name=INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    calls, transitions = [], []
    readers, readers_lock = [0], threading.Lock()
    original_cpu = radial._project_radial_block
    def cpu(*args):
        if not args[-3]:
            return original_cpu(*args)
        with readers_lock:
            readers[0] += 1
        try:
            return original_cpu(*args)
        finally:
            with readers_lock:
                readers[0] -= 1
    monkeypatch.setattr(radial, '_project_radial_block', cpu)
    original_ordered = radial._ordered_radial_blocks
    def ordered(*args, **kwargs):
        stream = original_ordered(*args, **kwargs)
        try:
            yield from stream
        finally:
            stream.close()
            transitions.append('cpu_readers_joined')
    monkeypatch.setattr(radial, '_ordered_radial_blocks', ordered)
    original_admit = radial._try_radial_cuda_stage
    attempts = [0]
    def admit(*args, **kwargs):
        attempts[0] += 1
        if attempts[0] <= 2:
            kwargs['retry_state']['retryable'] = True
            return None  # Deterministic unavailable admission, no fake GPU mathematics.
        stage = original_admit(*args, **kwargs)
        assert stage is not None, 'Real CUDA admission unavailable after the CPU prefix'
        encoded = stage.project_encoded
        def project(first, count, *args, **kwargs):
            assert readers[0] == 0 and 'cpu_readers_joined' in transitions
            block = encoded(first, count, *args, **kwargs)
            calls.append(dict(first=first, count=count, device=stage.device_index))
            return block
        stage.project_encoded = project
        return stage
    monkeypatch.setattr(radial, '_try_radial_cuda_stage', admit)
    monkeypatch.setattr(radial, '_radial_block_schedule', lambda *args, **kwargs: (1, 3))
    monkeypatch.setattr(radial, '_CUDA_RECHECK_SLICES', 1)
    actual = backing(source, view, tmp_path / 'promoted', shape)
    assert attempts[0] == 3 and calls[0]['first'] == 1
    assert sum(row['count'] for row in calls) == shape[0] - 1
    assert readers[0] == 0
    monkeypatch.setattr(radial, '_try_radial_cuda_stage', original_admit)
    monkeypatch.setenv('YOLO_TTA_GPU_RADIAL_BACKPROJECT', '0')
    expected = backing(source, view, tmp_path / 'cpu-reference', shape)
    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero(actual) > 0
    np.testing.assert_array_equal(parity.dense_store(source), data)
    report(dict(test='compiled_cpu_prefix_real_cuda_suffix', view=view.name,
        tilted=bool(view.radial_tilted_source), tilt_degrees=view.tilt_angle_deg,
        tilt_direction=view.tilt_direction,
        exact=True, changed_voxels=0, cpu_planes=1, gpu_planes=shape[0] - 1,
        gpu_encoded_calls=calls, admission_attempts=3, cpu_readers_joined=True,
        source_unchanged=True, limitation='Initial busy admissions and block schedule are controlled; GPU mathematics is real.'))


@pytest.mark.parametrize('view', OTHER_VIEWS, ids=lambda view: view.name)
def test_other_qualified_legacy_automatic_routes(tmp_path, monkeypatch, view):
    data = parity.source_for(view)
    source = tmp_path / 'source.cvol'
    write_raw_bbox_mask_store(data, source, format_name=INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    module = radial if view.family == 'radial' else spherical_projection
    name = '_try_radial_cuda_stage' if view.family == 'radial' else '_try_spherical_cuda_stage'
    calls = []
    original = getattr(module, name)
    def admit(*args, **kwargs):
        stage = original(*args, **kwargs)
        assert stage is not None, 'Available-GPU automatic legacy route silently fell back to CPU'
        encoded = stage.project_encoded
        def project(first, count, *args, **kwargs):
            block = encoded(first, count, *args, **kwargs)
            calls.append(dict(first=first, count=count, device=stage.device_index))
            return block
        stage.project_encoded = project
        return stage
    monkeypatch.setattr(module, name, admit)
    flag = 'YOLO_TTA_GPU_' + view.family.upper() + '_BACKPROJECT'
    assert flag not in os.environ
    actual = backing(source, view, tmp_path / 'automatic', parity.SOURCE_SHAPE)
    assert calls and sum(row['count'] for row in calls) == parity.SOURCE_SHAPE[0]
    monkeypatch.setattr(module, name, original)
    monkeypatch.setenv(flag, '0')
    expected = backing(source, view, tmp_path / 'cpu-reference', parity.SOURCE_SHAPE)
    np.testing.assert_array_equal(actual, expected)
    assert np.count_nonzero(actual) > 0
    np.testing.assert_array_equal(parity.dense_store(source), data)
    report(dict(test='additional_legacy_automatic', view=view.name, family=view.family,
        tilt_degrees=view.tilt_angle_deg, tilt_direction=view.tilt_direction,
        exact=True, changed_voxels=0, foreground=int(np.count_nonzero(actual)),
        gpu_encoded_calls=calls, flags_default_unset=True, source_unchanged=True))

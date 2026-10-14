"""Opt-in real GPU comparison; requires this process to own the shared GPU lock."""
import contextlib
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np
import pytest

from XTA import assembly, backprojection, cylindrical_projection, spherical_projection
from XTA.interpolation import (
    INTERNAL_PACKED_CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore,
    write_raw_bbox_mask_store)
from XTA.sam_sparse_shell_projection import project_shell_sparse_store
from tests.test_sam_view_orientations import SOURCE_SHAPE, VIEWS


pytestmark = pytest.mark.skipif(os.environ.get('XTA_TEST_SAM_SPARSE_GPU') != '1',
    reason='requires explicit available-GPU qualification and an owned GPU lock')


def representative_views():
    result = []
    for base in ('transverse', 'sagittal', 'coronal'):
        result.append(next(v for v in VIEWS if v.family == 'radial'
            and v.radial_base_view == base and not v.radial_tilted_source))
        result.append(next(v for v in VIEWS if v.family == 'radial'
            and v.radial_base_view == base and v.radial_tilted_source
            and v.tilt_direction == 'vertical' and v.tilt_angle_deg > 0))
    for face in range(6):
        result.append(next(v for v in VIEWS if v.family == 'spherical'
            and not v.spherical_tilted_source and v.spherical_face == face))
    for face, direction, positive in ((0, 'vertical', True), (2, 'horizontal', False), (5, 'vertical', False)):
        result.append(next(v for v in VIEWS if v.family == 'spherical'
            and v.spherical_tilted_source and v.spherical_face == face
            and v.tilt_direction == direction and (v.tilt_angle_deg > 0) == positive))
    return tuple(result)


CASES = representative_views()


@pytest.fixture(autouse=True)
def real_gpu_owner(monkeypatch):
    lock = Path(os.environ['XTA_GPU_LOCK_PATH'])
    owner = json.loads(lock.read_text())
    assert owner['pid'] == os.getpid() and owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    import torch
    assert torch.cuda.is_available(), 'Real GPU qualification cannot silently use CPU'
    for key in ('YOLO_TTA_GPU_BACKPROJECT', 'YOLO_TTA_GPU_RADIAL_BACKPROJECT', 'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT'):
        monkeypatch.setenv(key, '1')
    coordinator = backprojection._MainProcessGpuStageCoordinator()
    coordinator.configure_workers([0])
    coordinator.set_inference_priority_active(False)
    monkeypatch.setattr(backprojection, '_MAIN_PROCESS_GPU_STAGE_COORDINATOR', coordinator)
    yield
    assert not coordinator.snapshot()['stage_leases']


def dense_store(path):
    with contextlib.closing(RawBBoxMaskStore.open(path, mmap_payload=True)) as store:
        return np.stack([store.decode_slice(frame) for frame in range(store.shape[0])])


def source_for(view):
    seed = int(hashlib.sha256(view.name.encode()).hexdigest()[:8], 16)
    data = (np.random.default_rng(seed).random((view.num_slices, view.src_h, view.src_w)) < .23).astype(np.uint8)
    data[:, 0, 0] = data[:, -1, -1] = 1
    return data


def require_gpu_stage(monkeypatch, view):
    module = cylindrical_projection if view.family == 'radial' else spherical_projection
    name = '_try_radial_cuda_stage' if view.family == 'radial' else '_try_spherical_cuda_stage'
    original = getattr(module, name)
    calls = []
    def acquire(*args, **kwargs):
        stage = original(*args, **kwargs)
        assert stage is not None, 'Explicit GPU stage unavailable; this is not a GPU parity pass'
        encoded = stage.project_encoded
        def project(*args, **kwargs):
            result = encoded(*args, **kwargs)
            calls.append(dict(first=int(args[0]), count=int(args[1]), device=stage.device_index))
            return result
        stage.project_encoded = project
        return stage
    monkeypatch.setattr(module, name, acquire)
    return calls


@pytest.mark.parametrize('view', CASES, ids=lambda v:v.name)
@pytest.mark.parametrize('shape', (SOURCE_SHAPE, (9, 11, 13)), ids=('native', 'odd-resized'))
def test_encoded_cpu_matches_explicit_gpu_decoded_output(tmp_path, monkeypatch, view, shape):
    data = source_for(view)
    source = tmp_path/'input.cvol'
    write_raw_bbox_mask_store(data, source, format_name=INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    before = hashlib.sha256(data.tobytes()).hexdigest()
    started = time.perf_counter()
    cpu = project_shell_sparse_store(source, view, tmp_path/'cpu.cvol', out_shape_tyx=shape, workers=1)
    cpu_seconds = time.perf_counter()-started
    calls = require_gpu_stage(monkeypatch, view)
    writer = IncrementalRawBBoxMaskStoreWriter(shape=shape, store_dir=tmp_path/'gpu.cvol',
        format_name=INTERNAL_PACKED_CVOL_FORMAT, desc='SAM encoded CPU versus explicit GPU')
    project = (cylindrical_projection.backproject_radial_volume_to_volume if view.family == 'radial'
               else spherical_projection.backproject_spherical_volume_to_volume)
    started = time.perf_counter()
    try:
        project(data, view, tmp_path/'unused.dat', 'SAM sparse GPU parity', out_shape_tyx=shape,
            workers=1, sink_only=True, projection_block_callback=writer)
        writer.finalize()
    except BaseException as error:
        writer.abort(error)
        writer.discard()
        raise
    gpu_seconds = time.perf_counter()-started
    assert calls and sum(row['count'] for row in calls) == shape[0]
    actual, expected = dense_store(tmp_path/'cpu.cvol'), dense_store(tmp_path/'gpu.cvol')
    np.testing.assert_array_equal(actual, expected)
    assert hashlib.sha256(data.tobytes()).hexdigest() == before
    row = dict(view=view.name, family=view.family, output_shape=shape,
        exact=True, changed_voxels=0, foreground=int(np.count_nonzero(actual)),
        decoded_sha256=hashlib.sha256(actual.tobytes()).hexdigest(), gpu_encoded_calls=calls,
        cpu_backend=cpu['backend'], cpu_native_decode_bytes=cpu['native_decode_bytes'],
        cpu_seconds=cpu_seconds, gpu_seconds=gpu_seconds,
        timing_limit='Small local sanity case; initialization/JIT may be included, not cluster throughput.')
    with Path(os.environ['XTA_GPU_PARITY_ROWS']).open('a') as report:
        report.write(json.dumps(row)+'\n')


@pytest.mark.parametrize('family', ['radial', 'spherical'])
def test_explicit_gpu_sam_publication_does_not_enter_new_cpu_adapter(tmp_path, monkeypatch, family):
    view = next(v for v in CASES if v.family == family)
    source = tmp_path/'input.cvol'
    data = source_for(view)
    write_raw_bbox_mask_store(data, source, format_name=INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    monkeypatch.setattr('XTA.sam_sparse_shell_projection.project_shell_sparse_store',
        lambda *args, **kwargs: pytest.fail('Explicit GPU route entered CPU encoded adapter'))
    calls = require_gpu_stage(monkeypatch, view)
    work = tmp_path/'work'
    work.mkdir()
    old_shape = assembly.final_source_output_shape()
    assembly.set_final_source_output_shape(SOURCE_SHAPE)
    try:
        backing = assembly._sam_directional_source_backing(source, view=view, stage='sam_selected_forward',
            has_foreground=True, model_name='gpu-parity', source='fullframe', pass_index=1,
            tile_config_id='', workers=1, _work_root=work)
        assert calls and dense_store(backing['path']).shape == SOURCE_SHAPE
    finally:
        assembly.set_final_source_output_shape(old_shape)

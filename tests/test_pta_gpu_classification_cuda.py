"""Opt-in real-device count/classification parity; parent pool protocol is tested separately."""
from dataclasses import replace
import json
import os
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import pta, pta_cuda_masks
from XTA.pta_config import parse_pta_args
from XTA.pta_gpu_classification import GpuClassificationTask, classify_task, item_keys


pytestmark = pytest.mark.skipif(os.environ.get('XTA_TEST_PTA_CLASSIFICATION_GPU') != '1',
    reason='requires explicit GPU qualification and an owned GPU lock')


@pytest.fixture
def cuda_runtime():
    owner = json.loads(Path(os.environ['XTA_GPU_LOCK_PATH']).read_text())
    assert owner['pid'] == os.getpid() and owner['token'] == os.environ['XTA_GPU_LOCK_TOKEN']
    import torch
    assert torch.cuda.is_available()
    runtime = dict(torch=torch, device_id=0,
        azimuthal_renderer=SimpleNamespace(_stream=torch.cuda.Stream(device=0)))
    yield runtime
    pta_cuda_masks.retire_gpu_categorical_volume(runtime)


CASES = [
    ('transverse', ['--enable_cartesian', 'transverse'], 'transverse'),
    ('sagittal', ['--enable_cartesian', 'sagittal'], 'sagittal'),
    ('coronal', ['--enable_cartesian', 'coronal'], 'coronal'),
    ('tilted_transverse', ['--enable_tilted', 'transverse:30:vertical'], 'tilted_transverse'),
    ('tilted_sagittal', ['--enable_tilted', 'sagittal:30:horizontal'], 'tilted_sagittal'),
    ('tilted_coronal', ['--enable_tilted', 'coronal:30:vertical'], 'tilted_coronal'),
    ('azimuthal', ['--enable_azimuthal', 'transverse:90'], 'azimuthal'),
    ('azimuthal_tilted', ['--enable_tilted', 'sagittal:30:horizontal',
                         '--enable_azimuthal', 'tilted_sagittal:90'], 'azimuthal'),
    ('radial', ['--enable_radial', 'transverse'], 'radial'),
    ('radial_tilted', ['--enable_tilted', 'coronal:30:vertical',
                      '--enable_radial', 'tilted_coronal'], 'radial'),
    ('spherical', ['--enable_spherical', 'transverse'], 'spherical'),
    ('spherical_tilted', ['--enable_tilted', 'transverse:30:horizontal',
                         '--enable_spherical', 'tilted_transverse'], 'spherical'),
]


def make_plan(tmp_path, options, family, *, shape=(9, 11, 13), imgsz=17, tile_sizes=(5, 9)):
    config = parse_pta_args(['--input', 'dataset', '--imgsz', str(imgsz), *options])
    views, _ = pta.compile_v18_pta_views(t_dim=shape[0], h=shape[1], w=shape[2],
                                       config=config, azimuthal_native_raster=imgsz)
    if family.startswith('tilted_'):
        view = next(v for v in reversed(views)
                    if pta.shared_geometry.is_tilted_view(v.shared_view)
                    and v.shared_view.tilt_base_view == family.removeprefix('tilted_'))
    else:
        view = next(v for v in reversed(views) if v.family.startswith(family))
    aff = pta.build_affine(view.src_w, view.src_h, 0, view.pad_mode, imgsz,
                           shared_view=view.shared_view)
    return pta.build_render_plan(view=view, aff=aff, tag=view.name, out_dir=tmp_path,
        stem='sample', tile_configs=tuple(pta.TileConfig(size, size, f's{size}') for size in tile_sizes),
        save_overlay=False, imgsz=imgsz, label_enabled=True, publish_images=False, publish_labels=False)


def cpu_truth(mask, coverage, plan, *, semantic, frames, selected=None):
    prep = SimpleNamespace(label_enabled=True, semantic_foreground=semantic,
        semantic_coverage_for_render=coverage, mask_for_render=mask, plans=[plan],
        smoothing_stats=[], void_fill_stats=None,
        src=SimpleNamespace(stem='sample', label_source='nrrd', yolo_polygons_by_frame={}, labels_by_frame={}))
    candidates = [SimpleNamespace(parent_view_tag=plan.tag, frame_idx=index, item_key=key)
                  for index in frames for key in item_keys(plan, selected)]
    return pta.classify_original_foregrounds_for_volume(prep, workers=1, selected_candidates=candidates)


@pytest.mark.parametrize('case', CASES, ids=lambda c:c[0])
@pytest.mark.parametrize('semantic', [False, True])
def test_all_families_full_tiles_fraction_and_semantic_matches_cpu(tmp_path, cuda_runtime, case, semantic):
    _name, options, family = case
    plan = make_plan(tmp_path, options, family)
    rng = np.random.default_rng(118)
    mask = (rng.random((9, 11, 13)) < .27).astype(np.uint8)
    coverage = (rng.random(mask.shape) < .74).astype(np.uint8) if semantic else None
    frames = sorted({0, max(0, plan.view.num_slices - 1)})
    selected = ('full', *(tile.tile_tag for tile in plan.tile_layout[::2]))
    expected = cpu_truth(mask, coverage, plan, semantic=semantic, frames=frames, selected=selected)
    task = GpuClassificationTask(tuple((0, index, tuple(selected)) for index in frames), semantic)
    result, remaining, stats = classify_task(cuda_runtime, mask, [plan], task, identity='source', coverage=coverage)
    assert remaining == [] and result == expected
    assert stats['gpu_jobs'] == len(frames)
    owner = pta_cuda_masks._owner_for_runtime(cuda_runtime)
    pointer = owner.mask_gpu.data_ptr()
    assert owner.stream.query(), 'classifier returned before its CUDA reads settled'
    repeated, _, _ = classify_task(cuda_runtime, mask, [plan], task, identity='source', coverage=coverage)
    assert repeated == result and owner.mask_gpu.data_ptr() == pointer


@pytest.mark.parametrize('semantic', [False, True])
def test_tiny_masks_threshold_and_admission_fallback(tmp_path, cuda_runtime, monkeypatch, semantic):
    plan = make_plan(tmp_path, ['--enable_cartesian', 'transverse'], 'transverse',
                     shape=(4, 8, 8), imgsz=8, tile_sizes=())
    mask = np.zeros((4, 8, 8), np.uint8)
    mask[1, 2, 3] = 1
    mask[2, 1:4, 1:4] = 1
    mask[3, ::2, ::2] = 1
    expected = cpu_truth(mask, None, plan, semantic=semantic, frames=range(4))
    task = GpuClassificationTask(tuple((0, index, None) for index in range(4)), semantic)
    result, remaining, stats = classify_task(cuda_runtime, mask, [plan], task, identity='tiny')
    assert result == expected and not remaining
    assert stats['sparse_polygon_checks'] == (0 if semantic else 2)
    pta_cuda_masks.retire_gpu_categorical_volume(cuda_runtime)
    monkeypatch.setenv('PTA_GPU_CATEGORICAL_RESERVE_MIB', '999999')
    result, remaining, stats = classify_task(cuda_runtime, mask, [plan], task, identity='no-room')
    assert result == {} and remaining == [(0, index) for index in range(4)]
    assert stats['gpu_queries'] == 0 and stats['cpu_jobs'] == 4

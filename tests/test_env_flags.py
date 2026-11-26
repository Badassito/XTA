"""One lenient parser contract for every environment switch helper."""
from __future__ import annotations

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import env_flags, experimental_features, geometry_quality, workspace
from XTA.unification.sampling import forward_sampling_policy

# (module, helper, helper arguments, variable, default, helper returns the negation)
_SWITCH_HELPERS = (
    ('XTA.artifact_archive', '_lock_free_reads_enabled', (),
     'YOLO_TTA_ARCHIVE_LOCK_FREE_READS', True, False),
    ('XTA.cylindrical_owner', 'radial_owner_enabled', (), 'YOLO_TTA_RADIAL_OWNER', True, False),
    ('XTA.cylindrical_projection', 'radial_cuda_backproject_enabled', (),
     'YOLO_TTA_GPU_RADIAL_BACKPROJECT', True, False),
    ('XTA.lta_tracker_features', 'async_frame_io_enabled', (), 'YOLO_TTA_SAM_ASYNC_FRAME_IO', True, False),
    ('XTA.media', '_tail_frame_count_enabled', (), 'YOLO_TTA_FRAME_COUNT_FULL_SCAN', False, True),
    ('XTA.pta_workers', '_owner_option_enabled', ('YOLO_TTA_PTA_GPU_OWNER_PREWARM',),
     'YOLO_TTA_PTA_GPU_OWNER_PREWARM', True, False),
    ('XTA.sam_checkpoint_digest', 'prefetch_enabled', (),
     'YOLO_TTA_SAM_CHECKPOINT_DIGEST_PREFETCH', False, False),
    ('XTA.sam_parent_staging', '_cheap_overflow_shares_turns', (),
     'YOLO_TTA_SAM_PARENT_CHEAP_OVERFLOW_LAST', False, True),
    ('XTA.sam_tracker_runtime', '_largest_view_first_enabled', (),
     'YOLO_TTA_SAM_LARGEST_VIEW_FIRST', True, False),
    ('XTA.spherical_projection', 'spherical_cuda_backproject_enabled', (),
     'YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', True, False),
    ('XTA.sweep_native_projection', '_tilted_row_spans_enabled', (),
     'YOLO_TTA_SWEEP_TILTED_ROW_SPANS', True, False),
)
# Spellings that every reader of one variable must parse the same way.
_DIVERGENT_SPELLINGS = (('disabled', False), ('00', False), ('-0', False), ('off', False), ('1', True))


@pytest.mark.parametrize('default', (False, True))
@pytest.mark.parametrize('raw, expected', (
    ('', False), ('0', False), ('00', False), ('-0', False), ('false', False),
    (' OFF ', False), ('no', False), ('disabled', False), ('Disabled', False),
    ('1', True), ('2', True), ('yes', True), ('on', True), ('ture', True),
))
def test_set_values_parse_the_same_for_either_default(raw, expected, default):
    assert env_flags.env_flag('EXAMPLE_SWITCH', default, environ={'EXAMPLE_SWITCH': raw}) is expected


@pytest.mark.parametrize('default', (False, True))
def test_unset_value_returns_the_default(default):
    assert env_flags.env_flag('EXAMPLE_SWITCH', default, environ={}) is default


def test_numeric_parsers_fall_back_on_empty_or_invalid_values():
    assert env_flags.env_int('N', 7, environ={'N': ' 12 '}) == 12
    assert env_flags.env_int('N', 7, environ={'N': ''}) == 7
    assert env_flags.env_int('N', 7, environ={'N': '4.0'}) == 7
    assert env_flags.env_float('F', 1.5, environ={'F': '0.25'}) == 0.25
    assert env_flags.env_float('F', 1.5, environ={'F': 'x'}) == 1.5


def test_dependency_light_modules_share_one_parser():
    assert workspace._env_flag is env_flags.env_flag
    assert workspace._env_int is env_flags.env_int
    assert workspace._env_float is env_flags.env_float
    assert geometry_quality._flag is env_flags.env_flag
    assert experimental_features._env_flag is env_flags.env_flag


def test_disabled_turns_fast_geometry_off(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_GPU_SPHERICAL_FP32', raising=False)
    monkeypatch.setenv('YOLO_TTA_FAST_GEOMETRY', 'disabled')
    assert geometry_quality.fast_geometry_enabled() is False
    assert forward_sampling_policy().policy_version == 23


@pytest.mark.parametrize('module_name, helper_name, args, variable, default, negated', _SWITCH_HELPERS)
@pytest.mark.parametrize('raw', (None, '', '0', '00', '-0', 'false', 'disabled', '1', 'true', 'ture'))
def test_switch_helpers_use_the_shared_parser(
        monkeypatch, module_name, helper_name, args, variable, default, negated, raw):
    helper = getattr(importlib.import_module(module_name), helper_name)
    if raw is None:
        monkeypatch.delenv(variable, raising=False)
        expected = default
    else:
        monkeypatch.setenv(variable, raw)
        expected = env_flags.env_flag(variable, default, environ={variable: raw})
    assert helper(*args) is (not expected if negated else expected)


@pytest.mark.parametrize('raw, expected', _DIVERGENT_SPELLINGS)
def test_spherical_native_kernel_readers_agree(monkeypatch, raw, expected):
    from XTA import cuda_backend, spherical_cuda
    monkeypatch.setenv('YOLO_TTA_GPU_SPHERICAL_NATIVE_KERNEL', raw)
    monkeypatch.setattr(cuda_backend, 'is_spherical_view', lambda view: True)
    monkeypatch.setattr(cuda_backend, 'spherical_fp32_requested', lambda: False)
    monkeypatch.setattr(spherical_cuda, 'spherical_fp32_requested', lambda: False)
    monkeypatch.setattr(spherical_cuda, '_render_spherical_native_cuda', lambda engine, view, index: 'native')
    monkeypatch.setattr(spherical_cuda, '_render_spherical_native_torch', lambda engine, view, index: 'torch')
    engine = SimpleNamespace(_volume_gpu=SimpleNamespace(is_cuda=True))
    view = SimpleNamespace(name='spherical_face0', src_h=4, src_w=4)
    cache_key = cuda_backend._GpuWorkerRenderEngine._native_plane_cache_key(engine, view, 0)
    rendered = spherical_cuda.render_spherical_native_resident(engine, view, 0)
    assert (cache_key[-1] == 'reference_fp64') is expected
    assert (rendered == 'native') is expected


@pytest.mark.parametrize('raw, expected', _DIVERGENT_SPELLINGS)
def test_gaussian_smoothing_gpu_readers_agree(monkeypatch, raw, expected):
    from XTA import assembly, pta
    monkeypatch.setenv('YOLO_TTA_GAUSSIAN_SMOOTHING_GPU', raw)
    monkeypatch.setattr(pta, '_GAUSSIAN_GPU_MIN_VOXELS', 0)
    requests = []

    def accelerated(mask, **kwargs):
        requests.append(kwargs['operation'])
        return {'backend': 'cupy_chunked_multi_gpu'}

    monkeypatch.setattr('XTA.pta_preprocessing.run_gpu_preprocessing', accelerated)
    pta.apply_gaussian_smoothing(np.zeros((2, 2, 2), np.uint8), sigma=0.6, passes=1,
                                 warnings=pta.WarningLog(), device_ids=(0,))
    assert assembly.gaussian_smoothing_gpu_enabled() is expected
    assert (requests == ['gaussian']) is expected

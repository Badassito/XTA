"""SAM role preflight and its verified native image-coordinate contracts."""
from __future__ import annotations

import contextlib
from dataclasses import replace
import io
import sys
import types
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import cli
from XTA.config import (
    resolve_backend_devices, resolve_backend_models, resolve_interpolation_settings,
)
from XTA.geometry import ViewInfo
from XTA.sam_integration import SamInterpolationContext, validate_sam_interpolation_geometry


def test_explicit_pretracker_topology_cap_is_preserved_with_live_extra_credit(tmp_path):
    from XTA.sam_interpolation import prepare_sam_interpolation_pass, SamInterpolationInfrastructureError
    from XTA.sam_resources import admit_sam_parent_resources
    from tests.test_sam_selection_resources import Pool, GIB
    observations=np.zeros((3,7,9),np.uint8)
    observations[[0,2],3,4]=1
    with admit_sam_parent_resources(Pool(),GIB,'explicit-preflight',headroom_probe=lambda:64*GIB) as profile:
        assert profile.has_extra_credit and profile.assigned_topology_bytes>1024
        with pytest.raises(SamInterpolationInfrastructureError,match='topology exceeds'):
            prepare_sam_interpolation_pass(observations,gap_distance=5,min_radius=0,
                interpolation_walk_back=0,resource_profile=profile,
                policy={'sam_bridge_policy':{'max_group_bytes':1024}})


def test_inherited_pretracker_topology_cap_can_use_live_extra_credit(tmp_path):
    from XTA.sam_interpolation import prepare_sam_interpolation_pass
    from XTA.sam_resources import admit_sam_parent_resources
    from XTA.sam_branch_selection import branch_workspace_bytes
    from XTA.sam_policy import resolve_sam_bridge_policy
    from tests.test_sam_selection_resources import Pool, GIB
    shape=(18,1024,1024)
    observations=np.zeros(shape,np.uint8)
    frames=tuple(range(shape[0]))
    group=SimpleNamespace(group_id='large',context_bbox_yx=(0,0,1024,1024),frame_indices=frames)
    run=SimpleNamespace(run_id='large-run',group_id='large',pass_index=1,expected_frames=frames)
    plan=SimpleNamespace(groups=(group,),runs=(run,),needed_frames=frames,
        frame_crop_bounds={frame:group.context_bbox_yx for frame in frames},contract_lease_budget=None)
    required=branch_workspace_bytes(shape)
    assert required>resolve_sam_bridge_policy(environ={})['max_group_bytes']
    with admit_sam_parent_resources(Pool(),GIB,'inherited-preflight',headroom_probe=lambda:64*GIB) as profile:
        assert profile.assigned_topology_bytes>required
        with mock.patch('XTA.sam_bridge_planning.plan_sam_bridges',return_value=plan):
            prepared=prepare_sam_interpolation_pass(observations,gap_distance=17,min_radius=0,
                interpolation_walk_back=0,resource_profile=profile)
    assert prepared.runs==(run,)


def transverse(*, shape=(3, 4, 6)):
    return ViewInfo(
        name='transverse__tta_0', physical_view_name='transverse',
        num_slices=shape[0], src_h=shape[1], src_w=shape[2], pad_mode='clamp',
        family='orthogonal', tta_angle_deg=0.0,
    )


def context(tmp_path, source):
    return SamInterpolationContext(
        model_path='unused-bundle', device_ids=('cuda:0',), temp_dir=tmp_path,
        evidence_root=tmp_path / 'evidence', source_volume=source,
        source_identity='immutable-test-source',
    )


@pytest.mark.parametrize('changes', (
    {'physical_view_name': 'future_cartesian_axis'},
    {'family': 'future_projection'},
    {'tta_angle_deg': float('nan')},
    {'tilt_angle_deg': float('inf')},
    {'src_w': 0},
    {'num_slices': 0},
    {'family': 'tilted', 'tilt_base_view': 'future_axis', 'tilt_direction': 'vertical'},
    {'family': 'tilted', 'tilt_base_view': 'transverse', 'tilt_direction': 'diagonal'},
))
def test_active_sam_rejects_unknown_or_malformed_view_geometry_before_runtime(changes):
    view = replace(transverse(), **changes)
    with pytest.raises(ValueError, match='SAM'):
        validate_sam_interpolation_geometry([transverse(), view])


def test_native_rectangular_transverse_image_cache_is_exact_and_lazy(tmp_path):
    source = np.arange(72, dtype=np.uint8).reshape(3, 4, 6)
    owner = context(tmp_path, source)
    view = transverse()
    try:
        reference = owner.image_provider(view, source.shape)
        with mock.patch.object(owner, '_start', side_effect=AssertionError('GPU must stay lazy')):
            assert owner.image_provider(view, source.shape) is reference
        assert owner._runtime is None
        mapping = reference.open()
        np.testing.assert_array_equal(mapping, source)
        del mapping
    finally:
        owner.close()


def test_reduced_square_canvas_retains_the_detector_pixel_coordinates(tmp_path):
    # Linear image values make each bilinear sample independently predictable:
    # 4x6 -> 2x2 samples native x=(1,4), y=(0.5,2.5).
    yy, xx = np.indices((4, 6))
    source = np.stack([8 * yy + 2 * xx + frame * 30 for frame in range(3)]).astype(np.uint8)
    owner = context(tmp_path, source)
    try:
        reference = owner.image_provider(transverse(), (3, 2, 2))
        mapping = reference.open()
        expected = np.stack([np.array([[6, 12], [22, 28]]) + frame * 30 for frame in range(3)])
        np.testing.assert_array_equal(mapping, expected.astype(np.uint8))
        del mapping
        assert owner._runtime is None
    finally:
        owner.close()


@pytest.mark.parametrize('shape', ((3, 3, 2), (2, 4, 6), (3, 0, 0), (3, 4)))
def test_invalid_canvas_geometry_is_rejected_before_cache_allocation(tmp_path, shape):
    owner = context(tmp_path, np.zeros((3, 4, 6), dtype=np.uint8))
    try:
        with pytest.raises(ValueError):
            owner.image_provider(transverse(), shape)
        assert not (tmp_path / 'sam_image_cache').exists()
        assert owner._runtime is None
    finally:
        owner.close()


@pytest.mark.parametrize('extra, expected_fragment', (
    (['--interpolation_backend', 'both'], 'invalid choice'),
    (['--interpolation_backend', 'sam'], 'sam:PATH'),
    (['--model', 'sam:bundle'], 'detector entry'),
    (['--model', 'cpu:openvino', 'sam:bundle', 'sam:duplicate'], 'duplicate sam:'),
    (['--model', 'cpu:openvino', 'sam:'], 'tagged'),
    (['--model', 'cpu:openvino', 'sam:bundle', '--interpolation_backend', 'sam'], '--sam_device'),
))
def test_invalid_sam_role_selection_never_dispatches_runtime(extra, expected_fragment):
    runtime = types.ModuleType('XTA.tta_mode')
    runtime.run = mock.Mock(side_effect=AssertionError('runtime must not start'))
    arguments = ['--input', 'input.mkv', '--model', 'cpu:openvino', '--device', 'cpu']
    stderr = io.StringIO()
    with mock.patch.dict(sys.modules, {'XTA.tta_mode': runtime}):
        with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as error:
            cli._run_tta([*arguments, *extra])
    assert error.value.code == 2
    assert expected_fragment in stderr.getvalue()
    runtime.run.assert_not_called()


@pytest.mark.parametrize('backend, distance', (('sdf', 15), ('sam', 0)))
def test_unused_sam_is_not_required_by_real_cli_preflight(backend, distance):
    runtime = types.ModuleType('XTA.tta_mode')
    runtime.run = mock.Mock()
    arguments = [
        '--input', 'input.mkv', '--model', 'cpu:openvino', '--device', 'cpu',
        '--interpolation_backend', backend, '--interpolation_distance', str(distance),
    ]
    with mock.patch.dict(sys.modules, {'XTA.tta_mode': runtime}):
        cli._run_tta(arguments)
    runtime.run.assert_called_once_with()


def test_cpu_detector_and_gpu_sam_take_the_real_cli_resolution_path():
    arguments = [
        '--input', 'input.mkv', '--model', r'cpu:C:\Detector Models\model.xml',
        r'sam:C:\SAM Bundles\sam:3.1', '--device', 'cpu', '--sam_device', '2,0',
        '--interpolation_backend', 'sam', '--enable_cartesian', 'transverse', '--angle', '0',
    ]
    runtime = types.ModuleType('XTA.tta_mode')
    runtime.run = mock.Mock()
    with mock.patch.dict(sys.modules, {'XTA.tta_mode': runtime}):
        with mock.patch('XTA.config.resolve_interpolation_settings', wraps=resolve_interpolation_settings) as resolve:
            cli._run_tta(arguments)
    runtime.run.assert_called_once_with()
    resolve.assert_called_once()
    args, models, devices = resolve.call_args.args
    assert models == resolve_backend_models(args.model)
    assert devices == resolve_backend_devices(args.device)
    assert models.gpu is None and devices.gpu_devices == ()
    settings = resolve_interpolation_settings(args, models, devices)
    assert settings.sam_model == r'C:\SAM Bundles\sam:3.1'
    assert settings.sam_devices == ('cuda:2', 'cuda:0')


class _PreflightReached(RuntimeError):
    pass


@pytest.mark.parametrize('backend, distance', (('sdf', 15), ('sam', 0)))
def test_pipeline_unused_sam_skips_bundle_resolution(tmp_path, backend, distance):
    from XTA import pipeline
    source = tmp_path / 'input.mkv'
    detector = tmp_path / 'cpu-model.xml'
    source.touch()
    detector.touch()
    arguments = [
        'tta', '--input', str(source), '--device', 'cpu', '--model', f'cpu:{detector}',
        f'sam:{tmp_path / "does-not-exist"}', '--interpolation_backend', backend,
        '--interpolation_distance', str(distance),
    ]
    with mock.patch.object(sys, 'argv', arguments):
        with mock.patch.object(pipeline, 'initialize_runtime_observability'):
            with mock.patch.object(pipeline, 'configure_component_replay_capture'):
                with mock.patch('XTA.lta_sam.resolve_local_sam_bundle') as resolve_bundle:
                    with mock.patch('XTA.tta_augmentation_config.resolve_tta_augmentation', side_effect=_PreflightReached):
                        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(_PreflightReached):
                            pipeline._main_impl()
    resolve_bundle.assert_not_called()


def test_pipeline_cpu_detector_gpu_sam_resolves_sam_as_a_separate_bundle(tmp_path):
    from XTA import pipeline
    source = tmp_path / 'input.mkv'
    detector = tmp_path / 'cpu-model.xml'
    source.touch()
    detector.touch()
    bundle = tmp_path / 'sam bundle:3.1'
    arguments = [
        'tta', '--input', str(source), '--device', 'cpu', '--model', f'cpu:{detector}',
        f'sam:{bundle}', '--interpolation_backend', 'sam', '--sam_device', '0',
    ]
    with mock.patch.object(sys, 'argv', arguments):
        with mock.patch.object(pipeline, 'initialize_runtime_observability'):
            with mock.patch.object(pipeline, 'configure_component_replay_capture'):
                with mock.patch('XTA.lta_sam.resolve_local_sam_bundle', return_value=object()) as resolve_bundle:
                    with mock.patch('XTA.tta_augmentation_config.resolve_tta_augmentation', side_effect=_PreflightReached) as detector_preflight:
                        with contextlib.redirect_stdout(io.StringIO()), pytest.raises(_PreflightReached):
                            pipeline._main_impl()
    resolve_bundle.assert_called_once_with(str(bundle))
    assert detector_preflight.call_args.kwargs['gpu_devices'] == ()
    assert detector_preflight.call_args.kwargs['cpu_enabled'] is True

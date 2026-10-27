"""Only qualified packed continuations reduce the dense-output reservation."""

import ast
import math
from pathlib import Path
from types import SimpleNamespace

import pytest

from XTA.geometry import _build_tilted_view_infos
from XTA.packed_tilted_projection import terminal_native_source_packing_eligible


GIB = 1024**3
MIB = 1024**2


@pytest.fixture
def production(monkeypatch):
    for name in ('YOLO_TTA_PACKED_PARENT', 'YOLO_TTA_PACKED_PARENT_MIN_MIB',
                 'YOLO_TTA_NATIVE_PULL_BACKEND', 'YOLO_TTA_NATIVE_PULL_PLAN_MIB',
                 'YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB'):
        monkeypatch.delenv(name, raising=False)
    tree = ast.parse((Path(__file__).parents[1]/'XTA'/'pipeline.py').read_text(encoding='utf-8'))
    helper = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                  and node.name == '_parent_prepare_transient_bytes')
    submit = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == '_submit_view_prepare')
    assignments = [node for node in submit.body if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id in ('packed_source_eligible', 'transient_bytes')
                for target in node.targets)]
    assert len(assignments) == 2
    namespace = dict(__package__='XTA', __name__='XTA.pipeline', math=math, GIB=GIB)
    exec(compile(ast.Module(body=[helper], type_ignores=[]), '<production transient helper>', 'exec'), namespace)
    return namespace[helper.name], assignments


def shapes():
    return (3000, 2048, 2049), (1931, 3064, 3022)


def arguments():
    native, output = shapes()
    return dict(interpolation_enabled=False, packed_native_source=True,
                native_shape=native, source_shape=output,
                cleanup_workers=1, cleanup_is_simple=True)


def view():
    return _build_tilted_view_infos(9, 7, 9, tilt_views=('transverse',),
        tilt_angles=(30.,), tilt_directions=('horizontal',))[0]


def test_packed_charge_covers_full_initial_packed_overlap_and_capture_scratch(production):
    helper, _ = production
    from XTA.packed_dense_source import DENSE_PACKED_PAYLOAD_BYTES, DENSE_PACKED_CONTROL_BYTES
    native, output = shapes()
    kwargs = arguments()
    amount = helper(view(), math.prod(native), math.prod(output), **kwargs)
    packed = native[0]*native[1]*((native[2]+7)//8)
    assert amount >= (2*GIB + packed + (64+256)*MIB
                      + DENSE_PACKED_PAYLOAD_BYTES + DENSE_PACKED_CONTROL_BYTES)
    assert amount < math.prod(output) + 4*GIB
    assert helper(view(), math.prod(native), math.prod(output),
        confidence_workspace_bytes=256*MIB, **kwargs) == amount + 256*MIB


def test_row_padding_is_charged_for_each_row_not_only_the_whole_volume(production):
    helper, _ = production
    output = (3, 7, 11)
    narrow, odd = (7, 5, 8), (7, 5, 9)
    values = [helper(view(), math.prod(native), math.prod(output),
        interpolation_enabled=False, packed_native_source=True,
        native_shape=native, source_shape=output, cleanup_is_simple=True) for native in (narrow, odd)]
    assert values[1]-values[0] == 7*5


@pytest.mark.parametrize('variable,default', [('YOLO_TTA_NATIVE_PULL_PLAN_MIB', 64),
                                            ('YOLO_TTA_NATIVE_PULL_WORKSPACE_MIB', 256)])
def test_large_user_workspace_caps_are_charged_in_full(production, monkeypatch, variable, default):
    helper, _ = production
    native, output = shapes()
    before = helper(view(), math.prod(native), math.prod(output), **arguments())
    requested = 1024*1024
    monkeypatch.setenv(variable, str(requested))
    after = helper(view(), math.prod(native), math.prod(output), **arguments())
    assert after-before == (requested-default)*MIB
    assert after > math.prod(output)+4*GIB


def test_large_parallel_cleanup_is_charged_above_the_two_gib_control_floor(production):
    helper, _ = production
    native, output = (1024, 4096, 4096), (1024, 4096, 4096)
    amount = helper(view(), math.prod(native), math.prod(output),
        interpolation_enabled=False, packed_native_source=True,
        native_shape=native, source_shape=output, cleanup_workers=64, cleanup_is_simple=True)
    assert amount >= 32*4096*4096*64 + native[0]*native[1]*((native[2]+7)//8)


@pytest.mark.parametrize('overrides', [dict(packed_native_source=False),
    dict(cleanup_is_simple=False), dict(interpolation_enabled=True)])
def test_dense_sam_and_unqualified_cleanup_keep_the_existing_estimate(production, overrides):
    helper, _ = production
    native, output = shapes()
    kwargs = arguments() | overrides
    assert helper(view(), math.prod(native), math.prod(output), **kwargs) == math.prod(output)+4*GIB
    assert helper(view(), math.prod(native), math.prod(output),
                  interpolation_enabled=False) == math.prod(output)+4*GIB


@pytest.mark.parametrize('native,output', [(None, (2,3,4)), ((2,3), (2,3,4)),
    ((2,3,0), (2,3,4)), ((2,3,4), (2,3,5)), ((2,3,5), (2,3,4))])
def test_opted_in_reservation_requires_actual_valid_shape_accounting(production, native, output):
    helper, _ = production
    with pytest.raises(ValueError, match='actual native/source shapes'):
        helper(view(), 24, 24, interpolation_enabled=False, packed_native_source=True,
               native_shape=native, source_shape=output, cleanup_is_simple=True)


@pytest.mark.parametrize('change', ['tiles', 'debug', 'prepublished', 'no_layers', 'no_retirement',
    'no_callback', 'interpolation', 'extrapolation', 'numpy_backend', 'disabled', 'below_threshold',
    'complex_cleanup'])
def test_actual_submission_exclusions_never_reduce_the_reservation(production, monkeypatch, change):
    helper, assignments = production
    native, output = shapes()
    key = ('model', 'view')
    namespace = dict(math=math, view=view(), processing_bytes=math.prod(native),
        source_bytes=math.prod(output), capture_shape=native,
        input_T=output[0], input_H=output[1], input_W=output[2],
        capture_workspace_bytes=0, component_ref_dense_retirement_active=True,
        component_layers_needed=True, dense_tiling_active=False, keep_temp_artifacts=False,
        preinterpolation_layer_already_published=False, key=key,
        direct_union_backing_leases={key: object()}, confmap_mm=object(),
        parent_slice_postprocess_workers=1, angle_variant_streaming_cleanup_active=True,
        interpolation_settings=SimpleNamespace(extrapolation_enabled=False, extrapolation_distance=0),
        args=SimpleNamespace(interpolation_distance=0, interpolation_passes=1, min_conf=0., min_radius=0.),
        terminal_native_source_packing_eligible=terminal_native_source_packing_eligible,
        _parent_prepare_transient_bytes=helper,
        _view_uses_interpolation=lambda view, distance: distance > 0)
    if change == 'tiles': namespace['dense_tiling_active'] = True
    if change == 'debug': namespace['keep_temp_artifacts'] = True
    if change == 'prepublished': namespace['preinterpolation_layer_already_published'] = True
    if change == 'no_layers': namespace['component_layers_needed'] = False
    if change == 'no_retirement': namespace['component_ref_dense_retirement_active'] = False
    if change == 'no_callback': namespace['direct_union_backing_leases'] = {}
    if change == 'interpolation': namespace['args'].interpolation_distance = 1
    if change == 'extrapolation':
        namespace['interpolation_settings'] = SimpleNamespace(extrapolation_enabled=True, extrapolation_distance=1)
    if change == 'numpy_backend': monkeypatch.setenv('YOLO_TTA_NATIVE_PULL_BACKEND', 'numpy')
    if change == 'disabled': monkeypatch.setenv('YOLO_TTA_PACKED_PARENT', '0')
    if change == 'below_threshold':
        namespace['capture_shape'] = (1, 2, 3)
    if change == 'complex_cleanup':
        namespace['angle_variant_streaming_cleanup_active'] = False
        namespace['args'].min_conf = namespace['args'].min_radius = 1.
    exec(compile(ast.Module(body=assignments, type_ignores=[]), '<production transient submission>', 'exec'), namespace)
    assert namespace['transient_bytes'] == math.prod(output)+4*GIB


def test_actual_submission_reduces_only_the_shared_predicate_terminal_path(production):
    helper, assignments = production
    native, output = shapes()
    key = ('model', 'view')
    namespace = dict(math=math, view=view(), processing_bytes=math.prod(native), source_bytes=math.prod(output),
        capture_shape=native, input_T=output[0], input_H=output[1], input_W=output[2],
        capture_workspace_bytes=117, component_ref_dense_retirement_active=True, component_layers_needed=True,
        dense_tiling_active=False, keep_temp_artifacts=False, preinterpolation_layer_already_published=False,
        key=key, direct_union_backing_leases={key: object()}, confmap_mm=object(),
        parent_slice_postprocess_workers=1, angle_variant_streaming_cleanup_active=True,
        interpolation_settings=SimpleNamespace(extrapolation_enabled=False, extrapolation_distance=0),
        args=SimpleNamespace(interpolation_distance=0, interpolation_passes=1, min_conf=0., min_radius=0.),
        terminal_native_source_packing_eligible=terminal_native_source_packing_eligible,
        _parent_prepare_transient_bytes=helper, _view_uses_interpolation=lambda *_: False)
    exec(compile(ast.Module(body=assignments, type_ignores=[]), '<production transient submission>', 'exec'), namespace)
    assert namespace['packed_source_eligible']
    assert namespace['transient_bytes'] == helper(namespace['view'], math.prod(native), math.prod(output),
        confidence_workspace_bytes=117, **arguments())

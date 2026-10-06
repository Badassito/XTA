"""Parent confidence capture uses allocated workers and admitted bounded scratch."""
from contextlib import contextmanager
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import confidence_evidence, pipeline
from XTA.config import GIB
from XTA.interpolation import PreparedViewResult
from tests.test_sam_parent_staging_pipeline import submission


def test_disabled_capture_does_not_resolve_or_allocate_resources():
    with mock.patch.object(confidence_evidence, 'plan_confidence_capture', create=True) as resolve:
        assert pipeline._parent_confidence_capture_plan((7, 9, 11), 32, enabled=False) is None
    resolve.assert_not_called()


def test_capture_planner_receives_real_canvas_and_allocated_parent_workers():
    plan = SimpleNamespace(workspace_bytes=19)
    with mock.patch.object(confidence_evidence, 'plan_confidence_capture', return_value=plan,
                           create=True) as resolve:
        assert pipeline._parent_confidence_capture_plan((1931, 2048, 2048), 32, enabled=True) is plan
    resolve.assert_called_once_with((1931, 2048, 2048), 32, workspace_bytes=256*1024**2)


@pytest.mark.parametrize('failure', [False, True])
def test_prepare_scope_is_bounded_to_call_and_unwinds_failure(failure):
    plan, calls = object(), []
    @contextmanager
    def scope(value):
        assert value is plan
        calls.append('enter')
        try:
            yield
        finally:
            calls.append('exit')
    def prepare(**kwargs):
        assert kwargs == dict(slice_workers=32, source='immutable inputs')
        calls.append('prepare')
        if failure:
            raise RuntimeError('capture failed')
        return 'prepared'
    with mock.patch.object(confidence_evidence, 'confidence_capture_resources', scope, create=True):
        if failure:
            with pytest.raises(RuntimeError, match='capture failed'):
                pipeline._prepare_parent_with_confidence_capture(prepare, capture_plan=plan,
                    slice_workers=32, source='immutable inputs')
        else:
            assert pipeline._prepare_parent_with_confidence_capture(prepare, capture_plan=plan,
                slice_workers=32, source='immutable inputs') == 'prepared'
    assert calls == ['enter', 'prepare', 'exit']


def test_absent_capture_plan_preserves_unwrapped_prepare_contract():
    prepare = mock.Mock(return_value='prepared')
    with mock.patch.object(confidence_evidence, 'confidence_capture_resources', create=True) as scope:
        assert pipeline._prepare_parent_with_confidence_capture(prepare, capture_plan=None,
            source='original', slice_workers=3) == 'prepared'
    scope.assert_not_called()
    prepare.assert_called_once_with(source='original', slice_workers=3)


@pytest.mark.parametrize('view_name,interpolate,expected', [
    ('transverse', False, 2*GIB), ('transverse', True, 2*600+4*GIB),
    ('tilted_sagittal', False, 900+4*GIB),
    ('azimuthal_tilted_coronal', True, 900+4*GIB),
])
def test_actual_and_policy_reservations_share_exact_workspace_charge(view_name, interpolate, expected):
    view = SimpleNamespace(name=view_name)
    assert pipeline._parent_prepare_transient_bytes(view, 600, 900,
        interpolation_enabled=interpolate, confidence_workspace_bytes=719) == expected+719


def test_submitted_capture_scope_runs_inside_exact_parent_reservation(tmp_path):
    function, ns, view, key, _leases, _captured, executor, _future, _dispatch = submission(
        tmp_path, ready=True, staged=False)
    ns['args'].reconciliation_retain_confidence = True
    ns['parent_slice_postprocess_workers'] = 32
    ns['interpolation_settings'].backend = 'sdf'
    plan, events = SimpleNamespace(workspace_bytes=123456), []
    prepared = PreparedViewResult('model', view.name, '0', 0., None, None, [])
    ns['_parent_confidence_capture_plan'] = mock.Mock(return_value=plan)
    @contextmanager
    def reserve(nbytes, _description):
        events.append(('reserve', nbytes))
        try:
            yield
        finally:
            events.append(('release', nbytes))
    @contextmanager
    def scope(value):
        assert value is plan
        events.append(('scope', value.workspace_bytes))
        yield
        events.append(('scope_end', value.workspace_bytes))
    def prepare(**kwargs):
        assert kwargs['slice_workers'] == 32
        assert kwargs['union_mm'].shape == (3, 4, 5)
        assert kwargs['confmap_mm'] is None
        assert kwargs['confidence_owner'].pop().shape == (3, 4, 5)
        events.append(('prepare', 32))
        return prepared
    ns['parent_transient_admission'] = SimpleNamespace(reserve=reserve)
    ns['prepare_view_volume_after_fullframe'] = prepare
    function('model', view)
    task = executor.submit.call_args.args[0]
    ns['_parent_confidence_capture_plan'].assert_called_once_with((3, 4, 5), 32, enabled=True)
    assert task.transient_bytes == 2*60+4*GIB+plan.workspace_bytes
    with mock.patch.object(confidence_evidence, 'confidence_capture_resources', scope, create=True):
        assert task() is prepared
    assert events == [('reserve', task.transient_bytes), ('scope', 123456),
                      ('prepare', 32), ('scope_end', 123456), ('release', task.transient_bytes)]
    assert key in ns['view_processing_submitted']


def test_capture_preflight_failure_preserves_original_dense_owner(tmp_path):
    function, ns, view, key, leases, _captured, executor, _future, _dispatch = submission(
        tmp_path, ready=True, staged=False)
    original_mask = ns['baseline_union_by_model_view'][key]
    original_scores = ns['baseline_confmap_by_model_view'][key]
    ns['_parent_confidence_capture_plan'] = mock.Mock(side_effect=RuntimeError('insufficient workspace'))
    with pytest.raises(RuntimeError, match='insufficient workspace'):
        function('model', view)
    assert ns['baseline_union_by_model_view'][key] is original_mask
    assert ns['baseline_confmap_by_model_view'][key] is original_scores
    assert not ns['view_processing_submitted']
    assert leases.leases[key].phase == 'inference'
    executor.submit.assert_not_called()


@pytest.mark.parametrize('disabled,score_missing,d1_published', [
    (True,False,False), (False,True,False), (False,False,True),
])
def test_disabled_missing_score_or_d1_published_parent_has_zero_capture_charge(
        tmp_path, disabled, score_missing, d1_published):
    function, ns, view, key, _leases, _captured, executor, _future, _dispatch = submission(
        tmp_path, ready=True, staged=False)
    ns['args'].reconciliation_retain_confidence = not disabled
    if score_missing:
        ns['baseline_confmap_by_model_view'][key] = None
    if d1_published:
        ns['d1_view_shadow_path_by_parent'][key] = tmp_path/'immutable_shadow.cvol'
    ns['_parent_confidence_capture_plan'] = mock.Mock(return_value=None)
    function('model', view)
    assert ns['_parent_confidence_capture_plan'].call_args.kwargs == dict(enabled=False)
    task = executor.submit.call_args.args[0]
    assert task.transient_bytes == 2*60+4*GIB
    assert task.prepare.keywords['capture_plan'] is None


def test_policy_memory_preflight_uses_same_actual_charge_and_reuses_canvas_plans():
    shape = (7, 11, 13)
    view = SimpleNamespace(name='tilted_coronal')
    parent = dict(kind='fullframe', processing_shape=shape, view=view)
    d1_parent = {**parent, 'd1_output_shape': shape}
    tasks = [parent, dict(parent), d1_parent, dict(kind='tile')]
    plan = SimpleNamespace(workspace_bytes=87654)
    def resolve(canvas, workers, *, enabled):
        assert canvas == shape and workers == 32
        return plan if enabled else None
    with mock.patch.object(pipeline, '_parent_confidence_capture_plan', side_effect=resolve) as planner:
        with mock.patch.object(pipeline, '_view_uses_interpolation', return_value=False):
            actual = pipeline._policy_parent_prepare_transient_bytes(tasks, 123456, 32,
                interpolation_distance=0, retain_confidence=True)
    assert actual == pipeline._parent_prepare_transient_bytes(view, 7*11*13, 123456,
        interpolation_enabled=False, confidence_workspace_bytes=plan.workspace_bytes)
    assert planner.call_count == 2  # one enabled canvas, one already-published D1 canvas
    assert [call.kwargs['enabled'] for call in planner.call_args_list] == [True, False]


def test_real_pipeline_scope_preserves_payloads_and_readonly_input_lifetime(tmp_path):
    from XTA.confidence_capture import current_confidence_capture_plan
    from XTA.geometry import get_view_infos
    shape = (7, 137, 151)
    random = np.random.default_rng(790)
    scores = random.integers(0, 256, shape, dtype=np.uint8)
    mask = np.zeros(shape, np.uint8)
    mask[2:5, 17:130, 29:149] = random.integers(0, 2, (3, 113, 120), dtype=np.uint8)
    active = np.zeros(shape[0], bool)
    active[2:5] = True
    boxes = np.full((shape[0], 4), -1, np.int64)
    boxes[2:5] = [17, 130, 29, 149]
    original_mask, original_scores = mask.copy(), scores.copy()
    view = get_view_infos(*shape, cartesian_views=('transverse',))[0]
    plan = pipeline._parent_confidence_capture_plan(shape, 3, enabled=True)
    assert plan.workers <= 3 and plan.workspace_bytes <= 256*1024**2
    confidence_evidence.configure_confidence_evidence(tmp_path/'out', enabled=True,
        defer_projection=True)
    try:
        def prepare(**kwargs):
            assert current_confidence_capture_plan() is plan
            return confidence_evidence.capture_prediction_confidence(mask, scores, view=view,
                model_name='model', temp_dir=tmp_path, known_slice_any=active,
                known_slice_bboxes=boxes)
        ref = pipeline._prepare_parent_with_confidence_capture(prepare, capture_plan=plan,
            slice_workers=3)
        assert current_confidence_capture_plan() is None
        expected = confidence_evidence.write_block_confidence_evidence(tmp_path/'reference', shape,
            lambda z:np.where(mask[z] != 0, scores[z], np.uint8(0)),
            layer_key=ref.layer_key, model_name='model', coordinate_space='native_view_processing',
            source_shape_tyx=shape)
        for filename in ('index.bin', 'scores.u8.zlib'):
            assert (ref.path/filename).read_bytes() == (expected.path/filename).read_bytes()
        np.testing.assert_array_equal(mask, original_mask)
        np.testing.assert_array_equal(scores, original_scores)
        assert mask.flags.writeable and scores.flags.writeable
    finally:
        confidence_evidence.shutdown_confidence_publication(raise_errors=False)
        confidence_evidence.configure_confidence_evidence(None, enabled=False)

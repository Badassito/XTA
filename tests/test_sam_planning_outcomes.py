"""Resource refusals stay visible without confusing legitimate empty plans."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.sam_planning_outcomes import sam_planning_omission


@pytest.mark.parametrize('operation', ('interpolation', 'extrapolation'))
@pytest.mark.parametrize('status', ('planned', 'empty', 'disabled', 'exhausted'))
def test_complete_or_empty_planning_has_no_omission(operation, status):
    assert sam_planning_omission(dict(planning_status=status), operation=operation) is None


@pytest.mark.parametrize('status', ('unresolved', 'incomplete', 'invalid', 'partial'))
def test_all_incomplete_plan_statuses_are_explicit(status):
    record = sam_planning_omission(dict(planning_status=status), operation='interpolation')
    assert record['status'] == status


def test_planned_status_does_not_hide_unresolved_families():
    record = sam_planning_omission(dict(sam_planning_status='planned', pass_index=2,
        sam_group_planning_receipts=[dict(status='planned', reasons=[]),
            dict(status='unresolved', reasons=['family_endpoint_limit', 'group_contract_memory_limit']),
            dict(status='unresolved', reasons=['group_contract_memory_limit'])]),
        operation='interpolation', scope_id='model/radial', view_name='radial')
    assert record == dict(operation='interpolation', scope_id='model/radial', view='radial',
        pass_index=2, status='planned', reasons=['family_endpoint_limit', 'group_contract_memory_limit'],
        unresolved_plan=False, unresolved_group_count=2,
        group_refusal_reasons=dict(family_endpoint_limit=1, group_contract_memory_limit=2))


def test_summary_group_counts_remain_visible_without_detailed_receipts():
    record = sam_planning_omission(dict(sam_planning_status='planned', sam_unresolved_groups=11,
        sam_group_refusal_reasons=dict(family_endpoint_limit=6, group_contract_memory_limit=9,
                                      total_contract_memory_limit=9)), operation='interpolation')
    assert record['unresolved_group_count'] == 11
    assert record['group_refusal_reasons']['group_contract_memory_limit'] == 9
    assert record['reasons'] == ['family_endpoint_limit', 'group_contract_memory_limit',
                                 'total_contract_memory_limit']


@pytest.mark.parametrize('operation', ('interpolation', 'extrapolation'))
@pytest.mark.parametrize('refused', (False, True))
def test_inventory_refusal_warns_counts_and_retains_observations(tmp_path, monkeypatch, capsys,
                                                               operation, refused):
    from XTA import geometry, runtime, sam_integration
    from XTA.interpolation import close_memmap_array
    from XTA.sam_bridge_planning import SamPlanningLimits
    from tests.test_sam_view_image_cache import context_for
    observed = np.ones((3, 7, 9), np.uint8)
    view = geometry.get_view_infos(*observed.shape, cartesian_views=('transverse',))[0]
    context = context_for(tmp_path, observed)
    telemetry = runtime.RuntimeTelemetry()
    telemetry.enabled = True
    events = []
    monkeypatch.setattr(sam_integration, 'runtime_telemetry', lambda: telemetry)
    monkeypatch.setattr(telemetry, 'trace_event', lambda event, **fields: events.append((event, fields)))
    monkeypatch.setattr(context, '_start', lambda: pytest.fail('Inventory-only test must not start SAM'))
    kwargs = dict(work_dir=tmp_path/'pass', min_radius=0.,
        planner_limits=SamPlanningLimits(max_observation_bytes=1 if refused else 10000))
    method = context.interpolate if operation == 'interpolation' else context.extrapolate
    kwargs['gap_distance' if operation == 'interpolation' else 'distance'] = 2
    merged = None
    try:
        merged, stats, components = method(observed, view=view, scope='model/view', **kwargs)
        np.testing.assert_array_equal(merged, observed)
        assert stats['sam_requested_work_complete'] is (not refused)
        assert context.unresolved_plan_passes == int(refused)
        assert len(context.planning_omissions) == int(refused)
        assert telemetry.snapshot()['counters'].get('sam.planning_incomplete_passes', 0) == int(refused)
        warnings = capsys.readouterr().err
        if refused:
            assert 'requested SAM ' + operation + ' is incomplete' in warnings
            assert 'observation_inventory_memory_limit' in warnings
            record = stats['sam_planning_omission']
            assert record['view'] == view.name and record['scope_id'] == 'model/view'
            assert record['unresolved_plan'] and record['unresolved_group_count'] == 0
            assert events == [('sam_planning_incomplete', record)]
            assert stats['skip_reason'] == 'resource_limit_unresolved'
        else:
            assert 'requested SAM' not in warnings
            assert 'sam_planning_omission' not in stats
            assert events == []
    finally:
        if merged is not None and merged is not observed:
            close_memmap_array(merged)
        context.close()


def test_partial_plan_warning_precedes_tracker_work(tmp_path, monkeypatch, capsys):
    from XTA import sam_integration
    from tests.test_sam_view_image_cache import context_for
    context = context_for(tmp_path, np.zeros((3, 7, 9), np.uint8))
    telemetry = SimpleNamespace(add=lambda *args: None, trace_event=lambda *args, **kwargs: None)
    monkeypatch.setattr(sam_integration, 'runtime_telemetry', lambda: telemetry)
    try:
        record = context._record_planning_outcome(SimpleNamespace(status='planned', reasons=(),
            groups=(SimpleNamespace(status='unresolved', reasons=('family_endpoint_limit',)),)),
            dict(scope_id='model/radial', view_name='radial'), 'interpolation')
        assert not record['unresolved_plan'] and record['unresolved_group_count'] == 1
        assert context.planning_omissions == [record] and context.unresolved_plan_passes == 0
        assert 'unresolved groups=1' in capsys.readouterr().err
    finally:
        context.close()

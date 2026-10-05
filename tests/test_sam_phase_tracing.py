"""CPU-only attribution for planning and canonical image-cache work."""
from __future__ import annotations

import numpy as np
import pytest

from XTA import geometry, runtime, sam_integration, sam_interpolation


def _context(tmp_path):
    return sam_integration.SamInterpolationContext(model_path='unused', device_ids=(0,),
        temp_dir=tmp_path, evidence_root=tmp_path / 'evidence',
        source_volume=np.zeros((3, 4, 5), np.uint8), source_identity='trace-source')


@pytest.mark.parametrize('operation', ('interpolation', 'extrapolation'))
def test_context_failed_planning_is_balanced_and_restores_thread_scope(tmp_path, monkeypatch, operation):
    from XTA import sam_extrapolation
    context = _context(tmp_path)
    view = geometry.ViewInfo(name='transverse__tta_a0', physical_view_name='transverse',
        num_slices=3, src_h=4, src_w=5, pad_mode='clamp', family='orthogonal', tta_angle_deg=0.)
    events = []
    monkeypatch.setattr(runtime, 'runtime_trace_event', lambda event, **fields: events.append((event, fields)))

    def fail_plan(*_args, **_kwargs):
        assert context._resource_local.sam_phase_scope == (operation, 'local-scope')
        raise RuntimeError('planning failed')

    target = sam_interpolation if operation == 'interpolation' else sam_extrapolation
    monkeypatch.setattr(target, 'prepare_sam_' + operation + '_pass', fail_plan)
    context._resource_local.sam_phase_scope = ('prior', 'outer')
    method = context.interpolate if operation == 'interpolation' else context.extrapolate
    try:
        with pytest.raises(RuntimeError, match='planning failed'):
            method(np.zeros((3, 4, 5), np.uint8), view=view, scope='local-scope')
        assert context._resource_local.sam_phase_scope == ('prior', 'outer')
        assert context._active_passes == 0
        assert [event for event, _ in events] == ['sam_phase_begin', 'sam_phase_end']
        begin, end = events[0][1], events[1][1]
        assert begin == {key: value for key, value in end.items() if key != 'failed'}
        assert end['failed'] is True
        assert begin['sam_phase'] == 'planning'
        assert begin['sam_operation'] == operation
        assert begin['scope_id'] == 'local-scope'
    finally:
        context.close()


@pytest.mark.parametrize('failed', (False, True))
def test_image_provider_trace_preserves_result_or_error(tmp_path, monkeypatch, failed):
    context = _context(tmp_path)
    events = []
    result = object()
    context._resource_local.sam_phase_scope = ('extrapolation', 'cohort-scope')
    monkeypatch.setattr(runtime, 'runtime_trace_event', lambda event, **fields: events.append((event, fields)))

    def render(*_args, **_kwargs):
        if failed:
            raise RuntimeError('render failed')
        return result

    monkeypatch.setattr(context, '_image_provider', render)
    try:
        if failed:
            with pytest.raises(RuntimeError, match='render failed'):
                context.image_provider(None, (3, 4, 5))
        else:
            assert context.image_provider(None, (3, 4, 5)) is result
        assert [event for event, _ in events] == ['sam_phase_begin', 'sam_phase_end']
        assert events[1][1]['failed'] is failed
        assert events[0][1]['sam_phase'] == 'image_render'
        assert events[0][1]['sam_operation'] == 'extrapolation'
        assert events[0][1]['scope_id'] == 'cohort-scope'
        assert events[0][1]['sam_phase_id'] == events[1][1]['sam_phase_id']
    finally:
        context.close()

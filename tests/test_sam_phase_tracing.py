"""CPU-only attribution for planning and canonical image-cache work."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
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
        assert begin == {key: value for key, value in end.items()
                         if key not in ('failed', 'error', 'error_type')}
        assert end['failed'] is True
        assert end['error_type'] == 'RuntimeError' and end['error'] == 'planning failed'
        assert begin['sam_phase'] == 'planning'
        assert begin['sam_operation'] == operation
        assert begin['scope_id'] == 'local-scope'
        root = context.evidence_root if operation == 'interpolation' else context.extrapolation_evidence_root
        target = root / hashlib.sha256(b'local-scope').hexdigest()[:20] / 'context_preparation_failure.json'
        receipt = json.loads(target.read_text())
        assert receipt['phase'] == 'planning'
        assert receipt['error_type'] == 'RuntimeError' and receipt['error'] == 'planning failed'
        assert receipt['sam_operation'] == operation
        assert receipt['scope_id'] == 'local-scope' and receipt['view_name'] == view.name
        assert receipt['physical_view'] == 'transverse' and receipt['native_shape_tyx'] == [3, 4, 5]
        assert receipt['sam_resource_profile']['status'] == 'direct_declared_bounds'
        assert not receipt['complete'] and not receipt['prepared_plan_available']
        assert receipt['observation_snapshot_sha256'] is None
        # The actual failure is durable before sibling/global cancellation.
        assert not context._cancel.is_set()
    finally:
        context.close()


@pytest.mark.parametrize('operation', ('interpolation', 'extrapolation'))
def test_receipt_write_failure_preserves_exact_planning_error(tmp_path, monkeypatch, operation):
    from XTA import sam_extrapolation
    context = _context(tmp_path)
    view = geometry.get_view_infos(3, 4, 5, cartesian_views=('transverse',))[0]
    original = MemoryError('original planner resource failure')
    def fail_plan(*args, **kwargs):
        raise original
    target = sam_interpolation if operation == 'interpolation' else sam_extrapolation
    monkeypatch.setattr(target, 'prepare_sam_' + operation + '_pass', fail_plan)
    write = Path.write_text
    def fail_write(path, *args, **kwargs):
        if path.name == 'context_preparation_failure.json.tmp':
            raise OSError('diagnostic disk unavailable')
        return write(path, *args, **kwargs)
    monkeypatch.setattr(Path, 'write_text', fail_write)
    method = context.interpolate if operation == 'interpolation' else context.extrapolate
    try:
        with pytest.raises(MemoryError) as caught:
            method(np.zeros((3, 4, 5), np.uint8), view=view, scope='failed')
        assert caught.value is original
        assert any('diagnostic disk unavailable' in note for note in original.__notes__)
        assert context._active_passes == 0 and not context._cancel.is_set()
    finally:
        context.close()


def test_phase_error_is_bounded_and_caught_fallback_remains_local(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(runtime, 'runtime_trace_event', lambda event, **fields: events.append((event, fields)))
    with sam_interpolation._trace_sam_phase('image_render', 'fallback', operation='interpolation'):
        try:
            with sam_interpolation._trace_sam_phase('gpu_projection', 'fallback', operation='interpolation'):
                raise MemoryError('gpu fallback ' + 'x'*5000)
        except MemoryError:
            pass  # The existing CPU fallback completes the enclosing operation.
    ends = [fields for event, fields in events if event == 'sam_phase_end']
    assert ends[0]['failed'] and ends[0]['sam_phase'] == 'gpu_projection'
    assert ends[0]['error_type'] == 'MemoryError' and len(ends[0]['error']) == 4096
    assert not ends[1]['failed'] and ends[1]['sam_phase'] == 'image_render'
    assert 'error' not in ends[1] and 'error_type' not in ends[1]


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

"""Configured later SAM passes and disabled tails retain bounded publication order."""
import pytest

from XTA import assembly, geometry
from XTA.projection_queue import settle_prepared_view_components
from tests.test_sam_async_layer_publication import (
    SelectedContext, make_queue, make_submitter, prepare)


@pytest.mark.parametrize('distance', [0, 2])
def test_async_publication_keeps_two_interpolation_layers_with_configured_later_passes(
        tmp_path, monkeypatch, distance):
    shape = (5, 7, 9)
    view = geometry.get_view_infos(*shape, cartesian_views=('transverse',))[0]
    context = SelectedContext(tmp_path/'evidence', shape)
    calls, publications = [], []
    original_interpolate = context.interpolate
    def interpolate(*args, **kwargs):
        calls.append(kwargs['pass_index'])
        assert kwargs['interpolation_passes'] == 3
        return original_interpolate(*args, **kwargs)
    monkeypatch.setattr(context, 'interpolate', interpolate)
    actual_prepare = assembly.prepare_view_volume_after_fullframe
    def configured_prepare(**kwargs):
        kwargs.update(interpolate_passes=3, extrapolation_distance=distance)
        return actual_prepare(**kwargs)
    monkeypatch.setattr(assembly, 'prepare_view_volume_after_fullframe', configured_prepare)
    queue = make_queue()
    actual_submit = make_submitter(queue, shape)
    def submit(entry, **kwargs):
        publications.append((kwargs['layer_kind'], entry['direction']))
        if kwargs['layer_kind'] == 'interpolation':
            assert context.extrapolation_entered.is_set() == bool(distance)
        return actual_submit(entry, **kwargs)
    assembly.set_final_source_output_shape(shape)
    try:
        result = prepare(tmp_path/'parent', view, context, submit)
        queue.shutdown()
        assert calls == [1]
        expected = [('interpolation', 'forward'), ('interpolation', 'backward')]
        if distance:
            expected += [('extrapolation', 'forward'), ('extrapolation', 'backward')]
        assert publications == expected
        assert context.extrapolation_entered.is_set() == bool(distance)
        assert settle_prepared_view_components(result)
        assert [(ref.mask_kind, ref.interpolation_direction or ref.extrapolation_provenance['direction'])
                for ref in result.nrrd_layers] == [
                    ('bridge' if kind == 'interpolation' else kind, direction) for kind, direction in expected]
        state = queue.snapshot()
        assert state['pending'] == state['source_bytes'] == state['working_bytes'] == 0
    finally:
        queue.abort()
        queue.shutdown(cancel_futures=True)
        assembly.set_final_source_output_shape(None)

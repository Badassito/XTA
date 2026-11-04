"""Same-frame corridor reuse keeps every streamed SAM role plane byte-identical."""
from unittest import mock

import numpy as np
import pytest

from XTA import sam_interpolation_contracts as contracts
from tests.test_sam_streaming_contracts import expected, fixture


def publication_order_values(masks, metadata):
    """Replay add_group's access pattern: validation then frame-major publication."""
    frames = list(map(int, metadata['frame_indices']))
    values = {}
    for frame in frames:
        for name in (f'write:{frame}', f'acceptance:{frame}'):
            np.asarray(masks[name])
        for name in masks:
            if name.startswith('edge_write:') and name.endswith(f':{frame}'):
                np.asarray(masks[name])
    for name in masks:
        values[name] = np.array(masks[name], copy=True)
    return values


@pytest.mark.parametrize('case', ['simple', 'branching', 'inventory', 'walkback', 'cyclic', 'retry'])
@pytest.mark.parametrize('zero_margins', [False, True])
def test_cached_corridors_match_uncached_and_eager_planes(monkeypatch, case, zero_margins):
    _source, plan, recipe = fixture(case, zero_margins=zero_margins)
    wanted = expected(recipe, plan)
    calls = {}
    original = contracts._corridor

    def counted(*args, **kwargs):
        calls['n'] = calls.get('n', 0) + 1
        return original(*args, **kwargs)

    results = {}
    for planes in (0, contracts._CORRIDOR_CACHE_PLANES):
        calls['n'] = 0
        monkeypatch.setattr(contracts, '_CORRIDOR_CACHE_PLANES', planes)
        monkeypatch.setattr(contracts, '_corridor', counted)
        metadata, masks = contracts.stream_sam_group_contracts(recipe, 0.)
        assert metadata == wanted.metadata
        results[planes] = (publication_order_values(masks, metadata), calls['n'])
    (uncached, uncached_calls), (cached, cached_calls) = results[0], results[contracts._CORRIDOR_CACHE_PLANES]
    assert set(cached) == set(uncached) == set(wanted.masks)
    for name, value in wanted.masks.items():
        assert cached[name].dtype == uncached[name].dtype == value.dtype
        assert cached[name].tobytes() == uncached[name].tobytes() == value.tobytes(), name
    assert cached_calls <= uncached_calls
    if case != "inventory":  # its single restricted edge has no repeated same-frame corridor
        assert cached_calls < uncached_calls


def test_cache_is_frame_local_bounded_and_immutable():
    _source, _plan, recipe = fixture('branching')
    masks = contracts.StreamingSamGroupMasks(recipe)
    frame = recipe.descriptor.frame_indices[len(recipe.descriptor.frame_indices)//2]
    pairs = [(a, b) for a, b in masks.edges.values() if a.frame_index <= frame <= b.frame_index]
    assert pairs
    a, b = pairs[0]
    first = masks._corridor(a, b, frame, 2)
    assert not first.flags.writeable
    with pytest.raises(ValueError):
        first[...] = True
    assert masks._corridor(a, b, frame, 2) is first
    np.testing.assert_array_equal(first, contracts._corridor(a, b, frame, masks.crop, 2,
        raster_origin_yx=masks.origin))
    for margin in (3, 4, 5):
        masks._corridor(a, b, frame, margin)
    assert len(masks._corridor_cache) <= contracts._CORRIDOR_CACHE_PLANES
    assert masks._corridor(a, b, frame, 2) is not first
    other = next(value for value in recipe.descriptor.frame_indices if value != frame)
    masks._corridor(a, b, other, 2)
    assert masks._corridor_frame == other and len(masks._corridor_cache) == 1


def test_write_and_edge_planes_do_not_mutate_cached_corridor():
    _source, _plan, recipe = fixture('simple')
    masks = contracts.StreamingSamGroupMasks(recipe)
    (edge_id, (a, b)), = list(masks.edges.items())[:1]
    frame = next(value for value in recipe.descriptor.frame_indices if a.frame_index < value < b.frame_index)
    margin = recipe.bounds.curvature_margin_px
    corridor = masks._corridor(a, b, frame, margin)
    before = corridor.tobytes()
    for name in (f'edge_contract:{edge_id}:{frame}', f'edge_write:{edge_id}:{frame}',
                 f'write:{frame}', f'acceptance:{frame}'):
        value = masks[name]
        assert not value.flags.writeable
    assert masks._corridor(a, b, frame, margin).tobytes() == before

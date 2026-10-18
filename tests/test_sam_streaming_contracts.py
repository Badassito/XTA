"""Streamed role planes preserve eager SAM contracts without frame/edge stacks."""
from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest

from XTA.sam_bridge_planning import (SamPlanningLimits, _materialize_contract_group,
    expand_sam_bridge_group_context, plan_sam_bridges)
from XTA.sam_evidence import SamEvidenceWriter
from XTA.sam_interpolation import _write_materialized_group
from XTA.sam_interpolation_contracts import (StreamingSamGroupMasks,
    stream_sam_group_contracts, streaming_sam_contract_peak_bytes)


def fixture(case, *, zero_margins=False):
    options = dict(interpolation_distance=12, interpolation_candidates=3,
        interpolation_walk_back=0, interpolation_min_radius=0, interpolation_search_angle=30,
        spacing_zyx=(1., 1.5, .75), lazy_contracts=True,
        limits=SamPlanningLimits(context_margin_px=4,
            acceptance_margin_px=0 if zero_margins else 2,
            curvature_margin_px=0 if zero_margins else 2))
    if case in ('branching', 'inventory'):
        source = np.zeros((15, 65, 73), np.uint16)
        source[1, 28:33, 32:37] = 8
        source[7:12, 25:30, 28:33] = 8
        source[11, 32:37, 37:42] = 8
        source[6, 10:13, 10:13] = 9
    elif case == 'walkback':
        source = np.zeros((12, 41, 41), np.uint8)
        source[[0, 1, 2, 3, 9, 10, 11], 18:23, 18:23] = 1
        options['interpolation_walk_back'] = 2
    elif case == 'cyclic':
        source = np.zeros((8, 15, 27), np.uint8)
        source[[6, 7], 6:9, 4:7] = 1
        source[[1, 2], 6:9, 20:23] = 1
        options.update(wrap_axis=True, interpolation_distance=3, interpolation_walk_back=1)
    else:
        source = np.zeros((10, 101, 181), np.uint16)
        source[1, 30:34, 49:54] = 8
        source[8, 32:37, 54:58] = 8
        source[4, 50:53, 100:103] = 9
    plan = plan_sam_bridges(source, **options)
    groups = [group for group in plan.groups if group.contract_recipe is not None]
    assert groups
    group = max(groups, key=lambda value: len(value.observation_ids))
    if case == 'retry':
        group = expand_sam_bridge_group_context(group, (0, 0, *source.shape[1:]),
            max_crop_pixels=source.shape[1] * source.shape[2], retry_lineage={'attempt': 1})
    recipe = group.contract_recipe
    if case == 'inventory':
        assert len(recipe.pairs) >= 2
        descriptor = replace(recipe.descriptor, edges=recipe.descriptor.edges[:1])
        recipe = replace(recipe, descriptor=descriptor, pairs=tuple(sorted(recipe.pairs)[:1]))
    return source, plan, recipe


class Capture:
    def add_group(self, metadata, masks):
        self.metadata = metadata
        self.masks = {key: np.array(masks[key], copy=True) for key in masks}


def expected(recipe, plan):
    capture = Capture()
    _write_materialized_group(capture, _materialize_contract_group(recipe), plan.by_id, 0.)
    return capture


@pytest.mark.parametrize('case', ['simple', 'branching', 'inventory', 'walkback', 'cyclic', 'retry'])
@pytest.mark.parametrize('zero_margins', [False, True])
def test_every_streamed_role_and_metadata_matches_eager_exactly(case, zero_margins):
    source, plan, recipe = fixture(case, zero_margins=zero_margins)
    before = source.copy()
    wanted = expected(recipe, plan)
    metadata, masks = stream_sam_group_contracts(recipe, 0.)
    assert metadata == wanted.metadata
    assert set(masks) == set(wanted.masks) and len(masks) == len(wanted.masks)
    keys = list(masks)
    np.random.default_rng(23).shuffle(keys)
    for key in keys + list(reversed(keys)):
        value = masks[key]
        np.testing.assert_array_equal(value, wanted.masks[key], err_msg=key)
        assert value.dtype == np.bool_ and value.ndim == 2 and not value.flags.writeable
        with pytest.raises(ValueError):
            value.flags.writeable = True
    np.testing.assert_array_equal(source, before)
    if case == 'cyclic':
        assert max(metadata['frame_indices']) >= source.shape[0]
        assert any(endpoint['mirror_u'] for endpoint in metadata['endpoints'])
    if case == 'walkback':
        assert len(metadata['endpoint_ids']) < len(metadata['endpoints'])
    if case == 'retry':
        assert metadata['crop_contract']['retry_lineage'] == {'attempt': 1}


def test_writer_validates_and_serializes_exact_role_planes_without_eager_materialization(tmp_path, monkeypatch):
    from XTA import sam_bridge_planning
    source, plan, recipe = fixture('branching')
    wanted = expected(recipe, plan)
    metadata, masks = stream_sam_group_contracts(recipe, 0.)
    monkeypatch.setattr(sam_bridge_planning, '_materialize_contract_group',
        lambda *_a: pytest.fail('streaming cannot allocate eager contracts'))
    with SamEvidenceWriter(tmp_path/'streamed', {'shape_tyx': list(source.shape)}) as writer:
        writer.add_group(metadata, masks)
        bundle = writer.commit()
    with SamEvidenceWriter(tmp_path/'eager', {'shape_tyx': list(source.shape)}) as writer:
        writer.add_group(wanted.metadata, wanted.masks)
        writer.commit()
    for name in ('masks.bin', 'index.json', 'manifest.json'):
        assert (tmp_path/'streamed'/name).read_bytes() == (tmp_path/'eager'/name).read_bytes()
    assert set(bundle.groups[metadata['group_id']]['mask_keys']) == set(wanted.masks)
    for key, value in wanted.masks.items():
        np.testing.assert_array_equal(bundle.group_mask(metadata['group_id'], key), value, err_msg=key)


def test_candidate_inventory_is_built_once_for_real_writer_role_access(tmp_path):
    source, plan, recipe = fixture('inventory')
    wanted = expected(recipe, plan)
    class CountedGraph:
        calls = 0
        def get(self, key, default):
            self.calls += 1
            return recipe.graph.get(key, default)
    graph = CountedGraph()
    metadata, masks = stream_sam_group_contracts(replace(recipe, graph=graph), 0.)
    assert graph.calls == len(recipe.endpoints)
    inventory = masks._candidate_edge_inventory
    assert len(inventory) > len(masks.edges)
    assert all(a.frame_index <= b.frame_index for _source, _target, a, b in inventory)
    with SamEvidenceWriter(tmp_path/'inventory', {'shape_tyx': list(source.shape)}) as writer:
        writer.add_group(metadata, masks)
        bundle = writer.commit()
    assert graph.calls == len(recipe.endpoints)
    assert masks._candidate_edge_inventory is inventory
    for key, value in wanted.masks.items():
        np.testing.assert_array_equal(bundle.group_mask(metadata['group_id'], key), value, err_msg=key)


def test_last_frame_cache_prevents_quadratic_repeated_group_write_generation(monkeypatch):
    from XTA import sam_interpolation_contracts as streaming
    _source, _plan, recipe = fixture('branching')
    masks = StreamingSamGroupMasks(recipe)
    frame = next(frame for frame in recipe.descriptor.frame_indices
                 if sum(a.frame_index < frame < b.frame_index for a, b in masks.edges.values()) >= 2)
    calls = []
    original = streaming._corridor
    def counted(*args, **kwargs):
        calls.append(args[2])
        return original(*args, **kwargs)
    monkeypatch.setattr(streaming, '_corridor', counted)
    union = masks[f'write:{frame}']
    first_count = len(calls)
    assert first_count >= 2
    for edge in recipe.descriptor.edges:
        masks[f'edge_write:{edge.edge_id}:{frame}']
        assert masks[f'write:{frame}'] is union
    assert len(calls) == 2 * first_count
    for target in reversed(recipe.descriptor.frame_indices):
        for role in ('write', 'acceptance', 'known_foreground', 'unrelated'):
            masks[f'{role}:{target}']
        assert len(masks._cache) <= 4
        assert sum(value.nbytes for value in masks._cache.values()) <= 4 * np.prod(masks.shape)


def test_retained_planes_are_stable_and_old_cache_does_not_accumulate():
    source, plan, recipe = fixture('walkback')
    wanted = expected(recipe, plan)
    _, masks = stream_sam_group_contracts(recipe, 0.)
    first, last = recipe.descriptor.frame_indices[0], recipe.descriptor.frame_indices[-1]
    retained = masks[f'write:{first}']
    old_known = weakref.ref(masks[f'known_foreground:{first}'])
    masks[f'acceptance:{last}']
    gc.collect()
    assert old_known() is None
    np.testing.assert_array_equal(retained, wanted.masks[f'write:{first}'])
    source[:] = 1
    for key, value in wanted.masks.items():
        np.testing.assert_array_equal(masks[key], value)
    assert len(masks._cache) <= 4


def test_no_corridor_is_generated_outside_its_inclusive_edge_interval(monkeypatch):
    from XTA import sam_interpolation_contracts as streaming
    _source, _plan, recipe = fixture('walkback')
    _, masks = stream_sam_group_contracts(recipe, 0.)
    corridor = streaming._corridor
    def bounded(a, b, frame, *args, **kwargs):
        assert a.frame_index <= frame <= b.frame_index
        return corridor(a, b, frame, *args, **kwargs)
    monkeypatch.setattr(streaming, '_corridor', bounded)
    empty = []
    for key in masks:
        value = masks[key]
        if key.startswith(('edge_write:', 'edge_contract:')):
            kind, edge_id, frame = key.split(':')
            a, b = masks.edges[edge_id]
            inside = a.frame_index < int(frame) < b.frame_index if kind == 'edge_write' else a.frame_index <= int(frame) <= b.frame_index
            if not inside:
                empty.append(value)
    assert len(empty) > 1 and all(value is empty[0] for value in empty)
    assert not empty[0].any()
    with pytest.raises(ValueError):
        empty[0].flags.writeable = True


def test_peak_accounts_for_large_role_inventory_without_edge_frame_rasters():
    pixels, frames, edges, observations = 4_194_304, 1000, 128, 512
    charge = streaming_sam_contract_peak_bytes(pixels, frames, edges, observations)
    eager = pixels * (2 * frames * (5 + 2 * edges) + 4 * observations + 16)
    assert 96 * pixels < charge < 1024**3 < eager
    assert streaming_sam_contract_peak_bytes(1, frames * 1000, edges, observations) > 16 * 1024**3


@pytest.mark.parametrize('key', ['missing', 'write:01', 'write:-100', 'endpoint:missing', 'edge_write:missing:1'])
def test_unknown_role_keys_raise_keyerror(key):
    _source, _plan, recipe = fixture('simple')
    _, masks = stream_sam_group_contracts(recipe, 0.)
    with pytest.raises(KeyError):
        masks[key]

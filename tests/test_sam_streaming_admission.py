"""Production streaming preserves work, byte ownership and explicit bounds."""
from dataclasses import replace
import gc
import weakref

import numpy as np
import pytest

from XTA.sam_bridge_planning import (SamPlanningLimits, _contract_materialization,
    expand_sam_bridge_group_context, plan_sam_bridges)
from XTA.sam_evidence import SamEvidenceWriter
from XTA.sam_interpolation import _write_group


def planned(**overrides):
    source = np.zeros((16, 101, 181), np.uint8)
    source[1, 30:35, 49:54] = 1
    source[14, 31:36, 50:55] = 1
    options = dict(interpolation_distance=15, interpolation_walk_back=0,
                   interpolation_min_radius=0, lazy_contracts=True, _stream_contracts=True)
    options.update(overrides)
    plan = plan_sam_bridges(source, **options)
    return source, plan


def test_native_stream_writer_holds_credit_until_drain(tmp_path, monkeypatch):
    source, plan = planned()
    group, = plan.groups
    assert group.crop_contract['contract_materialization'] == 'streamed_planes'
    original = SamEvidenceWriter.add_group
    seen = []

    def checked(writer, metadata, masks):
        assert plan.contract_lease_budget.live_bytes == group.crop_contract['charged_contract_bytes']
        seen.append(weakref.ref(masks))
        original(writer, metadata, masks)
        assert not writer._packing_pending
        assert plan.contract_lease_budget.live_bytes > 0

    monkeypatch.setattr(SamEvidenceWriter, 'add_group', checked)
    with SamEvidenceWriter(tmp_path / 'evidence', {'shape_tyx': list(source.shape)}) as writer:
        _write_group(writer, group, plan.by_id, 0.)
        bundle = writer.commit()
    assert plan.contract_lease_budget.live_bytes == 0
    assert seen[0]() is None
    assert bundle.groups[group.group_id]['complete']


def test_stream_failure_does_not_retain_uncharged_planes_in_traceback(tmp_path, monkeypatch):
    source, plan = planned()
    group, = plan.groups
    owners = []

    def failing(writer, metadata, masks):
        plane = masks[f'acceptance:{group.frame_indices[2]}']
        owners.append(weakref.ref(plane))
        raise OSError('injected native evidence failure')

    monkeypatch.setattr(SamEvidenceWriter, 'add_group', failing)
    with SamEvidenceWriter(tmp_path / 'failure', {'shape_tyx': list(source.shape)}) as writer:
        with pytest.raises(OSError, match='injected') as caught:
            _write_group(writer, group, plan.by_id, 0.)
    gc.collect()
    assert caught.value.__traceback__ is not None
    assert owners[0]() is None
    assert plan.contract_lease_budget.snapshot()['live_bytes'] == 0


def test_custom_writer_keeps_eager_borrowed_plane_credit():
    _source, plan = planned()
    group, = plan.groups
    class Recorder:
        def add_group(self, metadata, masks):
            self.retained = masks[f'acceptance:{group.frame_indices[2]}']
    recorder = Recorder()
    _write_group(recorder, group, plan.by_id, 0.)
    gc.collect()
    assert plan.contract_lease_budget.live_bytes > 0
    recorder.retained = None
    gc.collect()
    assert plan.contract_lease_budget.live_bytes == 0


def test_streaming_does_not_raise_peak_for_simple_short_family():
    materialization = _contract_materialization(10000, 3, 1, 2, 2, True)
    assert materialization['contract_materialization'] == 'dense'
    assert materialization['charged_contract_bytes'] == 66 * 10000


def test_metadata_refusal_precedes_evidence_and_is_not_empty_success():
    _source, original = planned()
    assert original.contract_metadata_bytes > 1
    _source, refused = planned(limits=SamPlanningLimits(max_contract_metadata_bytes=1))
    assert refused.status == 'unresolved'
    assert refused.reasons == ('contract_metadata_memory_limit',)
    assert not refused.groups and not refused.runs
    assert refused.contract_metadata_bytes == original.contract_metadata_bytes
    assert len(refused.observations) == len(original.observations)


def test_resource_storage_change_preserves_existing_group_and_run_ids():
    legacy = SamPlanningLimits(max_endpoints_per_group=64,
        max_observations_per_group=512, max_edges_per_group=128)
    _source, old = planned(limits=legacy, _stream_contracts=False)
    _source, new = planned()
    assert old.planning_fingerprint != new.planning_fingerprint
    assert [group.group_id for group in old.groups] == [group.group_id for group in new.groups]
    assert [run.run_id for run in old.runs] == [run.run_id for run in new.runs]
    assert new.groups[0].crop_contract['selection_priority'] == (0, old.groups[0].group_id)


def test_tiled_metadata_includes_child_packets_before_tracker_admission():
    from XTA.geometry import get_view_infos
    from XTA.sam_interpolation import prepare_sam_interpolation_pass, SamInterpolationInfrastructureError
    source, plan = planned()
    view = get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
    with pytest.raises(SamInterpolationInfrastructureError, match='metadata budget'):
        prepare_sam_interpolation_pass(source, view=view, scope='tiled-metadata',
            gap_distance=15, min_radius=0., search_angle_deg=0., interpolation_walk_back=0,
            interpolation_candidates=1, interpolation_passes=1, crop_mode='tiled',
            planner_limits=SamPlanningLimits(max_contract_metadata_bytes=plan.contract_metadata_bytes + 1))


def test_retry_recomputes_nonproportional_stream_metadata_charge():
    source, plan = planned()
    group, = plan.groups
    expanded = expand_sam_bridge_group_context(group, (0, 0, *source.shape[1:]),
        max_crop_pixels=source.shape[1] * source.shape[2])
    expected = _contract_materialization(source.shape[1] * source.shape[2],
        len(group.frame_indices), len(group.edges), len(group.endpoint_ids),
        len(group.observation_ids), True)
    for key, value in expected.items():
        assert expanded.crop_contract[key] == value
    assert expanded.crop_contract['legacy_raster_origin_yx'] == group.crop_contract['legacy_raster_origin_yx']
    assert expanded.crop_contract['selection_priority'] == (0, expanded.group_id)


def test_complete_large_family_retained_with_explicit_lower_caps_respected():
    source = np.zeros((9, 512, 512), np.uint8)
    for row in range(10):
        for column in range(13):
            source[[0, 1, 7, 8], 200 + 4 * row, 200 + 4 * column] = 1
    bounds = SamPlanningLimits(context_margin_px=128)
    options = dict(interpolation_distance=6, interpolation_candidates=1,
        interpolation_walk_back=1, interpolation_min_radius=0,
        interpolation_search_angle=60, lazy_contracts=True, _stream_contracts=True)
    plan = plan_sam_bridges(source, limits=bounds, **options)
    group, = plan.groups
    assert group.status == 'planned'
    assert len(group.observation_ids) == 520 and len(group.edges) == 130
    assert len(group.endpoint_ids) == 260 and len(plan.runs) == 520
    assert group.crop_contract['selection_priority'] == (1, group.group_id)
    limited = plan_sam_bridges(source, limits=replace(bounds,
        max_endpoints_per_group=64, max_edges_per_group=128,
        max_observations_per_group=512), **options)
    assert limited.status == 'unresolved' and not limited.runs
    assert 'family_endpoint_limit' in limited.groups[0].reasons

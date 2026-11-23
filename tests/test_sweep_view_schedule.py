"""Detector ranges become immutable family products before later branches start."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from XTA import sweep_views
from XTA.sweep_view_schedule import SweepViewSchedule


def schedule(monkeypatch, tmp_path, *, role, interpolate=False, overrides=None, reconciliation_mode=None):
    for name, value in (
        ('XTA_YOLO_SWEEP_NATIVE_ROLE', role),
        ('XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS', '50,60'),
        ('XTA_YOLO_SWEEP_OUTPUT_ROOT', str(tmp_path)),
        ('XTA_YOLO_SWEEP_ID', 'fixture'),
    ):
        monkeypatch.setenv(name, value)
    values = dict(augmentation_ratio=1, min_conf=0, min_radius=0, enable_2D_hole_fill=[],
                  gpu_batch=1, cpu_batch=1, capture_component_replay=None)
    values.update(overrides or {})
    return SweepViewSchedule(SimpleNamespace(**values), cutoff=50 if role == 'produce' else 60,
        tiles=[], save_options={'nrrd'}, gpu=False, cpu=True,
        interpolation=SimpleNamespace(enabled=interpolate, extrapolation_enabled=False),
        reconciliation_mode=reconciliation_mode)


@pytest.mark.parametrize('interpolate, projection_supported', ((True, False), (False, True), (False, False)))
def test_completed_view_is_published_or_retained_for_independent_suffix(
        monkeypatch, tmp_path, interpolate, projection_supported):
    plan = schedule(monkeypatch, tmp_path, role='produce', interpolate=interpolate)
    view = SimpleNamespace(name='transverse', num_slices=2)
    receipt = {'status': 'complete', 'counters': {'rendered_frames': 2, 'forward_calls': 2}}
    plan.record_task('model', view, {'sweep_native_fanout': receipt})
    decision = dict(status='ready', prediction_counts={50: 2, 60: 1}, stored_bytes=75)
    seal = Mock(return_value=decision)
    retire = Mock()
    monkeypatch.setattr(sweep_views, 'seal_view_family', seal)
    monkeypatch.setattr(sweep_views, 'retire_view_payloads', retire)
    reader = SimpleNamespace(read_mask_plane=lambda z: np.full((2, 2), z, np.uint8))
    monkeypatch.setattr(sweep_views, 'open_view', lambda *args, **kwargs: nullcontext(reader))

    def publish(_view, *, read_mask_plane, **kwargs):
        np.testing.assert_array_equal(read_mask_plane(60, 1), np.ones((2,2)))
        return {60: 'ref'} if projection_supported else None

    publication = Mock()
    publication.publish_view_family.side_effect = publish
    plan.producer_view_ready('model', view, (2, 2, 2), publication)
    plan.close()
    seal.assert_called_once_with('model','transverse',(2,2,2),(50,60),receipts=[receipt],identity=None)
    assert publication.publish_view_family.call_count == int(not interpolate)
    assert retire.call_count == int(projection_supported)
    assert plan.summary(1)['produced_views'] == 1


def test_unsupported_filter_retains_scalar_path(monkeypatch, tmp_path):
    plan = schedule(monkeypatch, tmp_path, role='consume', overrides={'min_radius': 3})
    assert plan.consumer_source('model', SimpleNamespace(name='v'), (1,2,3), None) is None
    summary = plan.summary(4)
    assert summary['consumed_views'] == 0 and summary['scalar_views'] == 4
    assert summary['fallback_reasons']


def test_confidence_reconciliation_shares_detector_but_retains_source_evidence_tail(monkeypatch, tmp_path):
    plan = schedule(monkeypatch,tmp_path,role='produce',reconciliation_mode='confidence')
    assert plan.producer and not plan.eager_projection


def test_floor_marker_and_sparse_work_counters_aggregate_across_tasks(monkeypatch, tmp_path):
    plan = schedule(monkeypatch, tmp_path, role='produce')
    view = SimpleNamespace(name='view')
    counters = dict(scalar_owned_floor_frames=4, floor_count_d2h_bytes=32, native_packet_aliases=3,
                    bulk_empty_frames=12, empty_range_writes=2)
    for _ in range(2):
        plan.record_task('model', view, {'sweep_native_fanout': {'counters': counters}})
    assert plan.summary(1)['work'] == {key: 2 * value for key, value in counters.items()}


def test_previously_projected_view_requires_durable_publication_receipt(monkeypatch, tmp_path):
    plan = schedule(monkeypatch, tmp_path, role='consume')
    monkeypatch.setattr(sweep_views, 'view_family_status', lambda *args, **kwargs: 'projected')
    publication = Mock()
    publication.has_published_view.return_value = False
    with pytest.raises(sweep_views.SweepViewIntegrityError, match='lost its publication receipt'):
        plan.consumer_source('model', SimpleNamespace(name='transverse'), (3, 4, 5), publication)


@pytest.mark.parametrize('kinds, eligible', (
    (['projected', 'projected'], True),
    (['projected', 'native'], False),
    (['native', 'projected'], False),
    (['projected', None], False),
))
def test_skip_warmup_only_for_fully_prepublished_consumer(monkeypatch,tmp_path,kinds,eligible):
    plan = schedule(monkeypatch,tmp_path,role='consume')
    views = [SimpleNamespace(name=str(i), augmentation_pass=0) for i in range(len(kinds))]
    monkeypatch.setattr(plan,'consumer_source',lambda m,v,s,p:kinds[int(v.name)])
    scheduled, stored, fully_projected = plan.partition('model', views, lambda view:(2,3,4), None)
    assert fully_projected is eligible
    assert len(stored) + len(scheduled) == len(views)
    assert [source for _, source in stored] == [source for source in ('projected', 'native')
                                               for kind in kinds if kind == source]

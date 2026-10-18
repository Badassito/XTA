"""Persistent SAM metadata uses the identified fixed base, never CPU-wave credit."""
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import sam_resources as resources
from XTA.interpolation import _ByteAdmissionPool
from tests.test_sam_view_image_cache import context_for


GIB = resources.GIB
MIB = 1024**2


@pytest.mark.parametrize('headroom', (4*GIB, 128*GIB))
def test_fixed_base_partition_adds_no_parent_or_cpu_wave_credit(headroom):
    pool = _ByteAdmissionPool(64*GIB, 'metadata')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'parent', worker_count=1,
            base_allowance_bytes=4*GIB, headroom_probe=lambda: headroom) as profile:
        record = resources.validate_live_sam_resource_profile(profile)
        assert profile.assigned_observation_bytes == 512*MIB
        assert profile.assigned_contract_metadata_bytes == 512*MIB
        assert profile.base_other_non_cpu_allowance_bytes == GIB
        assert record['persistent_metadata_allowance_status'] == 'identified_non_cpu_base'
        assert (profile.assigned_observation_bytes+profile.assigned_contract_metadata_bytes
                +profile.base_other_non_cpu_allowance_bytes) == profile.non_cpu_base_allowance_bytes == 2*GIB
        assert profile.nominal_cpu_wave_base_bytes == profile.assigned_cpu_wave_base_bytes == 2*GIB
        assert profile.assigned_cpu_wave_bytes == 2*GIB+profile.reserved_extra_bytes
        assert pool.in_use == 4*GIB+profile.reserved_extra_bytes
        for name in ('assigned_observation_bytes', 'assigned_contract_metadata_bytes',
                     'base_other_non_cpu_allowance_bytes'):
            assert record[name] == record['effective_budgets'][name] == getattr(profile, name)
    assert pool.in_use == 0


@pytest.mark.parametrize('base,allowance,capacity,expected', (
    (4*GIB, 0, 64*GIB, 0),
    (GIB, 4*GIB, 64*GIB, 256*MIB),
    (4*GIB, 512*MIB, 64*GIB, 128*MIB),
    (10*GIB, 4*GIB, GIB, 256*MIB),
    (1, 1, 64*GIB, 0),
))
def test_allowances_follow_only_the_identified_charged_base(base, allowance, capacity, expected):
    pool = _ByteAdmissionPool(capacity, 'partial-metadata')
    with resources.admit_sam_parent_resources(pool, base, 'parent',
            base_allowance_bytes=allowance, headroom_probe=lambda:128*GIB) as profile:
        record = profile.metadata()
        assert profile.assigned_observation_bytes == profile.assigned_contract_metadata_bytes == expected
        assert expected*2+profile.base_other_non_cpu_allowance_bytes == profile.non_cpu_base_allowance_bytes
        assert profile.non_cpu_base_allowance_bytes <= min(base, allowance, capacity)
        assert record['persistent_metadata_allowance_status'] == (
            'identified_non_cpu_base' if allowance else 'no_identified_non_cpu_base')
        if not allowance:
            assert profile.has_extra_credit and profile.base_other_non_cpu_allowance_bytes == 0


def test_metadata_budget_changes_effective_identity_and_saved_receipts_grant_nothing():
    pool = _ByteAdmissionPool(64*GIB, 'metadata-identity')
    records = []
    for allowance in (0, GIB):
        with resources.admit_sam_parent_resources(pool, GIB, 'parent',
                base_allowance_bytes=allowance, headroom_probe=lambda:128*GIB) as profile:
            records.append(profile.metadata())
            with pytest.raises(TypeError, match='live SamResourceProfile'):
                resources.validate_live_sam_resource_profile(records[-1])
            with pytest.raises(RuntimeError, match='expired|another'):
                resources.validate_live_sam_resource_profile(replace(profile))
    assert records[0]['assigned_cpu_wave_bytes'] == records[1]['assigned_cpu_wave_bytes']
    assert records[0]['reserved_extra_bytes'] == records[1]['reserved_extra_bytes']
    assert records[0]['profile_id'] != records[1]['profile_id']
    with pytest.raises(RuntimeError, match='expired'):
        resources.validate_live_sam_resource_profile(profile)


@pytest.mark.parametrize('allowance', (0, 4*GIB))
def test_image_build_credit_cannot_spend_observation_or_contract_metadata(tmp_path, allowance):
    context = context_for(tmp_path, np.zeros((3, 7, 9), np.uint8))
    pool = _ByteAdmissionPool(64*GIB, 'image-partition')
    try:
        with resources.admit_sam_parent_resources(pool, 4*GIB, 'parent',
                base_allowance_bytes=allowance, headroom_probe=lambda:128*GIB) as profile:
            context._resource_local.profile = profile
            with context._idle:
                owned = context._admit_image_build(1, 0, 2*GIB)
                if allowance:
                    with pytest.raises(RuntimeError, match='separate image-only grant'):
                        context._admit_image_build(GIB+1, 0, 2*GIB)
                else:
                    oversized = context._admit_image_build(GIB+1, 0, 2*GIB)
            assert owned['credited'] is bool(allowance)
            assert owned['lease_id'] == (profile._lease.lease_id if allowance else None)
            if not allowance:
                assert not oversized['credited'] and oversized['lease_id'] is None
                assert oversized['fallback_limit'] == 2*GIB
            assert pool.in_use == profile.base_charged_bytes+profile.reserved_extra_bytes
    finally:
        context._resource_local.profile = None
        context.close()


def test_gpu_image_admission_rejects_metadata_partition_before_source_or_cuda_access(monkeypatch):
    from XTA.sam_gpu_rendering import try_gpu_crop_renderer
    monkeypatch.setenv('YOLO_TTA_SAM_GPU_IMAGES', '1')
    monkeypatch.setenv('YOLO_TTA_SAM_RENDER_MAX_BYTES', '1')
    telemetry = mock.Mock()
    monkeypatch.setattr('XTA.runtime.runtime_telemetry', lambda:telemetry)
    pool = _ByteAdmissionPool(64*GIB, 'image-admission')
    with resources.admit_sam_parent_resources(pool, 4*GIB, 'parent',
            base_allowance_bytes=4*GIB, headroom_probe=lambda:128*GIB) as profile:
        context = SimpleNamespace(detector_retirement_ready=True,
            _resource_local=SimpleNamespace(profile=profile))
        # Metadata-only demand: no corresponding image/source buffer is allocated.
        plan = SimpleNamespace(frame_crop_bounds={frame:(0, 0, 2048, 2048) for frame in range(257)})
        assert try_gpu_crop_renderer(context, SimpleNamespace(family='radial'),
                                    (257, 2048, 2048), plan) is None
        telemetry.gauge.assert_called_with('sam.gpu_images.last_cpu_reason',
                                          'insufficient_parent_image_credit')


def test_separately_funded_image_profile_spends_its_entire_grant_without_metadata_reserves():
    from XTA.sam_image_prefetch import _try_image_profile
    pool = _ByteAdmissionPool(64*GIB, 'separate-image-credit')
    with _try_image_profile(pool, 256*MIB, 'parent/images', lambda:128*GIB) as profile:
        assert profile is not None
        record = resources.validate_live_sam_resource_profile(profile)
        assert record['status'] == 'image_only_admission'
        assert record['persistent_metadata_allowance_status'] == 'image_only_no_persistent_metadata'
        assert record['assigned_observation_bytes'] == record['assigned_contract_metadata_bytes'] == 0
        assert record['base_other_non_cpu_allowance_bytes'] == 256*MIB
        assert profile.assigned_cpu_wave_bytes == 0
        assert pool.in_use == 0
        assert resources.sam_image_staging_snapshot(pool)['image_staging_in_use_bytes'] == 256*MIB
    assert resources.sam_parent_promised_bytes(pool) == 0

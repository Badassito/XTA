"""Cyclic SAM geometry and original-only planning; no model or GPU."""
from __future__ import annotations

from types import SimpleNamespace
from dataclasses import replace

import numpy as np
import pytest

from XTA.sam_bridge_planning import SamPlanningLimits, plan_sam_bridges
from XTA.sam_cyclic import (CyclicFrameAddresses, CyclicObservationVolume,
    address_for_unfolded_index, build_cyclic_frame_addressing, mirror_bbox_yx,
    transform_crop_between_frame_addresses, validate_cyclic_frame_addressing)
from XTA.sam_interpolation import interpolate_sam_view_volume_pass, selected_sam_plane
from XTA.sam_interpolation import prepare_sam_interpolation_pass


def _source():
    volume = np.zeros((8, 15, 27), np.uint8)
    volume[7, 6:9, 4:7] = 1
    volume[1, 6:9, 20:23] = 1
    return volume


def _plan(volume, **kwargs):
    settings = dict(interpolation_distance=3, interpolation_candidates=1,
                    interpolation_walk_back=0, interpolation_min_radius=0,
                    interpolation_search_angle=0, scope_id="cyclic_test")
    settings.update(kwargs)
    return plan_sam_bridges(volume, **settings)


@pytest.mark.parametrize("index,native,cycle,mirror", [(-1,7,-1,True),(0,0,0,False),
    (7,7,0,False),(8,0,1,True),(9,1,1,True),(16,0,2,False)])
def test_signed_half_turn_addresses(index, native, cycle, mirror):
    address = address_for_unfolded_index(index,8)
    assert tuple(address.values()) == (index,native,cycle,mirror)
    assert not address_for_unfolded_index(index,8,period_degrees=360)["mirror_u"]


def test_non_square_crop_mirror_and_neighbor_parity_xor_are_exact():
    mask = np.arange(15).reshape(3,5)
    bbox=(2,3,5,8)
    assert mirror_bbox_yx(bbox,27) == (2,19,5,24)
    primary=address_for_unfolded_index(0,8)
    alias=address_for_unfolded_index(8,8)
    flipped, transformed=transform_crop_between_frame_addresses(mask,bbox,primary,alias,27)
    np.testing.assert_array_equal(flipped,mask[:,::-1])
    assert transformed == (2,19,5,24)
    restored, restored_bbox=transform_crop_between_frame_addresses(flipped,transformed,alias,primary,27)
    np.testing.assert_array_equal(restored,mask)
    assert restored_bbox == bbox
    # The predecessor of primary 0 is -1, a mirrored occurrence of native7.
    b=address_for_unfolded_index(7,8)
    target=address_for_unfolded_index(-1,8)
    _, mirrored=transform_crop_between_frame_addresses(mask,bbox,b,target,27)
    assert mirrored != bbox


def test_alias_adapter_is_lazy_and_never_coerces_original_or_extended_volume():
    source=_source()
    class Trap:
        shape,dtype=source.shape,source.dtype
        def __array__(self,*args,**kwargs):
            raise AssertionError("No dense original coercion")
        def __getitem__(self,index):
            return source[index]
    adapted=CyclicObservationVolume(Trap(),3)
    assert adapted.shape == (11,15,27)
    assert isinstance(adapted.frame_addresses,CyclicFrameAddresses)
    np.testing.assert_array_equal(adapted[9],source[1,:,::-1])
    with pytest.raises(RuntimeError,match="slice-only"):
        np.asarray(adapted)
    huge=build_cyclic_frame_addressing((100_000_000,3,5),3)
    assert "addresses" not in huge
    addresses=validate_cyclic_frame_addressing(huge)
    assert isinstance(addresses,CyclicFrameAddresses)
    assert addresses[100_000_001]["native_index"] == 1
    assert _plan(Trap(),wrap_axis=True).runs


def test_seam_gap_generates_both_directions_from_original_seeds_without_alias_duplicates():
    source=_source()
    linear=_plan(source)
    cyclic=_plan(source,wrap_axis=True)
    assert not linear.runs
    assert cyclic.virtual_shape_tyx == (11,15,27)
    assert cyclic.native_shape_tyx == source.shape
    assert len(cyclic.groups) == 1 and len(cyclic.groups[0].edges) == 1
    assert {run.expected_frames for run in cyclic.runs} == {(7,8,9),(9,8,7)}
    originals={obs.native_frame_index:obs for obs in linear.observations}
    for observed in cyclic.observations:
        assert observed.original_observation_id == originals[observed.native_frame_index].observation_id
        if observed.frame_index < 8:
            assert observed.observation_id == observed.original_observation_id
        else:
            assert observed.observation_id != observed.original_observation_id
            assert observed.mirror_u
    group=cyclic.groups[0]
    assert len(group.frame_addresses) == 3
    validate_cyclic_frame_addressing(group.frame_addressing,expected_frames=group.frame_indices)
    assert all(min(cyclic.by_id[edge.source_id].frame_index,cyclic.by_id[edge.target_id].frame_index)<8
               for edge in group.edges)
    assert "addresses" not in cyclic.frame_addressing


def test_canonical_labels_survive_mirrored_detector_aliases_and_distance_stays_bounded():
    source=_source()
    labels=np.zeros_like(source,dtype=np.uint16)
    labels[7]=source[7]*22
    labels[1]=source[1]*11
    plan=_plan(source,wrap_axis=True,canonical_labels=labels,interpolation_distance=1000)
    assert plan.virtual_shape_tyx[0] == 15
    assert {obs.canonical_label for obs in plan.observations if obs.native_frame_index==1} == {11}
    assert {obs.canonical_label for obs in plan.observations if obs.native_frame_index==7} == {22}
    assert all(abs(plan.by_id[edge.source_id].frame_index-plan.by_id[edge.target_id].frame_index)<8
               for group in plan.groups for edge in group.edges)


def test_cyclic_walkback_uses_original_continuations_and_extends_only_bounded_alias_prefix():
    source=_source()
    source[6]=source[7]
    source[2]=source[1]
    plan=_plan(source,wrap_axis=True,interpolation_walk_back=1)
    assert plan.virtual_shape_tyx == (12,15,27)
    assert {run.expected_frames for run in plan.runs} == {(7,8,9),(6,7,8,9),(9,8,7),(10,9,8,7)}
    assert sorted(run.walk_back_index for run in plan.runs) == [0,0,1,1]
    for run in plan.runs:
        seed=plan.by_id[run.seed_ids[0]]
        assert seed.lineage["observation_source"] == "detector"
        assert seed.original_observation_id
    assert all(plan.by_id[edge.source_id].original_observation_id != plan.by_id[edge.target_id].original_observation_id
               for group in plan.groups for edge in group.edges)


def test_corrupted_closure_and_unbounded_address_records_are_explicit():
    metadata=dict(build_cyclic_frame_addressing((8,15,27),3))
    metadata["addresses"]={7:dict(address_for_unfolded_index(7,8)),8:dict(address_for_unfolded_index(8,8))}
    validate_cyclic_frame_addressing(metadata,expected_frames=(7,8))
    metadata["addresses"][8]["mirror_u"]=False
    with pytest.raises(ValueError,match="closure"):
        validate_cyclic_frame_addressing(metadata,expected_frames=(7,8))
    plan=_plan(_source(),wrap_axis=True,limits=SamPlanningLimits(max_frame_address_records=2))
    assert plan.status == "unresolved" and not plan.runs
    assert "cyclic_frame_address_record_limit" in plan.reasons


def test_cyclic_source_change_after_import_is_rejected_at_addressed_reads(tmp_path,monkeypatch):
    from XTA import sam_cyclic
    original=sam_cyclic._SOURCE_PATH
    changed=tmp_path/"changed_cyclic.py"
    changed.write_bytes(original.read_bytes()+b"\n# changed after import\n")
    monkeypatch.setattr(sam_cyclic,"_SOURCE_PATH",changed)
    with pytest.raises(RuntimeError,match="changed after loading"):
        address_for_unfolded_index(8,8)
    with pytest.raises(RuntimeError,match="changed after loading"):
        validate_cyclic_frame_addressing({})


def test_same_name_shape_changed_sampler_cannot_reuse_a_prepared_detector_snapshot(tmp_path):
    from XTA.geometry import ViewInfo
    view=ViewInfo(name="azimuthal_transverse",physical_view_name="azimuthal_transverse",
        family="azimuthal",num_slices=8,src_h=15,src_w=27,pad_mode="pad",
        azimuths_deg=tuple(22.5*i for i in range(8)),diameter=27,center_x=13,center_y=13,roi_radius=13,
        full_t=15,full_h=27,full_w=27,azimuthal_base_view="transverse")
    source=_source()
    settings=dict(gap_distance=3,min_radius=0,search_angle_deg=0,interpolation_walk_back=0,wrap_axis=True,
                  scope={"scope_id":"view_recipe_test"})
    prepared=prepare_sam_interpolation_pass(source,view=view,**settings)
    changed=replace(view,center_x=14)
    tracker=SimpleNamespace(run=lambda **_kwargs:pytest.fail("Stale view plan must fail before tracking"))
    with pytest.raises(ValueError,match="Prepared SAM plan differs"):
        interpolate_sam_view_volume_pass(source,view=changed,prepared_plan=prepared,runtime=tracker,
                                         work_dir=tmp_path,**settings)
    assert not list(tmp_path.iterdir())


def test_declared_canvas_sampler_change_invalidates_lightweight_prepared_views(tmp_path):
    source=_source()
    options=dict(gap_distance=3,min_radius=0,search_angle_deg=0,interpolation_walk_back=0,wrap_axis=True)
    scope={"scope_id":"canvas_recipe_test","canvas_transform":{"sampler_recipe":{"center_x":13}}}
    prepared=prepare_sam_interpolation_pass(source,scope=scope,**options)
    changed={"scope_id":"canvas_recipe_test","canvas_transform":{"sampler_recipe":{"center_x":14}}}
    with pytest.raises(ValueError,match="Prepared SAM plan differs"):
        interpolate_sam_view_volume_pass(source,scope=changed,prepared_plan=prepared,
                                         runtime=SimpleNamespace(),work_dir=tmp_path,**options)


def test_infinite_sampling_diagnostics_do_not_change_prepared_sampler_identity():
    from XTA.geometry import ViewInfo
    view=ViewInfo(name="azimuthal_transverse",physical_view_name="azimuthal_transverse",
        family="azimuthal",num_slices=8,src_h=15,src_w=27,pad_mode="pad",
        azimuths_deg=tuple(22.5*i for i in range(8)),diameter=27,center_x=13,center_y=13,roi_radius=13,
        full_t=15,full_h=27,full_w=27,azimuthal_base_view="transverse",sampling_error_bound_sq=float("inf"))
    options=dict(gap_distance=3,min_radius=0,search_angle_deg=0,interpolation_walk_back=0,wrap_axis=True,
                 scope="diagnostic_identity_test")
    first=prepare_sam_interpolation_pass(_source(),view=view,**options)
    second=prepare_sam_interpolation_pass(_source(),view=replace(view,sampling_error_bound_sq=0,
        sampling_certificate="different diagnostic",sampling_reason="different diagnostic"),**options)
    assert first.settings_sha256 == second.settings_sha256


def test_changed_cyclic_implementation_identity_invalidates_cached_prepared_plan(tmp_path,monkeypatch):
    from XTA import sam_cyclic
    source=_source()
    settings=dict(gap_distance=3,min_radius=0,search_angle_deg=0,interpolation_walk_back=0,wrap_axis=True,
                  scope="cyclic_implementation_identity")
    prepared=prepare_sam_interpolation_pass(source,**settings)
    monkeypatch.setattr(sam_cyclic,"IMPLEMENTATION_SHA256","different implementation identity")
    with pytest.raises(ValueError,match="Prepared SAM plan differs"):
        interpolate_sam_view_volume_pass(source,prepared_plan=prepared,runtime=SimpleNamespace(),
                                         work_dir=tmp_path,**settings)
    assert not list(tmp_path.iterdir())


def test_selected_publication_folds_every_retained_alias_with_exact_bbox():
    header=dict(build_cyclic_frame_addressing((8,15,27),3))
    header["addresses"]={8:dict(address_for_unfolded_index(8,8))}
    raw=np.ones((3,3),bool)
    bundle=SimpleNamespace(groups={"g":dict(group_id="g",context_bbox_yx=(6,4,9,7),
        frame_indices=[8],frame_addressing=header)},runs={"r":dict(group_id="g",direction="forward",expected_frames=[8])},
        candidate_mask=lambda _run,_frame:raw,raw_mask=lambda _run,_frame:raw)
    plane=selected_sam_plane(bundle,{"selected_run_ids":["r"]},0,(15,27))
    assert plane.sum()==9 and plane[6:9,20:23].all()
    assert not plane[6:9,4:7].any()


def test_primary_and_alias_owners_fold_without_duplicate_pixels_or_subtracting_shared_support():
    header=dict(build_cyclic_frame_addressing((8,15,27),3))
    groups={}
    for frame,bbox in ((0,(6,20,9,23)),(8,(6,4,9,7))):
        metadata={**header,"addresses":{frame:dict(address_for_unfolded_index(frame,8))}}
        groups[str(frame)]=dict(group_id=str(frame),context_bbox_yx=bbox,frame_indices=[frame],frame_addressing=metadata)
    raw=np.ones((3,3),bool)
    bundle=SimpleNamespace(groups=groups,runs={str(frame):dict(group_id=str(frame),direction="forward",expected_frames=[frame])
                                              for frame in (0,8)},candidate_mask=lambda _run,_frame:raw,raw_mask=lambda _run,_frame:raw)
    both=selected_sam_plane(bundle,{"selected_run_ids":["0","8"]},0,(15,27))
    retained=selected_sam_plane(bundle,{"selected_run_ids":["8"]},0,(15,27))
    np.testing.assert_array_equal(both,retained)
    assert both.sum()==9


@pytest.mark.parametrize("crop_mode", ["whole", "tiled"])
def test_controlled_seam_generator_publishes_only_native_missing_slice(tmp_path,crop_mode):
    class Tracker:
        def run(self,**request):
            frames=(range(request["seed_frame"],request["frame_stop"]) if request["direction"]=="forward"
                    else range(request["seed_frame"],request["frame_start"]-1,-1))
            return SimpleNamespace(frames={frame:request["seed_mask"].copy() for frame in frames},
                tracker_scores={},observation_status={},receipt={"prediction_valid":True,"coverage_complete":True})
    source=_source()
    merged,stats,components=interpolate_sam_view_volume_pass(source,work_dir=tmp_path,runtime=Tracker(),
        gap_distance=3,min_radius=0,search_angle_deg=0,interpolation_walk_back=0,wrap_axis=True,
        scope="cyclic_test",return_bridge_components=True,crop_mode=crop_mode)
    try:
        assert merged.shape==source.shape
        assert stats["added_voxels"]==9
        assert merged[0,6:9,20:23].all()
        np.testing.assert_array_equal(merged[1],source[1])
        np.testing.assert_array_equal(merged[7],source[7])
        assert len(components)==2
        assert stats["wrap_axis"] is True
        assert stats["sam_original_observation_count"]==2
        assert stats["sam_observation_alias_count"]==1
    finally:
        if isinstance(merged,np.memmap):
            merged._mmap.close()

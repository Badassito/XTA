"""Exercise the actual projected store, scalar sink, confidence writer and consumer together."""
from __future__ import annotations

import gzip
import hashlib
import json
from pathlib import Path
from unittest.mock import Mock
from zipfile import ZipFile

import numpy as np
import pytest

from XTA import outputs
from XTA.geometry import ViewInfo
from XTA.tta_sweep_view_publication import ViewMajorPublication, branch_run_id


CUTS = (50, 70, 100)
SHAPE = (3, 8, 8)


@pytest.fixture
def view():
    return ViewInfo(name="transverse__tta_a0", num_slices=3, src_h=8, src_w=8,
                    pad_mode="pad", family="orthogonal", physical_view_name="transverse",
                    full_t=3, full_h=8, full_w=8, tta_aug_id="a0")


def make_owner(root, cent, progress=None, *, retain_confidence=True, staged=False,
               native_formats=('nrrd',), mirror_formats=()):
    family, work = root/"family", root/"work"
    nrrd_stage_root = work / "nrrd_sources" if staged else None
    branch = (nrrd_stage_root if staged else family / "confidence_cutoffs") / f"conf_{cent:03d}"
    sink = outputs.NrrdLayerSink(nrrd_dir=branch/"nrrd", stem="case",
        output_shape_tyx=SHAPE, max_workers=2,
        low_quality_specs=[outputs.LowQualityDownbinSpec("0.5","half",.5,(2,4,4), formats=mirror_formats)],
        full_quality_formats=native_formats,
        low_quality_root=branch/"low_quality", run_id=branch_run_id("family-one",cent),
        relative_artifact_paths=not staged)
    owner = ViewMajorPublication(family_root=family, workspace=work, sweep_id="family-one",
        cutoffs_hundredths=CUTS,current_cutoff=cent,scalar_sink=sink,source_shape_tyx=SHAPE,
        model_name="detector",task="segment",min_conf=0.,confidence_floor_by_backend={"gpu":.5},
        progress=progress, retain_confidence=retain_confidence, nrrd_stage_root=nrrd_stage_root)
    return owner,sink


def publish(owner,view, *, d1=False, score_lanes=True):
    mask=np.zeros(SHAPE,np.uint8)
    mask[:,2:6,2:6]=1
    masks={70:mask,100:np.zeros_like(mask)}
    def sparse(cent,z):
        if cent==70:
            yield (2,6,2,6,masks[cent][z,2:6,2:6],np.full((4,4),214,np.uint8))
    def sparse_mask(cent,z):
        if cent==70:
            yield 2,6,2,6,masks[cent][z,2:6,2:6]
    def no_dense(*_):
        raise AssertionError("Sparse readers must avoid constructing full native input planes")
    return owner.publish_view_family(view,native_shape=SHAPE,read_mask_plane=no_dense,
        read_score_plane=no_dense,read_mask_crops=sparse_mask,
        read_native_crops=sparse if score_lanes else no_dense,
        counts_by_cutoff={50:9,70:6,100:0}, retain_empty_orthogonal=d1)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES","-1")
    monkeypatch.setenv("YOLO_TTA_NRRD_GPU_MIRROR_TEE","0")
    monkeypatch.setenv("YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN","0")
    from XTA import nrrd_compression
    monkeypatch.setattr(nrrd_compression,"_require_nrrd_member_codec",lambda **_:("zlib",1,gzip.compress))


@pytest.mark.parametrize("staged", [False, True])
def test_default_publication_projects_masks_without_opening_score_layers(tmp_path, view, monkeypatch, staged):
    from XTA import sweep_confidence, sweep_evidence_processes

    def unexpected(*args, **kwargs):
        raise AssertionError("An unrequested evidence writer was opened")

    monkeypatch.setenv("XTA_YOLO_SWEEP_NATIVE_ROLE", "produce")
    monkeypatch.setenv("XTA_YOLO_SWEEP_NRRD_PROCESSES", "0")
    monkeypatch.setenv("XTA_YOLO_SWEEP_EVIDENCE_PROCESSES", "1")
    monkeypatch.setattr(sweep_confidence, "SweepObservedConfidenceWriter", unexpected)
    monkeypatch.setattr(sweep_evidence_processes, "SweepEvidenceProcessPool", unexpected)
    owner, sink = make_owner(tmp_path, 50, retain_confidence=False, staged=staged)
    try:
        refs = publish(owner, view, score_lanes=False)
        assert refs[70] is not None and refs[100] is None
        assert owner._confidence_pool is None
        assert not list(owner.family_root.rglob("reconciliation_evidence"))
        if staged:
            assert len(list((owner.nrrd_stage_root / "conf_070").rglob("*.seg.nrrd"))) == 2
            assert not list(owner.family_root.rglob("*.seg.nrrd"))
    finally:
        owner.close()
        sink.shutdown()

    consumer, sink = make_owner(tmp_path, 70, retain_confidence=False, staged=staged)
    try:
        assert consumer.adopt_view(view) == (refs[70], 6)
    finally:
        consumer.close()
        sink.shutdown()


def test_real_sibling_outputs_visible_during_first_view_and_adopted_by_later_scalar(tmp_path,view,monkeypatch):
    from XTA import assembly
    from XTA.finalization import _union_projected_layer_refs_grouped_into_volume
    from XTA.reconciliation_policy import ReconciliationSettings
    from XTA.reconciliation_runtime import reconcile_tta_layers
    progress=Mock()
    producer,floor=make_owner(tmp_path,50,progress)
    original_files={}
    try:
        refs=publish(producer,view)
        assert refs[70] is not None and refs[100] is None
        assert progress.view_published.call_count==2
        for cent in CUTS[1:]:
            branch=tmp_path/"family/confidence_cutoffs"/f"conf_{cent:03d}"
            catalog=json.loads((branch/"reconciliation_evidence/manifest.json").read_text())
            assert catalog["run_id"]==branch_run_id("family-one",cent)
            assert len(catalog["layers"])==1
            native=list((branch/"nrrd").glob("*.seg.nrrd"))
            low=list((branch/"low_quality/half/nrrd").glob("*.seg.nrrd"))
            assert len(native)==len(low)==int(cent==70)
            for path in native+low:
                original_files[path]=(path.stat().st_mtime_ns,path.read_bytes())
        producer.finish_floor()
        storage = json.loads(producer.storage_path.read_text())
        assert storage["live_bytes"]==storage["peak_live_bytes"]>0
        assert storage["configured_cap_bytes"] is None
    finally:
        producer.close()
        floor.shutdown()

    for cent in CUTS[1:]:
        consumer,sink=make_owner(tmp_path,cent)
        try:
            assert consumer.has_published_view(view)
            restored, count=consumer.adopt_view(view)
            assert restored==refs[cent]
            assert count==(6 if cent==70 else 0)
            if restored is not None:
                assert restored.path.is_dir()
            # Feed exactly the same refs/assembler and union policy used by the scalar tail.
            adopted = [restored] if restored else []
            volume = np.zeros(SHAPE, np.uint8)
            _union_projected_layer_refs_grouped_into_volume(adopted, volume, workers=1)
            expected = np.zeros_like(volume)
            if cent == 70:
                expected[:,2:6,2:6] = 1
            np.testing.assert_array_equal(volume, expected)
            policy_path = Path(assembly.__file__).parent / "examples/external_reconciliation/union.py"
            policy = ReconciliationSettings(str(policy_path),hashlib.sha256(policy_path.read_bytes()).hexdigest(),4)
            result, report = reconcile_tta_layers(adopted, views=[view],source_shape_tyx=SHAPE,
                processing_shape_tyx=SHAPE,settings=policy,output_dir=sink.nrrd_dir.parent/"reconciliation",
                workspace=tmp_path/"tail",policy={"mode":"union"},assembled_union=volume)
            assert report["counts"]["retained_voxels"] == int(expected.sum())
            np.testing.assert_array_equal(result, expected)
            monkeypatch.setattr(assembly,"nrrd_layer_sink",lambda:sink)
            assembly.materialize_nrrd_global_layer(result,model_name="detector",source="global",
                mask_kind="union",stage="final_output_after_all_postprocessing",temp_dir=tmp_path/"tail",
                include_empty=True,volume_is_immutable=True)
            sink.wait()
            meta = json.loads(sink.write_manifest().read_text())
            assert meta["run_id"]==branch_run_id("family-one",cent)
            assert sum(row["stage"]=="final_output_after_all_postprocessing" for row in meta["layers"])==1
            for path, snapshot in original_files.items():
                assert (path.stat().st_mtime_ns,path.read_bytes())==snapshot
        finally:
            sink.shutdown()
            consumer.close()
        if refs[cent] is not None:
            assert not refs[cent].path.exists()
    storage = json.loads(producer.storage_path.read_text())
    assert storage["live_bytes"]==0 and storage["retired_bytes"]==storage["published_bytes"]


@pytest.mark.parametrize("partial_bundle", [False, True])
def test_legacy_receipt_adoption_and_partial_bundle_rejection(tmp_path, view, partial_bundle):
    producer, floor = make_owner(tmp_path, 50)
    try:
        ref = publish(producer, view)[70]
        (bundle_path,) = (producer.work / "published" / "by_view").glob("*.zip")
        with ZipFile(bundle_path) as bundle:
            assert bundle.namelist() == ["conf_070.json", "conf_100.json"]
            row = bundle.read("conf_070.json")
        with pytest.raises(RuntimeError, match="already exist"):
            publish(producer, view)
    finally:
        producer.close()
        floor.shutdown()

    legacy_path = producer.work / "published" / "conf_070" / f"{bundle_path.stem}.json"
    legacy_path.parent.mkdir()
    legacy_path.write_bytes(row)
    if partial_bundle:
        # A well-formed ZIP with the requested row cannot authorize a partial view.
        with ZipFile(bundle_path, "w") as bundle:
            bundle.writestr("conf_070.json", row)
    else:
        bundle_path.unlink()

    consumer, sink = make_owner(tmp_path, 70)
    try:
        assert consumer.has_published_view(view)
        if partial_bundle:
            with pytest.raises(ValueError, match="cutoff inventory"):
                consumer.adopt_view(view)
            assert sink.layer_count() == 0
        else:
            assert consumer.adopt_view(view) == (ref, 6)
    finally:
        sink.shutdown()
        consumer.close()


def test_changed_view_geometry_rejected_before_sink_adoption(tmp_path,view):
    from dataclasses import replace
    producer,floor=make_owner(tmp_path,50)
    try:
        publish(producer,view)
    finally:
        producer.close();floor.shutdown()
    consumer,sink=make_owner(tmp_path,70)
    try:
        with pytest.raises(ValueError,match="identity"):
            consumer.adopt_view(replace(view,center_x=4))
        assert sink.layer_count()==0
    finally:
        sink.shutdown();consumer.close()


def test_unsupported_view_leaves_no_sibling_files(tmp_path,view,monkeypatch):
    from XTA import assembly
    from XTA.sweep_backprojection import SweepProjectionUnsupported
    def unsupported(**kwargs):
        raise SweepProjectionUnsupported("not enough scalar-equivalent projection work memory")
    monkeypatch.setattr(assembly,"project_sweep_view",unsupported)
    producer,sink=make_owner(tmp_path,50)
    try:
        assert publish(producer,view) is None
        assert sum(producer.geometry_fallbacks.values())==1
        assert not list((tmp_path/"family/confidence_cutoffs/conf_070").rglob("*.seg.nrrd"))
        assert not list((tmp_path/"work/view_publication/published/by_view").glob("*.zip"))
    finally:
        producer.close();sink.shutdown()


def test_scalar_d1_empty_cartesian_slot_is_published_then_adopted_without_rewrite(tmp_path,view):
    producer, floor = make_owner(tmp_path,50)
    try:
        ref = publish(producer,view,d1=True)[100]
        assert ref is not None and ref.shape == SHAPE
        assert tuple(ref.segment_extent_ijk) == (0,-1,0,-1,0,-1)
    finally:
        producer.close(); floor.shutdown()
    consumer, sink = make_owner(tmp_path,100)
    try:
        files = list(consumer._branch_dir(100).rglob("*.seg.nrrd"))
        assert len(files) == 2
        before = {p:p.stat().st_mtime_ns for p in files}
        restored,count = consumer.adopt_view(view)
        assert restored == ref and count == 0
        manifest = json.loads(sink.write_manifest().read_text())
        row, = manifest["layers"]
        assert row["empty_segment"] and row["view_name"]==view.name
        # Cropped default: an empty layer stores one zero voxel at the grid origin.
        assert row["stored_shape_tyx"]==[1,1,1] and row["stored_offset_tyx"]==[0,0,0]
        assert row["output_shape_tyx"]==list(SHAPE)
        assert row["suffix"]=="Transverse_TTA_a0_fullframe_yolo"
        for p in files:
            assert p.stat().st_mtime_ns==before[p]
    finally:
        sink.shutdown(); consumer.close()


def test_opt_in_zero_projected_store_budget_falls_back_before_writing(tmp_path,view,monkeypatch):
    monkeypatch.setenv("XTA_YOLO_SWEEP_PROJECTED_GIB","0")
    producer, sink = make_owner(tmp_path, 50)
    try:
        assert publish(producer,view) is None
        assert not list((tmp_path/"family/confidence_cutoffs/conf_070").rglob("*.seg.nrrd"))
        counts=json.loads(producer.storage_path.read_text())
        assert counts["configured_cap_bytes"]==0 and counts["live_bytes"]==0
        assert sum(counts["fallbacks"].values())==1
    finally:
        producer.close(); sink.shutdown()



@pytest.mark.parametrize('native_formats,mirror_formats,worker_count,expect_pool', [
    (('nrrd', 'xtareview'), ('nrrd',), '2', False),
    (('nrrd',), ('xtareview',), '2', False),
    (('nrrd',), ('nrrd',), '2', True),
])
def test_view_major_process_pool_only_starts_with_eligible_sibling_layers(
        tmp_path, monkeypatch, native_formats, mirror_formats, worker_count, expect_pool):
    from XTA import sweep_nrrd_processes as processes

    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'produce')
    monkeypatch.setenv(processes.PROCESSES_ENV, worker_count)
    identity = Mock(return_value=('zlib', 3, 'codec'))
    construct = Mock()
    monkeypatch.setattr(processes, 'nrrd_codec_identity', identity)
    monkeypatch.setattr(processes, 'SweepNrrdProcessPool', construct)
    owner, sink = make_owner(tmp_path, 50, retain_confidence=False, native_formats=native_formats,
                             mirror_formats=mirror_formats)
    try:
        assert identity.called is expect_pool
        assert construct.called is expect_pool
        assert (owner._nrrd_processes is not None) is expect_pool
    finally:
        owner.close()
        sink.shutdown()


def test_view_major_mixed_review_still_validates_worker_count(tmp_path, monkeypatch):
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'produce')
    monkeypatch.setenv('XTA_YOLO_SWEEP_NRRD_PROCESSES', '-1')
    with pytest.raises(ValueError, match='nonnegative'):
        make_owner(tmp_path, 50, retain_confidence=False, mirror_formats=('xtareview',))

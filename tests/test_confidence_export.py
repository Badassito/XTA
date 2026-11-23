from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import pytest

from XTA.confidence_evidence import ConfidenceEvidenceRef, ConfidenceEvidenceReader, write_confidence_evidence, write_block_confidence_evidence
from XTA.confidence_export import compact_score_plane, export_compact_evidence, export_memory_plan
from XTA.reconciliation_io import write_seg_nrrd
from tools.compare_reconciliation import compare


POLICIES = Path(__file__).resolve().parents[1] / "XTA/examples/external_reconciliation"


def brute_pool(native, mask, target_shape):
    source_t, source_h, source_w = native.shape
    target_t, target_h, target_w = target_shape
    area = source_h >= target_h and source_w >= target_w
    result = np.zeros(target_shape, np.uint8)
    # Growing axes take the source cell containing the output cell center.
    for z in range(target_t):
        zs = range(math.floor(z*source_t/target_t), math.ceil((z+1)*source_t/target_t)) if source_t >= target_t else (((2*z+1)*source_t)//(2*target_t),)
        for y in range(target_h):
            y0 = math.floor(y*source_h/target_h) if area else ((2*y+1)*source_h)//(2*target_h)
            y1 = math.ceil((y+1)*source_h/target_h) if area else y0+1
            for x in range(target_w):
                x0 = math.floor(x*source_w/target_w) if area else ((2*x+1)*source_w)//(2*target_w)
                x1 = math.ceil((x+1)*source_w/target_w) if area else x0+1
                if mask[z,y,x]:
                    result[z,y,x] = max(int(native[k,y0:y1,x0:x1].max()) for k in zs)
    return result


class Crops:
    def __init__(self, values, *, split):
        self.values, self.split = values, split
        self.calls = []

    def iter_crops(self, z):
        self.calls.append(z)
        height, width = self.values.shape[1:]
        if self.split:
            ys = sorted({0, height//3, height})
            xs = sorted({0, width//2, width})
        else:
            ys, xs = [0,height], [0,width]
        for y0,y1 in zip(ys,ys[1:]):
            for x0,x1 in zip(xs,xs[1:]):
                yield y0,y1,x0,x1,self.values[z,y0:y1,x0:x1]


@pytest.mark.parametrize("source_shape,target_shape", [((5,9,13),(3,5,7)), ((2,3,11),(4,4,4)), ((1,8,8),(4,4,4)),
                                                       ((3,4,5),(10,6,8))])
def test_pooling_matches_independent_global_footprints_for_crops_and_blocks(source_shape,target_shape):
    values = np.random.default_rng(42).integers(0,201,source_shape,dtype=np.uint8)
    mask = np.ones(target_shape,np.uint8)
    mask[:,1:3,1:3] = 0
    expected = brute_pool(values,mask,target_shape)
    for split in (False,True):
        reader = Crops(values,split=split)
        actual = np.stack([compact_score_plane(reader,mask[z],source_shape=source_shape,target_shape=target_shape,z=z)
                           for z in range(target_shape[0])])
        np.testing.assert_array_equal(actual,expected)


def make_run(tmp_path):
    root = tmp_path / "run"
    mask_dir = root / "low_quality" / "0p50" / "nrrd"
    mask_dir.mkdir(parents=True)
    source_shape, target_shape = (8,12,16),(4,8,8)
    rng = np.random.default_rng(102)
    native = [rng.integers(0,201,source_shape,dtype=np.uint8) for _ in range(2)]
    for values in native:
        values[:2,:2,:2] = 0
    masks = [np.ones(target_shape,np.uint8) for _ in range(2)]
    masks[0][:,2:4,3:5] = 0
    masks[1][:,5:7,2:4] = 0
    layers, entries = [], []
    for i,(values,mask) in enumerate(zip(native,masks)):
        filename = f"prediction_{i}.seg.nrrd"
        write_seg_nrrd(mask_dir/filename,shape_tyx=target_shape,read_slab=lambda a,b,m=mask:m[a:b])
        name = ("transverse","coronal")[i]
        key,model = "shared_layer_key",f"model_{i}"
        layers.append(dict(filename=filename,layer_key=key,model_name=model,physical_view_name=name,view_name=name,
                           view_family="orthogonal",source="fullframe",mask_kind="yolo",layer_role="additive_component",
                           recomposition_op="union",output_shape_tyx=list(target_shape),stored_shape_tyx=list(target_shape),
                           downbin_scale=.5,empty_segment=False,exported_axes="(X, Y, t)"))
        folder = root/"reconciliation_evidence"/f"scores_{i}"
        ref = write_confidence_evidence(folder,source_shape,lambda z,v=values:v[z],layer_key=key,model_name=model)
        entries.append(dict(directory=folder.name,layer_key=key,model_name=model,output_shape_tyx=list(source_shape)))
        assert ref.metadata["schema"] == "xta.confidence_evidence/1"
    manifest = mask_dir/"sample_nrrd_manifest.json"
    manifest.write_text(json.dumps(dict(layout="one_single_layer_nrrd_per_component",quality="low_quality",
        downbin_scale=.5,downbin_value="0.50",full_quality_output_shape_tyx=list(source_shape),output_shape_tyx=list(target_shape),
        exported_axes="(X, Y, t)",layer_count=len(layers),layers=layers)))
    score_manifest = root/"reconciliation_evidence/manifest.json"
    score_manifest.write_text(json.dumps(dict(schema="xta.confidence_evidence/1",layers=entries)))
    run = root/"manifest.json"
    run.write_text(json.dumps(dict(status="complete",inputs=dict(source_shape_t_y_x=list(source_shape),processing_shape_t_y_x=[10,12,16]),
        outputs=dict(paths=dict(nrrd_dir="native_masks_intentionally_absent")),resolved_configuration=dict(conf=.99),
        geometry=dict(physical_views=[dict(name=name,family="orthogonal") for name in ("transverse","coronal")]))))
    return run,manifest,native,masks


def test_selected_observed_score_scalar_roundtrip_on_source_and_compact_grid(tmp_path):
    nrrd = pytest.importorskip("nrrd")
    from tools.export_observed_score import export_score
    run, mask_manifest, native, masks = make_run(tmp_path)
    for identity in (run, mask_manifest, run.parent / "reconciliation_evidence/manifest.json"):
        document=json.loads(identity.read_bytes())
        document["run_id"]="score-test"
        identity.write_text(json.dumps(document))
    out = tmp_path / "one-score.nrrd"
    details = export_score(run, "model_1", "shared_layer_key", out, memory_mib=8)
    scores, header = nrrd.read(str(out),index_order="C")
    np.testing.assert_array_equal(scores, native[1])
    assert scores.dtype == np.uint8 and "Segment0_ID" not in header
    np.testing.assert_array_equal(header["space directions"],np.eye(3))
    np.testing.assert_array_equal(header["space origin"],[0,0,0])
    assert header["space"] == "left-posterior-superior"
    assert json.loads(header["xta_observed_score"])["unknown"] == "score_zero"
    assert details["run_id"]=="score-test" and details["grid"]=="source"
    with pytest.raises(FileExistsError):
        export_score(run, "model_1","shared_layer_key",out)
    with pytest.raises(ValueError,match="found 0"):
        export_score(run,"other","shared_layer_key",tmp_path/"missing.nrrd")
    with pytest.raises(ValueError,match="memory_mib"):
        export_score(run,"model_1","shared_layer_key",tmp_path/"small.nrrd",memory_mib=0.0001)
    portable=tmp_path/"portable"
    export_compact_evidence(run,mask_manifest,portable,memory_mib=8)
    compact_nrrd=tmp_path/"compact.nrrd"
    compact=export_score(portable/"manifest.json","model_0","shared_layer_key",compact_nrrd,memory_mib=8)
    np.testing.assert_array_equal(nrrd.read(str(compact_nrrd),index_order="C")[0],
                                  brute_pool(native[0],masks[0],masks[0].shape))
    assert compact["grid"]=="compact_source" and compact["run_id"]=="score-test"


def test_compact_export_from_grouped_review_retains_exact_masks_and_scores(tmp_path):
    from XTA.reconciliation_io import read_layer_manifest
    from XTA.sweep_grouped_reader import locate_run_layer_catalog, open_grouped_catalog
    from XTA.sweep_publication import publish_sweep_review
    from tools.compare_reconciliation import discover_geometry_context

    original, compact, scores, masks = make_run(tmp_path)
    evidence = original.parent / "reconciliation_evidence" / "manifest.json"
    for path in (original, compact, evidence):
        data = json.loads(path.read_bytes())
        data["run_id"] = "grouped-export"
        if path == original:
            data["confidence_sweep"] = dict(sweep_id="export-test", cutoff_hundredths=50, nrrd_storage="grouped")
        path.write_text(json.dumps(data), encoding="utf-8")
    root = tmp_path / "sweep"
    branch = root / "confidence_cutoffs" / "conf_050"
    branch.parent.mkdir(parents=True)
    stage = tmp_path / "scratch"
    staged_branch = stage / branch.name
    staged_branch.mkdir(parents=True)
    original.parent.rename(branch)
    (branch / "low_quality").rename(staged_branch / "low_quality")
    review = publish_sweep_review(root, [(50, branch)], sweep_id="export-test", nrrd_stage_root=stage)
    (root / "manifest.json").write_text(json.dumps({
        "schema": "xta.confidence_sweep/1",
        "review_manifest": "review/manifest.json",
        "review_manifest_sha256": hashlib.sha256(review.read_bytes()).hexdigest(),
    }), encoding="utf-8")

    run = branch / "manifest.json"
    catalogue = locate_run_layer_catalog(run, "low_quality/0p50")
    context, linked_run, _ = discover_geometry_context(catalogue)
    assert linked_run == run and context["source_shape_tyx"] == list(scores[0].shape)
    grouped = open_grouped_catalog(catalogue)
    assert grouped.segment("prediction_0.seg.nrrd").path == grouped.segment("prediction_1.seg.nrrd").path
    with read_layer_manifest(catalogue, workspace=tmp_path, memory_mib=1) as layers:
        for index, layer in enumerate(layers):
            np.testing.assert_array_equal(layer.read_slab(1, 3), masks[index][1:3])
            np.testing.assert_array_equal(layer.read_slab(0, 4), masks[index])
    output = tmp_path / "portable"
    result = export_compact_evidence(run, catalogue, output, memory_mib=8)
    assert locate_run_layer_catalog(output / "manifest.json") == output / result["paths"]["nrrd_manifest"]
    with read_layer_manifest(output / result["paths"]["nrrd_manifest"], workspace=tmp_path) as layers:
        assert layers.grouped_catalog is None
        for index, layer in enumerate(layers):
            np.testing.assert_array_equal(layer.read_slab(0, 4), masks[index])
    for index, entry in enumerate(result["confidence_layers"]):
        ref = ConfidenceEvidenceRef.open(output / "reconciliation_evidence" / entry["directory"])
        with ref.reader() as reader:
            actual, _ = reader(0, 4)
        np.testing.assert_array_equal(actual, brute_pool(scores[index], masks[index], masks[index].shape))

    corrupt = grouped.segment("prediction_0.seg.nrrd").path
    with corrupt.open("r+b") as stream:
        stream.seek(-1, 2)
        value = stream.read(1)[0]
        stream.seek(-1, 2)
        stream.write(bytes([value ^ 1]))
    rejected = tmp_path / "rejected"
    with pytest.raises(ValueError, match="checksum"):
        export_compact_evidence(run, catalogue, rejected, memory_mib=8)
    assert not rejected.exists()
    review.write_bytes(review.read_bytes() + b" ")
    with pytest.raises(ValueError, match="family receipt"):
        open_grouped_catalog(catalogue)


def test_compact_export_rejects_cross_run_masks_with_identical_grid(tmp_path):
    run, compact, _, _ = make_run(tmp_path)
    evidence = run.parent / 'reconciliation_evidence' / 'manifest.json'
    for path, identity in ((run, 'run-a'), (compact, 'run-b'), (evidence, 'run-a')):
        record = json.loads(path.read_text())
        record['run_id'] = identity
        path.write_text(json.dumps(record))
    with pytest.raises(ValueError, match='different runs'):
        export_compact_evidence(run, compact, tmp_path / 'export')
    assert not (tmp_path / 'export').exists()


def test_legacy_export_uses_crops_and_remains_portable_after_move(tmp_path,monkeypatch):
    run,manifest,native,masks = make_run(tmp_path)
    before = {p:p.read_bytes() for p in run.parent.rglob("*") if p.is_file()}
    package = tmp_path/"export"
    with monkeypatch.context() as patch:
        patch.setattr(ConfidenceEvidenceReader,"__call__",lambda *a,**k: (_ for _ in ()).throw(AssertionError("Native slab read is forbidden")))
        report = export_compact_evidence(run,manifest,package,memory_mib=8)
    assert report["inputs_unchanged"] and report["confidence_layer_count"] == 2
    assert all(path.read_bytes()==data for path,data in before.items())
    moved = tmp_path/"portable_copy"
    assert package.resolve().parent == tmp_path.resolve() and moved.parent.resolve() == tmp_path.resolve()
    package.rename(moved)
    evidence = json.loads((moved/"reconciliation_evidence/manifest.json").read_text())
    expected = [brute_pool(v,m,m.shape) for v,m in zip(native,masks)]
    for entry in evidence["layers"]:
        ref = ConfidenceEvidenceRef.open(moved/"reconciliation_evidence"/entry["directory"])
        values,known = ref.reader()(0,4)
        index = int(ref.model_name[-1])
        np.testing.assert_array_equal(values,expected[index])
        np.testing.assert_array_equal(known,expected[index]>0)
        assert values[0,0,0]==0 and masks[index][0,0,0]==1
        assert values.max() <= 200  # The run's .99 threshold never replaces saved scores.
    replay = compare([moved/report["paths"]["nrrd_manifest"]],[POLICIES/"confidence_anchored.py"],tmp_path/"replay",
                     memory_mib=8,previews=False,progress=lambda _:None)
    assert replay["confidence_available"]
    assert all(method["status"]=="complete" for method in replay["datasets"][0]["methods"])
    paths=json.loads((moved/"manifest.json").read_text())["outputs"]["paths"]
    assert all(not Path(value).is_absolute() for value in paths.values())
    assert not list(tmp_path.glob(".export.export-*"))


def test_export_preserves_semantic_observed_contract_and_actual_capture_floor(tmp_path):
    from XTA.confidence_retention import observed_confidence_contract
    run, manifest, _, _ = make_run(tmp_path)
    contract = observed_confidence_contract(prediction_task='semantic', confidence_floor=.17,
        min_conf=.30, confidence_floor_by_backend={'cpu': .17})
    for metadata_path in (run.parent / 'reconciliation_evidence').glob('*/metadata.json'):
        metadata = json.loads(metadata_path.read_text())
        metadata['provenance']['observed_confidence'] = contract
        metadata_path.write_text(json.dumps(metadata))
    output = tmp_path / 'export'
    export_compact_evidence(run, manifest, output, memory_mib=8)
    published = json.loads((output / 'reconciliation_evidence' / 'manifest.json').read_text())
    for row in published['layers']:
        assert row['observed_confidence'] == contract
        ref = ConfidenceEvidenceRef.open(output / 'reconciliation_evidence' / row['directory'])
        assert ref.metadata['provenance']['observed_confidence'] == contract


@pytest.mark.parametrize("mutation",["incomplete","source_grid","scale","missing_model"])
def test_export_rejects_ambiguous_inputs_without_publishing(tmp_path,mutation):
    run,manifest,_,_ = make_run(tmp_path)
    if mutation=="incomplete":
        data=json.loads(run.read_text());data["status"]="running";run.write_text(json.dumps(data))
    else:
        data=json.loads(manifest.read_text())
        if mutation=="source_grid":data["full_quality_output_shape_tyx"]=[10,12,16]
        elif mutation=="scale":data["downbin_scale"]=.25
        else:data["layers"][0]["model_name"]="unmatched_model"
        manifest.write_text(json.dumps(data))
    output=tmp_path/"export"
    with pytest.raises(ValueError):export_compact_evidence(run,manifest,output,memory_mib=8)
    assert not output.exists() and not list(tmp_path.glob(".export.export-*"))


def test_export_rejects_noncanonical_origin(tmp_path):
    run,manifest,_,_ = make_run(tmp_path)
    data=json.loads(manifest.read_text())
    for layer in data["layers"]:
        path=manifest.parent/layer["filename"]
        path.write_bytes(path.read_bytes().replace(b"space origin: (0,0,0)",b"space origin: (1,0,0)"))
    with pytest.raises(ValueError,match="canonical"):
        export_compact_evidence(run,manifest,tmp_path/"export",memory_mib=8)


def test_empty_and_bridge_layers_do_not_acquire_fabricated_scores(tmp_path):
    run,manifest,_,masks = make_run(tmp_path)
    data=json.loads(manifest.read_text())
    for filename,kind,empty in (("empty.seg.nrrd","yolo",True),("bridge.seg.nrrd","bridge",False)):
        values=np.zeros_like(masks[0]) if empty else masks[0]
        write_seg_nrrd(manifest.parent/filename,shape_tyx=values.shape,read_slab=lambda a,b,v=values:v[a:b])
        data["layers"].append({**data["layers"][0],"filename":filename,"layer_key":filename,"mask_kind":kind,"empty_segment":empty})
    data["layer_count"]=len(data["layers"]);manifest.write_text(json.dumps(data))
    report=export_compact_evidence(run,manifest,tmp_path/"export",memory_mib=8)
    assert report["layer_count"]==4 and report["confidence_layer_count"]==3
    empty=next(layer for layer in report["layers"] if layer["filename"]=="empty.seg.nrrd")
    assert empty["known_voxels"]==0
    assert next(layer for layer in report["layers"] if layer["filename"]=="bridge.seg.nrrd")["confidence"]=="not a direct-prediction layer"


def test_workspace_budget_and_failure_cleanup(tmp_path,monkeypatch):
    with pytest.raises(ValueError,match="requires at least"):
        export_memory_plan((1931,3064,3022),(388,612,604),1)
    run,manifest,_,_ = make_run(tmp_path)
    import XTA.confidence_export as module
    def fail(*args,**kwargs):raise RuntimeError("injected crop failure")
    monkeypatch.setattr(module,"compact_score_plane",fail)
    with pytest.raises(RuntimeError,match="injected crop failure"):
        export_compact_evidence(run,manifest,tmp_path/"export",memory_mib=8)
    assert not (tmp_path/"export").exists() and not list(tmp_path.glob(".export.export-*"))


def replace_with_block_evidence(run,manifest,native,*,native_coordinates):
    from XTA.geometry import ViewInfo
    entries=[]
    layers=json.loads(manifest.read_text())["layers"]
    for index,(layer,values) in enumerate(zip(layers,native)):
        name=layer["physical_view_name"]
        storage=values if name=="transverse" or not native_coordinates else values.transpose(2,0,1)
        view=ViewInfo(name=name,num_slices=storage.shape[0],src_h=storage.shape[1],src_w=storage.shape[2],
                      pad_mode="clamp",family="orthogonal",full_t=values.shape[0],full_h=values.shape[1],full_w=values.shape[2],
                      physical_view_name=name)
        directory=run.parent/"reconciliation_evidence"/f"blocks_{index}"
        ref=write_block_confidence_evidence(directory,storage.shape,lambda z,v=storage:v[z],
            layer_key=layer["layer_key"],model_name=layer["model_name"],
            coordinate_space="native_view_processing" if native_coordinates else "source",
            source_shape_tyx=values.shape,provenance={"view":asdict(view)},block_size=4)
        entries.append(dict(directory=directory.name,layer_key=ref.layer_key,model_name=ref.model_name,output_shape_tyx=list(ref.shape)))
    (run.parent/"reconciliation_evidence/manifest.json").write_text(json.dumps(dict(schema="xta.confidence_evidence/1",layers=entries)))


def test_source_blocks_export_to_legacy_compact_format(tmp_path):
    run,manifest,native,masks=make_run(tmp_path)
    replace_with_block_evidence(run,manifest,native,native_coordinates=False)
    report=export_compact_evidence(run,manifest,tmp_path/"export",memory_mib=8)
    for entry in report["confidence_layers"]:
        ref=ConfidenceEvidenceRef.open(tmp_path/"export/reconciliation_evidence"/entry["directory"])
        assert ref.metadata["schema"]=="xta.confidence_evidence/1"
        index=int(ref.model_name[-1])
        with ref.reader() as reader:
            actual,_=reader(0,4)
        np.testing.assert_array_equal(actual,brute_pool(native[index],masks[index],masks[index].shape))


def test_native_export_requires_opt_in_and_projects_one_view_at_a_time(tmp_path,monkeypatch):
    run,manifest,native,masks=make_run(tmp_path)
    replace_with_block_evidence(run,manifest,native,native_coordinates=True)
    with pytest.raises(ValueError,match="allow_native_projection"):
        export_compact_evidence(run,manifest,tmp_path/"denied",memory_mib=8)
    original=ConfidenceEvidenceRef.source_reader
    active=[0,0]
    @contextmanager
    def counted(self,*args,**kwargs):
        active[0]+=1;active[1]=max(active[1],active[0])
        try:
            with original(self,*args,**kwargs) as reader:
                yield reader
        finally:active[0]-=1
    monkeypatch.setattr(ConfidenceEvidenceRef,"source_reader",counted)
    report=export_compact_evidence(run,manifest,tmp_path/"export",memory_mib=8,
                                   allow_native_projection=True,max_staging_mib=1)
    assert active==[0,1]
    assert all(layer["native_projection_applied"] for layer in report["layers"])
    for entry in report["confidence_layers"]:
        ref=ConfidenceEvidenceRef.open(tmp_path/"export/reconciliation_evidence"/entry["directory"])
        index=int(ref.model_name[-1])
        with ref.reader() as reader:actual,_=reader(0,4)
        np.testing.assert_array_equal(actual,brute_pool(native[index],masks[index],masks[index].shape))
    assert not list((tmp_path/"export").rglob("*.dat"))
    assert not (tmp_path/"export/.workspace").exists()


def test_export_checks_declared_confidence_checksums(tmp_path):
    run,manifest,_,_=make_run(tmp_path)
    metadata=run.parent/"reconciliation_evidence/scores_0/metadata.json"
    record=json.loads(metadata.read_text());record["payload_sha256"]="0"*64;metadata.write_text(json.dumps(record))
    with pytest.raises(ValueError,match="checksum mismatch"):
        export_compact_evidence(run,manifest,tmp_path/"export",memory_mib=8)
    assert not (tmp_path/"export").exists() and not list(tmp_path.glob(".export.export-*"))


def test_native_staging_limit_is_enforced(tmp_path):
    run,manifest,native,_=make_run(tmp_path)
    replace_with_block_evidence(run,manifest,native,native_coordinates=True)
    with pytest.raises(MemoryError,match="staging bytes"):
        export_compact_evidence(run,manifest,tmp_path/"export",memory_mib=8,
                                allow_native_projection=True,max_staging_mib=.0001)
    assert not (tmp_path/"export").exists() and not list(tmp_path.glob(".export.export-*"))


def test_selected_native_scalar_requests_explicit_bounded_source_projection(tmp_path):
    nrrd = pytest.importorskip("nrrd")
    from tools.export_observed_score import export_score
    run, manifest, native, _ = make_run(tmp_path)
    replace_with_block_evidence(run, manifest, native, native_coordinates=True)
    destination = tmp_path/"projected.nrrd"
    args = run, "model_1", "shared_layer_key", destination
    with pytest.raises(ValueError, match="allow-native-projection"):
        export_score(*args)
    with pytest.raises(MemoryError, match="staging bytes"):
        export_score(*args, allow_native_projection=True, max_staging_mib=.0001, temp_dir=tmp_path/"work")
    assert not destination.exists() and not list((tmp_path/"work").iterdir())
    result = export_score(*args, allow_native_projection=True, memory_mib=32, max_staging_mib=1,
                          temp_dir=tmp_path/"work")
    assert result["stored_coordinate_space"] == "native_view_processing" and result["native_projection_applied"]
    np.testing.assert_array_equal(nrrd.read(str(destination),index_order="C")[0], native[1])


def _layer_layout_run(root, layout):
    """make_run whose compact masks are stock single-layer NRRDs in the given storage layout."""
    import contextlib, io
    from XTA import outputs
    from XTA.interpolation import NrrdLayerRef
    run, manifest, native, masks = make_run(root)
    data = json.loads(manifest.read_text())
    target = tuple(data["output_shape_tyx"])
    boxes = ((slice(1, 3), slice(2, 7), slice(1, 6)), (slice(0, 4), slice(5, 8), slice(4, 8)))
    for index, (row, box) in enumerate(zip(data["layers"], boxes)):
        mask = np.zeros(target, np.uint8)
        mask[box] = np.random.default_rng(index).random(mask[box].shape) < .7
        mask[box][0, 0, 0] = mask[box][-1, -1, -1] = 1
        masks[index] = mask
        path = manifest.parent / row["filename"]
        path.unlink()
        ref = NrrdLayerRef(key=row["layer_key"], name=row["filename"], path=path.with_suffix(".live"), shape=target,
                           storage_format="live_u8", live_array=mask)
        with contextlib.redirect_stdout(io.StringIO()):
            outputs.write_single_layer_nrrd_from_ref(ref, target, path, segment_name=row["filename"], layout=layout)
        fields = dict(line.split(":=" if ":=" in line else ": ", 1) for line in
                      path.read_bytes().split(b"\n\n", 1)[0].decode().splitlines()[1:] if not line.startswith("#"))
        row["stored_shape_tyx"] = [int(v) for v in reversed(fields["sizes"].split())]
        if layout == "cropped":
            row["stored_offset_tyx"] = [int(v) for v in reversed(fields["Segmentation_ReferenceImageExtentOffset"].split())]
            row["stored_layout"] = "segment_bounding_box"
            assert row["stored_shape_tyx"] != list(target)
    # An empty detector layer may be listed without a file; export writes it on the full grid.
    missing = {**data["layers"][0], "filename": "missing_empty.seg.nrrd", "layer_key": "missing_empty",
               "model_name": "model_absent", "empty_segment": True, "stored_shape_tyx": [1, 1, 1]}
    if layout == "cropped":
        missing.update(stored_offset_tyx=[0, 0, 0], stored_layout="segment_bounding_box")
    data["layers"].append(missing)
    data["layer_count"] = len(data["layers"])
    manifest.write_text(json.dumps(data))
    return run, manifest, native, masks


def test_cropped_compact_masks_export_identically_to_full_grid_masks(tmp_path):
    from XTA.reconciliation_io import read_layer_manifest
    results = {}
    for layout in ("full", "cropped"):
        run, manifest, native, masks = _layer_layout_run(tmp_path / layout, layout)
        output = tmp_path / f"export_{layout}"
        report = export_compact_evidence(run, manifest, output, memory_mib=8)
        portable = json.loads((output / "nrrd" / manifest.name).read_text())
        rows = {row["filename"]: row for row in portable["layers"]}
        scores = {}
        for entry in json.loads((output / "reconciliation_evidence/manifest.json").read_text())["layers"]:
            ref = ConfidenceEvidenceRef.open(output / "reconciliation_evidence" / entry["directory"])
            scores[(entry["model_name"], entry["layer_key"])] = ref.reader()(0, masks[0].shape[0])
        with read_layer_manifest(output / "nrrd" / manifest.name, workspace=tmp_path / f"work_{layout}",
                                 include_all_layers=True) as collection:
            decoded = {layer.metadata["filename"]: layer.read_slab(0, collection.shape_tyx[0]) for layer in collection}
        results[layout] = report, rows, scores, decoded, masks, native
    full_report, full_rows, full_scores, full_decoded, masks, native = results["full"]
    report, rows, scores, decoded, cropped_masks, _ = results["cropped"]
    assert [layer["known_voxels"] for layer in report["layers"]] == [
        layer["known_voxels"] for layer in full_report["layers"]]
    assert scores.keys() == full_scores.keys() and len(scores) == 3
    for key, (values, known) in scores.items():
        np.testing.assert_array_equal(values, full_scores[key][0])
        np.testing.assert_array_equal(known, full_scores[key][1])
    for index, mask in enumerate(masks):
        np.testing.assert_array_equal(cropped_masks[index], mask)
        np.testing.assert_array_equal(scores[(f"model_{index}", "shared_layer_key")][0], brute_pool(native[index], mask, mask.shape))
    assert decoded.keys() == full_decoded.keys()
    for filename, array in decoded.items():
        np.testing.assert_array_equal(array, full_decoded[filename])
    target = list(masks[0].shape)
    rewritten, full_rewritten = rows["missing_empty.seg.nrrd"], full_rows["missing_empty.seg.nrrd"]
    assert rewritten["stored_shape_tyx"] == full_rewritten["stored_shape_tyx"] == target
    assert rewritten["stored_offset_tyx"] == [0, 0, 0] and rewritten["stored_layout"] == "full_reference"
    assert "stored_offset_tyx" not in full_rewritten and "stored_layout" not in full_rewritten
    assert rows["prediction_0.seg.nrrd"]["stored_layout"] == "segment_bounding_box"

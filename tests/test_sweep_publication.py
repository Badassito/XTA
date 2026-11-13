"""Stock sweep publications preserve independent canonical masks, identities and counts."""
import gzip
import hashlib
import json

import numpy as np
import pytest

from XTA.reconciliation_io import ReferenceGeometry, _read_header, write_seg_nrrd
from XTA.sweep_coverage import DirectCoverage
from XTA.sweep_publication import publish_sweep_review


def read_nrrd(path):
    h, offset = _read_header(path)
    sizes = tuple(reversed([int(x) for x in h["sizes"].split()]))
    dtypes = {"uint8":"u1", "uint16":"<u2", "uint32":"<u4"}
    with path.open("rb") as src:
        src.seek(offset)
        with gzip.GzipFile(fileobj=src) as encoded:
            return np.frombuffer(encoded.read(), dtype=dtypes[h["type"]]).reshape(sizes), h


def scalar(root, cent, *, grids=("native", "low_quality/half"), shifted=False, transfer_fixture=False):
    branch = root / "confidence_cutoffs" / f"conf_{cent:03d}"
    branch.mkdir(parents=True)
    (branch / "manifest.json").write_text(json.dumps({"status":"complete", "confidence_sweep":{
        "sweep_id":"sweep-test", "cutoff_hundredths":cent}}))
    expected = {}
    for grid_id in grids:
        folder = branch / ("nrrd" if grid_id == "native" else f"{grid_id}/nrrd")
        folder.mkdir(parents=True)
        shape = (7,10,11) if grid_id == "native" else (6,8,9)
        if transfer_fixture:
            shape = (16,24,24) if grid_id == "native" else (8,12,12)
            geometry = ReferenceGeometry(shape)
        else:
            geometry = ReferenceGeometry(shape, directions_xyz=((0.,2.,0.),(-3.,0.,0.),(0.,0.,.5)),
                                         origin_xyz=(4.+shifted,6.,-2.))
        data = np.zeros(shape, np.uint8)
        data[2:4,2:5,2:5] = 1
        if cent == 80:
            data[2,2,2] = 0
        tile = np.zeros_like(data)
        tile[3:5,3:6,3:6] = 1
        bridge = np.zeros_like(data)
        bridge[2,4:7,7] = 1
        if cent == 80:
            bridge[2,6,6] = 1
        audit = np.zeros_like(data)
        audit[3,1,1] = 1
        defs = [
            ("d","yolo","fullframe","pre_interpolation","additive_component","union",data),
            ("t","yolo","tile","cfg_pre_tile_interpolation","additive_component","union",tile),
            ("b","bridge","fullframe","interpolation","additive_component","union",bridge),
            ("a","yolo","audit","pre_interpolation","diagnostic_only","none",audit),
            ("g","union","global","final_output_after_all_postprocessing","checkpoint","select", data|tile|bridge),
        ]
        rows = []
        for key,kind,source,stage,role,op,a in defs:
            suffix = "Global_final_output" if key == "g" else key
            filename = f"volume_{suffix}.seg.nrrd"
            name = f"volume_{suffix}"
            color = (0.3,0.45,0.75)
            write_seg_nrrd(folder / filename, shape_tyx=shape, read_slab=lambda z0,z1:a[z0:z1],
                           geometry=geometry, segment_name=name, color_rgb=color)
            rows.append(dict(filename=filename, layer_key=key, segment_name=name, suffix=suffix, model_name="m",
                             view_name="transverse", mask_kind=kind, source=source, stage=stage,
                             layer_role=role, recomposition_op=op, segment_color_rgb=color))
            expected[grid_id,key] = a
        doc = dict(layout="one_single_layer_nrrd_per_component", output_shape_tyx=shape,
                   exported_axes="(X, Y, t)", layer_count=len(rows), layers=rows)
        if grid_id != "native":
            doc.update(quality="low_quality", full_quality_output_shape_tyx=(16,24,24) if transfer_fixture else (7,10,11),
                       downbin_token="half", downbin_scale=0.5)
        (folder / "volume_nrrd_manifest.json").write_text(json.dumps(doc))
    return (cent,branch),expected


IDENTITY_KEYS = ("segment_id", "logical_id", "cutoff_cent", "source_filename", "segment_name", "display_name",
                 "model_name", "layer_key", "role", "recomposition_op", "layer_role", "color_rgb", "view_name")


def check_grouped_grid(published, grid, expected_by_cut):
    """Every grouped segment is the exact scalar mask; every group holds one cutoff."""
    grid_id = grid["grid_id"]
    members = {}
    assert grid["payload"] == "grouped"
    for group in grid["groups"]:
        assert {row["cutoff_cent"] for row in group["segments"]} == {group["cutoff_cent"]}
        array, header = read_nrrd(published.parent / group["path"])
        x,y,z = group["offset_xyz"]
        for row in group["segments"]:
            index = row["layer"]
            assert header[f"Segment{index}_ID"] == row["segment_id"]
            assert header[f"Segment{index}_Name"] == row["display_name"]
            assert f"XTA.CutoffCent:{row['cutoff_cent']}|" in header[f"Segment{index}_Tags"]
            assert np.allclose([float(x) for x in header[f"Segment{index}_Color"].split()], row["color_rgb"])
            assert row["segment_id"] not in members
            expected = expected_by_cut[row["cutoff_cent"]][grid_id,row["layer_key"]]
            full = np.zeros(expected.shape,np.uint8)
            cropped = array[index]
            dz,dy,dx = cropped.shape
            full[z:z+dz,y:y+dy,x:x+dx] = cropped
            np.testing.assert_array_equal(full,expected)
            members[row["segment_id"]] = row
    assert {r["role"] for r in members.values()} == {"direct","generated","global","derived"}
    assert [c["cutoff_cent"] for c in grid["coverage"]] == sorted(expected_by_cut)
    for coverage in grid["coverage"]:
        as_array, h = read_nrrd(published.parent / coverage["path"])
        expected = expected_by_cut[coverage["cutoff_cent"]]
        np.testing.assert_array_equal(as_array,expected[grid_id,"d"] + expected[grid_id,"t"])
        assert coverage["maximum_count"] == 2
        assert coverage["voxel_votes"] == int(as_array.sum())
        assert coverage["direct_layer_count"] == 2
        assert {members[s]["layer_key"] for s in coverage["source_segment_ids"]} == {"d","t"}
        assert h["xta_role"] == "direct_layer_count"
        assert "0 Background 0 0 0 0" in (published.parent/coverage["color_table"]).read_text()
    return members


def test_real_scalar_branches_become_color_preserving_grouped_masks_and_direct_counts(tmp_path):
    b50, e50 = scalar(tmp_path, 50)
    b80, e80 = scalar(tmp_path, 80)
    published = publish_sweep_review(tmp_path, [b80,b50], sweep_id="sweep-test", memory_mib=2)
    output = json.loads(published.read_bytes())
    assert output["default_cutoff_cent"] == 50 and output["cutoffs_cent"] == [50,80]
    grids = {g["grid_id"]: g for g in output["grids"]}
    assert set(grids) == {"native","low_quality/half"}
    for grid in grids.values():
        assert grid["origin_xyz"] == [4.,6.,-2.]
        for branch in grid["branches"]:
            for key in ("source_manifest","run_manifest"):
                assert hashlib.sha256((published.parent / branch[key]).read_bytes()).hexdigest() == branch[key+"_sha256"]
    low = check_grouped_grid(published, grids["low_quality/half"], {50: e50, 80: e80})
    assert len(grids["low_quality/half"]["groups"]) == 2
    # With a preview grid, native is identities and origins only: no groups, counts or files.
    native = grids["native"]
    assert native["payload"] == "reference_only" and native["groups"] == [] and native["coverage"] == []
    assert native["shape_tyx"] == [7,10,11] and not (published.parent / "native").exists()
    rows = {row["segment_id"]: row for row in native["segments"]}
    assert len(rows) == len(native["segments"]) == 10
    assert not set(rows) & set(low)
    assert {(r["cutoff_cent"], r["logical_id"]) for r in rows.values()} == {
        (r["cutoff_cent"], r["logical_id"]) for r in low.values()}
    assert all("foreground_voxels" not in row and "layer" not in row for row in rows.values())
    assert len({r["logical_id"] for r in [*rows.values(), *low.values()]}) == 5
    with pytest.raises(FileExistsError):
        publish_sweep_review(tmp_path,[b50,b80],sweep_id="sweep-test")


def test_reference_native_rows_equal_grouped_native_identities(tmp_path):
    with_preview, only_native = tmp_path / "with-preview", tmp_path / "only-native"
    branches = {}
    for root, grids in ((with_preview, ("native", "low_quality/half")), (only_native, ("native",))):
        rows = [scalar(root, cent, grids=grids) for cent in (50, 80)]
        branches[root] = [branch for branch, _ in rows]
        expected = {cent: arrays for cent, ((_, _), arrays) in zip((50, 80), rows)}
    reference = json.loads(publish_sweep_review(with_preview, branches[with_preview], sweep_id="sweep-test").read_bytes())
    grouped_path = publish_sweep_review(only_native, branches[only_native], sweep_id="sweep-test")
    grouped = json.loads(grouped_path.read_bytes())
    # Without a preview grid, native stays the reviewable grouped grid.
    assert [g["payload"] for g in grouped["grids"]] == ["grouped"]
    members = check_grouped_grid(grouped_path, grouped["grids"][0], expected)
    native = next(g for g in reference["grids"] if g["grid_id"] == "native")
    project = lambda row: {key: row[key] for key in IDENTITY_KEYS}
    assert sorted(map(project, native["segments"]), key=lambda r: r["segment_id"]) == sorted(
        map(project, members.values()), key=lambda r: r["segment_id"])
    for key in ("shape_tyx", "space", "directions_xyz", "origin_xyz"):
        assert native[key] == grouped["grids"][0][key]


def corrupt_payloads(branch, grid_folder):
    for path in (branch / grid_folder).glob("*.seg.nrrd"):
        header = path.read_bytes().split(b"\n\n", 1)[0] + b"\n\n"
        path.write_bytes(header + b"\xff" * 64)


def test_native_payloads_are_never_opened_when_a_preview_grid_exists(tmp_path):
    roots = {}
    for name, grids in (("preview", ("native", "low_quality/half")), ("native-only", ("native",))):
        root = tmp_path / name
        roots[name] = (root, [scalar(root, cent, grids=grids)[0] for cent in (50, 80)])
        for _, branch in roots[name][1]:
            corrupt_payloads(branch, "nrrd")
    root, branches = roots["preview"]
    review = json.loads(publish_sweep_review(root, branches, sweep_id="sweep-test", memory_mib=2).read_bytes())
    assert {g["grid_id"]: g["payload"] for g in review["grids"]} == {
        "native": "reference_only", "low_quality/half": "grouped"}
    # The corruption is real: a native-only review must read those payloads and fails.
    root, branches = roots["native-only"]
    with pytest.raises(Exception):
        publish_sweep_review(root, branches, sweep_id="sweep-test", memory_mib=2)
    assert not (root / "review").exists()


def member(cutoff, view, extent=(0, 3, 0, 3, 0, 3)):
    from XTA.sweep_publication import _Member
    info = dict(model_name="m", view_name=view, layer_key=view, logical_id=view)
    return _Member(cutoff, None, info, extent)


def test_groups_never_mix_cutoffs_and_keep_views_adjacent():
    from XTA import sweep_publication as publisher
    members = [member(cut, view) for view in ("c", "a", "b") for cut in (80, 50, 65)]
    packs = list(publisher._packs(members))
    assert [{m.cutoff for m in pack} for pack in packs] == [{50}, {65}, {80}]
    assert [[m.info["view_name"] for m in pack] for pack in packs] == [["a", "b", "c"]] * 3
    many = [member(50, f"v{i:03d}") for i in range(publisher._MAX_CHANNELS + 1)]
    assert [len(pack) for pack in publisher._packs(many)] == [publisher._MAX_CHANNELS, 1]


def published_files(review):
    return {p.relative_to(review).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in review.rglob("*") if p.is_file() and p.name != "manifest.json"}


def test_parallel_cutoffs_publish_identical_files_and_records(tmp_path):
    reviews = []
    for workers in (1, 3):
        root = tmp_path / f"workers-{workers}"
        branches = [scalar(root, cent)[0] for cent in (50, 65, 80)]
        reviews.append(publish_sweep_review(root, branches, sweep_id="sweep-test", memory_mib=2, workers=workers))
    first, second = (json.loads(path.read_bytes()) for path in reviews)
    strip = lambda doc: [{k: v for k, v in g.items() if k != "branches"} for g in doc["grids"]]
    assert strip(first) == strip(second)
    assert published_files(reviews[0].parent) == published_files(reviews[1].parent)
    groups = next(g for g in first["grids"] if g["grid_id"] == "low_quality/half")["groups"]
    assert [g["cutoff_cent"] for g in groups] == [50, 65, 80]
    assert [g["path"] for g in groups] == [f"low_quality/half/group_{i:04d}.seg.nrrd" for i in (1, 2, 3)]


def test_one_failed_cutoff_worker_publishes_nothing(tmp_path, monkeypatch):
    from XTA import sweep_publication as publisher
    branches = [scalar(tmp_path, cent)[0] for cent in (50, 65, 80)]
    original = publisher.write_grouped_seg_nrrd
    def failing(path, **kwargs):
        if any(s.tags["XTA.CutoffCent"] == "65" for s in kwargs["segments"]):
            raise OSError("cutoff 65 write fault")
        return original(path, **kwargs)
    monkeypatch.setattr(publisher, "write_grouped_seg_nrrd", failing)
    with pytest.raises(OSError, match="cutoff 65"):
        publish_sweep_review(tmp_path, branches, sweep_id="sweep-test", memory_mib=2, workers=3)
    assert not (tmp_path / "review").exists()
    assert not list(tmp_path.glob(".xta-review-*"))


@pytest.mark.parametrize("workers", [0, -1, True, 1.5])
def test_worker_count_must_be_a_positive_integer(tmp_path, workers):
    branch, _ = scalar(tmp_path, 50, grids=("native",))
    with pytest.raises(ValueError, match="workers"):
        publish_sweep_review(tmp_path, [branch], sweep_id="sweep-test", workers=workers)


def test_empty_saves_and_failures_cannot_publish_partial_family(tmp_path):
    b50,_ = scalar(tmp_path,50,grids=())
    assert publish_sweep_review(tmp_path,[b50],sweep_id="sweep-test") is None
    with pytest.raises(ValueError,match="identity"):
        publish_sweep_review(tmp_path,[(60,b50[1])],sweep_id="sweep-test")
    b80,_ = scalar(tmp_path,80,grids=("native",))
    with pytest.raises(ValueError,match="different review grids"):
        publish_sweep_review(tmp_path,[b50,b80],sweep_id="sweep-test")
    b81,_ = scalar(tmp_path,81,grids=("native",),shifted=True)
    with pytest.raises(ValueError,match="geometry"):
        publish_sweep_review(tmp_path,[b80,b81],sweep_id="sweep-test")
    assert not (tmp_path/"review").exists()
    assert not list(tmp_path.glob(".xta-review-*"))


@pytest.mark.parametrize("explicit_temp", [True,False])
@pytest.mark.parametrize("fail", [False,True])
def test_review_spools_are_private_on_fast_temp_and_publication_is_adjacent(
    tmp_path, monkeypatch, explicit_temp, fail,
):
    import tempfile
    from XTA import sweep_publication as publisher

    output = tmp_path / "shared"
    fast = tmp_path / "fast"
    fast.mkdir()
    monkeypatch.setattr(tempfile, "tempdir", str(fast))
    branch, _ = scalar(output, 50, grids=("native",))
    captured = []
    original = publisher.write_grouped_seg_nrrd
    def checked(path, **kwargs):
        assert path.is_relative_to(output)
        assert kwargs["temp_dir"].is_relative_to(fast)
        captured.append(kwargs["temp_dir"])
        result = original(path, **kwargs)
        if fail:
            raise OSError("staging fault")
        return result
    monkeypatch.setattr(publisher, "write_grouped_seg_nrrd", checked)
    override = fast / "explicit" if explicit_temp else None
    if fail:
        with pytest.raises(OSError, match="staging fault"):
            publish_sweep_review(output,[branch],sweep_id="sweep-test",temp_dir=override)
        assert not (output/"review").exists()
    else:
        result = publish_sweep_review(output,[branch],sweep_id="sweep-test",temp_dir=override)
        assert result.is_file()
    assert captured and not any(p.exists() for p in captured)
    assert not list(output.glob(".xta-review-*"))
    assert list((override or fast).iterdir()) == []


def test_overlap_count_does_not_wrap_at_255(tmp_path):
    geo = ReferenceGeometry((3,4,5))
    counter = DirectCoverage(tmp_path, geo, 256)
    try:
        ones = np.ones((1,4,5), np.uint8)
        for _ in range(256):
            counter.observe(1,ones,(0,4,0,3,1,1))
        path = tmp_path/"count.nrrd"
        record = counter.write(path,sweep_id="sweep-test",grid_id="native",cutoff_cent=50)
        result, h = read_nrrd(path)
        assert h["endian"]=="little" and result.dtype==np.uint16
        assert np.all(result[1] == 256) and not result[[0,2]].any()
        assert record["voxel_votes"] == 256*20
        assert record["gzip_spans"]["zero_bytes"] == 2*4*5*2
    finally:
        counter.close()


def test_large_empty_count_grid_does_not_allocate_or_encode_dense_planes(tmp_path):
    geometry = ReferenceGeometry((16,4096,4096))
    counter = DirectCoverage(tmp_path,geometry,0)
    try:
        result = counter.write(tmp_path/"empty.nrrd",sweep_id="test",grid_id="native",cutoff_cent=100,
                               memory_mib=1)
        assert result["row_band_bytes"] == 0
        assert result["gzip_spans"]["zero_bytes"] == 256*1024**2
        assert result["gzip_spans"]["data_bytes"] == 0
        assert result["maximum_count"] == result["voxel_votes"] == 0
        assert (tmp_path/"empty.nrrd").stat().st_size < 1024**2
    finally:
        counter.close()


@pytest.mark.parametrize("run_id,layer_id", [("r",None),(None,"r"),("r","other"),("r","r")])
def test_optional_scalar_run_id_binds_grid_when_present(tmp_path,run_id,layer_id):
    branch,_ = scalar(tmp_path,50,grids=("native",))
    def set_id(path, value):
        if value is not None:
            doc = json.loads(path.read_bytes())
            doc["run_id"] = value
            path.write_text(json.dumps(doc))
    set_id(branch[1]/"manifest.json",run_id)
    set_id(branch[1]/"nrrd"/"volume_nrrd_manifest.json",layer_id)
    if run_id == layer_id:
        index = publish_sweep_review(tmp_path,[branch],sweep_id="sweep-test")
        assert json.loads(index.read_bytes())["grids"][0]["branches"][0]["run_id"] == run_id
    else:
        with pytest.raises(ValueError,match="run_id"):
            publish_sweep_review(tmp_path,[branch],sweep_id="sweep-test")


def make_independent(root, cent, name, *, model_hash="f"*64):
    (_, old), _ = scalar(root, cent, grids=("native",))
    branch = root / name
    old.rename(branch)
    run = {
        "schema":"xta.v21.run_manifest/1", "status":"complete", "launcher":{"mode":"tta"},
        "resolved_configuration":{"conf":cent/100, "output_dir":name},
        "inputs":{"source":{"path":"source.mkv","size_bytes":1024,"modified_time_ns":100,
                            "change_or_creation_time_ns":101,"device":1,"file_id":7},
                  "models":{"gpu":{"artifacts":[{"sha256":model_hash}]},"cpu":None},
                  "source_shape_t_y_x":[7,10,11]}
    }
    (branch/"manifest.json").write_text(json.dumps(run))
    return branch


def test_offline_packer_uses_recorded_cutoffs_model_and_source_identity_without_rewriting(tmp_path, capsys):
    from tools.pack_confidence_sweep import main

    study = tmp_path/"study"
    study.mkdir()
    low = make_independent(study, 50, "ordinary result A")
    high = make_independent(study, 80, "ordinary result B")
    original = {p: p.read_bytes() for b in (low,high) for p in b.rglob("*") if p.is_file()}
    assert main(["--root",str(study),"--name","compared","--branch",".80",high.name,"--branch",".50",low.name]) == 0
    packaged = study/"compared"/"manifest.json"
    assert str(packaged) in capsys.readouterr().out
    result = json.loads(packaged.read_bytes())
    assert result["source_mode"] == "independent_scalar"
    assert result["independent_scalar_configuration_differences"] == ["output_dir"]
    assert all(p.read_bytes() == data for p,data in original.items())
    moved = tmp_path/"moved"
    study.rename(moved)
    for b in result["grids"][0]["branches"]:
        assert b["source_mode"] == "independent_scalar"
        path = moved/"compared"/b["run_manifest"]
        assert json.loads(path.read_bytes())["resolved_configuration"]["conf"] == b["cutoff_cent"]/100
        assert hashlib.sha256(path.read_bytes()).hexdigest() == b["run_manifest_sha256"]


def test_offline_packer_rejects_unrelated_incomplete_or_unverified_scalar_sources(tmp_path):
    low = make_independent(tmp_path, 50, "fifty")
    high = make_independent(tmp_path, 75, "seventy-five", model_hash="a"*64)
    kwargs = dict(sweep_id="offline", source_mode="independent_scalar")
    with pytest.raises(ValueError, match="identities differ"):
        publish_sweep_review(tmp_path,[(50,low),(75,high)],**kwargs)
    with pytest.raises(ValueError, match="Recorded scalar cutoff"):
        publish_sweep_review(tmp_path,[(51,low)],**kwargs)
    doc = json.loads((low/"manifest.json").read_bytes())
    del doc["inputs"]["source"]["size_bytes"]
    (low/"manifest.json").write_text(json.dumps(doc))
    with pytest.raises(ValueError,match="complete file identity"):
        publish_sweep_review(tmp_path,[(50,low)],**kwargs)
    assert not (tmp_path/"review").exists()

def test_empty_final_global_is_published_for_a_cutoff_without_detections(tmp_path, monkeypatch):
    from unittest.mock import Mock
    from XTA import assembly

    sink = Mock()
    monkeypatch.setattr(assembly, "nrrd_layer_sink", lambda: sink)
    mask = np.zeros((3, 4, 5), np.uint8)
    result = assembly.materialize_nrrd_global_layer(
        mask, model_name="best", source="global", mask_kind="union",
        stage="final_output_after_all_postprocessing", temp_dir=tmp_path,
        volume_is_immutable=True, include_empty=True,
    )
    assert result is not None and result.live_array is mask
    assert result.shape == mask.shape and result.layer_role == "checkpoint"
    assert sink.submit_layer.call_args.args[0] is result


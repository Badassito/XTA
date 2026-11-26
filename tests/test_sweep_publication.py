"""Stock sweep publications preserve independent canonical masks, identities and counts."""
import gzip
import hashlib
import json
import os

import numpy as np
import pytest

from XTA.reconciliation_io import ReferenceGeometry, _read_header, read_layer_manifest, write_seg_nrrd
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


def scalar(root, cent, *, grids=("native", "low_quality/half"), shifted=False, transfer_fixture=False,
           stage_root=None, incomplete=False):
    branch = root / "confidence_cutoffs" / f"conf_{cent:03d}"
    branch.mkdir(parents=True)
    ownership = {"sweep_id": "sweep-test", "cutoff_hundredths": cent}
    if stage_root is not None:
        ownership["nrrd_storage"] = "grouped"
    (branch / "manifest.json").write_text(json.dumps({
        "status": "incomplete" if incomplete else "complete", "confidence_sweep": ownership,
    }))
    backing = stage_root / branch.name if stage_root is not None else branch
    expected = {}
    for grid_id in grids:
        folder = backing / ("nrrd" if grid_id == "native" else f"{grid_id}/nrrd")
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


@pytest.mark.parametrize("incomplete", [False, True])
def test_staged_sweep_groups_native_and_preview_with_indexed_mask_payloads(tmp_path, incomplete):
    from XTA.sweep_grouped_reader import (
        LayerCatalogueRef, layer_catalogue_bytes, locate_run_layer_catalog, open_grouped_catalog,
    )

    root = tmp_path / "output"
    staging = tmp_path / "scratch"
    branches, masks = [], {}
    for cent in (50, 80):
        branch, expected = scalar(root, cent, stage_root=staging, incomplete=incomplete and cent == 80)
        branches.append(branch)
        masks[cent] = expected
        private_catalog = staging / branch[1].name / "nrrd" / "volume_nrrd_manifest.json"
        doc = json.loads(private_catalog.read_bytes())
        archive = branch[1] / "sam-artifacts.tar"
        doc["layers"][2]["proposal_evidence_path"] = str(archive) + "#/sam_interpolation/fixture"
        doc["layers"][2]["extrapolation"] = {
            "evidence_path": os.path.relpath(archive, private_catalog.parent) + "#/sam_extrapolation/fixture",
            "evidence_path_base": "manifest_directory",
        }
        private_catalog.write_text(json.dumps(doc))
    published = publish_sweep_review(root, branches, sweep_id="sweep-test", memory_mib=2,
                                     nrrd_stage_root=staging,
                                     sweep_status="incomplete" if incomplete else "complete")
    review = json.loads(published.read_bytes())
    assert review["status"] == "complete"
    assert review["sweep_status"] == ("incomplete" if incomplete else "complete")
    assert review["source_storage"] == "grouped"
    assert {grid["grid_id"] for grid in review["grids"]} == {"native", "low_quality/half"}
    assert not list((root / "confidence_cutoffs").rglob("*.nrrd"))
    assert list(published.parent.rglob("*.json")) == [published]
    anchor = LayerCatalogueRef(published, "native", 50)
    branch_manifest = dict(branches)[50] / "manifest.json"
    for access in (lambda: open_grouped_catalog(anchor), lambda: layer_catalogue_bytes(anchor),
                   lambda: locate_run_layer_catalog(branch_manifest)):
        with pytest.raises(ValueError, match="matching terminal family receipt"):
            access()
    family_path = root / "manifest.json"
    family = dict(schema="xta.confidence_sweep/1", sweep_id="sweep-test", status=review["sweep_status"],
                  review_manifest="review/manifest.json",
                  review_manifest_sha256=hashlib.sha256(published.read_bytes()).hexdigest())
    for changes in ({"schema": "unknown"}, {"sweep_id": "different"}, {"status": "in_progress"},
                    {"review_manifest_sha256": None}):
        family_path.write_bytes(json.dumps({**family, **changes}).encode())
        with pytest.raises(ValueError, match="matching terminal family receipt"):
            open_grouped_catalog(anchor)
    family_path.write_bytes(json.dumps(family).encode())
    catalogue_hashes = set()
    for grid in review["grids"]:
        check_grouped_grid(published, grid, masks)
        for origin in grid["branches"]:
            assert origin["source_storage"] == "grouped"
            assert origin["source_catalog"] == "embedded"
            assert origin["source_manifest"] == "manifest.json"
            branch = dict(branches)[origin["cutoff_cent"]]
            catalogue = locate_run_layer_catalog(branch / "manifest.json", grid["grid_id"])
            raw_catalogue = layer_catalogue_bytes(catalogue)
            assert hashlib.sha256(raw_catalogue).hexdigest() == origin["source_manifest_sha256"]
            catalogue_hashes.add(origin["source_manifest_sha256"])
            source = json.loads(raw_catalogue)
            assert source["layout"] == "grouped_sweep_segments"
            assert source["source_manifest_name"] == "volume_nrrd_manifest.json"
            assert source["grid_id"] == grid["grid_id"]
            assert source["cutoff_cent"] == origin["cutoff_cent"]
            assert (catalogue.parent / source["review_manifest"]).resolve() == published
            with read_layer_manifest(catalogue, workspace=staging, include_all_layers=True) as layers:
                for layer in layers:
                    expected = masks[origin["cutoff_cent"]][grid["grid_id"], layer.metadata["layer_key"]]
                    np.testing.assert_array_equal(layer.read_slab(0, expected.shape[0]), expected)
            layer_row = source["layers"][-1]
            selection = dict(source, layers=[layer_row], layer_count=1)
            with read_layer_manifest(catalogue, workspace=staging, include_all_layers=True,
                                     manifest_bytes=json.dumps(selection).encode()) as layers:
                assert [layer.layer_id for layer in layers] == [layer_row["filename"]]
            selection["layers"] = [dict(layer_row, description="changed")]
            with pytest.raises(ValueError, match="metadata differs from its authenticated catalogue"):
                read_layer_manifest(catalogue, workspace=staging, include_all_layers=True,
                                    manifest_bytes=json.dumps(selection).encode())
            if grid["grid_id"] == "native":
                bridge = next(layer for layer in source["layers"] if layer["layer_key"] == "b")
                relative, member = bridge["proposal_evidence_path"].split("#/")
                assert (catalogue.parent / relative).resolve() == branch / "sam-artifacts.tar"
                assert member == "sam_interpolation/fixture"
                assert bridge["proposal_evidence_path_base"] == "manifest_directory"
                tail = bridge["extrapolation"]
                tail_path, tail_member = tail["evidence_path"].split("#/")
                assert (catalogue.parent / tail_path).resolve() == branch / "sam-artifacts.tar"
                assert tail_member == "sam_extrapolation/fixture"
                assert tail["evidence_path_base"] == "manifest_directory"
        for group in grid["groups"]:
            path = published.parent / group["path"]
            raw_group = path.read_bytes()
            independent, _ = read_nrrd(path)
            assert hashlib.sha256(raw_group).hexdigest() == group["sha256"]
            for segment in group["segments"]:
                start = segment["gzip_offset"]
                stop = start + segment["gzip_bytes"]
                assert segment["label_value"] == 1
                assert gzip.decompress(raw_group[start:stop]) == independent[segment["layer"]].tobytes()
    assert len(catalogue_hashes) == 4


def test_grouped_publication_refuses_a_destination_created_during_staging(tmp_path, monkeypatch):
    from XTA import json_publication
    from XTA.run_transport import TransportError

    root = tmp_path / "output"
    branch, _ = scalar(root, 50, stage_root=tmp_path / "scratch")
    destination = root / "review"
    before_publication = json_publication._publish_sweep_directory

    def compete(stage, target):
        target.mkdir()
        return before_publication(stage, target)

    monkeypatch.setattr(json_publication, "_publish_sweep_directory", compete)
    with pytest.raises((FileExistsError, TransportError), match="Destination appeared|already exists"):
        publish_sweep_review(root, [branch], sweep_id="sweep-test", nrrd_stage_root=tmp_path / "scratch")
    assert destination.is_dir() and not list(destination.iterdir())


def test_complete_review_stage_survives_directory_promotion_failure(tmp_path, monkeypatch):
    import errno
    from XTA import json_publication
    from XTA.run_transport import TransportError

    branch, _ = scalar(tmp_path, 50, grids=("native",))

    def unavailable(stage, destination):
        raise OSError(errno.ENOTSUP, "cannot publish directory", str(destination))

    monkeypatch.setattr(json_publication, "_publish_sweep_directory", unavailable)
    with pytest.raises(TransportError, match="Completed sweep products remain at") as failure:
        publish_sweep_review(tmp_path, [branch], sweep_id="sweep-test")
    [stage] = tmp_path.glob(".review.atomic-result-*/review")
    assert str(stage) in str(failure.value)
    record = json.loads((stage / "manifest.json").read_bytes())
    assert record["status"] == "complete"
    assert (stage / record["grids"][0]["groups"][0]["path"]).is_file()
    assert not (tmp_path / "review").exists()


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


@pytest.mark.parametrize("resident", [False, True])
def test_parallel_groups_within_one_cutoff_preserve_the_managed_review(tmp_path, monkeypatch, resident):
    import threading
    from XTA import sam_resources, sweep_publication as publisher, workspace

    stage = tmp_path / "staged"
    branches = [scalar(tmp_path, 80, stage_root=stage, grids=("native",))[0]]
    monkeypatch.setattr(publisher, "_MAX_CHANNELS", 1)
    monkeypatch.setattr(sam_resources, "physical_sam_headroom", lambda: 128 << 20)
    if not resident:
        monkeypatch.setattr(publisher, "_native_coverage_plans", lambda *args: {})
    args = dict(sweep_id="sweep-test", memory_mib=2, nrrd_stage_root=stage)
    expected = publish_sweep_review(tmp_path, branches, workers=1, review_name="serial", **args)

    rendezvous = threading.Barrier(2, timeout=10)
    owner_by_group = {}
    lock = threading.Lock()
    original = publisher.write_grouped_seg_nrrd
    def overlapping(path, **kwargs):
        with lock:
            owner_by_group[path.name] = kwargs["segments"][0].source_layer._owner
            first_pair = len(owner_by_group) <= 2
        if first_pair:
            rendezvous.wait()
        return original(path, **kwargs)
    monkeypatch.setattr(publisher, "write_grouped_seg_nrrd", overlapping)
    monkeypatch.setattr(workspace, "_cpu_count", lambda: 64)
    monkeypatch.setattr(workspace, "_slurm_allocated_cpu_count", lambda: 4)
    created_workers = []
    make_pool = publisher.ThreadPoolExecutor
    def record_pool(*, max_workers, thread_name_prefix):
        created_workers.append(max_workers)
        return make_pool(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
    monkeypatch.setattr(publisher, "ThreadPoolExecutor", record_pool)
    actual = publish_sweep_review(tmp_path, branches, review_name="parallel", **args)

    assert created_workers == [2]
    assert len(owner_by_group) == len(set(owner_by_group.values())) == 5
    assert all(owner._closed and owner._open_count == 0 for owner in owner_by_group.values())
    assert actual.read_bytes() == expected.read_bytes()
    assert published_files(actual.parent) == published_files(expected.parent)


def test_managed_group_failure_drains_active_writers_before_releasing_coverage(tmp_path, monkeypatch):
    import threading
    from XTA import sam_resources, sweep_publication as publisher

    stage = tmp_path / "staged"
    branches = [scalar(tmp_path, 80, stage_root=stage, grids=("native",))[0]]
    inputs = published_files(stage)
    monkeypatch.setattr(publisher, "_MAX_CHANNELS", 1)
    # An explicit request for 20 group writers fits only two 2 MiB reservations.
    monkeypatch.setattr(sam_resources, "physical_sam_headroom", lambda: 6 << 20)
    barrier = threading.Barrier(2, timeout=10)
    lock = threading.Lock()
    active = peak = started = 0
    original = publisher.write_grouped_seg_nrrd
    close = DirectCoverage.close
    def injected(path, **kwargs):
        nonlocal active, peak, started
        with lock:
            active += 1
            peak = max(peak, active)
            started += 1
            order = started
        try:
            if order <= 2:
                barrier.wait()
            if order == 1:
                raise OSError("injected pack write failure")
            return original(path, **kwargs)
        finally:
            with lock:
                active -= 1
    def checked_close(self):
        assert active == 0
        close(self)
    monkeypatch.setattr(publisher, "write_grouped_seg_nrrd", injected)
    monkeypatch.setattr(DirectCoverage, "close", checked_close)
    with pytest.raises(OSError, match="injected pack write failure"):
        publish_sweep_review(tmp_path, branches, sweep_id="sweep-test", memory_mib=2,
                             nrrd_stage_root=stage, workers=20)
    assert peak == 2
    assert inputs == published_files(stage)
    assert not (tmp_path / "review").exists()
    assert not list(tmp_path.glob(".review.atomic-result-*"))


def test_masks_and_coverage_share_one_compression_pool_across_cutoffs_and_grids(tmp_path, monkeypatch):
    from XTA import sweep_publication as publisher

    pools = []
    write_group = publisher.write_grouped_seg_nrrd
    write_coverage = DirectCoverage.write

    def group(*args, member_pool, **kwargs):
        pools.append(member_pool)
        return write_group(*args, member_pool=member_pool, **kwargs)

    def coverage(*args, member_pool, **kwargs):
        pools.append(member_pool)
        return write_coverage(*args, member_pool=member_pool, **kwargs)

    monkeypatch.setattr(publisher, "write_grouped_seg_nrrd", group)
    monkeypatch.setattr(DirectCoverage, "write", coverage)
    stage = tmp_path / "staged-nrrds"
    branches = [scalar(tmp_path, cent, stage_root=stage)[0] for cent in (50, 80)]
    index = publish_sweep_review(tmp_path, branches, sweep_id="sweep-test", workers=2, nrrd_stage_root=stage)

    grids = json.loads(index.read_bytes())["grids"]
    assert len(grids) == 2
    assert len(pools) == sum(len(g["groups"]) + len(g["coverage"]) for g in grids)
    assert all(pool is pools[0] for pool in pools)
    for grid in grids:
        totals = grid["group_compression"]
        assert 1 <= totals["worker_limit"] <= 32
        assert totals["reused_bytes"] + totals["data_bytes"] + totals["zero_bytes"] == sum(
            int(np.prod(g["stored_shape_tyx"])) * len(g["segments"]) for g in grid["groups"])
        assert all(c["gzip_spans"]["data_codec"] == totals["data_codec"] for c in grid["coverage"])


def test_review_links_only_the_captured_cutoff_receipts_confidence(tmp_path):
    stage = tmp_path / 'staged'
    branches = [scalar(tmp_path, cent, stage_root=stage)[0] for cent in (50,80)]
    for cent, branch in branches:
        folder = branch / 'reconciliation_evidence'
        folder.mkdir()
        (folder / 'manifest.json').write_text('{"layers":[]}')
        if cent == 50:
            p = branch / 'manifest.json'
            receipt = json.loads(p.read_bytes())
            receipt['prediction_processing'] = {'observed_confidence':None}
            p.write_text(json.dumps(receipt))
    index = publish_sweep_review(tmp_path, branches, sweep_id='sweep-test', nrrd_stage_root=stage)
    for grid in json.loads(index.read_bytes())['grids']:
        transient, legacy = grid['branches']
        assert transient['observed_confidence_manifest'] is None
        assert (index.parent / legacy['observed_confidence_manifest']).resolve() == (
            branches[1][1] / 'reconciliation_evidence/manifest.json').resolve()


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
    assert not list(tmp_path.glob(".review.atomic-result-*"))


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
    assert not list(tmp_path.glob(".review.atomic-result-*"))


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
    assert not list(output.glob(".review.atomic-result-*"))
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


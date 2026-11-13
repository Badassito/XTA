from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pytest

from XTA.grouped_seg_nrrd import GroupedSegment, write_grouped_seg_nrrd
from XTA.reconciliation_io import ReferenceGeometry


def read_stock(path):
    header, payload = Path(path).read_bytes().split(b"\n\n", 1)
    fields = {}
    for line in header.decode("utf-8").splitlines()[1:]:
        if not line or line.startswith("#"):
            continue
        key, value = line.split(":", 1)
        fields[key] = value.removeprefix("=").strip()
    sizes = tuple(map(int, fields["sizes"].split()))
    raw = gzip.decompress(payload)
    assert len(raw) == int(np.prod(sizes))
    return fields, np.frombuffer(raw, np.uint8).reshape(sizes[::-1])


def segment(data, sid="stable-1", *, name="Threshold 0.50", color=(0.2, 0.4, 0.8), extent=None, calls=None, tags=None):
    def read(z0, z1):
        if calls is not None:
            calls.append((z0, z1))
        return data[z0:z1]
    return GroupedSegment(sid, name, color, read, tags or {}, extent)


def test_stock_layers_preserve_arbitrary_overlap_names_ids_colors_and_empty(tmp_path):
    a = np.zeros((4, 5, 7), np.uint8)
    b = a.copy()
    a[0:3, 1:4, 2:5] = 1
    b[1:4, 2:5, 3:7] = 1
    empty = np.zeros_like(a)
    tags = {"XTA.Stage": "raw_yolo", "XTA.Confidence": "0.50"}
    sources = [segment(a, tags=tags),
               segment(b, "stable-2", name="Other branch", color=(0.9, 0.1, 0.3)),
               segment(empty, "stable-empty", name="Empty branch")]
    output = tmp_path / "ordinary.seg.nrrd"
    result = write_grouped_seg_nrrd(output, shape_tyx=a.shape, segments=sources, chunk_slices=2)
    fields, data = read_stock(output)
    assert fields["dimension"] == "4"
    assert fields["kinds"] == "domain domain domain list"
    assert fields["space directions"] == "(1,0,0) (0,1,0) (0,0,1) none"
    np.testing.assert_array_equal(data, np.stack([a, b, empty]))
    assert np.count_nonzero(data[0] & data[1]) > 0
    for index, source in enumerate(sources):
        assert fields[f"Segment{index}_ID"] == source.segment_id
        assert fields[f"Segment{index}_Name"] == source.name
        assert fields[f"Segment{index}_Layer"] == str(index)
        assert fields[f"Segment{index}_LabelValue"] == "1"
        np.testing.assert_allclose(list(map(float, fields[f"Segment{index}_Color"].split())), source.color_rgb)
        assert result["segments"][index]["foreground_voxels"] == int(np.count_nonzero(data[index]))
    assert fields["Segment0_Tags"] == "XTA.Stage:raw_yolo|XTA.Confidence:0.50|"
    assert fields["Segment2_Extent"] == "0 -1 0 -1 0 -1"
    assert result["payload_uncompressed_bytes"] == a.size * 3


@pytest.mark.parametrize("shape,voxel", [((2,3,4),(1,2,3)), ((1,3,1),(0,0,0)), ((3,1,1),(2,0,0))])
def test_nonempty_singleton_storage_pads_one_in_grid_neighbor_without_moving_mask(tmp_path, shape, voxel):
    data = np.zeros(shape, np.uint8)
    data[voxel] = 1
    z, y, x = voxel
    extent = (x,x,y,y,z,z)
    path = tmp_path/"singleton.seg.nrrd"
    result = write_grouped_seg_nrrd(path, shape_tyx=shape, segments=[segment(data, extent=extent)])
    fields, stored = read_stock(path)
    assert stored.size == 2 and stored.sum() == 1
    assert result["segments"][0]["extent_xyz"] == list(extent)
    offset = np.array(result["offset_xyz"][::-1])
    np.testing.assert_array_equal(np.argwhere(stored[0])[0] + offset, voxel)
    assert np.all(offset >= 0) and np.all(offset + np.array(stored.shape[1:]) <= shape)


def test_conservative_extent_union_crops_oblique_grid_and_reads_source_once(tmp_path):
    shape = (9, 11, 13)
    a = np.zeros(shape, np.uint8)
    b = a.copy()
    a[3:5, 4:8, 5:9] = 1
    b[4:7, 6:10, 7:11] = 1
    geometry = ReferenceGeometry(shape, directions_xyz=((0., 2., 0.), (-3., 0., 0.), (0., 0., .5)),
                                 origin_xyz=(11., -17., 23.))
    calls_a, calls_b = [], []
    def never(*_args):
        raise AssertionError("trusted empty segment should not be read")
    sources = [segment(a, extent=(4, 9, 3, 8, 2, 5), calls=calls_a),
               segment(b, "b", extent=(6, 11, 5, 10, 3, 7), calls=calls_b),
               GroupedSegment("empty", "Empty", (0., 0., 0.), never, extent_xyz=(0, -1, 0, -1, 0, -1))]
    output = tmp_path / "crop.seg.nrrd"
    result = write_grouped_seg_nrrd(output, shape_tyx=shape, segments=sources, geometry=geometry, chunk_slices=2)
    fields, actual = read_stock(output)
    assert result["stored_shape_tyx"] == [6, 8, 8]
    assert result["offset_xyz"] == [4, 3, 2]
    np.testing.assert_array_equal(actual[:2], np.stack([a, b])[:, 2:8, 3:11, 4:12])
    assert fields["space origin"] == "(2,-9,24)"
    assert fields["Segmentation_ReferenceImageExtentOffset"] == "4 3 2"
    assert fields["Segmentation_ReferenceImageExtent"] == "0 12 0 10 0 8"
    assert fields["Segment0_Extent"] == "1 4 1 4 1 2"
    assert fields["Segment1_Extent"] == "3 6 3 6 2 4"
    assert [z for lo, hi in calls_a for z in range(lo, hi)] == list(range(2, 6))
    assert [z for lo, hi in calls_b for z in range(lo, hi)] == list(range(3, 8))


def test_numeric_buffer_admission_bounds_full_reference_callback_slabs(tmp_path):
    source = np.ones((7, 20, 30), np.uint8)
    calls = []
    result = write_grouped_seg_nrrd(tmp_path / "bounded.seg.nrrd", shape_tyx=source.shape,
        segments=[segment(source, calls=calls)], memory_mib=4096/(1024*1024), chunk_slices=100)
    assert result["chunk_slices"] == 1
    assert calls == [(z, z+1) for z in range(7)]
    assert 3 * source[0].nbytes + min(64*1024, source[0].nbytes) <= result["numeric_buffer_budget_bytes"]
    with pytest.raises(ValueError, match="at least 2400"):
        write_grouped_seg_nrrd(tmp_path / "small.seg.nrrd", shape_tyx=source.shape,
            segments=[segment(source)], memory_mib=2048/(1024*1024))
    assert not (tmp_path / "small.seg.nrrd").exists()


def test_all_trusted_empty_segments_are_retained_in_minimum_stock_grid(tmp_path):
    def never(*_args):
        raise AssertionError("no source decoding expected")
    sources = [GroupedSegment(f"empty-{i}", f"Empty {i}", (.1, .2, .3), never,
                              extent_xyz=(0, -1, 0, -1, 0, -1)) for i in range(6)]
    output = tmp_path / "empty.seg.nrrd"
    result = write_grouped_seg_nrrd(output, shape_tyx=(11, 13, 17), segments=sources)
    fields, data = read_stock(output)
    assert data.shape == (6, 1, 1, 1)
    assert not data.any()
    assert result["segment_count"] == 6
    assert all(s["empty_segment"] for s in result["segments"])
    assert fields["Segmentation_ReferenceImageExtent"] == "0 16 0 12 0 10"
    # Empty declarations must not require a hypothetical full-reference plane.
    huge = write_grouped_seg_nrrd(tmp_path / "huge-empty.seg.nrrd", shape_tyx=(10000,100000,100000),
        segments=sources, memory_mib=1024/(1024*1024))
    assert huge["stored_shape_tyx"] == [1,1,1]
    assert huge["minimum_numeric_buffer_bytes"] == 4


def test_noncontiguous_bool_callbacks_and_many_ordinary_layers(tmp_path):
    rng = np.random.default_rng(91)
    masks = (rng.random((51, 3, 4, 9)) > .72)[..., ::-1]
    assert not masks.flags.c_contiguous
    output = tmp_path / "fifty-one.seg.nrrd"
    result = write_grouped_seg_nrrd(output, shape_tyx=masks.shape[1:],
        segments=[segment(mask, str(i), name=f"Cut {i}") for i, mask in enumerate(masks)], chunk_slices=1)
    fields, data = read_stock(output)
    np.testing.assert_array_equal(data, masks)
    assert fields["Segment50_Layer"] == "50"
    assert result["segment_count"] == 51


@pytest.mark.parametrize("bad", [
    np.full((2, 3, 4), 2, np.uint8),
    np.zeros((2, 3, 4), np.float32),
    np.zeros((1, 3, 4), np.uint8),
])
def test_invalid_callback_never_replaces_existing_output_and_cleans_stage(tmp_path, bad):
    output = tmp_path / "existing.seg.nrrd"
    output.write_bytes(b"retained-existing-output")
    source = GroupedSegment("id", "Name", (.1, .2, .3), lambda _z0, _z1: bad)
    with pytest.raises(ValueError):
        write_grouped_seg_nrrd(output, shape_tyx=(2, 3, 4), segments=[source], overwrite=True)
    assert output.read_bytes() == b"retained-existing-output"
    assert not list(tmp_path.glob(".grouped-seg-write-*"))


def test_contradictory_returned_extent_and_midstream_failure_do_not_publish(tmp_path):
    source = np.ones((3, 4, 5), np.uint8)
    with pytest.raises(ValueError, match="exceeds extent_xyz"):
        write_grouped_seg_nrrd(tmp_path / "wrong.seg.nrrd", shape_tyx=source.shape,
            segments=[segment(source, extent=(1, 2, 1, 2, 0, 2))])
    def fail(z0, z1):
        if z0:
            raise RuntimeError("producer failed")
        return source[z0:z1]
    with pytest.raises(RuntimeError, match="producer failed"):
        write_grouped_seg_nrrd(tmp_path / "failed.seg.nrrd", shape_tyx=source.shape,
            segments=[GroupedSegment("id", "Name", (.1, .2, .3), fail)], chunk_slices=1)
    assert not list(tmp_path.glob("*.seg.nrrd"))
    assert not list(tmp_path.glob(".grouped-seg-write-*"))


@pytest.mark.parametrize("change", [
    {"segment_id": "bad\nID"}, {"name": "Injected\nSegment1_ID:=bad"},
    {"color_rgb": (.1, float("nan"), .3)}, {"tags": {"bad:key": "value"}},
    {"tags": {"key": "bad|value"}}, {"extent_xyz": (0, 9, 0, 2, 0, 1)},
    {"extent_xyz": (0, -1, 0, 2, 0, 1)},
])
def test_invalid_metadata_is_rejected_before_callback_or_publication(tmp_path, change):
    def never(*_args):
        raise AssertionError("validation must precede reading")
    fields = dict(segment_id="stable", name="Name", color_rgb=(.1, .2, .3), read_slab=never)
    fields.update(change)
    with pytest.raises(ValueError):
        write_grouped_seg_nrrd(tmp_path / "bad.seg.nrrd", shape_tyx=(2, 3, 4),
            segments=[GroupedSegment(**fields)])
    assert not list(tmp_path.iterdir())


def test_duplicate_ids_overwrite_policy_and_geometry_mismatch(tmp_path):
    data = np.zeros((2, 3, 4), np.uint8)
    source = segment(data)
    with pytest.raises(ValueError, match="Duplicate"):
        write_grouped_seg_nrrd(tmp_path / "duplicate.seg.nrrd", shape_tyx=data.shape, segments=[source, source])
    with pytest.raises(ValueError, match="spatial reference"):
        write_grouped_seg_nrrd(tmp_path / "geometry.seg.nrrd", shape_tyx=data.shape, segments=[source],
            geometry=ReferenceGeometry((3, 3, 4)))
    output = tmp_path / "exists.seg.nrrd"
    output.write_bytes(b"preserved")
    with pytest.raises(FileExistsError):
        write_grouped_seg_nrrd(output, shape_tyx=data.shape, segments=[source])
    assert output.read_bytes() == b"preserved"


def test_large_known_empty_z_ranges_are_one_zero_descriptor_without_source_scan(tmp_path):
    # Three distant one-plane sources force a tall stored group. Only their
    # declared source planes may be read; the 99,997 padded planes are encoded
    # as reusable complete members rather than sent through the data compressor.
    shape = (100000, 1, 1)
    calls = []
    def one(z0, z1):
        calls.append((z0, z1))
        assert z1-z0 == 1
        return np.ones((1, 1, 1), np.uint8)
    def never(*_args):
        raise AssertionError("empty layer should not be decoded")
    sources = [
        GroupedSegment("first", "First", (.1,.2,.3), one, extent_xyz=(0,0,0,0,0,0)),
        GroupedSegment("last", "Last", (.3,.2,.1), one, extent_xyz=(0,0,0,0,99999,99999)),
        GroupedSegment("empty", "Empty", (.2,.2,.2), never, extent_xyz=(0,-1,0,-1,0,-1)),
    ]
    output = tmp_path / "padded.seg.nrrd"
    result = write_grouped_seg_nrrd(output, shape_tyx=shape, segments=sources, chunk_slices=1)
    fields, data = read_stock(output)
    assert calls == [(0,1), (99999,100000)]
    assert data.shape == (3,100000,1,1)
    assert int(data.sum()) == 2
    assert data[0,0,0,0] == data[1,-1,0,0] == 1
    assert result["gzip_spans"]["data_bytes"] == 2
    assert result["gzip_spans"]["zero_bytes"] == 299998
    assert result["gzip_spans"]["zero_spans"] == 2


@pytest.mark.parametrize("fail", [False, True])
def test_payload_fast_scratch_and_adjacent_final_stage_are_private_and_clean(tmp_path, fail):
    shape = (2,3,4)
    output_dir = tmp_path / "destination"
    scratch = tmp_path / "fast"
    scratch.mkdir()
    retained = scratch / "keep.txt"
    retained.write_text("caller-owned")
    output = output_dir / "result.seg.nrrd"
    seen = []
    def read(z0,z1):
        payload_dirs = list(scratch.glob(".grouped-seg-payload-*"))
        final_dirs = list(output_dir.glob(".grouped-seg-write-*"))
        assert len(payload_dirs) == len(final_dirs) == 1
        assert (payload_dirs[0] / "payload.gz").exists()
        assert payload_dirs[0].parent == scratch
        assert final_dirs[0].parent == output_dir
        assert not (final_dirs[0] / "payload.gz").exists()
        seen.append((payload_dirs[0],final_dirs[0]))
        if fail:
            raise RuntimeError("source failed")
        return np.ones((z1-z0,3,4),np.uint8)
    source = GroupedSegment("id","Name",(.1,.2,.3),read)
    if fail:
        with pytest.raises(RuntimeError,match="source failed"):
            write_grouped_seg_nrrd(output,shape_tyx=shape,segments=[source],temp_dir=scratch)
        assert not output.exists()
    else:
        write_grouped_seg_nrrd(output,shape_tyx=shape,segments=[source],temp_dir=scratch)
        _, data = read_stock(output)
        assert data.sum() == 24
    assert seen and all(not path.exists() for pair in seen for path in pair)
    assert retained.read_text() == "caller-owned"
    assert not list(scratch.glob(".grouped-seg-payload-*"))
    assert not list(output_dir.glob(".grouped-seg-write-*"))


def test_default_payload_spool_uses_system_temp_root(tmp_path, monkeypatch):
    import tempfile
    default_root = tmp_path / "system-temp"
    default_root.mkdir()
    monkeypatch.setattr(tempfile,"tempdir",str(default_root))
    def read(z0,z1):
        assert len(list(default_root.glob(".grouped-seg-payload-*"))) == 1
        return np.ones((z1-z0,2,3),np.uint8)
    write_grouped_seg_nrrd(tmp_path / "default.seg.nrrd",shape_tyx=(1,2,3),
        segments=[GroupedSegment("id","Name",(.1,.2,.3),read)])
    assert not list(default_root.iterdir())


def test_concurrent_publications_do_not_share_or_remove_each_others_private_spools(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    import threading
    scratch = tmp_path / "shared-fast"
    scratch.mkdir()
    destination = tmp_path / "outputs"
    barrier = threading.Barrier(2)
    def publish(index):
        def read(z0,z1):
            barrier.wait(timeout=10)
            assert len(list(scratch.glob(".grouped-seg-payload-*"))) == 2
            barrier.wait(timeout=10)
            return np.full((z1-z0,2,3),index,np.uint8)
        return write_grouped_seg_nrrd(destination / f"{index}.seg.nrrd",shape_tyx=(1,2,3),
            segments=[GroupedSegment(str(index),f"Name {index}",(.1,.2,.3),read)],temp_dir=scratch)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(publish,(0,1)))
    assert len(results) == 2
    for index in (0,1):
        _, data = read_stock(destination / f"{index}.seg.nrrd")
        assert np.all(data == index)
    assert not list(scratch.iterdir())
    assert not list(destination.glob(".grouped-seg-write-*"))

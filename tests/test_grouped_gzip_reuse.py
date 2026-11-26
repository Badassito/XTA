from __future__ import annotations

import gzip
import hashlib
import json
import struct
import zlib

import numpy as np
import pytest

from XTA.grouped_seg_nrrd import GroupedSegment, write_grouped_seg_nrrd
from XTA.reconciliation_io import ReferenceGeometry, read_layer_manifest, write_seg_nrrd


def _scalar_layer(path, raster, offset, shape):
    """Write the same embedded crop geometry that scalar publication supplies."""
    z, y, x = offset
    write_seg_nrrd(path, shape_tyx=raster.shape, read_slab=lambda start, stop: raster[start:stop],
                   geometry=ReferenceGeometry(raster.shape, origin_xyz=(x, y, z)))
    header, original = path.read_bytes().split(b"\n\n", 1)
    reference_extent = " ".join(str(bound) for size in reversed(shape) for bound in (0, size-1))
    header += (f"\nSegmentation_ReferenceImageExtentOffset:={x} {y} {z}"
               f"\nSegmentation_ReferenceImageExtent:={reference_extent}\n\n").encode()
    path.write_bytes(header + original)
    return header


def _catalogue(tmp_path, shape, filenames):
    path = tmp_path / "catalogue.json"
    path.write_text(json.dumps({"output_shape_tyx": shape, "exported_axes": "(x,y,t)",
                                "layers": [{"filename": name} for name in filenames]}),
                    encoding="utf-8")
    return path


def _descriptor(layer, observer=None):
    extent = list(map(int, layer.header["Segment0_Extent"].split()))
    reference_extent = tuple(v + base for axis, base in enumerate(reversed(layer.offset_tyx))
                             for v in extent[2*axis:2*axis+2])
    return GroupedSegment(layer.layer_id, layer.layer_id, (0, 0, 0), None,
                          extent_xyz=reference_extent, source_layer=layer, observe_slab=observer)


def test_group_releases_each_partial_decoder_before_the_next_channel(tmp_path):
    shape = (6, 4, 5)
    masks = np.zeros(shape, np.uint8)
    masks[1:3] = 1
    names = ["a.seg.nrrd", "b.seg.nrrd"]
    for name in names:
        _scalar_layer(tmp_path / name, masks, (0, 0, 0), shape)
    catalogue = _catalogue(tmp_path, shape, names)
    with read_layer_manifest(catalogue, workspace=tmp_path, include_all_layers=True, max_open=1) as layers:
        result = write_grouped_seg_nrrd(tmp_path / "group.seg.nrrd", shape_tyx=shape,
                                       segments=[_descriptor(layer) for layer in layers])
        assert all(row["foreground_voxels"] == 40 for row in result["segments"])
        assert layers._open_count == 0
        np.testing.assert_array_equal(layers[0].read_slab(0, shape[0]), masks)


@pytest.mark.parametrize("fast", [False, True])
def test_copy_aligned_members_with_z_padding_and_reencode_mismatched_crops(tmp_path, fast):
    if fast:
        pytest.importorskip("isal")
    shape = (7, 9, 10)
    padded_z = np.zeros((4, 4, 5), dtype=np.uint8)
    padded_z[1:3] = 1
    rasters = [np.ones((2, 4, 5), dtype=np.uint8),
               np.ones((2, 4, 5), dtype=np.uint8),
               np.ones((2, 2, 2), dtype=np.uint8),
               padded_z]
    offsets = [(1, 2, 3), (4, 2, 3), (3, 3, 5), (1, 2, 3)]
    masks = np.zeros((len(rasters), *shape), dtype=np.uint8)
    votes = np.zeros(shape, dtype=np.uint8)
    filenames = []
    payloads = []
    for index, (raster, (z, y, x)) in enumerate(zip(rasters, offsets)):
        path = tmp_path / f"layer{index}.seg.nrrd"
        filenames.append(path.name)
        header = _scalar_layer(path, raster, (z, y, x), shape)
        # Miscellaneous gzip producer, with valid multiple members and the
        # zero padding that legacy readers allow between/after members.
        raw = raster.tobytes()
        members = [gzip.compress(raw[:7], mtime=5), gzip.compress(raw[7:], mtime=13)]
        header_with_checksum = members[1][:3] + b"\x02" + members[1][4:10]
        checksum = struct.pack("<H", zlib.crc32(header_with_checksum) & 0xffff)
        members[1] = header_with_checksum + checksum + members[1][10:]
        path.write_bytes(header + (b"\0" * 7).join(members) + b"\0" * 129)
        payloads.append(b"".join(members))
        dz, dy, dx = raster.shape
        masks[index, z:z+dz, y:y+dy, x:x+dx] = raster

    catalogue = _catalogue(tmp_path, shape, filenames)
    result_path = tmp_path / "group.seg.nrrd"
    with read_layer_manifest(catalogue, workspace=tmp_path, include_all_layers=True,
                             prefer_fast_gzip=fast) as layers:
        descriptors = []
        for layer in layers:
            _, y, x = layer.offset_tyx
            _, dy, dx = layer.stored_shape_tyx

            def observe(z, slab, y=y, x=x, dy=dy, dx=dx):
                assert not slab.flags.writeable
                votes[z:z+len(slab), y:y+dy, x:x+dx] += slab

            descriptors.append(_descriptor(layer, observe))
        result = write_grouped_seg_nrrd(result_path, shape_tyx=shape, segments=descriptors,
                                        chunk_slices=1, index_segments=True)

    data = result_path.read_bytes()
    expected = masks[:, 1:6, 2:6, 3:8]
    np.testing.assert_array_equal(votes, masks.sum(axis=0))
    assert result["sha256"] == hashlib.sha256(data).hexdigest()
    assert result["gzip_spans"]["reused_spans"] == 2
    assert result["gzip_spans"]["reused_bytes"] == 80
    assert result["gzip_spans"]["reused_compressed_bytes"] == sum(map(len, payloads[:2]))
    assert result["gzip_spans"]["data_bytes"] == 80
    assert gzip.decompress(data.split(b"\n\n", 1)[1]) == expected.tobytes()
    for index, row in enumerate(result["segments"]):
        raw = data[row["gzip_offset"]:row["gzip_offset"] + row["gzip_bytes"]]
        if index < 2:
            assert payloads[index] in raw
        assert gzip.decompress(raw) == expected[index].tobytes()
        assert row["foreground_voxels"] == int(expected[index].sum())


@pytest.mark.parametrize("fast", [False, True])
@pytest.mark.parametrize("failure", ["crc", "size_checksum", "truncated", "extra_voxel", "nonbinary"])
def test_invalid_reused_source_keeps_existing_output_and_cleans_stages(tmp_path, fast, failure):
    if fast:
        pytest.importorskip("isal")
    shape = (2, 3, 4)
    path = tmp_path / "layer.seg.nrrd"
    raster = np.ones(shape, dtype=np.uint8)
    header = _scalar_layer(path, raster, (0, 0, 0), shape)
    payload = gzip.compress(raster.tobytes())
    if failure == "crc":
        payload = payload[:-8] + bytes([payload[-8] ^ 1]) + payload[-7:]
    elif failure == "size_checksum":
        payload = payload[:-4] + bytes([payload[-4] ^ 1]) + payload[-3:]
    elif failure == "truncated":
        payload = payload[:-1]
    elif failure == "extra_voxel":
        payload = gzip.compress(raster.tobytes() + b"\0")
    else:
        payload = gzip.compress(b"\2" + raster.tobytes()[1:])
    path.write_bytes(header + payload)
    catalogue = _catalogue(tmp_path, shape, [path.name])
    output = tmp_path / "group.seg.nrrd"
    output.write_bytes(b"retained")
    with read_layer_manifest(catalogue, workspace=tmp_path, include_all_layers=True,
                             prefer_fast_gzip=fast) as layers:
        with pytest.raises((gzip.BadGzipFile, EOFError, ValueError)):
            write_grouped_seg_nrrd(output, shape_tyx=shape, segments=[_descriptor(layers[0])],
                                   chunk_slices=1, overwrite=True, temp_dir=tmp_path)
        assert layers._open_count == 0
    assert output.read_bytes() == b"retained"
    assert not list(tmp_path.glob(".grouped-seg-*"))

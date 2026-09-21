from __future__ import annotations

import gzip
import json
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA.reconciliation_io import ReferenceGeometry, read_layer_manifest, write_seg_nrrd


def layer_file(directory, name, data, *, fields=None, payload=None, member_bytes=None):
    data = np.asarray(data, dtype=np.uint8)
    nonzero = np.nonzero(data)
    extent = [v for axis in (2, 1, 0) for v in (int(nonzero[axis].min()), int(nonzero[axis].max()))] if data.any() else [0, -1, 0, -1, 0, -1]
    header = {
        "type": "uint8", "dimension": "3", "sizes": " ".join(str(x) for x in data.shape[::-1]),
        "space": "left-posterior-superior", "kinds": "domain domain domain",
        "space directions": "(1,0,0) (0,1,0) (0,0,1)", "space origin": "(0,0,0)",
        "encoding": "gzip", "Segment0_LabelValue": "1", "Segment0_Layer": "0",
        "Segment0_Extent": " ".join(str(v) for v in extent),
    }
    header.update(fields or {})
    lines = ["NRRD0005"] + [f"{k}{':=' if k.startswith('Segment') else ': '}{v}" for k,v in header.items()]
    if payload is None:
        raw = data.tobytes()
        if header["encoding"] == "raw":
            payload = raw
        elif member_bytes:
            payload = b"".join(gzip.compress(raw[i:i+member_bytes], mtime=0) for i in range(0, len(raw), member_bytes))
        else:
            payload = gzip.compress(raw, mtime=0)
    path = directory / name
    path.write_bytes(("\n".join(lines)+"\n\n").encode()+payload)
    return {"filename": name, "output_shape_tyx": list(data.shape), "stored_shape_tyx": list(data.shape),
            "exported_axes": "(X, Y, t)", "layer_role": "additive_component", "recomposition_op": "union",
            "empty_segment": not bool(data.any()), "physical_view_name": name, "source": "fullframe", "mask_kind": "yolo"}


def manifest_file(directory, entries, *, shape=None):
    path = directory / "layers.json"
    payload = {"output_shape_tyx": shape or entries[0]["output_shape_tyx"], "exported_axes": "(X, Y, t)",
               "layer_count": len(entries), "layers": entries}
    path.write_text(json.dumps(payload))
    return path


def test_concatenated_gzip_slab_major_reads_once_per_layer_and_rewinds(tmp_path):
    a = np.zeros((7, 4, 5), dtype=np.uint8)
    b = a.copy()
    a[::2, 1:3, 2] = 1
    b[1::2, 2, 1:4] = 1
    entries = [layer_file(tmp_path, "a.seg.nrrd", a, member_bytes=17), layer_file(tmp_path, "b.seg.nrrd", b, member_bytes=23)]
    manifest = manifest_file(tmp_path, entries)
    with mock.patch("XTA.reconciliation_io.gzip.GzipFile", wraps=gzip.GzipFile) as decoder:
        with read_layer_manifest(manifest, workspace=tmp_path / "workspace") as layers:
            for z in range(7):
                np.testing.assert_array_equal(layers[0].read_slab(z, z+1), a[z:z+1])
                np.testing.assert_array_equal(layers[1].read_slab(z, z+1), b[z:z+1])
            assert decoder.call_count == 2
            assert layers._open_count == 0
            np.testing.assert_array_equal(layers[0].read_slab(0, 2), a[:2])
            assert decoder.call_count == 3
            np.testing.assert_array_equal(layers[0].read_slab(0, 1), a[:1])
            assert decoder.call_count == 4
    assert not (tmp_path / "workspace").exists()


def test_filters_checkpoints_and_keeps_metadata_without_confidence_inference(tmp_path):
    data = np.ones((2, 3, 4), np.uint8)
    entry = layer_file(tmp_path, "direct.seg.nrrd", data)
    entry["conf"] = 0.65
    checkpoint = {**entry, "filename": "omitted-checkpoint.seg.nrrd", "layer_role": "checkpoint", "recomposition_op": "select"}
    audit = {**entry, "filename": "omitted-audit.seg.nrrd", "layer_role": "diagnostic_only", "recomposition_op": "none"}
    with read_layer_manifest(manifest_file(tmp_path, [entry, checkpoint, audit]), workspace=tmp_path) as layers:
        assert len(layers) == 1
        assert layers[0].metadata == entry
        assert "confidence" not in layers[0].metadata
        assert layers[0].shape_tyx == data.shape


def test_crop_offset_uses_spatial_origin_and_not_segment_extent(tmp_path):
    full = np.zeros((8, 9, 10), np.uint8)
    full[0, 0, 0] = 1
    crop = np.zeros((2, 3, 4), np.uint8)
    crop[1, 1, 2] = 1
    full_entry = layer_file(tmp_path, "reference.seg.nrrd", full)
    crop_entry = layer_file(tmp_path, "crop.seg.nrrd", crop, fields={
        "space origin": "(3,2,4)", "Segmentation_ReferenceImageExtentOffset": "3 2 4",
        "Segmentation_ReferenceImageExtent": "0 9 0 8 0 7",
    })
    crop_entry["output_shape_tyx"] = list(full.shape)
    expected = np.zeros_like(full)
    expected[4:6, 2:5, 3:7] = crop
    with read_layer_manifest(manifest_file(tmp_path, [full_entry, crop_entry]), workspace=tmp_path) as layers:
        assert layers[1].offset_tyx == (4, 2, 3)
        with mock.patch("XTA.reconciliation_io.gzip.GzipFile", wraps=gzip.GzipFile) as decoder:
            for z in range(8):
                np.testing.assert_array_equal(layers[1].read_slab(z, z+1), expected[z:z+1])
            assert decoder.call_count == 1


@pytest.mark.parametrize("change", [
    {"space origin": "(1,0,0)"}, {"space directions": "(2,0,0) (0,1,0) (0,0,1)"},
    {"space": "right-anterior-superior"}, {"kinds": "domain domain list"},
    {"space directions": "none (0,1,0) (0,0,1)"}, {"Segment1_LabelValue": "2"},
    {"Segmentation_ReferenceImageExtentOffset": "1 0 0"},
])
def test_rejects_misaligned_or_ambiguous_geometry(tmp_path, change):
    data = np.ones((2, 3, 4), np.uint8)
    a = layer_file(tmp_path, "a.seg.nrrd", data)
    b = layer_file(tmp_path, "b.seg.nrrd", data, fields=change)
    with pytest.raises(ValueError):
        read_layer_manifest(manifest_file(tmp_path, [a, b]), workspace=tmp_path)


def test_unexplained_crop_is_rejected(tmp_path):
    entry = layer_file(tmp_path, "a.seg.nrrd", np.ones((2, 3, 4), np.uint8))
    entry["output_shape_tyx"] = [3, 4, 5]
    with pytest.raises(ValueError, match="explicit reference offset"):
        read_layer_manifest(manifest_file(tmp_path, [entry]), workspace=tmp_path)


def test_full_reference_empty_placeholders_and_documented_missing_empty(tmp_path):
    empty = np.zeros((4, 5, 6), np.uint8)
    entry = layer_file(tmp_path, "empty.seg.nrrd", empty)
    absent = {**entry, "filename": "absent-empty.seg.nrrd"}
    with read_layer_manifest(manifest_file(tmp_path, [entry, absent]), workspace=tmp_path) as layers:
        for layer in layers:
            result = layer.read_slab(0, 4)
            assert result.shape == empty.shape and not result.any()
    absent["empty_segment"] = False
    with pytest.raises(FileNotFoundError):
        read_layer_manifest(manifest_file(tmp_path, [entry, absent]), workspace=tmp_path)


@pytest.mark.parametrize("raw", [b"\x00" * 23, b"\x00" * 25, b"\x00" * 23 + b"\x02"])
def test_payload_size_and_binary_validation(tmp_path, raw):
    entry = layer_file(tmp_path, "a.seg.nrrd", np.zeros((2, 3, 4), np.uint8), payload=gzip.compress(raw))
    with read_layer_manifest(manifest_file(tmp_path, [entry]), workspace=tmp_path) as layers:
        with pytest.raises(ValueError):
            layers[0].read_slab(0, 2)
        assert layers._open_count == 0


def test_crc_corruption_is_detected_at_eof(tmp_path):
    payload = bytearray(gzip.compress(bytes(24)))
    payload[-8] ^= 1
    entry = layer_file(tmp_path, "a.seg.nrrd", np.zeros((2, 3, 4), np.uint8), payload=payload)
    with read_layer_manifest(manifest_file(tmp_path, [entry]), workspace=tmp_path) as layers:
        with pytest.raises(gzip.BadGzipFile):
            layers[0].read_slab(0, 2)
        assert layers._open_count == 0


def test_missing_manifest_shape_axes_and_duplicate_filenames_fail(tmp_path):
    entry = layer_file(tmp_path, "a.seg.nrrd", np.ones((2, 3, 4), np.uint8))
    with pytest.raises(ValueError, match="unique"):
        read_layer_manifest(manifest_file(tmp_path, [entry, entry]), workspace=tmp_path)
    path = manifest_file(tmp_path, [entry])
    content = json.loads(path.read_text())
    content["exported_axes"] = "(t,Y,X)"
    path.write_text(json.dumps(content))
    with pytest.raises(ValueError, match="exported_axes"):
        read_layer_manifest(path, workspace=tmp_path)


def test_memory_budget_open_limit_raw_payload_and_close(tmp_path):
    data = np.ones((4, 32, 32), np.uint8)
    a = layer_file(tmp_path, "a.seg.nrrd", data, fields={"encoding": "raw"})
    b = layer_file(tmp_path, "b.seg.nrrd", data)
    with read_layer_manifest(manifest_file(tmp_path, [a, b]), workspace=tmp_path, memory_mib=0.001, max_open=1) as layers:
        with pytest.raises(ValueError, match="memory_mib"):
            layers[0].read_slab(0, 2)
        np.testing.assert_array_equal(layers[0].read_slab(0, 1), data[:1])
        with pytest.raises(OSError, match="max_open"):
            layers[1].read_slab(0, 1)
        layers[0].close()
        np.testing.assert_array_equal(layers[1].read_slab(0, 1), data[:1])
    with pytest.raises(RuntimeError, match="closed"):
        layers[1].read_slab(0, 1)


def test_stream_writer_roundtrip_preserves_geometry_and_reports_extent(tmp_path):
    data = np.zeros((7, 5, 9), np.uint8)
    data[2:6, 1:4, 3:8] = 1
    geometry = ReferenceGeometry(data.shape, directions_xyz=((0, 0.5, 0), (-2, 0, 0), (0, 0, 4)), origin_xyz=(3, 4, 5))
    calls = []
    def slab(z0, z1):
        calls.append((z0, z1))
        return data[z0:z1]
    path = tmp_path / "reconciled.seg.nrrd"
    result = write_seg_nrrd(path, shape_tyx=data.shape, read_slab=slab, geometry=geometry, chunk_slices=2, segment_name="Example\nmask")
    assert calls == [(0, 2), (2, 4), (4, 6), (6, 7)]
    assert result["foreground_voxels"] == 60
    assert result["segment_extent_xyz"] == [3, 7, 1, 3, 2, 5]
    entry = {"filename": path.name, "output_shape_tyx": list(data.shape), "stored_shape_tyx": list(data.shape),
             "layer_role": "additive_component", "recomposition_op": "union"}
    with read_layer_manifest(manifest_file(tmp_path, [entry]), workspace=tmp_path) as layers:
        assert layers.geometry == geometry
        assert layers[0].header["Segment0_Name"] == "Example mask"
        np.testing.assert_array_equal(layers[0].read_slab(0, 7), data)
    assert not list(tmp_path.glob(".reconciliation-write-*"))


def test_writer_failure_preserves_previous_output_and_cleans_staging(tmp_path):
    path = tmp_path / "existing.seg.nrrd"
    path.write_bytes(b"original")
    def broken(z0, z1):
        if z0:
            raise RuntimeError("reader failed")
        return np.zeros((z1-z0, 3, 4), np.uint8)
    with pytest.raises(FileExistsError):
        write_seg_nrrd(path, shape_tyx=(3, 3, 4), read_slab=broken)
    with pytest.raises(RuntimeError, match="reader failed"):
        write_seg_nrrd(path, shape_tyx=(3, 3, 4), read_slab=broken, chunk_slices=1, overwrite=True)
    assert path.read_bytes() == b"original"
    assert not list(tmp_path.glob(".reconciliation-write-*"))


def test_writer_empty_output_and_nonbinary_rejection(tmp_path):
    path = tmp_path / "empty.seg.nrrd"
    result = write_seg_nrrd(path, shape_tyx=(3, 3, 4), read_slab=lambda a,b: np.zeros((b-a, 3, 4), np.uint8))
    assert result["empty_segment"] is True
    assert result["segment_extent_xyz"] == [0, -1, 0, -1, 0, -1]
    with pytest.raises(ValueError, match="binary"):
        write_seg_nrrd(tmp_path / "bad.seg.nrrd", shape_tyx=(3, 3, 4), read_slab=lambda a,b: np.full((b-a, 3, 4), 2, np.uint8))
    assert not (tmp_path / "bad.seg.nrrd").exists()

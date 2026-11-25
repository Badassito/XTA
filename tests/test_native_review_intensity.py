from contextlib import closing
import gzip
import json
from pathlib import Path
import shutil
import sqlite3
import subprocess
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.file_hashing import sha256_file
from XTA.native_review_intensity import intensity_overview, read_intensity_region, write_native_intensity
from XTA.native_review_overview import native_ijk_to_ras, overview_layout
from XTA.unification.tta_manifest import _file_identity


GEOMETRY = {
    "space": "left-posterior-superior",
    "directions_xyz": ((0, 1.25, 0), (-0.8, 0, 0), (0, 0, 2.5)),
    "origin_xyz": (-2.5, 7, 22),
}


def _write_nrrd(path, values, *, compressed, origin="(-2.5,7,22)"):
    dimensions = " ".join(map(str, reversed(values.shape)))
    encoding = "gzip" if compressed else "raw"
    header = (
        "NRRD0005\ntype: uint8\ndimension: 3\nspace: left-posterior-superior\n"
        f"sizes: {dimensions}\nspace directions: (0,1.25,0) (-0.8,0,0) (0,0,2.5)\n"
        f"space origin: {origin}\nencoding: {encoding}\n\n"
    ).encode("ascii")
    raw = values.tobytes()
    path.write_bytes(header + (gzip.compress(raw) if compressed else raw))


def _store(path, shape):
    return SimpleNamespace(path=path, shape=shape, geometry=GEOMETRY,
                           read_connection=lambda: closing(sqlite3.connect(path)))


@pytest.mark.parametrize("compressed", [False, True])
def test_native_windows_and_overview_preserve_intensity_and_lattice(tmp_path, compressed):
    shape = (7, 135, 271)
    z, y, x = np.ogrid[:shape[0], :shape[1], :shape[2]]
    values = (z * 83 + y * 7 + x * 13).astype(np.uint8)
    input_path = tmp_path / "intensity.nrrd"
    _write_nrrd(input_path, values, compressed=compressed)
    store = _store(tmp_path / "review.xtareview", shape)

    with closing(sqlite3.connect(store.path)) as db:
        assert intensity_overview(store) is None
        assert read_intensity_region(store, ((0, 1), (0, 1), (0, 1))) is None
        receipt = write_native_intensity(db, input_path, shape, GEOMETRY, (3, 64, 64))
        db.commit()

    assert receipt["alignment"] == "verified_nrrd_geometry"
    assert receipt["sha256"] == sha256_file(input_path)
    assert receipt["chunk_shape_zyx"] == [3, 128, 128]
    bounds = ((2, 7), (127, 135), (125, 260))
    result = read_intensity_region(store, bounds)
    np.testing.assert_array_equal(result, values[tuple(slice(*axis) for axis in bounds)])
    assert result.flags.writeable

    overview, affine = intensity_overview(store)
    step, overview_shape = overview_layout(shape)
    centers = [np.minimum(np.arange(d) * step + step // 2, n - 1) for d, n in zip(overview_shape, shape)]
    np.testing.assert_array_equal(overview, values[np.ix_(*centers)])
    np.testing.assert_allclose(affine[:3, :3], native_ijk_to_ras(GEOMETRY)[:3, :3] * step)
    np.testing.assert_allclose(affine[:, 3], native_ijk_to_ras(GEOMETRY) @ [0.5, 0.5, 0.5, 1.0])

    with closing(sqlite3.connect(store.path)) as db:
        db.execute("DELETE FROM native_intensity_blocks WHERE z=1 AND y=1 AND x=1")
        db.commit()
    with pytest.raises(ValueError, match="Missing native intensity blocks"):
        read_intensity_region(store, bounds)


def test_nrrd_rejects_wrong_lattice_and_payload_lengths(tmp_path):
    values = np.ones((3, 5, 7), dtype=np.uint8)
    input_path = tmp_path / "intensity.nrrd"
    with closing(sqlite3.connect(":memory:")) as db:
        _write_nrrd(input_path, values, compressed=False, origin="(0,7,22)")
        with pytest.raises(ValueError, match="geometry differs"):
            write_native_intensity(db, input_path, values.shape, GEOMETRY, (4, 8, 8))
    _write_nrrd(input_path, values, compressed=False)
    original = input_path.read_bytes()
    for sample, message in ((original[:-1], "before the last native voxel"),
                            (original + b"\x00", "more voxels")):
        input_path.write_bytes(sample)
        with closing(sqlite3.connect(":memory:")) as db, pytest.raises(ValueError, match=message):
            write_native_intensity(db, input_path, values.shape, GEOMETRY, (4, 8, 8))


@pytest.mark.skipif(not (shutil.which("ffprobe") and shutil.which("ffmpeg")), reason="FFmpeg tools unavailable")
def test_ffv1_native_frames_and_recorded_source_identity(tmp_path):
    shape = (11, 17, 19)
    values = np.arange(np.prod(shape), dtype=np.uint16).reshape(shape).astype(np.uint8)
    input_path = tmp_path / "native.mkv"
    subprocess.run(
        ["ffmpeg", "-nostdin", "-v", "error", "-f", "rawvideo", "-pixel_format", "gray",
         "-video_size", "19x17", "-framerate", "10", "-i", "-", "-c:v", "ffv1", str(input_path)],
        input=values.tobytes(), capture_output=True, check=True,
    )
    store = _store(tmp_path / "review.xtareview", shape)
    expected = _file_identity(input_path, content_digest=True, use_sweep_cache=False)
    with closing(sqlite3.connect(":memory:")) as db:
        with pytest.raises(ValueError, match="SHA-256 differs"):
            write_native_intensity(db, input_path, shape, GEOMETRY, (4, 16, 16),
                                   source_record={**expected, "sha256": "f" * 64})
    with closing(sqlite3.connect(store.path)) as db:
        receipt = write_native_intensity(db, input_path, shape, GEOMETRY, (4, 16, 16), source_record=expected)
        db.commit()
        recorded = json.loads(db.execute("SELECT source FROM native_intensity_info").fetchone()[0])
    assert receipt["alignment"] == "recorded_source_sha256"
    assert recorded["sha256"] == expected["sha256"]
    np.testing.assert_array_equal(read_intensity_region(store, ((2, 10), (1, 15), (2, 18))), values[2:10, 1:15, 2:18])

"""Store and read independently compressed grayscale blocks on the native lattice."""

from contextlib import contextmanager, nullcontext
import gzip
import json
import math
from pathlib import Path, PurePosixPath
import subprocess
import zlib

import numpy as np

from .native_review_overview import native_ijk_to_ras, overview_ijk_to_ras, overview_layout


_SLAB_BYTES = 64 * 1024 * 1024
_MAX_REGION_BYTES = 128 * 1024 * 1024


def _video_alignment(path, source_record, actual):
    if source_record is None:
        return "declared_native_frame_grid"
    digest = source_record.get("sha256")
    if digest is not None:
        if not isinstance(digest, str) or digest.lower() != actual["sha256"]:
            raise ValueError("Intensity video SHA-256 differs from the recorded TTA source")
        return "recorded_source_sha256"
    if all(source_record.get(key) == value for key, value in actual.items() if key != "sha256"):
        return "recorded_source_file_identity"
    # A copied video has a different device, inode, and creation time. Compare
    # the facts that a metadata-preserving copy can actually retain.
    recorded_path = source_record.get("path", "")
    if (isinstance(recorded_path, str)
            and PurePosixPath(recorded_path.replace("\\", "/")).name.casefold() == path.name.casefold()
            and source_record.get("size_bytes") == actual["size_bytes"]
            and source_record.get("modified_time_ns") == actual["modified_time_ns"]):
        return "recorded_source_copy_metadata"
    raise ValueError("Intensity video differs from the source recorded in the TTA review")


@contextmanager
def _video_frames(path, shape):
    from .media import (_require_bin, _spawn_decode_process, _synchronous_decode_stdout,
                        ffmpeg_decode_threads)

    try:
        _require_bin("ffprobe")
        _require_bin("ffmpeg")
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=width,height", "-of", "json", str(path)],
            capture_output=True, text=True, encoding="utf-8", errors="replace", check=True,
        )
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Could not probe intensity video {path}: {error.stderr.strip()}") from error
    except RuntimeError as error:
        raise ValueError(str(error)) from error
    streams = json.loads(probe.stdout).get("streams", ())
    if (len(streams) != 1 or (streams[0].get("height"), streams[0].get("width")) != shape[1:]):
        raise ValueError("Intensity video frame dimensions differ from the native review grid")
    decoder_threads = str(min(4, ffmpeg_decode_threads()))
    command = [
        "ffmpeg", "-nostdin", "-v", "error", "-xerror", "-threads", decoder_threads, "-i", str(path),
        "-map", "0:v:0", "-threads", "1", "-f", "rawvideo", "-pix_fmt", "gray", "-vsync", "0", "-",
    ]
    try:
        with _synchronous_decode_stdout(_spawn_decode_process(command)) as stream:
            yield stream
    except RuntimeError as error:
        raise ValueError(f"Could not decode intensity video {path}: {error}") from error


@contextmanager
def _nrrd_frames(path, shape, geometry):
    from .reconciliation_io import _ints, _read_header_stream, _U8_TYPES, _vectors

    with path.open("rb") as stream:
        fields, _ = _read_header_stream(stream, path)
        if (fields.get("type", "").lower() not in _U8_TYPES or fields.get("dimension") != "3"
                or any(key in fields for key in ("data file", "datafile", "byte skip", "line skip"))):
            raise ValueError("Intensity NRRD must be an embedded scalar uint8 volume")
        kinds = fields.get("kinds", "domain domain domain").lower().split()
        if len(kinds) != 3 or set(kinds) - {"domain", "space"}:
            raise ValueError("Intensity NRRD must have three spatial axes")
        if tuple(reversed(_ints(fields.get("sizes", ""), 3, "NRRD sizes"))) != shape:
            raise ValueError("Intensity NRRD shape differs from the native review grid")
        input_geometry = {
            "space": fields.get("space", ""),
            "directions_xyz": _vectors(fields.get("space directions", ""), 3, "space directions"),
            "origin_xyz": _vectors(fields.get("space origin", ""), 1, "space origin")[0],
        }
        if not np.allclose(native_ijk_to_ras(input_geometry), native_ijk_to_ras(geometry), rtol=0, atol=1e-7):
            raise ValueError("Intensity NRRD geometry differs from the native review grid")
        encoding = fields.get("encoding", "").lower()
        if encoding not in ("raw", "gzip", "gz"):
            raise ValueError("Intensity NRRD encoding must be raw or gzip")
        source = nullcontext(stream) if encoding == "raw" else gzip.GzipFile(fileobj=stream, mode="rb")
        with source as payload:
            yield payload


def _read_slab(stream, target):
    with memoryview(target).cast("B") as output:
        position = 0
        while position < len(output):
            read = stream.readinto(output[position:])
            if not read:
                raise ValueError("Intensity ended before the last native voxel")
            position += read


def write_native_intensity(connection, intensity_path, shape_zyx, geometry, chunk_shape, *, source_record=None):
    """Add aligned grayscale to a staged review database and return its source receipt."""
    from .unification.tta_manifest import _file_identity

    path = Path(intensity_path).expanduser().resolve(strict=True)
    before = _file_identity(path, content_digest=True, use_sweep_cache=False)
    shape = tuple(map(int, shape_zyx))
    native_ijk_to_ras(geometry)
    overview_step, overview_shape = overview_layout(shape)
    planes, rows, columns = shape
    # Keep at most one XY plane when it alone exceeds the slab budget.
    step_z = min(int(chunk_shape[0]), max(1, _SLAB_BYTES // (rows * columns)), planes)
    step_y = min(max(128, int(chunk_shape[1])), rows)
    step_x = min(max(128, int(chunk_shape[2])), columns)
    steps = (step_z, step_y, step_x)
    slab = np.empty((step_z, rows, columns), dtype=np.uint8)
    overview = np.empty(overview_shape, dtype=np.uint8)
    center_z, center_y, center_x = (
        np.minimum(np.arange(count) * overview_step + overview_step // 2, dimension - 1)
        for count, dimension in zip(overview_shape, shape)
    )
    yx = np.ix_(center_y, center_x)

    if path.suffix.lower() == ".nrrd":
        alignment = "verified_nrrd_geometry"
        frames = _nrrd_frames(path, shape, geometry)
    else:
        alignment = _video_alignment(path, source_record, before)
        frames = _video_frames(path, shape)
    source = {**before, "alignment": alignment}
    connection.execute(
        "CREATE TABLE native_intensity_info "
        "(id INTEGER PRIMARY KEY CHECK (id = 1), chunk_shape TEXT NOT NULL, source TEXT NOT NULL, "
        "overview BLOB NOT NULL)"
    )
    connection.execute(
        "CREATE TABLE native_intensity_blocks "
        "(z INTEGER NOT NULL, y INTEGER NOT NULL, x INTEGER NOT NULL, payload BLOB NOT NULL, "
        "PRIMARY KEY(z,y,x)) WITHOUT ROWID"
    )

    next_overview = 0
    with frames as stream:
        for z in range(0, planes, step_z):
            raster = slab[: min(step_z, planes - z)]
            _read_slab(stream, raster)
            while next_overview < len(center_z) and center_z[next_overview] < z + len(raster):
                overview[next_overview] = raster[int(center_z[next_overview]) - z][yx]
                next_overview += 1
            connection.executemany(
                "INSERT INTO native_intensity_blocks VALUES (?, ?, ?, ?)",
                ((z // step_z, y // step_y, x // step_x,
                  zlib.compress(np.ascontiguousarray(raster[:, y:y + step_y, x:x + step_x]), 1))
                 for y in range(0, rows, step_y) for x in range(0, columns, step_x)),
            )
        if stream.read(1):
            raise ValueError("Intensity contains more voxels than the native review grid")
    current = _file_identity(path, use_sweep_cache=False)
    if current != {key: value for key, value in before.items() if key != "sha256"}:
        raise ValueError("Intensity source changed while preparing the native review")
    connection.execute("INSERT INTO native_intensity_info VALUES (1, ?, ?, ?)",
                       (json.dumps(steps), json.dumps(source), zlib.compress(overview, 1)))
    return {**source, "chunk_shape_zyx": list(steps)}


def _present(connection):
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='native_intensity_info'"
    ).fetchone() is not None


def _compressed_bound(size):
    return size + (size >> 12) + (size >> 14) + (size >> 25) + 13


def _decode(row, shape):
    size = math.prod(shape)
    if row is None or row[-2] is None or row[-2] != len(row[-1]):
        raise ValueError("Missing or oversized native intensity payload")
    decoder = zlib.decompressobj()
    try:
        data = decoder.decompress(row[-1], size + 1)
    except zlib.error as error:
        raise ValueError("Invalid native intensity payload") from error
    if len(data) != size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Invalid native intensity payload")
    return np.frombuffer(data, dtype=np.uint8).reshape(shape)


def intensity_overview(store):
    """Return the optional grayscale navigation overview and its IJK-to-RAS matrix."""
    step, shape = overview_layout(store.shape)
    with store.read_connection() as connection:
        if not _present(connection):
            return None
        row = connection.execute(
            "SELECT length(overview), substr(overview, 1, ?) FROM native_intensity_info WHERE id = 1",
            (_compressed_bound(math.prod(shape)),),
        ).fetchone()
    return _decode(row, shape), overview_ijk_to_ras(store.geometry, step)


def read_intensity_region(store, bounds):
    """Return uint8 native ZYX voxels for half-open bounds, or None without embedded intensity."""
    region = tuple(tuple(values) for values in bounds)
    if (len(region) != 3 or any(len(pair) != 2 or
            any(not isinstance(value, (int, np.integer)) or isinstance(value, bool) for value in pair)
            or not 0 <= pair[0] < pair[1] <= extent
            for pair, extent in zip(region, store.shape))):
        raise ValueError("Native intensity bounds must be inside the review lattice")
    lo = tuple(int(pair[0]) for pair in region)
    hi = tuple(int(pair[1]) for pair in region)
    requested = tuple(stop - start for start, stop in zip(lo, hi))
    if math.prod(requested) > _MAX_REGION_BYTES:
        raise ValueError("Requested native intensity regions may not exceed 128 MiB")
    with store.read_connection() as connection:
        if not _present(connection):
            return None
        row = connection.execute("SELECT chunk_shape FROM native_intensity_info WHERE id = 1").fetchone()
        if row is None:
            raise ValueError("Missing native intensity layout")
        steps = json.loads(row[0])
        if (not isinstance(steps, list) or len(steps) != 3 or
                any(type(step) is not int or not 0 < step <= n for step, n in zip(steps, store.shape))
                or math.prod(steps) > max(_SLAB_BYTES, math.prod(store.shape[1:]))):
            raise ValueError("Invalid native intensity layout")
        ranges = tuple((a // step, (b - 1) // step) for a, b, step in zip(lo, hi, steps))
        blocks = connection.execute(
            "SELECT z, y, x, length(payload), substr(payload, 1, ?) FROM native_intensity_blocks "
            "WHERE z BETWEEN ? AND ? AND y BETWEEN ? AND ? AND x BETWEEN ? AND ?",
            (_compressed_bound(math.prod(steps)), *(i for pair in ranges for i in pair)),
        )
        result = np.empty(requested, dtype=np.uint8)
        count = 0
        for row in blocks:
            origin = tuple(index * step for index, step in zip(row[:3], steps))
            block_shape = tuple(min(step, n - a) for step, n, a in zip(steps, store.shape, origin))
            block = _decode(row, block_shape)
            start = tuple(max(a, b) for a, b in zip(lo, origin))
            stop = tuple(min(b, a + sz) for b, a, sz in zip(hi, origin, block_shape))
            target = tuple(slice(a - o, b - o) for a, b, o in zip(start, stop, lo))
            source = tuple(slice(a - o, b - o) for a, b, o in zip(start, stop, origin))
            result[target] = block[source]
            count += 1
        if count != math.prod(b - a + 1 for a, b in ranges):
            raise ValueError("Missing native intensity blocks")
    return result

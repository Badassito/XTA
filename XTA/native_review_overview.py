"""Bounded, non-editable navigation over an exact native review."""
from contextlib import closing, contextmanager
import json
import math
from pathlib import Path
import sqlite3
import stat
import tempfile
import zlib

import numpy as np


MAX_OVERVIEW_EDGE = 192
MAX_SURFACE_EDGE = 384
MAX_SURFACE_SAMPLES = 24 * 1024**2
SURFACE_ISOVALUE = 127.5
_CACHE_APPLICATION_ID = 0x5854414f  # XTAO


def overview_layout(shape_zyx):
    shape = tuple(map(int, shape_zyx))
    if len(shape) != 3 or min(shape) < 1:
        raise ValueError("Expected positive native ZYX dimensions")
    step = (max(shape) + MAX_OVERVIEW_EDGE - 1) // MAX_OVERVIEW_EDGE
    return step, tuple((n + step - 1) // step for n in shape)


def native_ijk_to_ras(geometry):
    """NRRD direction vectors are rows; Slicer IJK-to-world uses columns."""
    directions = np.asarray(geometry["directions_xyz"], dtype=np.float64)
    origin = np.asarray(geometry["origin_xyz"], dtype=np.float64)
    if (directions.shape != (3, 3) or origin.shape != (3,)
            or not np.isfinite(directions).all() or not np.isfinite(origin).all()
            or np.linalg.det(directions) == 0):
        raise ValueError("Invalid native grid geometry")
    space = geometry["space"].strip().lower()
    signs = {
        "left-posterior-superior": (-1, -1, 1), "lps": (-1, -1, 1),
        "right-anterior-superior": (1, 1, 1), "ras": (1, 1, 1),
        "left-anterior-superior": (-1, 1, 1), "las": (-1, 1, 1),
    }
    if space not in signs:
        raise ValueError(f"Unsupported native anatomical space {space!r}")
    flip = np.asarray(signs[space])
    affine = np.eye(4)
    affine[:3, :3] = flip[:, None] * directions.T
    affine[:3, 3] = flip * origin
    return affine


def overview_ijk_to_ras(geometry, step):
    # Cell 0 covers native centers [0, step). Empty padding past the final
    # native slice keeps every block on the same regular navigation lattice.
    to_native = np.eye(4)
    to_native[:3, :3] *= step
    to_native[:3, 3] = (step - 1) / 2
    return native_ijk_to_ras(geometry) @ to_native


class NativeOverviewAccumulator:
    """Pool foreground by presence, including blocks misaligned to overview cells."""

    def __init__(self, shape_zyx, *, extent_xyz=None, max_count_bytes=None):
        self.shape = tuple(map(int, shape_zyx))
        self.step, size = overview_layout(self.shape)
        self.data = np.zeros(size, dtype=np.bool_)
        self._counts = _counts_for_extent(self.shape, extent_xyz, max_count_bytes=max_count_bytes)

    def add(self, origin_zyx, binary_block):
        origin = tuple(map(int, origin_zyx))
        mask = np.asarray(binary_block)
        if (mask.ndim != 3 or len(origin) != 3 or min(origin) < 0
                or any(lo + n > maximum for lo, n, maximum in zip(origin, mask.shape, self.shape))):
            raise ValueError("Navigation overview block extends outside the native grid")
        if not mask.size:
            return
        if self._counts is not None:
            self._counts.add(origin, mask)
        region = []
        for axis, (lo, n) in enumerate(zip(origin, mask.shape)):
            first_cell = lo // self.step
            region.append(slice(first_cell, (lo + n - 1) // self.step + 1))
            first_boundary = self.step - lo % self.step
            starts = [0, *range(first_boundary, n, self.step)]
            mask = np.logical_or.reduceat(mask, starts, axis=axis)
        self.data[tuple(region)] |= mask

    def save(self, connection, cutoff):
        connection.execute("CREATE TABLE IF NOT EXISTS native_overviews "
                           "(cutoff INTEGER PRIMARY KEY, packed BLOB NOT NULL)")
        packed = np.packbits(self.data, bitorder="little")
        connection.execute("INSERT INTO native_overviews VALUES (?, ?)",
                           (int(cutoff), zlib.compress(packed, 1)))
        if self._counts is not None:
            connection.execute("CREATE TABLE IF NOT EXISTS native_surface_counts_v1 "
                               "(cutoff INTEGER PRIMARY KEY, layout TEXT NOT NULL, packed BLOB NOT NULL)")
            connection.execute("INSERT INTO native_surface_counts_v1 VALUES (?, ?, ?)",
                               (int(cutoff), self._counts.layout_key(), zlib.compress(self._counts.data, 1)))


def overview_volume(store, cutoff):
    """Read the immutable navigation Global; edited data lives on the native grid."""
    step, shape = overview_layout(store.shape)
    if not any(layer["role"] == "global" and cutoff in layer["cutoffs"] for layer in store.layers.values()):
        return np.zeros(shape, np.uint8), overview_ijk_to_ras(store.geometry, step)
    voxels = math.prod(shape)
    packed_size = (voxels + 7) // 8
    with store.read_connection() as db:
        row = db.execute("SELECT length(packed), substr(packed, 1, ?) FROM native_overviews WHERE cutoff=?",
                         (packed_size + packed_size // 100 + 128, int(cutoff))).fetchone()
    if row is None or len(row[1]) != row[0]:
        raise ValueError(f"Missing or oversized navigation overview for cutoff {cutoff}")
    decoder = zlib.decompressobj()
    packed = decoder.decompress(row[1], packed_size + 1)
    if len(packed) != packed_size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Invalid navigation overview")
    data = np.unpackbits(np.frombuffer(packed, np.uint8), count=voxels, bitorder="little")
    return data.reshape(shape), overview_ijk_to_ras(store.geometry, step)


class NativeOverviewCounts:
    """Exact numbers of native foreground voxels inside navigation cells."""

    def __init__(self, native_shape, *, step=None, bounds_zyx=None, max_count_bytes=None):
        self.native_shape = tuple(map(int, native_shape))
        if bounds_zyx is None:
            bounds_zyx = tuple((0, size) for size in self.native_shape)
        self.bounds_zyx = tuple((int(lo), int(hi)) for lo, hi in bounds_zyx)
        if (len(self.bounds_zyx) != 3 or len(self.native_shape) != 3
                or not all(0 <= lo < hi <= size for (lo, hi), size in zip(self.bounds_zyx, self.native_shape))):
            raise ValueError("Navigation bounds lie outside the native grid")

        if step is None:
            # A power of two also keeps blocks aligned with the standard
            # native archive layout.
            minimum = (max(self.native_shape) + MAX_SURFACE_EDGE - 1) // MAX_SURFACE_EDGE
            step = 1 << (minimum - 1).bit_length()
            while math.prod(self._layout(step)[1]) > MAX_SURFACE_SAMPLES:
                step *= 2
        self.step = int(step)
        if self.step <= 0 or self.step**3 > np.iinfo(np.uint32).max:
            raise ValueError("Invalid native navigation cell size")
        self.origin_zyx, shape = self._layout(self.step)
        if math.prod(shape) > MAX_SURFACE_SAMPLES:
            raise ValueError("Native navigation field exceeds its sample budget")
        dtype = np.dtype("<u2" if self.step**3 <= np.iinfo(np.uint16).max else "<u4")
        if max_count_bytes is not None and math.prod(shape) * dtype.itemsize > max_count_bytes:
            raise MemoryError("Native navigation counts exceed their preparation budget")
        self.data = np.zeros(shape, dtype=dtype)

    def _layout(self, step):
        starts = tuple(lo // step for lo, _ in self.bounds_zyx)
        stops = tuple((hi + step - 1) // step for _, hi in self.bounds_zyx)
        return tuple(a * step for a in starts), tuple(b - a for a, b in zip(starts, stops))

    def add(self, origin_zyx, binary_block, *, subtract=False):
        """Accumulate the block's intersection with the field's aligned bounds."""
        origin = tuple(map(int, origin_zyx))
        block = np.asarray(binary_block)
        low = tuple(max(a, b) for a, b in zip(origin, self.origin_zyx))
        high = tuple(min(a + size, b + cells * self.step)
                     for a, size, b, cells in zip(origin, block.shape, self.origin_zyx, self.data.shape))
        if any(a >= b for a, b in zip(low, high)):
            return
        block = block[tuple(slice(a - start, b - start) for a, b, start in zip(low, high, origin))]
        region = []
        for axis, (a, b, start) in enumerate(zip(low, high, self.origin_zyx)):
            region.append(slice((a - start) // self.step, (b - 1 - start) // self.step + 1))
        for axis in reversed(range(3)):
            a = low[axis]
            first_boundary = self.step - a % self.step
            starts = [0, *range(first_boundary, block.shape[axis], self.step)]
            if len(starts) != block.shape[axis]:
                block = np.add.reduceat(block, starts, axis=axis, dtype=self.data.dtype)
        dest = self.data[tuple(region)]
        if subtract:
            if np.any(dest < block):
                raise ValueError("A navigation edit removed more support than the original cells contained")
            dest -= block
        else:
            dest += block

    def affine(self, geometry):
        affine = overview_ijk_to_ras(geometry, self.step)
        affine[:3, 3] += affine[:3, :3] @ (np.asarray(self.origin_zyx[::-1]) / self.step)
        return affine

    def layout_key(self):
        return json.dumps((self.step, self.origin_zyx, self.data.shape, self.data.dtype.str))

    def scalar_field(self):
        """Resolve dense boundaries by occupancy and boost isolated thin structure."""
        from scipy.ndimage import maximum_filter

        counts = self.data
        field = np.empty(counts.shape, dtype=np.uint8)
        nz, ny, nx = counts.shape
        width_z, width_y, width_x = [
            np.minimum(self.step, size - (origin + self.step * np.arange(cells))).astype(np.float32)
            for origin, cells, size in zip(self.origin_zyx, counts.shape, self.native_shape)
        ]
        band_depth = max(1, (256 * 1024) // (ny * nx))
        for z in range(0, nz, band_depth):
            stop = min(nz, z + band_depth)
            first = max(0, z - 1)
            bounds = slice(first, min(nz, stop + 1))
            density = counts[bounds].astype(np.float32)
            density /= width_z[bounds, None, None]
            density /= width_y[None, :, None]
            density /= width_x[None, None, :]
            peak = maximum_filter(density, size=3, mode="constant")
            density = density[z - first:stop - first]
            peak = peak[z - first:stop - first]
            # A dense surface is close to the ordinary half-occupancy contour.
            # For thin support with no dense neighbor, keep local peaks just
            # above the contour. Full normalization would make them too wide.
            amplitude = 0.52 + 0.48 * np.sqrt(peak)
            np.maximum(peak, 1e-12, out=peak)
            density /= peak
            density *= amplitude * 255
            np.clip(density, 0, 255, out=density)
            np.rint(density, out=density)
            field[z:stop] = density
        return field


def _counts_for_extent(shape, extent_xyz, *, max_count_bytes=None):
    if extent_xyz is None or extent_xyz[1] < extent_xyz[0]:
        return None
    try:
        return NativeOverviewCounts(shape, max_count_bytes=max_count_bytes, bounds_zyx=tuple(
            (int(extent_xyz[axis]), int(extent_xyz[axis + 1]) + 1) for axis in (4, 2, 0)
        ))
    except MemoryError:
        if max_count_bytes is None:
            raise
        return None


def _restore_counts(field, row, expected):
    if row is None or row[0] != len(row[1]):
        return False
    decoder = zlib.decompressobj()
    raw = decoder.decompress(row[1], field.data.nbytes + 1)
    if len(raw) != field.data.nbytes or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        return False
    values = np.frombuffer(raw, dtype=field.data.dtype)
    if values.max(initial=0) > field.step**3 or (expected is not None and values.sum(dtype=np.uint64) != expected):
        return False
    field.data[:] = values.reshape(field.data.shape)
    return True


@contextmanager
def _open_cache(path, mode):
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode) or before.st_nlink > 1:
        raise ValueError("Unsafe native overview cache path")
    with closing(sqlite3.connect(path.absolute().as_uri() + "?mode=" + mode, uri=True, timeout=1)) as db:
        db.execute("PRAGMA trusted_schema=OFF")
        after = path.lstat()
        if (after.st_nlink != before.st_nlink or after.st_mode != before.st_mode
                or (after.st_dev, after.st_ino) != (before.st_dev, before.st_ino)
                or db.execute("PRAGMA application_id").fetchone()[0] != _CACHE_APPLICATION_ID):
            raise ValueError("Foreign native overview cache path")
        yield db


def _save_cached_counts(path, key, field):
    from .run_transport import _publish_fresh

    payload = zlib.compress(field.data, 1)
    insert = "INSERT OR REPLACE INTO surface_counts_v1 VALUES (?, ?, ?, ?)"
    if path.exists() or path.is_symlink():
        with _open_cache(path, "rw") as db, db:
            db.execute(insert, (*key, payload))
        return
    # Initialize the private database before publishing its application ID.
    # A competing process may already have created the destination on exit.
    with tempfile.TemporaryDirectory(prefix=".xta-overview-", dir=path.parent) as staging:
        temporary = Path(staging) / "overview.sqlite"
        with closing(sqlite3.connect(temporary)) as db, db:
            db.execute(f"PRAGMA application_id={_CACHE_APPLICATION_ID}")
            db.execute("CREATE TABLE surface_counts_v1 "
                       "(identity TEXT, layout TEXT, cutoff INTEGER, packed BLOB NOT NULL, "
                       "PRIMARY KEY(identity,layout,cutoff))")
            db.execute(insert, (*key, payload))
        _publish_fresh(temporary, path)


def original_overview_counts(store, cutoff):
    """Read or transparently cache the original Global's fractional overview."""
    from .run_transport import TransportError

    layer = store.layers[store.global_layer_id] if store.global_layer_id is not None else None
    field = _counts_for_extent(store.shape, None if layer is None else layer["extent_by_cutoff"].get(cutoff))
    if field is None:
        return NativeOverviewCounts(store.shape, bounds_zyx=((0, 1),) * 3)
    metadata = layer.get("segments_by_cutoff", {})
    expected = metadata.get(str(cutoff), metadata.get(cutoff, {})).get("foreground_voxels")
    limit = field.data.nbytes + field.data.nbytes // 100 + 128
    layout = field.layout_key()
    try:
        with store.read_connection() as db:
            row = db.execute(
                "SELECT length(packed), substr(packed, 1, ?) FROM native_surface_counts_v1 "
                "WHERE cutoff=? AND layout=?", (limit, int(cutoff), layout),
            ).fetchone()
            if _restore_counts(field, row, expected):
                return field
    except (sqlite3.Error, zlib.error):
        pass

    path = Path(store.path)
    cache = path.with_name(path.name + ".overview.sqlite")
    key = (store.identity, layout, int(cutoff))
    try:
        with _open_cache(cache, "ro") as db:
            row = db.execute(
                "SELECT length(packed), substr(packed, 1, ?) FROM surface_counts_v1 "
                "WHERE identity=? AND layout=? AND cutoff=?",
                (limit, *key),
            ).fetchone()
            if _restore_counts(field, row, expected):
                return field
    except (OSError, ValueError, sqlite3.Error, zlib.error):
        pass

    for index in store.iter_blocks(store.global_layer_id, cutoff):
        bounds = store.block_bounds(index)
        field.add(tuple(a for a, _ in bounds), store.read_block(store.global_layer_id, cutoff, index))
    try:
        _save_cached_counts(cache, key, field)
    except (OSError, ValueError, sqlite3.Error, TransportError):
        # A read-only study remains usable with its in-memory overview.
        pass
    return field

"""Reconstruct an edited binary preview without consulting native predictions.

The signed Euclidean distance field is measured in native voxel units. Linear
interpolation samples that field at native voxel centers and its zero contour
defines an inward-smoothed subset of the nearest-cell edited foreground. Edited
background cells remain entirely background. A one-cell background border
supplies the boundary condition at the edge of the preview. This reconstructs
edited shape without recovering detail absent from the preview.

The distance field is preview-sized; native floats exist only in bounded output
blocks. scipy's EDT workspace is also preview-sized. Linear distance interpolation
can split very thin shapes. The writer checks every edited voxel center and the
6-connected foreground-component count against the actual written native grid.
It refuses publication if either changes. The background-cell restriction prevents
connections across edited background; no arbitrary subvoxel topology is promised.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import gzip
import hashlib
import math
import os
from pathlib import Path
import tempfile

import numpy as np
from scipy import ndimage

try:
    from isal.igzip import IGzipFile as _IGzipFile
except ImportError:
    _IGzipFile = None


VERSION = '1.0.0'
ALGORITHM = 'edited-binary-signed-distance-inward-cell-centers/2'
POLICY = ALGORITHM
MIB = 1024 ** 2


def _shape3(value):
    if (not isinstance(value, (tuple, list)) or len(value) != 3
            or any(isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer))
                   or v <= 0 for v in value)):
        raise ValueError('Expected three positive integer TYX dimensions.')
    return tuple(int(v) for v in value)


def _parameters(preview, full_shape, block_mib, workers):
    a = np.asarray(preview)
    if a.ndim != 3 or any(n == 0 for n in a.shape):
        raise ValueError('The edited preview must be a nonempty three-dimensional array.')
    if a.dtype.kind not in 'buif' or np.any((a != 0) & (a != 1)):
        raise ValueError('The edited preview must contain only binary values 0 and 1.')
    full = _shape3(full_shape)
    if any(n < l for n, l in zip(full, a.shape)):
        raise ValueError('Edited-shape reconstruction supports upsampling only.')
    if not math.isfinite(block_mib) or block_mib <= 0:
        raise ValueError('block_mib must be positive and finite.')
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    return a, full


def _cell_indices(source_size, target_size, start=0, stop=None):
    """Exact nearest-cell mapping, including centers landing on a cell face."""
    if stop is None:
        stop = target_size
    return ((2 * np.arange(start, stop, dtype=np.int64) + 1) * source_size
            // (2 * target_size))


def signed_distance_field(preview, full_shape):
    """Return padded float32 signed distances, positive inside edited foreground.

    Opposite centers adjacent to a planar boundary have distances +pitch and
    -pitch. Their interpolated zero lies at the cell face; subtracting a half
    voxel or extending a binary label by one cell would incorrectly change size.
    """
    a, full = _parameters(preview, full_shape, 1, 1)
    return _signed_distance_field(a, full)


def _signed_distance_field(a, full):
    """Build the field after the public entry point has validated its inputs."""
    pitch = np.asarray(full, np.float64) / np.asarray(a.shape, np.float64)
    padded = np.pad(a.astype(bool, copy=False), 1, mode='constant')
    if not np.any(padded):
        return np.full(padded.shape, -float(max(pitch)), np.float32)
    distance = ndimage.distance_transform_edt(padded, sampling=pitch)
    field = distance.astype(np.float32)
    del distance
    np.logical_not(padded, out=padded)
    distance = ndimage.distance_transform_edt(padded, sampling=pitch)
    np.subtract(field, distance, out=field, casting='unsafe')
    return field


def block_plan(full_shape, block_mib=64, workers=1):
    """Describe row-task granularity without allocating or processing a volume.

    The budget covers aggregate native interpolation/result buffers, not the
    preview distance field or scipy EDT workspace. Large worker allocations need
    a correspondingly larger explicit budget to avoid very small row tasks.
    """
    full = _shape3(full_shape)
    if not math.isfinite(block_mib) or block_mib <= 0:
        raise ValueError('block_mib must be positive and finite.')
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError('workers must be a positive integer.')
    rows = max(1, min(full[1], int(block_mib * MIB // (full[2] * 10 * workers))))
    return {'requested_workers': workers, 'row_block_height': rows,
            'scheduled_blocks': full[0] * math.ceil(full[1] / rows),
            'aggregate_native_buffer_budget_mib': block_mib}


def _blocks(field, preview, full, block_mib, workers):
    # Bound the aggregate float/binary result buffers across workers. One row is
    # the unavoidable minimum; no full native float volume is ever allocated.
    rows = block_plan(full, block_mib, workers)['row_block_height']
    scale = np.asarray(preview.shape, np.float64) / np.asarray(full, np.float64)
    matrix = np.diag(scale)
    offset = .5 * scale + .5  # cell centers plus the padded input border
    source_x = _cell_indices(preview.shape[2], full[2])

    def work(item):
        t, y, n = item
        source_t = (2 * t + 1) * preview.shape[0] // (2 * full[0])
        source_y = _cell_indices(preview.shape[1], full[1], y, y + n)
        trusted = preview[source_t][np.ix_(source_y, source_x)].astype(bool)
        foreground_rows = np.flatnonzero(np.any(trusted, axis=1))
        output = np.zeros((n, full[2]), np.uint8)
        if not foreground_rows.size:
            return t, y, output
        # Keep the original affine origin exactly. Cropping leading rows/columns
        # would reassociate floating-point coordinate arithmetic at zero ties.
        # Trailing background needs no interpolation because POLICY/2 excludes it.
        height = int(foreground_rows[-1]) + 1
        width = int(np.flatnonzero(np.any(trusted, axis=0))[-1]) + 1
        start = np.array([t, y, 0], np.float64)
        values = ndimage.affine_transform(
            field, matrix, offset=offset + scale * start,
            output_shape=(1, height, width), output=np.float32,
            order=1, mode='constant', cval=-float(max(full)), prefilter=False)
        output[:height, :width] = (values[0] > 0) & trusted[:height, :width]
        return t, y, output

    tasks = ((t, y, min(rows, full[1] - y))
             for t in range(full[0]) for y in range(0, full[1], rows))
    if workers == 1:
        for task in tasks:
            yield work(task)
        return
    # Executor.map eagerly submits all tasks on older Python versions. Keep a
    # bounded ordered queue instead so completed native blocks cannot accumulate.
    with ThreadPoolExecutor(max_workers=workers) as executor:
        pending = []
        for _ in range(workers):
            task = next(tasks, None)
            if task is not None:
                pending.append(executor.submit(work, task))
        while pending:
            result = pending.pop(0).result()
            task = next(tasks, None)
            if task is not None:
                pending.append(executor.submit(work, task))
            yield result


def iter_edited_shape_slabs(preview, full_shape, block_mib=64, workers=1):
    """Yield ordered ``(t, y_start, binary_rows)`` at native voxel spacing."""
    a, full = _parameters(preview, full_shape, block_mib, workers)
    field = _signed_distance_field(a, full)
    yield from _blocks(field, a, full, block_mib, workers)


class _ConnectedComponents6:
    """Count 3D face-connected foreground using two labeled planes at a time.

    Previously visited components absent from the current plane cannot reconnect
    later, so finalize them and compact the remaining component IDs every plane.
    Memory is bounded by plane area rather than native volume or path length.
    """
    def __init__(self):
        self.previous = None
        self.active = 0
        self.finished = 0

    def add(self, plane, foreground_voxels=None):
        if foreground_voxels == 0:
            self.finished += self.active
            self.active = 0
            self.previous = None
            return
        labels, count = ndimage.label(plane)
        if not count:
            self.finished += self.active
            self.active = 0
            self.previous = None
            return
        if self.previous is None:
            self.previous = labels
            self.active = count
            return
        old_count = self.active
        parent = np.arange(old_count + count + 1, dtype=np.int64)

        def root(i):
            while parent[i] != i:
                parent[i] = parent[parent[i]]
                i = parent[i]
            return int(i)

        overlap = (self.previous != 0) & (labels != 0)
        if old_count == 1 and count == 1:
            pairs = (count + 2,) if np.any(overlap) else ()
        else:
            pairs = np.unique(self.previous[overlap].astype(np.uint64) * (count + 1)
                              + labels[overlap].astype(np.uint64))
        for pair in pairs:
            old, new = divmod(int(pair), count + 1)
            left, right = root(old), root(old_count + new)
            if left != right:
                parent[right] = left
        current_roots = np.array([root(old_count + i) for i in range(1, count + 1)])
        unique_roots, current_ids = np.unique(current_roots, return_inverse=True)
        self.finished += len({root(i) for i in range(1, old_count + 1)} - set(unique_roots))
        lookup = np.zeros(count + 1, dtype=np.int32)
        lookup[1:] = current_ids + 1
        self.previous = lookup[labels]
        self.active = len(unique_roots)

    @property
    def count(self):
        return self.finished + self.active


def _component_count(preview):
    counter = _ConnectedComponents6()
    for plane in preview:
        counter.add(plane)
    return counter.count


def _digest(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 * MIB), b''):
            h.update(data)
    return h.hexdigest()


def _gzip_type():
    """Select gzip before I/O; decoding failures never trigger a codec retry."""
    selected = os.environ.get('XTA_NATIVE_GZIP_BACKEND', 'auto').lower()
    if selected not in ('auto', 'stdlib', 'isal'):
        raise ValueError('XTA_NATIVE_GZIP_BACKEND must be auto, stdlib, or isal.')
    if selected == 'isal' and _IGzipFile is None:
        raise RuntimeError('XTA_NATIVE_GZIP_BACKEND=isal requires the optional isal package.')
    return _IGzipFile if selected != 'stdlib' and _IGzipFile is not None else gzip.GzipFile


def _validate_written(path, preview, full, block_mib):
    """Read the gzip bytes and sample actual native centers, not the input field."""
    low = preview.shape
    indices = [_cell_indices(n, l) for l, n in zip(low, full)]
    # Each source center has a unique nearest native center during upsampling.
    by_t = {int(t): i for i, t in enumerate(indices[0])}
    rows = max(1, min(full[1], int(block_mib * MIB // full[2])))
    digest = hashlib.sha256()
    foreground = false_positive = false_negative = sampled = 0
    background_intrusion = 0
    components = _ConnectedComponents6()
    source_components = _component_count(preview)
    guard_y = _cell_indices(low[1], full[1])
    guard_x = _cell_indices(low[2], full[2])
    with path.open('rb') as raw:
        header_bytes = 0
        while True:
            line = raw.readline()
            header_bytes += len(line)
            if not line or header_bytes > 65536:
                raise ValueError('Invalid written NRRD header.')
            if line == b'\n':
                break
        with _gzip_type()(fileobj=raw, mode='rb') as compressed:
            for t in range(full[0]):
                plane = np.empty(full[1:], np.uint8)
                plane_foreground = 0
                trusted_t = (2 * t + 1) * low[0] // (2 * full[0])
                for y in range(0, full[1], rows):
                    n = min(rows, full[1] - y)
                    data = compressed.read(n * full[2])
                    if len(data) != n * full[2]:
                        raise ValueError('Truncated written NRRD payload.')
                    digest.update(data)
                    block = np.frombuffer(data, np.uint8).reshape(n, full[2])
                    nonzero = int(np.count_nonzero(block))
                    if nonzero and np.any(block > 1):
                        raise ValueError('Written NRRD payload is nonbinary.')
                    foreground += nonzero
                    plane_foreground += nonzero
                    plane[y:y + n] = block
                    if nonzero:
                        trusted = preview[trusted_t][np.ix_(guard_y[y:y + n], guard_x)]
                        background_intrusion += int(np.count_nonzero((block != 0) & (trusted == 0)))
                    source_t = by_t.get(t)
                    if source_t is not None:
                        source_y = np.flatnonzero((indices[1] >= y) & (indices[1] < y + n))
                        if source_y.size:
                            actual = block[np.ix_(indices[1][source_y] - y, indices[2])].astype(bool)
                            expected = preview[source_t, source_y, :].astype(bool)
                            false_positive += int(np.count_nonzero(actual & ~expected))
                            false_negative += int(np.count_nonzero(~actual & expected))
                            sampled += actual.size
                components.add(plane, foreground_voxels=plane_foreground)
            if compressed.read(1):
                raise ValueError('Written NRRD payload exceeds its declared shape.')
    audit = {'sampled_voxels': sampled, 'false_positive_voxels': false_positive,
             'false_negative_voxels': false_negative,
             'sampling': 'nearest-native-voxel-center-to-each-edited-voxel-center'}
    if sampled != preview.size or false_positive or false_negative:
        raise ValueError('Edited-shape center roundtrip failed: '
                         f'{false_positive} false positives, {false_negative} false negatives; '
                         'the native grid cannot safely represent this interpolated edit.')
    if background_intrusion:
        raise ValueError(f'Written reconstruction entered {background_intrusion} edited background cells.')
    if components.count != source_components:
        raise ValueError('Edited-shape foreground component count changed: '
                         f'{source_components} edited, {components.count} reconstructed; '
                         'the interpolation cannot safely preserve this edited separation.')
    topology = {'connectivity': 6, 'source_foreground_components': source_components,
                'output_foreground_components': components.count,
                'foreground_voxels_in_edited_background_cells': background_intrusion,
                'sampling': 'actual-written-native-grid'}
    return foreground, digest.hexdigest(), audit, topology


def write_edited_shape(path, preview, full_shape, block_mib=64, workers=1):
    """Atomically write a faithful, smooth binary native-size NRRD.

    An existing destination is refused. The temporary output is removed on any
    write/roundtrip failure. No native component prediction is read or combined.
    """
    a, full = _parameters(preview, full_shape, block_mib, workers)
    path = Path(path)
    if os.path.lexists(path):
        raise FileExistsError(path)
    codec = _gzip_type()
    t, y, x = full
    header = ('NRRD0005\n# Authoritative edited preview signed-distance reconstruction\n'
              'type: unsigned char\ndimension: 3\nspace: left-posterior-superior\n'
              f'sizes: {x} {y} {t}\nspace directions: (1,0,0) (0,1,0) (0,0,1)\n'
              'kinds: domain domain domain\nencoding: gzip\nspace origin: (0,0,0)\n\n')
    temporary = None
    count = 0
    payload_hash = hashlib.sha256()
    try:
        with tempfile.NamedTemporaryFile(mode='wb', prefix=path.name + '.', suffix='.tmp',
                                         dir=path.parent, delete=False) as raw:
            temporary = Path(raw.name)
            raw.write(header.encode('ascii'))
            with codec(filename='', fileobj=raw, mode='wb', compresslevel=1, mtime=0) as z:
                for _, _, block in _blocks(_signed_distance_field(a, full), a, full, block_mib, workers):
                    data = memoryview(np.ascontiguousarray(block)).cast('B')
                    z.write(data)
                    payload_hash.update(data)
                    count += int(np.count_nonzero(block))
        actual_count, actual_hash, audit, topology = _validate_written(temporary, a, full, block_mib)
        if actual_count != count or actual_hash != payload_hash.hexdigest():
            raise ValueError('Written reconstruction differs from generated payload.')
        metadata = {
            'shape_tyx': list(full), 'foreground_voxels': count,
            'binary_payload_sha256': actual_hash, 'sha256': _digest(temporary),
            'bytes': temporary.stat().st_size,
            'reconstruction': {
                'algorithm': ALGORITHM, 'version': VERSION,
                'source_shape_tyx': list(a.shape),
                'source_foreground_voxels': int(np.count_nonzero(a)),
                'preview_pitch_native_tyx': [n / l for l, n in zip(a.shape, full)],
                'gzip_backend': 'isal' if codec is _IGzipFile else 'stdlib',
                'execution': block_plan(full, block_mib, workers),
                'center_roundtrip': audit,
                'separation_audit': topology,
                'smoothing': 'inward only; edited background cells remain background',
                'topology_scope': 'background cells, voxel centers and foreground 6-components audited; arbitrary subvoxel topology is not guaranteed',
            },
        }
        # Exclusive publication also refuses dangling symlinks and a destination
        # created while reconstruction was running. Both paths share a parent,
        # so the hard link publishes the complete file atomically on one volume.
        os.link(temporary, path)
        return metadata
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()

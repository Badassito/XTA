#!/usr/bin/env python3
"""Transfer deletion-only Slicer preview edits to ORIGINAL XTA binary volumes.

Native connected components are selected, not resized into existence. Processing
uses cropped slab labeling, exact inter-slab face equivalences, foreground-run
storage, and concurrent layers. See ../docs/slicer_pta_cleanup.md for the ambiguity
and memory contracts. It runs without importing XTA inference code.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import gzip
import hashlib
import json
import math
import mmap
import multiprocessing
import os
from pathlib import Path
import re
import shutil
import sys
import tempfile
import threading
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from datetime import datetime, timezone

import numpy as np
from scipy import ndimage as ndi
from xta_native_io import gzip_backend, open_gzip_reader

VERSION = '0.3.1'
FORMAT = 'xta-slicer-edits-v1'
FACE = ndi.generate_binary_structure(3, 1)
MIB = 1024 ** 2


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(4 * MIB), b''):
            h.update(b)
    return h.hexdigest()


def load_json(path: Path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def save_json(path: Path, obj):
    temp = path.with_name(path.name + '.tmp')
    with open(temp, 'w', encoding='utf-8') as f:
        json.dump(obj, f, indent=2, ensure_ascii=False, allow_nan=False)
        f.write('\n')
    os.replace(temp, path)


def child(root: Path, name: str) -> Path:
    """Manifest and bundle members must be relative files within their directory."""
    root = root.resolve()
    raw = Path(name)
    if raw.is_absolute() or '..' in raw.parts:
        raise ValueError(f'Unsafe relative filename: {name!r}')
    p = (root / raw).resolve()
    if not p.is_relative_to(root):
        raise ValueError(f'Path escapes its manifest directory: {name!r}')
    return p


def shape3(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f'Invalid (t,Y,X) shape: {value!r}')
    if any(isinstance(v, (bool, np.bool_)) or
           not (isinstance(v, (int, np.integer)) or
                isinstance(v, str) and re.fullmatch(r'[0-9]+', v)) for v in value):
        raise ValueError(f'Invalid (t,Y,X) shape: {value!r}')
    shape = tuple(int(v) for v in value)
    if any(v < 1 for v in shape):
        raise ValueError(f'Invalid (t,Y,X) shape: {value!r}')
    return shape


def read_header(path: Path):
    fields = {}
    with open(path, 'rb') as f:
        if not f.readline().startswith(b'NRRD000'):
            raise ValueError(f'Not an NRRD: {path}')
        while True:
            raw = f.readline(4 * MIB + 1)
            if f.tell() > 4 * MIB or not raw:
                raise ValueError(f'Missing/oversized NRRD header: {path}')
            if not raw.strip():
                offset = f.tell()
                break
            line = raw.decode('utf-8').strip()
            if line.startswith('#'):
                continue
            sep = ':=' if ':=' in line else ':'
            if sep not in line:
                raise ValueError(f'Invalid NRRD header line in {path}: {line!r}')
            key, value = line.split(sep, 1)
            key = key.strip()
            if key in fields:
                raise ValueError(f'Duplicate NRRD header field: {key}')
            fields[key] = value.strip()
    if fields.get('dimension') != '3':
        raise ValueError(f'Expected a single-layer, 3D XTA NRRD: {path}')
    if fields.get('type', '').lower() not in {'uchar', 'unsigned char', 'uint8', 'uint8_t'}:
        raise ValueError(f'Only XTA uint8 binary NRRDs are supported: {path}')
    if fields.get('encoding', '').lower() not in {'raw', 'gzip', 'gz'}:
        raise ValueError(f'Only inline raw/gzip NRRDs are supported: {path}')
    for field in ('data file', 'datafile', 'byte skip', 'byteskip', 'line skip', 'lineskip'):
        if field in fields:
            raise ValueError(f'Unsupported NRRD field {field!r} in {path}')
    if fields.get('Segment0_LabelValue', '1') != '1' or fields.get('Segment0_Layer', '0') != '0':
        raise ValueError(f'Expected binary label 1 in layer 0: {path}')
    if any(re.match(r'Segment[1-9]\d*_', k) for k in fields):
        raise ValueError(f'Multiple segments in an original XTA layer: {path}')
    shape = shape3(list(reversed(fields['sizes'].split())))
    return fields, offset, shape


def ijk_to_ras(fields):
    space = fields.get('space', '').lower()
    if space in ('left-posterior-superior', 'lps'):
        convention = np.diag([-1., -1., 1.])
    elif space in ('right-anterior-superior', 'ras'):
        convention = np.eye(3)
    else:
        raise ValueError(f'Unsupported/missing NRRD coordinate space: {space!r}')
    def vector(s):
        return [float(x) for x in s.strip('()').split(',')]
    vectors = re.findall(r'\([^)]*\)', fields.get('space directions', ''))
    if len(vectors) != 3:
        raise ValueError('Three explicit NRRD space-direction vectors are required.')
    matrix = np.eye(4)
    matrix[:3, :3] = convention @ np.asarray([vector(v) for v in vectors]).T
    matrix[:3, 3] = convention @ np.asarray(vector(fields.get('space origin', '')))
    if not np.all(np.isfinite(matrix)) or abs(np.linalg.det(matrix[:3, :3])) < 1e-12:
        raise ValueError('Invalid NRRD geometry.')
    return matrix


def validate_xta_geometry(fields):
    # XTA's published preview/native coordinates each use their own unit lattice.
    # A generic world-space resampler would incorrectly assume they have equal FOV.
    matrix = ijk_to_ras(fields)
    expected = np.diag([-1., -1., 1., 1.])
    if not np.allclose(matrix, expected, rtol=0, atol=1e-7):
        raise ValueError('Expected the original XTA unit-LPS output lattice; geometry was changed.')


class Nrrd:
    def __init__(self, path, shape=None):
        self.path = Path(path)
        self.header, self.offset, self.shape = read_header(self.path)
        validate_xta_geometry(self.header)
        if shape is not None and tuple(shape) != self.shape:
            raise ValueError(f'NRRD/manifest shape mismatch: {self.path}: {self.shape} != {shape}')

    def slabs(self, depth):
        plane = self.shape[1] * self.shape[2]
        with open(self.path, 'rb') as raw:
            raw.seek(self.offset)
            with context_payload(raw, self.header['encoding']) as f:
                for z in range(0, self.shape[0], depth):
                    count = min(depth, self.shape[0] - z)
                    need = count * plane
                    data = bytearray(need)
                    view = memoryview(data)
                    pos = 0
                    while pos < need:
                        n = f.readinto(view[pos:])
                        if not n:
                            raise ValueError(f'Truncated NRRD payload: {self.path}')
                        pos += n
                    a = np.frombuffer(data, dtype=np.uint8).reshape(count, *self.shape[1:])
                    if a.max(initial=0) > 1:
                        raise ValueError(f'Nonbinary original XTA data: {self.path}')
                    yield z, a
                if f.read(1):
                    raise ValueError(f'Extra bytes after NRRD payload: {self.path}')

    def array(self):
        a = np.empty(self.shape, np.uint8)
        for z, block in self.slabs(8):
            a[z:z + len(block)] = block
        return a


@contextmanager
def context_payload(raw, encoding):
    if encoding.lower() in ('gzip', 'gz'):
        # Both supported readers consume concatenated members and verify CRCs.
        with open_gzip_reader(raw) as f:
            yield f
    else:
        yield raw


def write_nrrd(path, shape, slabs, name, *, color='0.8 0.3 0.2', level=1):
    """Write a full-reference binary NRRD, one bounded slab at a time."""
    path = Path(path)
    shape = shape3(shape)
    clean_name = str(name).replace('\r', ' ').replace('\n', ' ')
    temp = path.with_name(path.name + '.tmp')
    t, y, x = shape
    text = ('NRRD0004\n'
            'type: unsigned char\ndimension: 3\nspace: left-posterior-superior\n'
            f'sizes: {x} {y} {t}\n'
            'space directions: (1,0,0) (0,1,0) (0,0,1)\n'
            'kinds: domain domain domain\nencoding: gzip\nspace origin: (0,0,0)\n'
            f'content: binary segmentation mask; source_shape_tyx=({t},{y},{x}); exported_axes=(X,Y,t)\n'
            'Segmentation_ContainedRepresentationNames:=Binary labelmap|\n'
            'Segmentation_SourceRepresentation:=Binary labelmap\n'
            'Segment0_ID:=Segment_1\n'
            f'Segment0_Name:={clean_name}\nSegment0_NameAutoGenerated:=0\n'
            f'Segment0_Color:={color}\nSegment0_ColorAutoGenerated:=0\n'
            'Segment0_LabelValue:=1\nSegment0_Layer:=0\n'
            f'Segment0_Extent:=0 {x-1} 0 {y-1} 0 {t-1}\n\n')
    total = 0
    try:
        with open(temp, 'wb') as raw:
            raw.write(text.encode('utf-8'))
            with gzip.GzipFile(filename='', fileobj=raw, mode='wb', compresslevel=level, mtime=0) as f:
                for z, a in slabs:
                    if z != total or a.shape[1:] != shape[1:]:
                        raise ValueError('Noncontiguous output slab sequence.')
                    a = np.ascontiguousarray(a, dtype=np.uint8)
                    if a.max(initial=0) > 1:
                        raise ValueError('Output must be binary.')
                    f.write(memoryview(a).cast('B'))
                    total += len(a)
        if total != t:
            raise ValueError(f'Incomplete output: {total} of {t} planes.')
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


@contextmanager
def mapped(path, shape, dtype=np.uint8):
    a = np.memmap(path, mode='w+', dtype=dtype, shape=shape)
    try:
        yield a
    finally:
        a.flush()
        a._mmap.close()
        Path(path).unlink(missing_ok=True)


def evict(a):
    """Best effort only: file-backed pages are still managed by the operating system."""
    if isinstance(a, np.memmap):
        a.flush()
        if hasattr(a._mmap, 'madvise') and hasattr(mmap, 'MADV_DONTNEED'):
            try:
                a._mmap.madvise(mmap.MADV_DONTNEED)
            except (OSError, ValueError):
                pass


def array_slabs(a, depth):
    for z in range(0, a.shape[0], depth):
        yield z, a[z:z + depth]


class Support:
    """Inverse POSITIVE sampling footprint, used only for decision evidence.

    For true downbins, XTA's CPU area and GPU adaptive-max paths share this
    potential footprint. Quantization may suppress CPU contributors; consequently
    this envelope is conservative, not a claim to invert the published resize.
    """
    def __init__(self, full, low):
        self.full, self.low = tuple(full), tuple(low)
        if any(m > n for n, m in zip(full, low)):
            raise ValueError('Preview axes larger than native axes are not supported.')
        self.indices = []
        for n, m in zip(full, low):
            i = np.arange(n, dtype=np.int64)
            lo = i * m // n
            hi = ((i + 1) * m + n - 1) // n - 1
            self.indices.append((lo, hi))

    def gather(self, low_mask, z, depth):
        if tuple(low_mask.shape) != self.low:
            raise ValueError('Decision-mask shape mismatch.')
        out = np.zeros((depth, *self.full[1:]), dtype=bool)
        zz, yy, xx = self.indices
        for zs in (zz[0][z:z+depth], zz[1][z:z+depth]):
            for ys in yy:
                for xs in xx:
                    np.logical_or(out, low_mask[np.ix_(zs, ys, xs)], out=out)
        return out


class PreviewSupportCheck:
    """Require every positive ORIGINAL preview cell to have native source support.

    This necessary condition detects incompatible source pairs and unsupported
    direct-from-backing mirror geometry. It is NOT proof of the same TTA run and
    does not assume which area/max reduction backend produced the preview.
    """
    def __init__(self, full_shape, original_preview):
        self.full = tuple(full_shape)
        self.original = original_preview
        self.seen = np.zeros(original_preview.shape, dtype=bool)
        self.indices = Support(full_shape, original_preview.shape).indices
        self.inverse = self.indices[0]
        axes = []
        for n, m in zip(full_shape[1:], original_preview.shape[1:]):
            i = np.arange(m, dtype=np.int64)
            axes.append((i*n//m, ((i+1)*n+m-1)//m))
        self.y, self.x = axes
        self.y_overlap = np.flatnonzero(self.y[1][:-1] > self.y[0][1:])
        self.x_overlap = np.flatnonzero(self.x[1][:-1] > self.x[0][1:])

    def observe(self, z, block):
        y0,y1 = self.y
        x0,x1 = self.x
        occupied = False
        for offset, frame in enumerate(block):
            rows = np.flatnonzero(np.any(frame, axis=1))
            if not len(rows):
                continue
            occupied = True
            sparse_rows = len(rows) * 8 < frame.shape[0]
            source = frame[rows] if sparse_rows else frame
            columns = np.logical_or.reduceat(source, x0, axis=1)
            # Fractional cell boundaries include the first pixel of the next bin.
            columns[:, self.x_overlap] |= source[:, x1[self.x_overlap]-1] != 0
            if sparse_rows:
                projected = np.zeros(self.original.shape[1:], dtype=bool)
                # Only occupied native rows contribute. Both inverse endpoints
                # preserve fractional overlap with adjacent preview cells.
                for endpoints in self.indices[1]:
                    np.logical_or.at(projected, endpoints[rows], columns)
            else:
                projected = np.logical_or.reduceat(columns, y0, axis=0)
                projected[self.y_overlap] |= columns[y1[self.y_overlap]-1]
            zz = z + offset
            lo, hi = (int(a[zz]) for a in self.inverse)
            for target in range(lo,hi+1):
                self.seen[target] |= projected
        return occupied

    def observe_points(self, z, y, x):
        choices = [(lo[coordinates], hi[coordinates])
                   for (lo, hi), coordinates in zip(self.indices, (z, y, x))]
        for zz in choices[0]:
            for yy in choices[1]:
                for xx in choices[2]:
                    np.logical_or.at(self.seen, (zz, yy, xx), True)

    def finish(self):
        unsupported = int(np.count_nonzero((self.original != 0) & ~self.seen))
        unseen_native = int(np.count_nonzero((self.original == 0) & self.seen))
        if unsupported:
            raise ValueError(f'{unsupported} original preview foreground cells have NO corresponding '
                             'native sampling support. Wrong source pair, modified files, or unsupported '
                             'mirror fallback geometry; refusing transfer.')
        return {'unsupported_preview_foreground_cells': unsupported,
                'native_support_cells_absent_from_preview': unseen_native,
                'contract': 'necessary positive-support check, not exact preview reconstruction or run identity'}


class Components:
    """Union/find tables are O(provisional component count), with an explicit cap."""
    def __init__(self, maximum):
        self.maximum = int(maximum)
        self.capacity = min(65536, maximum + 1)
        self.parent = np.zeros(self.capacity, np.int32)
        self.count = np.zeros(self.capacity, np.uint64)
        self.flags = np.zeros(self.capacity, np.uint8)
        self.n = 0

    def add(self, counts, flags):
        n = len(counts) - 1
        if self.n + n > self.maximum:
            raise ValueError(f'Provisional component cap {self.maximum:,} exceeded; '
                             'increase --max-components or use larger slabs. No final output published.')
        needed = self.n + n + 1
        if needed > self.capacity:
            capacity = min(self.maximum + 1, max(needed, self.capacity * 2))
            for attr in ('parent', 'count', 'flags'):
                old = getattr(self, attr)
                new = np.zeros(capacity, old.dtype)
                new[:self.n+1] = old[:self.n+1]
                setattr(self, attr, new)
            self.capacity = capacity
        ids = np.arange(self.n + 1, self.n + n + 1, dtype=np.int32)
        self.parent[ids] = ids
        self.count[ids] = counts[1:]
        self.flags[ids] = flags[1:]
        offset = self.n
        self.n += n
        return offset

    def root(self, i):
        p = self.parent
        i = int(i)
        while p[i] != i:
            p[i] = p[p[i]]
            i = int(p[i])
        return i

    def join(self, a, b):
        a, b = self.root(a), self.root(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)

    def finish(self):
        for i in range(1, self.n + 1):
            self.parent[i] = self.root(i)
        roots = self.parent[:self.n+1].copy()
        counts = np.zeros(self.n+1, np.uint64)
        flags = np.zeros(self.n+1, np.uint8)
        np.add.at(counts, roots, self.count[:self.n+1])
        np.bitwise_or.at(flags, roots, self.flags[:self.n+1])
        return roots, counts, flags


def label_stream(slabs, labels, *, maximum, keep=None, removed=None, support=None, clip=None, observer=None):
    """6-connectivity within slabs AND across every slab boundary, without halos."""
    table = Components(maximum)
    previous = None
    input_count = 0
    clipped_count = 0
    total = 0
    touched = 0
    next_progress = 128
    for z, block in slabs:
        if z != total:
            raise ValueError('Slabs are not sequential.')
        if observer is not None:
            observer(z, block)
        a = np.asarray(block, dtype=np.uint8).copy()
        input_count += int(np.count_nonzero(a))
        if clip is not None:
            a &= clip[z:z + len(a)]
        clipped_count += int(np.count_nonzero(a))
        local, n = ndi.label(a, structure=FACE)
        counts = np.bincount(local.ravel(), minlength=n+1).astype(np.uint64)
        counts[0] = 0
        flags = np.zeros(n+1, np.uint8)
        if keep is not None:
            hit = support.gather(keep, z, len(a))
            flags[np.unique(local[hit])] |= 1
            del hit
            hit = support.gather(removed, z, len(a))
            flags[np.unique(local[hit])] |= 2
            flags[0] = 0
            del hit
        offset = table.add(counts, flags)
        if offset:
            np.add(local, offset, out=local, where=local != 0)
        if previous is not None:
            active = (previous != 0) & (local[0] != 0)
            pairs = np.column_stack((previous[active], local[0][active]))
            for a_id, b_id in np.unique(pairs, axis=0):
                table.join(a_id, b_id)
        labels[z:z+len(local)] = local
        previous = local[-1].copy()
        total += len(local)
        touched += local.nbytes
        del a, local, counts, flags
        if total >= next_progress:
            print(f'  labeled {total}/{labels.shape[0]} planes', flush=True)
            next_progress = total + 128
        if touched >= 64*MIB:
            evict(labels)
            if clip is not None:
                evict(clip)
            touched = 0
    if total != labels.shape[0]:
        raise ValueError('Incomplete labeling input.')
    roots, counts, flags = table.finish()
    del table
    return roots, counts, flags, input_count, clipped_count


def load_edit(bundle_dir, entry, original):
    path = child(bundle_dir, entry['file'])
    if digest(path) != entry['sha256']:
        raise ValueError(f'Edited snapshot checksum mismatch: {path}')
    out = np.zeros(original.shape, np.uint8)
    with np.load(path, allow_pickle=False) as f:
        a = f['mask']
        if a.ndim != 3 or a.dtype != np.uint8 or a.max(initial=0) > 1:
            raise ValueError('Edited snapshot must be a three-dimensional uint8 binary mask.')
        if not a.size:
            if entry.get('foreground_voxels') != 0:
                raise ValueError('Invalid empty-snapshot metadata.')
            return out
        extent = np.asarray(f['extent_xyz'], dtype=np.int64)
        transform = np.linalg.solve(ijk_to_ras(original.header), f['ijk_to_ras'])
        if not np.allclose(transform[:3, :3], np.eye(3), rtol=0, atol=1e-5):
            raise ValueError(f'Edited lattice was resampled/rotated: {entry["name"]}')
        if not np.allclose(transform[3], [0, 0, 0, 1], rtol=0, atol=1e-7):
            raise ValueError('Invalid homogeneous image matrix.')
        translation = transform[:3, 3]
        rounded = np.rint(translation).astype(np.int64)
        if not np.allclose(translation, rounded, rtol=0, atol=1e-5):
            raise ValueError('Edited origin is not on the original preview voxel lattice.')
        if extent.shape != (6,) or tuple((extent[1::2]-extent[0::2]+1)[::-1]) != a.shape:
            raise ValueError('Edited snapshot extent/shape mismatch.')
        start = (extent[::2] + rounded)[::-1]
        end = start + np.asarray(a.shape)
        # Slicer may pad with zero outside the reference; only foreground outside is invalid.
        lo = np.maximum(start, 0)
        hi = np.minimum(end, original.shape)
        if np.all(hi > lo):
            src = tuple(slice(int(l-s), int(h-s)) for l,h,s in zip(lo,hi,start))
            dst = tuple(slice(int(l), int(h)) for l,h in zip(lo,hi))
            out[dst] = a[src]
        if np.count_nonzero(out) != np.count_nonzero(a):
            raise ValueError(f'Edited foreground lies outside its original reference: {entry["name"]}')
        if int(np.count_nonzero(a)) != int(entry['foreground_voxels']):
            raise ValueError('Edited snapshot voxel-count mismatch.')
    return out


def manifest_entries(manifest):
    if manifest.get('layout') != 'one_single_layer_nrrd_per_component':
        raise ValueError('Expected an XTA per-layer NRRD manifest.')
    by_filename = {}
    for e in manifest['layers']:
        key = e['filename']
        if key in by_filename:
            raise ValueError(f'Duplicate manifest filename: {key}')
        by_filename[key] = e
    return by_filename


def ensure_disjoint(output, inputs):
    output = output.resolve()
    for p in inputs:
        p = p.resolve()
        if output == p or p.is_relative_to(output) or output.is_relative_to(p):
            raise ValueError(f'Output must be separate from input directory {p}')


def run(args):
    # Validate programmatic calls too, before creating output or opening native data.
    if args.block_mib < 1 or not (1 <= args.max_components < np.iinfo(np.int32).max):
        raise ValueError('Invalid slab target or provisional component limit.')
    if args.workers < 1:
        raise ValueError('Worker count must be positive.')
    if args.keep_objects < 1 or args.auto_max_radius < 1 or args.auto_max_roi_voxels < 1:
        raise ValueError('Object count, auto radius and ROI voxel limit must be positive.')
    if not 0 <= args.auto_max_keep_loss < 1 or not 0 < args.auto_min_remove_fraction <= 1:
        raise ValueError('Invalid auto evidence guard fractions.')
    if args.conflicts == 'auto' and not getattr(args, 'validate_only', False):
        try:
            from skimage.segmentation import watershed
        except ImportError as exc:
            raise RuntimeError('Auto mode needs scikit-image; run: python -m pip install scikit-image') from exc
    bundle_dir = Path(args.edits).resolve()
    bundle_path = bundle_dir / 'edits.json'
    bundle = load_json(bundle_path)
    if bundle.get('format') != FORMAT:
        raise ValueError('Unsupported Slicer edit-bundle format.')
    low_path = Path(args.low_manifest).resolve()
    full_path = Path(args.full_manifest).resolve()
    if digest(low_path) != bundle['source_manifest_sha256']:
        raise ValueError('This is not the original preview manifest selected during export.')
    low, full = load_json(low_path), load_json(full_path)
    if low.get('quality') != 'low_quality' or full.get('quality') == 'low_quality':
        raise ValueError('Supply a low-quality manifest and a full-quality manifest, respectively.')
    low_shape, full_shape = shape3(low['output_shape_tyx']), shape3(full['output_shape_tyx'])
    if shape3(low['full_quality_output_shape_tyx']) != full_shape:
        raise ValueError('Full/preview manifest geometry mismatch.')
    support = Support(full_shape, low_shape)
    le, fe = manifest_entries(low), manifest_entries(full)
    edits = bundle['segments']
    if len({e['name'] for e in edits}) != len(edits) or len({e['source_filename'] for e in edits}) != len(edits):
        raise ValueError('Duplicate segment names/source files in the edit bundle.')
    globals_ = [e for e in edits if le.get(e['source_filename'], {}).get('suffix') == 'Global_final_output']
    if len(globals_) != 1:
        raise ValueError('Exactly one Global_final_output segment is required.')
    global_edit = globals_[0]
    layers = [e for e in edits if e is not global_edit]
    if not layers:
        raise ValueError('No nonglobal segments were exported.')
    nonadditive = []
    for e in edits:
        key = e['source_filename']
        if key not in le or key not in fe:
            raise ValueError(f'Layer missing from a manifest: {key}')
        for identity in ('suffix', 'segment_name', 'layer_key', 'model_name'):
            if le[key].get(identity) != fe[key].get(identity):
                raise ValueError(f'Layer identity mismatch ({identity}): {key}')
        if e is not global_edit and le[key].get('recomposition_op') != 'union':
            nonadditive.append(e['name'])
        source = child(low_path.parent, key)
        if digest(source) != e['source_sha256']:
            raise ValueError(f'Original preview changed since export: {source}')
        Nrrd(source, low_shape)
        Nrrd(child(full_path.parent, key), full_shape)
    if nonadditive and not args.allow_nonadditive:
        raise ValueError('Loaded nonglobal checkpoints/diagnostic layers are not additive XTA components: '
                         + ', '.join(nonadditive) + '. Remove these from the Slicer working segmentation '
                         'and repeat its update steps, or explicitly use --allow-nonadditive to reproduce '
                         'the literal toolbar union of the exported layers.')
    union_sources = None
    receipt, pairs = getattr(args, 'union_receipt', None), getattr(args, 'union_source_pairs', None)
    if bool(receipt) != bool(pairs):
        raise ValueError('--union-receipt and --union-source-pairs must be supplied together.')
    if receipt:
        from xta_native_union import UnionSources
        union_sources = UnionSources(receipt, pairs, low_shape, full_shape,
                                     low_path, full_path, edits, global_edit)
    output = Path(args.output).resolve()
    ensure_disjoint(output, [bundle_dir, low_path.parent, full_path.parent])
    if union_sources is not None:
        ensure_disjoint(output, [p['paths'][kind].parent for p in union_sources.pairs
                                 for kind in ('low', 'full')])
        if union_sources.reference_dir is not None:
            ensure_disjoint(output, [union_sources.reference_dir])
    if output.exists():
        raise ValueError('Output must be a NEW directory. Nothing will be overwritten.')
    depth = max(1, min(64, int(args.block_mib * MIB // (32 * full_shape[1] * full_shape[2]))))
    size = math.prod(full_shape)
    scratch_root = Path(args.scratch).resolve() if args.scratch else output.parent
    scratch_root.mkdir(parents=True, exist_ok=True)
    # Two shared uint8 rasters plus packed per-source Globals for mixed-source recovery.
    global_cache_bytes = union_sources.native_global_cache_bytes if union_sources is not None else 0
    required = 2 * size + global_cache_bytes + 256 * MIB
    if not getattr(args, 'validate_only', False) and shutil.disk_usage(scratch_root).free < required:
        raise ValueError(f'Scratch needs at least {required / 1024**3:.2f} GiB free (plus output storage).')
    output.mkdir(parents=True)
    report = {'tool_version': VERSION, 'status': 'in_progress',
              'created_utc': datetime.now(timezone.utc).isoformat(),
              'bundle_sha256': digest(bundle_path), 'preview_manifest_sha256': digest(low_path),
              'full_manifest_sha256': digest(full_path), 'shape_tyx': list(full_shape),
              'preview_shape_tyx': list(low_shape), 'slab_depth': depth,
              'scratch_array_bytes': 2 * size, 'max_provisional_components': args.max_components,
              'native_global_cache_bytes': global_cache_bytes,
              'native_global_cache_encoding': 'row_packed_little_bitorder' if union_sources is not None else None,
              'native_gzip_backend': gzip_backend(),
              'label_storage': 'foreground_runs', 'workers': args.workers,
              'worker_backend': 'process' if args.workers > 1 else 'thread',
              'union_write_policy': 'foreground_only_shared',
              'conflict_policy': args.conflicts, 'nonadditive_layers_explicitly_included': nonadditive,
              'keep_objects': args.keep_objects,
              'auto_parameters': {'max_radius_native_voxels': args.auto_max_radius,
                                  'max_padded_roi_voxels': args.auto_max_roi_voxels,
                                  'max_keep_evidence_loss': args.auto_max_keep_loss,
                                  'min_remove_evidence_fraction': args.auto_min_remove_fraction,
                                  'fallback': ('keep_and_report' if args.conflicts == 'keep' else
                                               'stop_without_publication'),
                                  'uses_intensity': False},
              'layers': [], 'warnings': [], 'source_full_sha256': {}}
    if union_sources is not None:
        report['source_union'] = union_sources.report()
        report['source_full_unions'] = {}
        report['warnings'].append('Mixed-source recovery uses the explicitly declared inventory-based '
                                  'source identification; edited-member cryptographic ancestry is unavailable.')
    save_json(output / 'report.json', report)
    print(f'Native {full_shape}; preview {low_shape}; slabs {depth} plane(s).', flush=True)
    print(f'Native gzip reader: {report["native_gzip_backend"]}.', flush=True)
    print(f'Shared array scratch {2*size/1024**3:.2f} GiB, packed source Globals '
          f'{global_cache_bytes/1024**3:.2f} GiB, plus sparse runs/output; {args.workers} layer worker(s).', flush=True)
    try:
        result = execute(args, output, scratch_root, report, global_edit, layers, le, fe,
                         low_path, full_path, bundle_dir, full_shape, low_shape, depth, support,
                         union_sources=union_sources)
        return result
    except BaseException as exc:
        if report['status'] == 'in_progress':
            report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        save_json(output / 'report.json', report)
        raise


_native_worker_context = None


def initialize_native_worker(context):
    global _native_worker_context
    context = dict(context)
    shape = context['full_shape']
    initial = np.memmap(context['tmp']/'initial-global.dat', mode='r', dtype=np.uint8, shape=shape)
    context['clip'] = initial if context.pop('clip_nonempty') else None
    context['union'] = np.memmap(context['tmp']/'union.dat', mode='r+', dtype=np.uint8, shape=shape)
    low_global_path = context.pop('low_global_path')
    sources = context.get('union_sources')
    context['low_global'] = (sources.low(sources.global_name) if sources is not None else
                             Nrrd(low_global_path, context['low_shape']).array())
    context['support'] = Support(shape, context['low_shape'])
    _native_worker_context = context


def process_native_worker(i, entry):
    return process_native_layer(i, entry, **_native_worker_context)


@contextmanager
def native_executor(workers, context):
    if workers == 1:
        with ThreadPoolExecutor(max_workers=1) as executor:
            yield executor
        return
    # Each worker owns one CPU task; avoid a BLAS thread pool in every process.
    names = ('OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')
    original = {name: os.environ.get(name) for name in names}
    try:
        for name in names:
            os.environ[name] = '1'
        with ProcessPoolExecutor(max_workers=workers,
                mp_context=multiprocessing.get_context('spawn'),
                initializer=initialize_native_worker, initargs=(context,)) as executor:
            yield executor
    finally:
        for name, value in original.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


def process_native_layer(i, entry, *, args, output, tmp, low_path, full_path,
                         bundle_dir, full_shape, low_shape, depth, support,
                         low_global, clip, union, edited_global_empty, union_sources=None):
    from xta_native_sparse import SparseLabels, sparse_label_stream
    started = time.perf_counter()
    print(f'Started [{i+1}] {entry["name"]}', flush=True)
    key = entry['source_filename']
    low_src = Nrrd(child(low_path.parent, key), low_shape)
    original_low = low_src.array() if union_sources is None else union_sources.low(entry['name'])
    after = load_edit(bundle_dir, entry, low_src)
    removed = (original_low != 0) & (after == 0)
    if union_sources is None and np.any(low_global):
        removed &= low_global != 0
    native = (Nrrd(child(full_path.parent, key), full_shape) if union_sources is None else
              union_sources.native(entry['name']))
    source_check = PreviewSupportCheck(full_shape, original_low) if union_sources is None else native
    source_hash = digest(native.path) if union_sources is None else native.descriptor()
    row = {'name': entry['name'], 'source_filename': key,
           'preview_removed_voxels': int(np.count_nonzero(removed)),
           'rejected_components': 0, 'rejected_voxels': 0,
           'conflicting_components': 0, 'conflicting_voxels_preserved': 0,
           'unresolved_components': 0, 'auto_resolved_components': 0,
           'auto_removed_voxels': 0, 'auto_components': []}
    component_path = tmp/f'components-{i}.tsv'
    with open(component_path, 'w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream, delimiter='\t')
        # Without deletion evidence every clipped native component is retained.
        if not np.any(removed) and not edited_global_empty:
            input_n = clipped_n = 0
            for z, block in native.slabs(depth):
                source_check.observe(z, block)
                input_n += int(np.count_nonzero(block))
                if clip is not None:
                    block &= clip[z:z+len(block)]
                clipped_n += int(np.count_nonzero(block))
                if np.any(block):
                    # Concurrent writers only store 1; no read/modify/write can lose another layer.
                    union[z:z+len(block)][block != 0] = 1
            row.update(input_voxels=input_n, after_native_global_intersection=clipped_n,
                       native_components=None, unobserved_components_preserved=None,
                       component_analysis='skipped_no_deletion_evidence', sparse_label_bytes=0)
        else:
            with SparseLabels(full_shape, tmp/f'labels-{i}.runs') as labels:
                label_started = time.perf_counter()
                roots, counts, flags, input_n, clipped_n = sparse_label_stream(
                    native.slabs(depth), labels, maximum=args.max_components,
                    keep=after, removed=removed, support=support, clip=clip,
                    observer=source_check.observe)
                row['label_seconds'] = time.perf_counter()-label_started
                row['sparse_label_bytes'] = labels.path.stat().st_size
                active = np.flatnonzero(counts)
                rejected = flags == 2
                mixed = (flags == 3) & (counts > 0)
                auto_resolved = np.zeros(len(roots), bool)
                auto_records, auto_preview = [], None
                auto_started = time.perf_counter()
                if np.any(mixed) and args.conflicts == 'auto':
                    from xta_auto import resolve_conflicts
                    auto_resolved, auto_records, auto_preview = resolve_conflicts(
                        labels, roots, mixed, support, after, removed, depth=depth, args=args)
                row['auto_seconds'] = time.perf_counter()-auto_started
                remaining_mixed = mixed & ~auto_resolved
                remaining = int(np.count_nonzero(remaining_mixed))
                for component in active:
                    flag = int(flags[component])
                    status = ('unobserved', 'keep', 'remove', 'conflict')[flag]
                    decision = ('remove_empty_global' if edited_global_empty else
                                'auto_split' if auto_resolved[component] else
                                'remove' if flag == 2 else 'keep')
                    writer.writerow([entry['name'], int(component), int(counts[component]), status, decision])
                keep_ids = np.zeros(len(roots), bool) if edited_global_empty else ~rejected
                keep_ids[0] = False
                lookup = keep_ids[roots].astype(np.uint8)
                row.update(input_voxels=input_n, after_native_global_intersection=clipped_n,
                           native_components=int(len(active)), component_analysis='sparse_foreground_runs',
                           rejected_components=int(np.count_nonzero(rejected & (counts > 0))),
                           rejected_voxels=int(counts[rejected].sum()),
                           conflicting_components=int(np.count_nonzero(mixed)),
                           conflicting_voxels_preserved=int(counts[remaining_mixed].sum()),
                           unresolved_components=remaining,
                           auto_resolved_components=int(np.count_nonzero(auto_resolved)),
                           auto_removed_voxels=sum(a.get('removed_native_voxels', 0) for a in auto_records),
                           auto_components=auto_records,
                           unobserved_components_preserved=(0 if edited_global_empty else
                               int(np.count_nonzero((flags == 0) & (counts > 0)))))
                conflict_projection = PreviewSupportCheck(full_shape, after) if remaining else None
                merge_started = time.perf_counter()
                labels.merge_into(union, lookup,
                    observer=conflict_projection.observe_points if remaining else None,
                    observer_lookup=remaining_mixed[roots] if remaining else None)
                row['merge_seconds'] = time.perf_counter()-merge_started
                for preview, prefix, field, color in (
                    (conflict_projection.seen if remaining else None, 'conflicts', 'review_preview_file', '1 0.5 0'),
                    (auto_preview, 'auto_removed', 'auto_removed_preview_file', '0.8 0.2 0.8')):
                    if preview is not None and np.any(preview):
                        review_dir = output/'review_previews'
                        review_dir.mkdir(exist_ok=True)
                        name = f'{i:05d}_{prefix}.seg.nrrd'
                        write_nrrd(review_dir/name, low_shape, array_slabs(preview, 8),
                                   'REVIEW_' + prefix.upper() + '_' + entry['name'], color=color)
                        row[field] = 'review_previews/' + name
    row['source_support_check'] = source_check.finish()
    row['elapsed_seconds'] = time.perf_counter()-started
    print(f'  finished [{i+1}] {entry["name"]}: {row["elapsed_seconds"]:.1f}s, '
          f'{row["sparse_label_bytes"]/MIB:.2f} MiB labels, '
          f'{row["unresolved_components"]} unresolved', flush=True)
    return i, row, source_hash, component_path


def execute(args, output, scratch_root, report, global_edit, layers, le, fe,
            low_path, full_path, bundle_dir, full_shape, low_shape, depth, support, union_sources=None):
    key = global_edit['source_filename']
    low_global_source = Nrrd(child(low_path.parent, key), low_shape)
    low_global = (low_global_source.array() if union_sources is None else
                  union_sources.low(global_edit['name']))
    edited_global = load_edit(bundle_dir, global_edit, low_global_source)
    low_global_nonempty = bool(np.any(low_global))
    union_low = np.zeros(low_shape, np.uint8)
    # Validate the entire edited low-resolution state BEFORE native work.
    print(f'Validating {len(layers)} edited previews with {args.workers} worker(s).', flush=True)
    low_union_lock = threading.Lock()
    def validate_low_layer(e):
        src = Nrrd(child(low_path.parent, e['source_filename']), low_shape)
        before = src.array() if union_sources is None else union_sources.low(e['name'])
        if union_sources is None and low_global_nonempty:
            before &= low_global
        after = load_edit(bundle_dir, e, src)
        if union_sources is not None:
            union_sources.verify_edit(e['name'], after, edited_global)
        if np.any((after != 0) & (before == 0)):
            raise ValueError(f'Edited {e["name"]} adds foreground, or initial Update Nonglobal '
                             'was not performed on this preview. Deletion-only transfer refused.')
        with low_union_lock:
            np.bitwise_or(union_low, after, out=union_low)
    with ThreadPoolExecutor(max_workers=args.workers) as executor:
        list(executor.map(validate_low_layer, layers))
    if union_sources is not None and union_sources.reference_entries is not None:
        report['source_union']['edit_reference_audit'] = union_sources.reference_audit
        report['source_union']['reference_deleted_segment_voxels'] = sum(
            row['deleted_voxels'] for row in union_sources.reference_audit.values())
    if not np.array_equal(union_low, edited_global):
        raise ValueError('Exported Global differs from the union of exported nonglobal segments. '
                         'Run Update Both in Slicer and export again; export ALL working segments.')
    low_labels, n = ndi.label(union_low, structure=FACE)
    del low_labels
    if n > 1:
        raise ValueError('Edited Global still contains multiple face-connected islands; run Update Both.')
    edited_global_empty = not np.any(union_low)
    report['edited_global_empty'] = bool(edited_global_empty)
    del union_low, edited_global
    if getattr(args, 'validate_only', False):
        report['status'] = 'validated_preview_only'
        report['native_source_validation'] = 'not_run; full-quality headers checked, payloads not read'
        report['pta_label_file'] = None
        save_json(output/'report.json', report)
        print('Preview edits validated; no native payload processing or PTA publication requested.', flush=True)
        return 0
    native_global_src = (Nrrd(child(full_path.parent, key), full_shape) if union_sources is None else
                         union_sources.native(global_edit['name']))
    if union_sources is None:
        report['source_full_sha256'][key] = digest(native_global_src.path)
    else:
        report['source_full_unions'][key] = native_global_src.descriptor()
    unresolved = 0
    with tempfile.TemporaryDirectory(prefix='xta-native-', dir=scratch_root) as tmp:
        tmp = Path(tmp)
        if union_sources is not None:
            print('Decoding and validating each original source Global once into shared packed scratch.', flush=True)
            union_sources.prepare_native_globals(tmp, depth)
        with mapped(tmp/'initial-global.dat', full_shape) as initial, mapped(tmp/'union.dat', full_shape) as union:
            print('Loading and checking the original native Global.', flush=True)
            global_count = 0
            source_check = PreviewSupportCheck(full_shape, low_global) if union_sources is None else native_global_src
            for z, block in native_global_src.slabs(depth):
                source_check.observe(z, block)
                initial[z:z+len(block)] = block
                global_count += int(np.count_nonzero(block))
                union[z:z+len(block)] = 0

            if low_global_nonempty != bool(global_count):
                raise ValueError('Original native/preview Global emptiness disagrees. Source pair needs review.')
            report['global_source_support_check'] = source_check.finish()
            del source_check
            initial.flush()
            # Union native readers apply each run's own Global before OR; using
            # the union Global here would admit cross-run foreground.
            clip = initial if global_count and union_sources is None else None
            if not global_count:
                report['warnings'].append('Original native Global is empty: initial native clipping was SKIPPED.')
            report['initial_global_voxels'] = global_count
            completed = {}
            with open(output/'components.tsv', 'w', newline='', encoding='utf-8') as tsv:
                csv.writer(tsv, delimiter='\t').writerow(
                    ['segment', 'component_id', 'native_voxels', 'evidence', 'decision'])
                context = dict(args=args, output=output, tmp=tmp,
                        low_path=low_path, full_path=full_path,
                        bundle_dir=bundle_dir, full_shape=full_shape, low_shape=low_shape,
                        depth=depth, edited_global_empty=edited_global_empty, union_sources=union_sources)
                worker_context = dict(context, low_global_path=low_global_source.path,
                                      clip_nonempty=bool(global_count) and union_sources is None)
                with native_executor(args.workers, worker_context) as executor:
                    if args.workers > 1:
                        futures = [executor.submit(process_native_worker, i, entry)
                                   for i, entry in enumerate(layers)]
                    else:
                        futures = [executor.submit(process_native_layer, i, entry, **context,
                            support=support, low_global=low_global, clip=clip, union=union)
                            for i, entry in enumerate(layers)]
                    try:
                        for future in as_completed(futures):
                            i, row, source_hash, component_path = future.result()
                            completed[i] = row
                            report['layers'] = [completed[k] for k in sorted(completed)]
                            field = 'source_full_sha256' if union_sources is None else 'source_full_unions'
                            report[field][row['source_filename']] = source_hash
                            unresolved += row['unresolved_components']
                            with open(component_path, encoding='utf-8', newline='') as part:
                                shutil.copyfileobj(part, tsv)
                            component_path.unlink()
                            tsv.flush()
                            save_json(output/'report.json', report)
                            print(f'Completed {len(completed)}/{len(layers)} layers.', flush=True)
                    except BaseException:
                        for future in futures:
                            future.cancel()
                        raise
            report['unresolved_components'] = unresolved
            report['auto_resolved_components'] = sum(r['auto_resolved_components'] for r in report['layers'])
            report['auto_removed_voxels'] = sum(r['auto_removed_voxels'] for r in report['layers'])
            if unresolved and args.conflicts != 'keep':
                report['status'] = 'needs_review'
                report['pta_label_file'] = None
                message = (f'{unresolved} native component(s) still have conflicting retained/deleted '
                           'preview evidence. Native and edited low-resolution objects do not agree; '
                           'no PTA label was published. Inspect components.tsv and review_previews. '
                           'Only explicit --conflicts keep permits publishing unresolved components.')
                report['warnings'].append(message)
                save_json(output/'report.json', report)
                print(f'NEEDS REVIEW: {message}', flush=True)
                return 2
            if unresolved:
                report['warnings'].append('Unresolved components were preserved under the selected conflict policy. '
                                          'Result is a conservative candidate, NOT a fully resolved transfer.')
            # PTA consumes only the native union; no native layer collection is written.
            print(f'Native Update Both: retaining up to {args.keep_objects} largest face-connected union component(s).', flush=True)
            from xta_native_sparse import SparseLabels, sparse_label_stream
            with SparseLabels(full_shape, tmp/'union-labels.runs') as labels:
                roots, counts, flags, _, _ = sparse_label_stream(
                    array_slabs(union, depth), labels, maximum=args.max_components)
                counts[0] = 0
                active = np.flatnonzero(counts)
                # Stable decreasing size; equal-size ties prefer the earliest label.
                order = np.lexsort((active, -counts[active].astype(np.int64)))
                kept_roots = active[order[:args.keep_objects]]
                largest = int(kept_roots[0]) if len(kept_roots) else 0
                tied = int(np.count_nonzero(counts == counts[largest])) if largest else 0
                report['largest_union_tie_count'] = tied
                if tied > 1:
                    report['warnings'].append('Equal-size native largest islands: earliest provisional label wins deterministically.')
                keep_root = np.zeros(len(counts), bool)
                keep_root[kept_roots] = True
                keep_lut = keep_root[roots]
                report['union_cleanup'] = {'input_components': int(len(active)),
                                           'kept_components': int(len(kept_roots)),
                                           'removed_components': int(len(active)-len(kept_roots)),
                                           'removed_voxels': int(counts.sum()-counts[kept_roots].sum())}
                for z in range(0, full_shape[0], depth):
                    union[z:z+depth] = 0
                labels.merge_into(union, keep_lut)
                final_voxels = int(counts[kept_roots].sum())
            evict(union)
            stem = str(low_path.name)
            suffix = '_nrrd_manifest.json'
            if not stem.endswith(suffix):
                raise ValueError('Expected an XTA <source-stem>_nrrd_manifest.json filename.')
            stem = stem[:-len(suffix)]
            # PTA accepts ONE ordinary 3D NRRD named after the original image/video stem.
            pta_name = stem + '.nrrd'
            write_nrrd(child(output, pta_name), full_shape, array_slabs(union, depth), global_edit['name'],
                       color=native_global_src.header.get('Segment0_Color', '0.8 0.3 0.2'))
            report['pta_label_file'] = pta_name
            report['final_global_voxels'] = final_voxels
            report['auto_resolved_components'] = sum(r['auto_resolved_components'] for r in report['layers'])
            report['auto_removed_voxels'] = sum(r['auto_removed_voxels'] for r in report['layers'])
            report['status'] = ('completed_with_unresolved_components' if unresolved else
                                'completed_with_heuristics' if report['auto_resolved_components'] else 'complete')
            if args.conflicts == 'auto':
                report['warnings'].append('Auto is a geometry/evidence heuristic, not a biological verdict. '
                                          'Radius uses native voxel indices, not calibrated physical units. '
                                          'ROI-limit or unsuitable-core cases stay intact and are reported.')
            report['warnings'].append('No exact low-resolution equality or biological correctness is implied. '
                                      'Native topology governs both Update steps; preview/native connectivity can differ.')
            report['warnings'].append('Components with no preview evidence are preserved until native Update Both '
                                      'unless edited Global is explicitly empty. Preview decisions cannot distinguish '
                                      'subvoxel structures merged by downsampling.')
            save_json(output/'report.json', report)
    print(f'{report["status"]}: {output/report["pta_label_file"]}', flush=True)
    return 0


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--version', action='version', version=VERSION)
    p.add_argument('--edits', required=True, help='Directory exported by Export XTA Edits in Slicer.')
    p.add_argument('--low-manifest', required=True, help='Original low_quality/<bin>/nrrd/*_nrrd_manifest.json.')
    p.add_argument('--full-manifest', required=True, help='Matching original full-quality nrrd/*_nrrd_manifest.json.')
    p.add_argument('--union-receipt', help='Explicit exact-name preview union manifest; requires --union-source-pairs.')
    p.add_argument('--union-source-pairs', help='Hashed original source pairs and receipt member assignments JSON.')
    p.add_argument('--validate-only', action='store_true',
                   help='Validate preview identities, deletion evidence and updated Global; do not read native payloads or publish a PTA mask.')
    p.add_argument('--output', required=True, help='New, separate output directory; never overwritten.')
    p.add_argument('--scratch', help='Fast local disk for temporary disk-backed arrays.')
    p.add_argument('--block-mib', type=int, default=128, help='Approximate slab-working target, not an RSS hard cap (default: 128).')
    p.add_argument('--workers', type=int, default=1, help='Concurrent native layer workers (default: 1); share Global/union arrays with foreground-only writes.')
    p.add_argument('--max-components', type=int, default=2000000, help='Cap on provisional slab components (default: 2 million).')
    conflict_options = p.add_mutually_exclusive_group()
    conflict_options.add_argument('--conflicts', choices=('error', 'keep', 'auto'), default='error',
                   help='Mixed native evidence: stop (default), explicitly keep/report unresolved components, or auto-cut thin necks and stop without a PTA mask if any conflicts remain.')
    conflict_options.add_argument('--auto', dest='conflicts', action='store_const', const='auto',
                   help='Shorthand for --conflicts auto; unresolved conflicts prevent PTA mask publication. Use only one conflict policy option.')
    p.add_argument('--auto-max-radius', type=int, default=2,
                   help='Maximum core-erosion distance, in native voxel-index units (default: 2).')
    p.add_argument('--auto-max-roi-voxels', type=int, default=8000000,
                   help='Maximum padded component bounding box for auto watershed (default: 8 million); not a RAM cap.')
    p.add_argument('--auto-max-keep-loss', type=float, default=0.01,
                   help='Reject a proposed cut losing more than this fraction of exclusive keep evidence (default: .01).')
    p.add_argument('--auto-min-remove-fraction', type=float, default=0.90,
                   help='Require this fraction of exclusive delete evidence removed (default: .90).')
    p.add_argument('--keep-objects', type=int, default=1,
                   help='Keep up to N largest native union components at the final step (default: 1).')
    p.add_argument('--allow-nonadditive', action='store_true',
                   help='Explicitly reproduce toolbar union of loaded checkpoint/audit layers too; NOT normal XTA recomposition.')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if args.block_mib < 1 or not (1 <= args.max_components < np.iinfo(np.int32).max):
        raise SystemExit('--block-mib must be positive; --max-components must fit a positive int32.')
    try:
        return run(args)
    except (Exception, KeyboardInterrupt) as exc:
        print(f'ERROR: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

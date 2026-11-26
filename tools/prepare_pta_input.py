#!/usr/bin/env python3
"""Prepare a PTA intensity/mask pair from edited preview NPZs, without training.

Dependencies: Python 3.10+, NumPy and ffprobe. Edited-shape reconstruction also
uses SciPy and the adjacent xta_edited_shape.py helper. Continuous-field
reconstruction uses SciPy, scikit-image and the adjacent
continuous_field_reconstruction package. Legacy preview upscaling is separate
from optional strict native evidence transfer and both edited-Global modes.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
from fractions import Fraction
import gzip
import hashlib
import importlib.util
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import uuid
import zipfile
from collections import OrderedDict
import numpy as np

VERSION = '1.0.0'
SCHEMA = 'pta-preview-input/1'
POLICY = 'preview-cubic-cell-center-threshold-0.5'
CONTINUOUS_FIELD_PACKAGE = 'continuous_field_reconstruction'
MIB = 1024**2
REFERENCE_RAS = np.diag([-1., -1., 1., 1.])

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*MIB), b''): h.update(b)
    return h.hexdigest()

# Hashed once as loaded, so a receipt never names a later edit of this helper.
HELPER_SHA256 = digest(__file__)

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def save_json(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False)+'\n', encoding='utf-8')

def shape3(value):
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError('Expected three positive TYX dimensions.')
    if any(isinstance(v, bool) or not isinstance(v, int) or v <= 0 for v in value):
        raise ValueError('Shape entries must be positive integers.')
    return tuple(value)

def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]*', value) or value in ('.', '..'):
        raise ValueError(f'Expected a plain volume stem/generation, not a path: {value!r}')
    return value

def child(directory, filename):
    if not isinstance(filename, str) or Path(filename).name != filename or '/' in filename or '\\' in filename:
        raise ValueError(f'Bundle member must be a plain filename: {filename!r}')
    p = directory / filename
    if p.is_symlink() or not p.is_file():
        raise ValueError(f'Missing/nonregular bundle member: {p}')
    return p

def preview_cut_columns(marker_ras, low_xyz, native_xyz):
    """First voxel centers strictly to the right in a standard axial XTA view."""
    r, a, s = marker_ras
    half = Fraction(1, 2)
    for index, size in zip((-r, -a, s), low_xyz):
        if not -half <= index < size - half:
            raise ValueError(f"Marker RAS lies outside the original preview (XYZ {low_xyz}).")
    native_i = (-r + half) * native_xyz[0] / low_xyz[0] - half
    return math.floor(-r) + 1, math.floor(native_i) + 1


def axial_crop_geometry(edits, manifest, low, full):
    """Validate a derived crop while retaining the original sampling lattice."""
    crop=edits.get('axial_crop')
    if not isinstance(crop,dict):raise ValueError('Cropped edit bundle needs axial_crop geometry.')
    original_low=shape3(crop.get('original_preview_shape_tyx'))
    original_full=shape3(crop.get('original_full_quality_shape_tyx'))
    if (original_low!=shape3(manifest['output_shape_tyx'])
            or original_full!=shape3(manifest['full_quality_output_shape_tyx'])
            or any(l>f for l,f in zip(original_low,original_full))):
        raise ValueError('Cropped bundle original geometry differs from its source manifest.')
    preview_i=crop.get('first_removed_preview_i');native_i=crop.get('first_removed_native_i')
    if (type(preview_i) is not int or type(native_i) is not int
            or not 0<preview_i<=original_low[2] or not 0<native_i<original_full[2]
            or low!=(*original_low[:2],preview_i) or full!=(*original_full[:2],native_i)):
        raise ValueError('Cropped bundle widths disagree with its declared cut columns.')
    marker=crop.get('marker_preview_ras')
    if (not isinstance(marker,list) or len(marker)!=3
            or any(type(v) not in (int,float) or not math.isfinite(v) for v in marker)):
        raise ValueError('Cropped bundle needs three finite marker RAS coordinates.')
    exact=crop.get('marker_preview_ras_exact')
    if exact is not None:
        if not isinstance(exact,list) or len(exact)!=3 or any(not isinstance(v,str) for v in exact):
            raise ValueError('Exact crop marker RAS needs three rational strings.')
        coordinates=tuple(Fraction(v) for v in exact)
        if not np.allclose([float(v) for v in coordinates],marker,rtol=1e-12,atol=0):
            raise ValueError('Exact crop marker RAS differs from its numeric coordinates.')
    else:coordinates=tuple(Fraction(str(v)) for v in marker)
    if preview_cut_columns(coordinates, original_low[::-1], original_full[::-1]) != (preview_i, native_i):
        raise ValueError('Crop columns do not match the marker plane on the original lattice.')
    if not isinstance(crop.get('source_edits_sha256'),str) or not re.fullmatch(r'[0-9a-f]{64}',crop['source_edits_sha256']):
        raise ValueError('Cropped bundle needs its original edits.json SHA256.')
    return crop

def _preview_archive_store(path, *, expected_identity=None):
    try:
        from XTA.native_review_store import NativeReviewStore
    except ModuleNotFoundError as error:
        if error.name != 'XTA':
            raise
        checkout = Path(__file__).resolve().parents[1]
        if not (checkout / 'XTA' / 'native_review_store.py').is_file():
            raise ValueError('Install XTA to read a preview archive, or use the standalone NPZ edit bundle.') from error
        sys.path.insert(0, str(checkout))
        from XTA.native_review_store import NativeReviewStore
    return NativeReviewStore(path, expected_identity=expected_identity)


def _load_preview_archive(bundle_dir, stem, mask_source, allow_empty_global):
    if mask_source != 'global':
        raise ValueError('Preview archives supply an explicitly selected Global snapshot.')
    path = child(bundle_dir, 'preview.xtareview')
    sha = digest(path)
    with _preview_archive_store(path) as store:
        source = store.source_record
        edit = source.get('preview_edit') if isinstance(source, dict) else None
        if (store.grid_role != 'preview' or not isinstance(edit, dict) or len(store.cutoffs) != 1
                or edit.get('cutoff_cent') != store.cutoffs[0]
                or not isinstance(edit.get('archive_identity'), str)
                or not re.fullmatch(r'[0-9a-f]{64}', edit['archive_identity'])
                or len(store.layers) != 1 or store.global_layer_id is None):
            raise ValueError('Expected an exported edited Global snapshot from a preview review.')
        low, full = store.shape, store.native_shape_tyx
        if not np.allclose(store.affine_ijk_to_ras, REFERENCE_RAS, rtol=0, atol=1e-7):
            raise ValueError('Preview archive needs the XTA unit-LPS preview lattice.')
        global_layer = store.layers[store.global_layer_id]
        cutoff = store.cutoffs[0]
        name = global_layer['name']
        declared_stem = source.get('source_stem')
        if (declared_stem is not None and declared_stem != stem
                or declared_stem is None and not name.startswith(stem + '_')):
            raise ValueError('Bundle contains a different volume stem.')
        count = global_layer['segments_by_cutoff'][str(cutoff)]['foreground_voxels']
        if not count and not allow_empty_global:
            raise ValueError('Global is empty: update it in Slicer and export it again.')
        crop = store.axial_crop
        if crop is not None:
            axial_crop_geometry({'axial_crop': crop},
                                {'output_shape_tyx': crop.get('original_preview_shape_tyx'),
                                 'full_quality_output_shape_tyx': crop.get('original_full_quality_shape_tyx')},
                                low, full)
        selected = dict(file=path.name, name=name, foreground_voxels=count, archive_identity=store.identity,
                        cutoff_cent=cutoff, sha256=sha)
        provenance = dict(bundle_format='xta-preview-archive-edits/1', archive_sha256=sha,
                          archive_identity=store.identity, preview_edit=edit, preview_shape_tyx=list(low),
                          source_shape_tyx=list(full), selected_segment_names=[name])
        if crop is not None:
            provenance['axial_crop'] = crop
    if sha != digest(path):
        raise ValueError('Preview export changed while loading.')
    return low, full, [selected], provenance


def load_bundle(bundle_dir, stem, mask_source='global', *, allow_empty_global=False):
    if (bundle_dir / 'preview.xtareview').exists():
        if (bundle_dir / 'edits.json').exists():
            raise ValueError('Choose a single preview export or a standalone NPZ bundle.')
        return _load_preview_archive(bundle_dir, stem, mask_source, allow_empty_global)
    edits_path = child(bundle_dir, 'edits.json')
    manifest_path = child(bundle_dir, 'source_manifest.json')
    e, m = read_json(edits_path), read_json(manifest_path)
    bundle_format=e.get('format')
    if bundle_format not in {'xta-slicer-edits-v1','xta-slicer-edits-axial-crop-v1'} or m.get('quality') != 'low_quality':
        raise ValueError('Expected a Slicer edit bundle and its low_quality source manifest.')
    if digest(manifest_path) != e.get('source_manifest_sha256'):
        raise ValueError('Source-manifest hash does not match edits.json.')
    low, full = shape3(e['preview_shape_tyx']), shape3(e['full_quality_shape_tyx'])
    crop=axial_crop_geometry(e,m,low,full) if bundle_format=='xta-slicer-edits-axial-crop-v1' else None
    if crop is None and (low != shape3(m['output_shape_tyx']) or full != shape3(m['full_quality_output_shape_tyx'])):
        raise ValueError('Bundle and manifest shapes disagree.')
    if any(l > f for l, f in zip(low, full)):
        raise ValueError('Preview exceeds the requested source dimensions.')
    entries = m.get('layers', [])
    byfile = {x['filename']: x for x in entries}
    if len(byfile) != len(entries): raise ValueError('Duplicate manifest filenames.')
    segments = e.get('segments', [])
    if not segments: raise ValueError('No edited segments.')
    for key in ('name', 'file', 'source_filename'):
        if len({x[key] for x in segments}) != len(segments):
            raise ValueError(f'Duplicate edited segment {key}.')
    components, globals_, excluded, snapshots = [], [], [], []
    for s in segments:
        original = byfile.get(s['source_filename'])
        if original is None or original.get('segment_name') != s['name']:
            raise ValueError(f'Segment identity does not match source manifest: {s["name"]}')
        if not s['name'].startswith(stem+'_'):
            raise ValueError('Bundle contains a different volume stem.')
        if s.get('recomposition_op') != original.get('recomposition_op'):
            raise ValueError('Edited and source recomposition operations disagree.')
        path = child(bundle_dir, s['file'])
        sha = digest(path)
        if sha != s.get('sha256'): raise ValueError(f'NPZ checksum mismatch: {path}')
        snapshots.append({'file': s['file'], 'sha256': sha, 'bytes': path.stat().st_size})
        if original.get('suffix') == 'Global_final_output':
            globals_.append(s)
        elif original.get('recomposition_op') == 'union':
            components.append(s)
        else:
            excluded.append({'name':s['name'],'reason':'Nonadditive checkpoint/diagnostic is not selected'})
    if len(globals_) != 1:raise ValueError('Expected exactly one Global_final_output in the edited bundle.')
    if mask_source == 'global':
        selected=globals_
        if not allow_empty_global and not globals_[0].get('foreground_voxels',0):
            raise ValueError('Global is empty: update Global from the intended edited layers in Slicer, inspect it, and re-export the bundle. Components mode is explicit, never an automatic fallback.')
        excluded += [{'name':s['name'],'reason':'Global selected; component snapshots are provenance only'} for s in components]
    else:
        selected=components
        excluded += [{'name':s['name'],'reason':'Explicit component union excludes Global select checkpoint',
                   'foreground_voxels':s.get('foreground_voxels')} for s in globals_]
    if not selected: raise ValueError('No selected edited masks.')
    provenance = {'edits_sha256': digest(edits_path), 'source_manifest_sha256': digest(manifest_path),
                  'preview_shape_tyx': list(low), 'source_shape_tyx': list(full),
                  'snapshots': snapshots, 'excluded_segments': excluded,
                  'selected_segment_names': [s['name'] for s in selected]}
    if crop is not None:provenance.update(bundle_format=bundle_format,axial_crop=crop)
    return low, full, selected, provenance

def read_snapshot(path, entry, preview_shape_tyx, max_snapshot_bytes=512*MIB):
    """Read a validated mask and its extent on the original preview lattice.

    Background padding outside the preview is discarded. Any foreground outside
    that field of view is rejected before a caller can mutate its target.
    """
    # Check decompressed member sizes before NumPy allocates an NPZ member.
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        if len(names) != len(set(names)) or set(names) != {'mask.npy', 'extent_xyz.npy', 'ijk_to_ras.npy'}:
            raise ValueError('Unexpected or duplicate NPZ members.')
        if archive.getinfo('mask.npy').file_size > max_snapshot_bytes + MIB:
            raise ValueError('Snapshot exceeds configured decompression budget.')
        if any(archive.getinfo(k).file_size > MIB for k in ('extent_xyz.npy','ijk_to_ras.npy')):
            raise ValueError('Oversized geometry metadata.')
    with np.load(path, allow_pickle=False) as z:
        mask, extent, affine = z['mask'], z['extent_xyz'], z['ijk_to_ras']
    if mask.dtype != np.uint8 or mask.ndim != 3 or mask.nbytes > max_snapshot_bytes:
        raise ValueError('Snapshot must be a bounded 3D uint8 binary mask.')
    count = 0
    for plane in mask:
        if np.any(plane > 1):
            raise ValueError('Snapshot is not binary.')
        count += int(np.count_nonzero(plane))
    if count != entry.get('foreground_voxels'):
        raise ValueError('Snapshot foreground count differs from edits.json.')
    empty_extent = np.asarray([0, -1, 0, -1, 0, -1], dtype=np.int64)
    # Slicer may give an empty segment an arbitrary image geometry.
    if mask.size == 0:
        return np.empty((0, 0, 0), dtype=np.uint8), empty_extent
    if extent.shape != (6,) or extent.dtype.kind not in 'iu':
        raise ValueError('Extent must have six integer XYZ entries.')
    limits = tuple(int(value) for value in extent)
    if any(abs(value) > 2**40 for value in limits):
        raise ValueError('Snapshot extent is outside the supported preview lattice.')
    lengths = tuple(limits[i + 1] - limits[i] + 1 for i in (0, 2, 4))
    if lengths[::-1] != mask.shape:
        raise ValueError('Snapshot extent/array shape mismatch.')
    if affine.shape != (4, 4) or affine.dtype.kind not in 'iuf' or not np.isfinite(affine).all():
        raise ValueError('Invalid snapshot affine.')
    relation = REFERENCE_RAS @ affine
    if (not np.allclose(relation[:3, :3], np.eye(3), rtol=0, atol=1e-5)
            or not np.allclose(relation[3], [0, 0, 0, 1], rtol=0, atol=1e-7)):
        raise ValueError('Edited lattice was rotated/resampled; expected XTA unit-LPS preview lattice.')
    offset = np.rint(relation[:3, 3])
    if not np.allclose(offset, relation[:3, 3], rtol=0, atol=1e-5) or np.max(np.abs(offset)) > 2**40:
        raise ValueError('Edited origin is not an integer preview-lattice offset.')
    start = (np.asarray(limits[::2], dtype=np.int64) + offset.astype(np.int64))[::-1]
    low = np.maximum(start, 0)
    high = np.minimum(start + np.asarray(mask.shape), preview_shape_tyx)
    if np.any(high <= low):
        if count:
            raise ValueError('Edited foreground lies outside the declared preview FOV; no clipping allowed.')
        return np.empty((0, 0, 0), dtype=np.uint8), empty_extent
    region = tuple(slice(int(lo - origin), int(hi - origin)) for lo, hi, origin in zip(low, high, start))
    inside = mask[region]
    if int(np.count_nonzero(inside)) != count:
        raise ValueError('Edited foreground lies outside the declared preview FOV; no clipping allowed.')
    world_extent = np.empty(6, dtype=np.int64)
    world_extent[::2] = low[::-1]
    world_extent[1::2] = high[::-1] - 1
    return inside, world_extent


def insert_snapshot(target, path, entry, max_snapshot_bytes):
    if 'archive_identity' in entry:
        with _preview_archive_store(path, expected_identity=entry['archive_identity']) as store:
            cutoff = entry['cutoff_cent']
            if store.axial_crop is not None and tuple(target.shape) == tuple(
                    store.axial_crop['original_preview_shape_tyx']):
                target = target[:, :, :store.shape[2]]
            if tuple(target.shape) != store.shape or cutoff not in store.cutoffs:
                raise ValueError('Edited archive does not match the target preview lattice.')
            count = 0
            for index in store.iter_blocks(store.global_layer_id, cutoff):
                bounds = store.block_bounds(index)
                if math.prod(b - a for a, b in bounds) > max_snapshot_bytes:
                    raise ValueError('Preview archive chunk exceeds the snapshot workspace budget.')
                block = store.read_block(store.global_layer_id, cutoff, index)
                region = tuple(slice(a, b) for a, b in bounds)
                np.bitwise_or(target[region], block, out=target[region])
                count += int(np.count_nonzero(block))
            if count != entry['foreground_voxels']:
                raise ValueError('Edited archive foreground count does not match the receipt.')
        if digest(path) != entry['sha256']:
            raise ValueError('Preview export changed while reading its Global.')
        return count
    mask, extent = read_snapshot(path, entry, target.shape, max_snapshot_bytes)
    if mask.size:
        region = tuple(slice(int(lo), int(hi) + 1) for lo, hi in zip(extent[::2][::-1], extent[1::2][::-1]))
        np.bitwise_or(target[region], mask, out=target[region])
    return entry['foreground_voxels']

def cubic_axis(low, full):
    """Keys/Catmull-Rom cubic weights (a=-0.5), zero outside input FOV."""
    pos=(np.arange(full,dtype=np.float64)+.5)*(low/full)-.5
    base=np.floor(pos).astype(np.int64);f=pos-base;f2=f*f;f3=f2*f
    indices=base[:,None]+np.arange(-1,3,dtype=np.int64)[None,:]
    weights=np.stack((-.5*f+f2-.5*f3,1-2.5*f2+1.5*f3,.5*f+2*f2-1.5*f3,-.5*f2+.5*f3),axis=1)
    weights[(indices<0)|(indices>=low)]=0
    return np.clip(indices,0,low-1),weights

def cubic_slabs(preview,full,block_mib=64):
    """Yield (T,Y start,float32 rows) with <=4 low-T X-resliced planes cached."""
    ti,tw=cubic_axis(preview.shape[0],full[0]);yi,yw=cubic_axis(preview.shape[1],full[1]);xi,xw=cubic_axis(preview.shape[2],full[2])
    cache=OrderedDict();rows=max(1,min(full[1],int(block_mib*MIB//max(1,full[2]*32))))
    def xplane(t):
        if t in cache:
            cache.move_to_end(t);return cache[t]
        a=np.zeros((preview.shape[1],full[2]),np.float64)
        for k in range(4):a+=preview[t][:,xi[:,k]]*xw[:,k]
        cache[t]=a
        if len(cache)>4:cache.popitem(last=False)
        return a
    for t in range(full[0]):
        combined=np.zeros((preview.shape[1],full[2]),np.float64)
        for k in range(4):
            if tw[t,k]!=0:combined+=xplane(int(ti[t,k]))*tw[t,k]
        for y in range(0,full[1],rows):
            n=min(rows,full[1]-y);slab=np.zeros((n,full[2]),np.float64)
            for k in range(4):slab+=combined[yi[y:y+n,k]]*yw[y:y+n,k,None]
            yield t,y,slab.astype(np.float32)

def write_upscaled(path, preview, full, block_mib=64):
    t,y,x=full
    header=(f'NRRD0005\n# Edited preview cubic/0.5 upscaling, not native recovery\n'
            f'type: unsigned char\ndimension: 3\nspace: left-posterior-superior\n'
            f'sizes: {x} {y} {t}\nspace directions: (1,0,0) (0,1,0) (0,0,1)\n'
            f'kinds: domain domain domain\nencoding: gzip\nspace origin: (0,0,0)\n\n')
    count=0; h=hashlib.sha256()
    with path.open('wb') as f:
        f.write(header.encode('ascii'))
        with gzip.GzipFile(filename='',fileobj=f,mode='wb',compresslevel=1,mtime=0) as z:
            for _,_,values in cubic_slabs(preview,full,block_mib):
                slab=(values>=.5).astype(np.uint8)
                data=slab.tobytes(order='C');z.write(data);h.update(data)
                count+=int(np.count_nonzero(slab))
    return {'shape_tyx':list(full),'foreground_voxels':count,'binary_payload_sha256':h.hexdigest(),
            'sha256':digest(path),'bytes':path.stat().st_size}

def validate_nrrd(path, full, expected=None):
    h=hashlib.sha256(); count=size=0
    with path.open('rb') as f:
        lines=[]
        while True:
            line=f.readline()
            if not line or sum(map(len,lines))+len(line)>65536: raise ValueError('Invalid NRRD header.')
            if line in (b'\n',b'\r\n'): break
            lines.append(line)
        fields={}
        for line in b''.join(lines).decode('ascii').splitlines():
            if ':' in line and not line.startswith('#'):
                k,v=line.split(':',1);fields[k]=v.strip()
        if not lines[0].startswith(b'NRRD000'):raise ValueError('Missing NRRD magic.')
        types={'unsigned char','uchar','uint8','uint8_t'}
        def vectors(text):return [[float(x.strip()) for x in v.split(',')] for v in re.findall(r'\(([^)]+)\)',text)]
        directions=np.asarray(vectors(fields.get('space directions','')),dtype=float)
        origin=np.asarray(vectors(fields.get('space origin','')),dtype=float)
        if (fields.get('type','').lower() not in types or fields.get('dimension')!='3'
            or fields.get('space','').lower() not in {'left-posterior-superior','lps'}
            or tuple(map(int,fields.get('sizes','').split()))!=tuple(full[::-1])
            or fields.get('encoding','').lower() not in {'gzip','gz'}
            or directions.shape!=(3,3) or not np.allclose(directions,np.eye(3),rtol=0,atol=1e-7)
            or origin.shape!=(1,3) or not np.allclose(origin,0,rtol=0,atol=1e-7)):
            raise ValueError('Prepared NRRD geometry/type mismatch.')
        with gzip.GzipFile(fileobj=f,mode='rb') as z:
            for b in iter(lambda:z.read(4*MIB),b''):
                a=np.frombuffer(b,np.uint8)
                if np.any(a>1): raise ValueError('Prepared NRRD is nonbinary.')
                size+=len(b)
                if size>math.prod(full):raise ValueError('Prepared NRRD exceeds declared shape.')
                count+=int(np.count_nonzero(a));h.update(b)
    actual={'shape_tyx':list(full),'foreground_voxels':count,'binary_payload_sha256':h.hexdigest(),
            'sha256':digest(path),'bytes':path.stat().st_size}
    if size!=math.prod(full) or (expected is not None and any(actual[k]!=expected[k] for k in actual)):
        raise ValueError('Prepared NRRD length/count/hash validation failed.')
    return actual

def crop_reconstructed_nrrd(source, destination, original_full, full, original_mask, crop, block_mib):
    """Crop validated native rows without changing their original sample positions."""
    original_width=original_full[2];width=full[2]
    header=(f'NRRD0005\n# Native-grid axial crop after original-lattice reconstruction\n'
            f'type: unsigned char\ndimension: 3\nspace: left-posterior-superior\n'
            f'sizes: {width} {full[1]} {full[0]}\nspace directions: (1,0,0) (0,1,0) (0,0,1)\n'
            f'kinds: domain domain domain\nencoding: gzip\nspace origin: (0,0,0)\n\n')
    verify_compressed_identity(source,original_mask)
    rows=max(1,int(min(8, block_mib)*MIB//original_width))
    remaining=original_full[0]*original_full[1]
    input_hash=hashlib.sha256();output_hash=hashlib.sha256();count=0
    with source.open('rb') as raw, destination.open('xb') as output:
        header_bytes=0
        while True:
            line=raw.readline();header_bytes+=len(line)
            if not line or header_bytes>65536:raise ValueError('Invalid reconstruction header during crop.')
            if line in (b'\n',b'\r\n'):break
        output.write(header.encode('ascii'))
        with gzip.GzipFile(fileobj=raw,mode='rb') as decoded, gzip.GzipFile(filename='',fileobj=output,mode='wb',compresslevel=1,mtime=0) as encoded:
            while remaining:
                n=min(rows,remaining);data=decoded.read(n*original_width)
                if len(data)!=n*original_width:raise ValueError('Truncated reconstruction during crop.')
                input_hash.update(data)
                retained=np.frombuffer(data,np.uint8).reshape(n,original_width)[:,:width]
                written=retained.tobytes();encoded.write(written);output_hash.update(written)
                count+=int(np.count_nonzero(retained));remaining-=n
            if decoded.read(1):raise ValueError('Reconstruction exceeds its original shape during crop.')
    if input_hash.hexdigest()!=original_mask['binary_payload_sha256']:
        raise ValueError('Reconstruction changed while its cropped payload was written.')
    result={'shape_tyx':list(full),'foreground_voxels':count,'binary_payload_sha256':output_hash.hexdigest(),
            'sha256':digest(destination),'bytes':destination.stat().st_size}
    validate_nrrd(destination,full,result)
    # The uncropped staging file is deleted below, so its path is not recorded.
    result.update(axial_crop=crop,pre_crop_mask={k:v for k,v in original_mask.items() if k!='path'},
                  crop_audit={'sampling':'unchanged native voxel values; high-X columns removed',
                              'original_shape_tyx':list(original_full),'retained_width':width,
                              'written_payload_verified':True})
    source.unlink()
    return result

def probe_source(path, ffprobe):
    cmd=[ffprobe,'-v','error','-select_streams','v:0','-show_entries','stream=width,height,nb_frames','-of','json',str(path)]
    result=json.loads(subprocess.check_output(cmd,text=True))
    streams=result.get('streams',[])
    if len(streams)!=1:raise ValueError('Expected one selected video stream.')
    stream=streams[0]; frames=stream.get('nb_frames')
    counted=False
    if frames in (None,'N/A','0',0):
        cmd=[ffprobe,'-v','error','-select_streams','v:0','-count_frames','-show_entries','stream=width,height,nb_read_frames','-of','json',str(path)]
        stream=json.loads(subprocess.check_output(cmd,text=True))['streams'][0]
        frames=stream.get('nb_read_frames');counted=True
    shape=shape3([int(frames),int(stream['height']),int(stream['width'])])
    return {'shape_tyx':list(shape),'frame_count_method':'ffprobe counted frames' if counted else 'ffprobe nb_frames',
            'command':cmd}

def publish_new(stage, output):
    # The caller holds the per-volume prep lock and both paths have the same
    # parent. Ordinary rename works on filesystems without renameat2 flags.
    # Other writers must respect that lock; this is not atomic NOREPLACE.
    if os.path.lexists(output):
        raise FileExistsError(f'Prepared input already exists; not replacing it: {output}')
    os.rename(stage,output)

def source_identity(path, *, hashed=False):
    before=path.stat()
    result={'path':str(path),'bytes':before.st_size,'mtime_ns':before.st_mtime_ns}
    if hashed:result['sha256']=digest(path)
    after=path.stat()
    if (before.st_size,before.st_mtime_ns,before.st_ino)!=(after.st_size,after.st_mtime_ns,after.st_ino):
        raise ValueError('Original intensity changed while its identity was being read.')
    return result

def publish_intensity(source, destination, mode, identity):
    if mode=='reference':
        return {'method':'reference','source':str(source),
                'bytes':identity['bytes'],'sha256':identity['sha256']}
    if mode=='symlink':
        destination.symlink_to(source)
        return {'method':'symlink','source':str(source)}
    # Exclusive creation also avoids accidentally replacing an existing member.
    h=hashlib.sha256()
    with source.open('rb') as src, destination.open('xb') as dst:
        for block in iter(lambda:src.read(4*MIB),b''):
            dst.write(block);h.update(block)
    if (destination.stat().st_size!=identity['bytes'] or h.hexdigest()!=identity['sha256']
            or digest(destination)!=identity['sha256']):
        raise ValueError('Intensity copy differs from the original source identity.')
    return {'method':'copy','source':str(source),'bytes':destination.stat().st_size,'sha256':h.hexdigest()}

def native_contract(args,stem,bundle):
    if not all((args.native_filter,args.native_low_template,args.native_full_template)):
        raise ValueError('Native mode requires --native-filter, --native-low-template and --native-full-template; no confidence or source-path guessing.')
    tool=Path(args.native_filter).resolve()
    low=Path(args.native_low_template.format(stem=stem)).resolve()
    full=Path(args.native_full_template.format(stem=stem)).resolve()
    if any(not p.is_file() for p in (tool,low,full)):raise ValueError('Explicit native tool/manifests must exist.')
    edits=read_json(bundle/'edits.json')
    embedded=edits.get('sweep_context', {}).get('source_catalog') == 'embedded'
    if tool.is_relative_to(bundle) or (low==full and not embedded):raise ValueError('Invalid native tool/source paths.')
    version=subprocess.check_output([args.python,str(tool),'--version'],text=True).strip()
    if version!='0.3.1':raise ValueError(f'This wrapper is qualified for native_filter 0.3.1, found {version!r}.')
    if embedded:
        if __package__:
            from .xta_sweep_edits import validate_sweep_source_pair
        else:
            from xta_sweep_edits import validate_sweep_source_pair
        binding=validate_sweep_source_pair(bundle,edits,low,full)
    elif digest(low)!=edits['source_manifest_sha256']:
        raise ValueError('Native low manifest is not the exact edit-bundle source manifest.')
    contract={'filter':str(tool),'filter_sha256':digest(tool),'filter_version':version,
              'low_manifest':str(low),'low_sha256':digest(low),'full_manifest':str(full),'full_sha256':digest(full)}
    if embedded:
        contract['catalogue_selection']={key:binding[key] for key in (
            'grid_id','cutoff_cent','preview_manifest_sha256','native_manifest_sha256')}
    if args.native_union_receipt:
        for key,value in (('union_receipt',args.native_union_receipt),('union_source_pairs',args.native_union_source_pairs)):
            path=Path(value).resolve()
            if not path.is_file():raise ValueError(f'Explicit native {key} must exist: {path}')
            contract[key]=str(path);contract[key+'_sha256']=digest(path)
        # The union extension is separate from the pinned core executable.
        helpers={}
        for name in ('xta_native_union.py','xta_native_sparse.py','xta_native_io.py','xta_auto.py'):
            path=tool.with_name(name)
            if path.is_file():helpers[name]={'path':str(path),'sha256':digest(path)}
        contract['helper_files']=helpers
        pairs_path=Path(contract['union_source_pairs'])
        source_manifests=[]
        for pair in read_json(pairs_path).get('pairs',[]):
            for kind in ('low','full'):
                path=Path(pair[kind+'_manifest'])
                if not path.is_absolute():path=pairs_path.parent/path
                path=path.resolve()
                actual=digest(path)
                row={'path':str(path),'sha256':actual}
                if 'grid_id' in pair or 'cutoff_cent' in pair:
                    if __package__:
                        from .xta_sweep_edits import _embedded_catalogue_digest
                    else:
                        from xta_sweep_edits import _embedded_catalogue_digest
                    grid=pair['grid_id'] if kind=='low' else 'native'
                    actual=_embedded_catalogue_digest(path,grid,pair['cutoff_cent'])
                    row.update(grid_id=grid,cutoff_cent=pair['cutoff_cent'],catalogue_sha256=actual)
                if actual!=pair[kind+'_manifest_sha256']:
                    raise ValueError(f'Union {kind} source-manifest checksum mismatch: {path}')
                source_manifests.append(row)
        contract['union_source_manifests']=source_manifests
    return contract


def edited_shape_contract():
    """Bind the reconstruction implementation, rather than native predictions."""
    path=Path(__file__).with_name('xta_edited_shape.py').resolve()
    if not path.is_file():raise ValueError(f'Missing edited-shape reconstruction helper: {path}')
    before=digest(path)
    spec=importlib.util.spec_from_file_location('xta_edited_shape',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if digest(path)!=before:raise ValueError('Edited-shape helper changed while being loaded.')
    return module,{'helper':str(path),'helper_sha256':before,
                   'version':module.VERSION,'policy':module.POLICY}


def continuous_field_contract():
    """Load the adjacent package by path, binding its fixed policy and sources."""
    package=Path(__file__).resolve().with_name(CONTINUOUS_FIELD_PACKAGE)
    required=('__init__.py','policy.py','field_reference.py','radius_upscaler.py','nrrd_writer.py')
    if any(not (package/name).is_file() for name in required):
        raise ValueError(f'Missing continuous-field package beside this helper: {package}')
    def sources():
        return {p.relative_to(package).as_posix():digest(p) for p in sorted(package.rglob('*.py'))
                if not any(part.startswith('test') or part=='__pycache__' for part in p.relative_to(package).parts)}
    before=sources()
    key=hashlib.sha256(json.dumps({'path':str(package),'sources':before},sort_keys=True).encode()).hexdigest()[:24]
    # A path-and-content keyed name lets relative imports work without making
    # tools/ a package, and never reuses a module loaded from other sources.
    name='_pta_continuous_field_'+key
    if name not in sys.modules:
        spec=importlib.util.spec_from_file_location(name,package/'__init__.py',submodule_search_locations=[str(package)])
        module=importlib.util.module_from_spec(spec)
        sys.modules[name]=module
        try:
            spec.loader.exec_module(module)
        except BaseException:
            for loaded in list(sys.modules):
                if loaded==name or loaded.startswith(name+'.'):sys.modules.pop(loaded,None)
            raise
    module=sys.modules[name]
    import scipy
    import skimage
    if sources()!=before:raise ValueError('Continuous-field package changed while being loaded.')
    return module,{'policy':module.POLICY,'version':module.ALGORITHM_VERSION,'sources_sha256':before,
                   'runtime':{'python':'.'.join(map(str,sys.version_info[:2])),'numpy':np.__version__,
                              'scipy':scipy.__version__,'scikit_image':skimage.__version__}}


def reuse_identity(identity):
    """Return the identity compared before reusing a prepared input.

    Continuous-field package sources and library versions are provenance; its
    fixed policy/version and every data identity must still match. Other modes
    compare their complete identity, including helper hashes.
    """
    contract=identity.get('continuous_field')
    if identity.get('mode')!='continuous_field' or not isinstance(contract,dict):return identity
    return {**identity,'continuous_field':{key:contract.get(key) for key in ('policy','version')}}


def write_continuous_field(module, args, stem, label_stem, preview, full, destination, log_dir):
    """Build or validate the preview-sized field, then stream the native NRRD.

    A shared cache is keyed by output stem: one source video can have several
    edit bundles, such as uncropped and cropped labels.
    """
    cache=Path(args.field_cache).resolve()/label_stem if args.field_cache else log_dir/(stem+'.field-cache')
    last={'stage':None,'time':0.,'percent':-1}
    def progress(event):
        now=time.monotonic()
        stage=event.get('stage') if isinstance(event,dict) else None
        total=event.get('total_voxels',0) if isinstance(event,dict) else 0
        done=event.get('completed_voxels',0) if isinstance(event,dict) else 0
        percent=int(100*done/total) if total else None
        if (stage!=last['stage'] or now-last['time']>=30
                or percent==100 or (percent is not None and percent>=last['percent']+5)):
            print(f'{stem}: {event}',flush=True)
            last.update(stage=stage,time=now,percent=percent if percent is not None else -1)
    if cache.exists() or cache.is_symlink():
        reference=module.validate_field_reference(preview,full,cache)
        print(f'{stem}: validated existing field cache: {cache}',flush=True)
    else:
        reference=module.build_field_reference(preview,full,cache,memory_budget_mib=args.field_memory_mib,progress=progress)
    mask=module.write_upscaled_nrrd(cache,destination,native_shape_tyx=full,block_mib=args.block_mib,
                                    workers=args.workers,progress=progress)
    save_json(log_dir/(stem+'.field-reference.json'),reference)
    return mask


def native_quality(report):
    """A zero child exit code alone does not establish usable native labels."""
    status=report.get('status')
    unresolved=report.get('unresolved_components')
    if (status not in {'complete','completed_with_heuristics'}
            or isinstance(unresolved,bool) or not isinstance(unresolved,int) or unresolved!=0
            or any(row.get('unresolved_components',0) for row in report.get('layers',[]))):
        raise ValueError(f'Native recovery did not establish resolved success: status={status!r}, unresolved={unresolved!r}; diagnostics retained, no PTA input published.')
    return {key:report[key] for key in ('status','unresolved_components','auto_resolved_components',
             'auto_removed_voxels','final_global_voxels','warnings') if key in report}


def pre_crop_mask(mask, full, crop, label):
    """Return the original-lattice writer receipt behind a native-grid crop."""
    original_full=shape3(crop['original_full_quality_shape_tyx'])
    audit=mask.get('crop_audit',{})
    if (mask.get('axial_crop')!=crop or mask.get('shape_tyx')!=list(full)
            or audit.get('original_shape_tyx')!=list(original_full)
            or audit.get('retained_width')!=full[2] or audit.get('written_payload_verified') is not True
            or 'reconstruction' in mask):
        raise ValueError(f'Cropped {label} result lacks its native-grid crop audit.')
    return mask.get('pre_crop_mask',{}),original_full


def edited_shape_quality(mask, low, full, foreground, contract, crop=None):
    """Require the known writer's actual-byte audit before trusting its result.

    This checks the writer/receipt contract, not a second numerical audit. The
    in-process helper already rereads every native payload byte and checks its
    hash, binary values, edited centers, background cells and 6-components.
    """
    if crop is not None:
        source,original_full=pre_crop_mask(mask,full,crop,'edited-shape')
        edited_shape_quality(source,shape3(crop['original_preview_shape_tyx']),original_full,foreground,contract)
        return
    reconstruction=mask.get('reconstruction',{})
    centers=reconstruction.get('center_roundtrip',{})
    separation=reconstruction.get('separation_audit',{})
    components=separation.get('source_foreground_components')
    if (mask.get('shape_tyx')!=list(full)
            or reconstruction.get('algorithm')!=contract['policy']
            or reconstruction.get('version')!=contract['version']
            or reconstruction.get('source_shape_tyx')!=list(low)
            or reconstruction.get('source_foreground_voxels')!=foreground
            or centers.get('sampled_voxels')!=math.prod(low)
            or centers.get('false_positive_voxels')!=0
            or centers.get('false_negative_voxels')!=0
            or centers.get('sampling')!='nearest-native-voxel-center-to-each-edited-voxel-center'
            or separation.get('connectivity')!=6
            or isinstance(components,bool) or not isinstance(components,int) or components<1
            or separation.get('output_foreground_components')!=components
            or separation.get('foreground_voxels_in_edited_background_cells')!=0
            or separation.get('sampling')!='actual-written-native-grid'):
        raise ValueError('Edited-shape result lacks the required actual-written-payload quality audit.')


def continuous_field_quality(mask, full, contract, crop=None):
    """Require the continuous-field writer's complete readback receipt.

    The policy gives no edited-center, background-cell or topology guarantee,
    so none is checked. The writer already decoded every written byte, so this
    checks its receipt instead of decoding the file again.
    """
    if crop is not None:mask,full=pre_crop_mask(mask,full,crop,'continuous-field')
    validation=mask.get('validation',{})
    if (mask.get('status')!='complete' or mask.get('algorithm')!=contract['policy']
            or mask.get('shape_tyx')!=list(full)
            or isinstance(mask.get('foreground_voxels'),bool) or not isinstance(mask.get('foreground_voxels'),int)
            or not mask.get('binary_payload_sha256') or not mask.get('sha256')
            or isinstance(mask.get('bytes'),bool) or not isinstance(mask.get('bytes'),int)
            or validation.get('native_voxels_read')!=math.prod(full)
            or validation.get('binary_values_verified') is not True
            or validation.get('gzip_readback_to_end') is not True
            or validation.get('header_and_global_origin_verified') is not True):
        raise ValueError('Continuous-field writer did not return a complete native NRRD identity.')


def verify_compressed_identity(path, mask):
    """Detect a changed audited file without decompressing its payload again."""
    before=path.stat()
    actual=digest(path)
    after=path.stat()
    if (before.st_size!=mask['bytes'] or actual!=mask['sha256']
            or (before.st_size,before.st_mtime_ns,before.st_ino)!=
               (after.st_size,after.st_mtime_ns,after.st_ino)):
        raise ValueError('Audited reconstructed NRRD changed before publication.')


def training_base(args):
    # The legacy layout's TRAINING_ROOT/GENERATION; None when only explicit paths are given.
    return Path(args.training_root).resolve()/safe_name(args.generation) if args.training_root else None

def legacy_root(args):
    # Folder of STEM.mkv, the legacy intensity location used without --source-video.
    if not args.root:raise ValueError('--root is required without --source-video.')
    return Path(args.root)

def output_path(args,stem,base):
    # Keep the final path unresolved so an existing symlink is still rejected.
    if args.output_dir:
        path=Path(args.output_dir).absolute()
        return path.parent.resolve()/path.name
    if base is None:raise ValueError('--training-root is required without --output-dir.')
    return base/(stem+'_Input')

def bundle_path(args,stem,base):
    # Preserve the final component so an explicit symlink cannot hide a bundle.
    if args.bundle_dir:
        path=Path(args.bundle_dir).absolute()
        return path.parent.resolve()/path.name
    if base is None:raise ValueError('--training-root is required without --bundle-dir.')
    return base/(stem+'_Filtered')

def prepare(args, stem, log_dir):
    safe_name(stem)
    base=training_base(args)
    bundle=bundle_path(args,stem,base); output=output_path(args,stem,base)
    label_stem=safe_name(args.output_stem) if args.output_stem else stem
    # Continuous-field receipts always hash the video so staged pairs can be verified.
    hashed_source=args.intensity_mode in {'copy','reference'} or args.mode=='continuous_field'
    source=Path(args.source_video).resolve() if args.source_video else legacy_root(args).resolve()/(stem+'.mkv')
    if not source.is_file():raise ValueError(f'Missing original intensity: {source}')
    if bundle.is_symlink() or not bundle.is_dir():raise ValueError(f'Missing regular filtered bundle: {bundle}')
    output.parent.mkdir(parents=True,exist_ok=True)
    lock=output.parent/('.'+output.name+'.prep.lock')
    fd=os.open(lock,os.O_CREAT|os.O_EXCL|os.O_WRONLY,0o600)
    os.write(fd,json.dumps({'pid':os.getpid(),'host':socket.gethostname(),'started_utc':datetime.now(timezone.utc).isoformat()}).encode());os.close(fd)
    stage=None
    preserve_completed_stage=False
    try:
        low,full,selected,bundle_prov=load_bundle(bundle,stem,args.mask_source)
        crop=bundle_prov.get('axial_crop')
        if args.mode=='native' and bundle_prov.get('bundle_format')=='xta-preview-archive-edits/1':
            raise ValueError('A preview edit archive has no original native masks for native transfer.')
        if crop is not None and args.mode=='native':
            raise ValueError('Derived cropped edit bundles support edited_shape, continuous_field or upscale preparation.')
        reconstruction_low=shape3(crop['original_preview_shape_tyx']) if crop else low
        reconstruction_full=shape3(crop['original_full_quality_shape_tyx']) if crop else full
        if math.prod(reconstruction_low)>args.max_preview_mib*MIB:raise ValueError('Preview exceeds configured union allocation budget.')
        source_stat=source_identity(source,hashed=hashed_source)
        probe=probe_source(source,args.ffprobe)
        if tuple(probe['shape_tyx'])!=full:raise ValueError(f'Original intensity dimensions differ from bundle: {probe["shape_tyx"]} != {full}')
        native=native_contract(args,stem,bundle) if args.mode=='native' else None
        shape_module,shape_contract=edited_shape_contract() if args.mode=='edited_shape' else (None,None)
        continuous_module,continuous=continuous_field_contract() if args.mode=='continuous_field' else (None,None)
        policy=(POLICY if args.mode=='upscale' else shape_contract['policy'] if shape_contract
                else continuous['policy'] if continuous else 'strict-native-filter-0.3.1-auto')
        identity={'stem':stem,'mode':args.mode,'mask_source':args.mask_source,'policy':policy,
                  'preparation_algorithm_version':VERSION,'generation':args.generation,'bundle':bundle_prov,'intensity':source_stat,'native':native}
        if shape_contract:identity['edited_shape']=shape_contract
        if continuous:identity['continuous_field']=continuous
        if args.intensity_mode!='symlink':identity['intensity_mode']=args.intensity_mode
        if args.output_stem:identity['output_stem']=label_stem
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_dir():raise ValueError('Existing Input must be a regular directory.')
            receipt=read_json(output/'preparation.json')
            prior=receipt.get('identity')
            if not isinstance(prior,dict):prior={}
            if (receipt.get('format')!=SCHEMA or receipt.get('status')!='complete'
                    or reuse_identity(prior)!=reuse_identity(identity)):
                prepared=prior.get('mode')
                detail=(f' (prepared mode {prepared!r}, requested {args.mode!r})'
                        if isinstance(prepared,str) and prepared!=args.mode else '')
                raise ValueError(f'Existing input provenance differs{detail}; nothing overwritten. Use a new generation or inspect it.')
            if args.mode=='native':native_quality(receipt.get('native_quality',{}))
            if shape_contract:
                edited_shape_quality(receipt['mask'],low,full,selected[0]['foreground_voxels'],shape_contract,crop)
            if continuous:continuous_field_quality(receipt['mask'],full,continuous,crop)
            expected_files={label_stem+'.nrrd','preparation.json'}
            if args.intensity_mode!='reference':expected_files.add(label_stem+'.mkv')
            if {p.name for p in output.iterdir()}!=expected_files:
                raise ValueError('Existing Input contains unexpected files; refuse ambiguous PTA discovery.')
            intensity=output/(label_stem+'.mkv')
            if args.intensity_mode=='symlink':
                if not intensity.is_symlink() or intensity.resolve()!=source.resolve():raise ValueError('Existing intensity link has changed.')
            elif args.intensity_mode=='reference':
                if receipt.get('intensity_publication')!={'method':'reference','source':str(source),
                        'bytes':source_stat['bytes'],'sha256':source_stat['sha256']}:
                    raise ValueError('Existing referenced intensity identity has changed.')
            elif (intensity.is_symlink() or not intensity.is_file() or intensity.stat().st_size!=source_stat['bytes']
                  or digest(intensity)!=source_stat['sha256'] or receipt.get('intensity_publication')!=
                  {'method':'copy','source':str(source),'bytes':source_stat['bytes'],'sha256':source_stat['sha256']}):
                raise ValueError('Existing copied intensity identity has changed.')
            validate_nrrd(output/(label_stem+'.nrrd'),full,receipt['mask'])
            if source_identity(source,hashed=hashed_source)!=source_stat:
                raise ValueError('Inputs changed during existing-input verification.')
            _,_,_,after=load_bundle(bundle,stem,args.mask_source)
            if (after!=bundle_prov or (args.mode=='native' and native_contract(args,stem,bundle)!=native)
                    or (shape_contract and edited_shape_contract()[1]!=shape_contract)
                    or (continuous and continuous_field_contract()[1]!=continuous)):
                raise ValueError('Inputs changed during existing-input verification.')
            save_json(log_dir/(stem+'.result.json'),{'status':'validated_existing','input':str(output),'preparation_sha256':digest(output/'preparation.json')})
            print(f'Validated existing input: {output}',flush=True);return output
        stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.prep-',dir=output.parent))
        count=None
        quality=None
        if args.mode in {'upscale','edited_shape','continuous_field'}:
            preview=np.zeros(reconstruction_low,np.uint8)
            for i,entry in enumerate(selected,1):
                insert_snapshot(preview[:,:,:low[2]],child(bundle,entry['file']),entry,args.max_snapshot_mib*MIB)
                print(f'{stem}: selected snapshot {i}/{len(selected)} {entry["name"]}',flush=True)
            count=int(np.count_nonzero(preview))
            if not count:raise ValueError('Selected preview is empty; no output published.')
            reconstructed_path=stage/(label_stem+('.uncropped.nrrd' if crop else '.nrrd'))
            if continuous:
                mask=write_continuous_field(continuous_module,args,stem,label_stem,preview,reconstruction_full,reconstructed_path,log_dir)
            else:
                mask=(shape_module.write_edited_shape(reconstructed_path,preview,reconstruction_full,
                        block_mib=args.block_mib,workers=args.workers) if shape_module else
                      write_upscaled(reconstructed_path,preview,reconstruction_full,args.block_mib))
            del preview
            if shape_contract:
                edited_shape_quality(mask,reconstruction_low,reconstruction_full,count,shape_contract)
            elif continuous:
                continuous_field_quality(mask,reconstruction_full,continuous)
            else:
                validate_nrrd(reconstructed_path,reconstruction_full,mask)
            if crop:
                mask=crop_reconstructed_nrrd(reconstructed_path,stage/(label_stem+'.nrrd'),
                                            reconstruction_full,full,mask,crop,args.block_mib)
                if shape_contract:edited_shape_quality(mask,low,full,count,shape_contract,crop)
                if continuous:continuous_field_quality(mask,full,continuous,crop)
        else:
            native_out=stage/'native-output'
            scratch_root=Path(args.scratch_root).resolve();scratch_root.mkdir(parents=True,exist_ok=True)
            scratch=Path(tempfile.mkdtemp(prefix='pta-native-'+stem+'-',dir=scratch_root))
            command=[args.python,native['filter'],'--auto','--edits',str(bundle),'--low-manifest',native['low_manifest'],
                     '--full-manifest',native['full_manifest'],'--output',str(native_out),'--scratch',str(scratch),
                     '--workers',str(args.workers),'--block-mib',str(args.block_mib)]
            if args.native_union_receipt:
                command+=['--union-receipt',native['union_receipt'],'--union-source-pairs',native['union_source_pairs']]
            save_json(log_dir/(stem+'.native-command.json'),{'command':command,
                'warning':'Mixed-confidence unions or additions may fail the deletion-evidence contract; no bypass is enabled.'})
            native_success=False
            try:
                with (log_dir/(stem+'.native.log')).open('w',encoding='utf-8') as native_log:
                    subprocess.run(command,check=True,stdout=native_log,stderr=subprocess.STDOUT)
                quality=native_quality(read_json(native_out/'report.json'))
                mask=validate_nrrd(native_out/(stem+'.nrrd'),full)
                if not mask['foreground_voxels']:raise ValueError('Native output is empty; diagnostics retained, no PTA input published.')
                shutil.move(str(native_out/(stem+'.nrrd')),str(stage/(label_stem+'.nrrd')))
                native_success=True
            except BaseException:
                if (native_out/'report.json').is_file():shutil.copyfile(native_out/'report.json',log_dir/(stem+'.native-report.json'))
                if native_out.is_dir():shutil.move(str(native_out),str(log_dir/(stem+'.native-failed-diagnostics')))
                save_json(log_dir/(stem+'.native-failed-scratch.json'),{'retained_scratch':str(scratch)})
                raise
            finally:
                if (native_out/'report.json').is_file():shutil.copyfile(native_out/'report.json',log_dir/(stem+'.native-report.json'))
                if native_success and scratch.parent==scratch_root and scratch.name.startswith('pta-native-'+stem+'-'):shutil.rmtree(scratch)
            shutil.move(str(native_out),str(log_dir/(stem+'.native-diagnostics')))
        if not mask['foreground_voxels']:raise ValueError('Prepared mask is empty; no PTA input published.')
        intensity_publication=publish_intensity(source,stage/(label_stem+'.mkv'),args.intensity_mode,source_stat)
        # Detect source and bundle changes before making output visible.
        _,_,_,after=load_bundle(bundle,stem,args.mask_source)
        if (after!=bundle_prov or source_identity(source,hashed=hashed_source)!=source_stat
                or (args.mode=='native' and native_contract(args,stem,bundle)!=native)
                or (shape_contract and edited_shape_contract()[1]!=shape_contract)
                or (continuous and continuous_field_contract()[1]!=continuous)):
            raise ValueError('Inputs changed during preparation; output not published.')
        if shape_contract or continuous:verify_compressed_identity(stage/(label_stem+'.nrrd'),mask)
        # The writer recorded its private staging path; record the published one.
        if continuous:mask['path']=str(output/(label_stem+'.nrrd'))
        receipt={'format':SCHEMA,'tool_version':VERSION,'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                 'identity':identity,'source_probe':probe,'mask':mask,'preview_union_foreground_voxels':count,
                 'algorithm':('Selected preview mask; separable Keys cubic a=-0.5, zero exterior, coordinate (j+.5)*L/H-.5; float32 output >=0.5' if args.mode=='upscale' else
                              shape_contract['policy'] if shape_contract else continuous['policy'] if continuous else
                              'Existing strict native_filter 0.3.1 --auto; its evidence checks and resolved-success requirement apply'),
                 'source_intensity_hash':source_stat.get('sha256','not calculated; path/size/mtime and probed dimensions recorded'),
                 'intensity_publication':intensity_publication,
                 'native_component_recovery':args.mode=='native','original_source_nrrds_required':args.mode=='native',
                 'notes':['Default selects the updated nonempty Global. Components OR requires explicit mask-source selection.',
                          'Upscale does not apply largest-object filtering, smoothing or filling. Native mode retains its tool defaults.',
                          ('Historical source NRRD hashes in edits.json are provenance only; native/source layer bytes were not read.' if args.mode=='upscale' else 'Existing native tool reads and verifies original preview/native layer bytes; native diagnostics are retained in the run log directory.'),
                          'Output uses uncalibrated XTA source-grid unit LPS, not physical millimetre calibration.',
                          ('Preview upscale cannot recover boundaries removed by downsampling.' if args.mode=='upscale' else 'Native transfer is conditional on the unmodified deletion-evidence contract; no mixed-confidence bypass.')],
                 'helper_sha256':HELPER_SHA256}
        if shape_contract:
            receipt.update(edited_shape_reconstruction=True,label_authority='edited_preview_global',
                           restored_native_detail=False)
            receipt['notes']=['The selected edited Global is authoritative for shape and separation.',
                              'Signed-distance reconstruction samples the source-grid cell centers; original prediction NRRDs are not read or restored.',
                              'Output uses uncalibrated XTA source-grid unit LPS, not physical millimetre calibration.']
        if continuous:
            receipt.update(continuous_reconstruction=True,label_authority='edited_preview_sampled_support',
                           restored_native_detail=False)
            receipt['notes']=[
                'Full native source dimensions; radius-aware linear/cardinal-B3 reconstruction with guarded cap0.5.',
                'The edited binary preview is sampled support, not a native hard-positive/negative edit map.',
                'No GT/SAM correction, native prediction recovery, or explicit native edit-authority override is applied.',
                'No guarantee of coarse-center retention, edited-background exclusion, topology or anatomical accuracy.',
                'Output uses uncalibrated XTA source-grid unit LPS; not physical millimetres.',
                'Reuse checks data/geometry/fixed policy; code and runtime versions are recorded as provenance, not lockouts.']
        if crop:
            receipt['axial_crop']=crop
            receipt['notes'].append('Reconstruction uses the original preview/native lattice, padded with zero beyond the cropped preview; final native high-X columns are removed after reconstruction. The pre_crop_mask reconstruction audits apply before that final crop.')
        if quality is not None:receipt['native_quality']=quality
        save_json(stage/'preparation.json',receipt)
        preserve_completed_stage=True
        try:
            publish_new(stage,output)
        except OSError as exc:
            print(f'Publication failed. Completed, validated input retained at: {stage}',flush=True)
            save_json(log_dir/(stem+'.publication-failed.json'),
                      {'status':'publication_failed','completed_stage':str(stage),
                       'intended_output':str(output),'error':f'{type(exc).__name__}: {exc}'})
            raise
        stage=None
        save_json(log_dir/(stem+'.result.json'),{'status':'prepared','input':str(output),'preparation_sha256':digest(output/'preparation.json')})
        print(f'Published input: {output}',flush=True);return output
    finally:
        if (stage is not None and not preserve_completed_stage and stage.parent==output.parent
                and stage.name.startswith('.'+output.name+'.prep-')):
            shutil.rmtree(stage)
        lock.unlink()

def validate_only(args,stem,log_dir):
    bundle=bundle_path(args,stem,training_base(args))
    if bundle.is_symlink() or not bundle.is_dir():raise ValueError(f'Missing regular filtered bundle: {bundle}')
    low,full,selected,provenance=load_bundle(bundle,stem,args.mask_source)
    crop=provenance.get('axial_crop')
    if args.mode=='native' and provenance.get('bundle_format')=='xta-preview-archive-edits/1':
        raise ValueError('A preview edit archive has no original native masks for native transfer.')
    if crop is not None and args.mode=='native':
        raise ValueError('Derived cropped edit bundles support edited_shape, continuous_field or upscale preparation.')
    allocation_shape=shape3(crop['original_preview_shape_tyx']) if crop else low
    if math.prod(allocation_shape)>args.max_preview_mib*MIB:raise ValueError('Preview exceeds union allocation budget.')
    preview=np.zeros(low,np.uint8)
    for entry in selected:insert_snapshot(preview,child(bundle,entry['file']),entry,args.max_snapshot_mib*MIB)
    report={'status':'validated_only','bundle':str(bundle),'provenance':provenance,
            'selected_foreground_voxels':int(np.count_nonzero(preview)),'no_hq_output_or_input_directory_created':True}
    if not args.skip_source_probe:
        source=Path(args.source_video).resolve() if args.source_video else legacy_root(args)/(stem+'.mkv')
        report['source_probe']=probe_source(source,args.ffprobe)
        if tuple(report['source_probe']['shape_tyx'])!=full:raise ValueError('Intensity dimensions do not match bundle.')
    else:report['source_probe']='explicitly skipped for bundle-only validation; not permitted for preparation'
    if args.mode=='native':report['native']=native_contract(args,stem,bundle)
    if args.mode=='edited_shape':report['edited_shape']=edited_shape_contract()[1]
    if args.mode=='continuous_field':report['continuous_field']=continuous_field_contract()[1]
    save_json(log_dir/(stem+'.result.json'),report)
    print(f'Validated bundle only: {bundle}; selected FG={report["selected_foreground_voxels"]}',flush=True)
    return str(bundle)

def smoke(args,inputs,log_dir):
    if not args.smoke_script or not Path(args.smoke_script).is_file():raise ValueError('--smoke requires an existing --smoke-script entrypoint.')
    # Per-run input copies on local scratch: removed after a passing smoke, kept for diagnosis otherwise.
    scratch_root=Path(args.scratch_root).resolve();scratch_root.mkdir(parents=True,exist_ok=True)
    smoke_input=Path(tempfile.mkdtemp(prefix='pta-smoke-',dir=scratch_root))
    for prepared in map(Path,inputs):
        identity=read_json(prepared/'preparation.json')['identity']
        stem=identity.get('output_stem',identity['stem'])
        shutil.copy2(identity['intensity']['path'],smoke_input/(stem+'.mkv'))
        shutil.copy2(prepared/(stem+'.nrrd'),smoke_input/(stem+'.nrrd'))
    command=[args.python,str(Path(args.smoke_script).resolve()),'--mode','pta','--task','segment',
             '--input',str(smoke_input),'--output',str(log_dir/'pta-smoke'),'--imgsz',str(args.smoke_imgsz),
             '--background_percent','1','--channel_format','gray','--output_format','png',
             '--worker_backend','thread','--enable_cartesian','transverse',
             '--fraction',str(args.smoke_fraction),'--save','images','labels','summary']
    save_json(log_dir/'smoke-command.json',{'command':command,'training':False,'requested_gpu':False})
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['YOLO_TTA_TELEMETRY_DIR']=str(log_dir/'smoke-telemetry')
    with (log_dir/'smoke.log').open('w',encoding='utf-8') as f:
        subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,env=env)
    if not (log_dir/'pta-smoke'/'manifest.json').is_file():raise ValueError('PTA smoke returned without manifest.json.')
    shutil.rmtree(smoke_input)

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stems',nargs='*')
    p.add_argument('--volumes-file',type=Path,help='One stem per line; blank lines and # comments ignored')
    p.add_argument('--root',help='Legacy layout: folder holding STEM.mkv; required only without --source-video, '
                   'except with --validate-only --skip-source-probe')
    p.add_argument('--source-video',type=Path,help='Explicit original intensity path for exactly one volume; otherwise ROOT/STEM.mkv')
    p.add_argument('--output-dir',type=Path,help='Explicit prepared-input directory for exactly one volume; existing incompatible data is never overwritten')
    p.add_argument('--output-stem',help='Explicit output basename for exactly one volume; the positional stem still identifies source segments')
    p.add_argument('--intensity-mode',choices=['symlink','copy','reference'],default='symlink',help='Publish an intensity link (default), verified copy, or a hashed source reference with labels only')
    p.add_argument('--training-root',help='Legacy layout: TRAINING_ROOT/GENERATION/STEM_Filtered and STEM_Input; '
                   'required only without --bundle-dir, or without --output-dir when preparing')
    p.add_argument('--generation',default='v1')
    p.add_argument('--mode',choices=['upscale','native','edited_shape','continuous_field'],default='upscale')
    p.add_argument('--mask-source',choices=['global','components'],default='global')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--bundle-dir',type=Path,
                   help='Edited bundle directory with preview.xtareview or legacy NPZs for one volume')
    p.add_argument('--skip-source-probe',action='store_true',help='Bundle-only validation; forbidden for preparation')
    p.add_argument('--native-filter')
    p.add_argument('--native-low-template',help='Explicit original low manifest path; {stem} substitution allowed')
    p.add_argument('--native-full-template',help='Explicit matching full manifest path; {stem} substitution allowed')
    p.add_argument('--native-union-receipt',type=Path,help='Original preview-union receipt; native mode only, paired with --native-union-source-pairs')
    p.add_argument('--native-union-source-pairs',type=Path,help='Explicit native union source-pair JSON; native mode only, paired with --native-union-receipt')
    p.add_argument('--python',default=os.sys.executable)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--scratch-root',default=os.environ.get('SLURM_TMPDIR',tempfile.gettempdir()),help='Local fast storage for native intermediate arrays and per-run --smoke input copies')
    p.add_argument('--smoke',action='store_true',help='Run optional CPU PNG transverse PTA smoke after preparation; never training')
    p.add_argument('--smoke-script',help='Installed PTA CLI entrypoint, required only for --smoke')
    p.add_argument('--smoke-fraction',type=float,default=.001)
    p.add_argument('--smoke-imgsz',type=int,default=2048)
    p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--ffprobe',default='ffprobe')
    p.add_argument('--max-preview-mib',type=int,default=512)
    p.add_argument('--max-snapshot-mib',type=int,default=512)
    p.add_argument('--block-mib',type=int,default=64)
    p.add_argument('--field-memory-mib',type=int,default=2048,help='continuous_field field-build RAM budget; separate from the aggregate native --block-mib budget')
    p.add_argument('--field-cache',type=Path,help='continuous_field only: persistent field-cache parent; an output-stem subdirectory is built or validated')
    a=p.parse_args(argv)
    if min(a.max_preview_mib,a.max_snapshot_mib,a.block_mib,a.field_memory_mib)<=0:p.error('Memory budgets must be positive.')
    if a.workers<1:p.error('workers must be positive')
    if a.smoke and a.validate_only:p.error('smoke cannot be combined with validate-only')
    if not 0<a.smoke_fraction<=1 or a.smoke_imgsz<1:p.error('Invalid smoke sampling/image size')
    if a.skip_source_probe and not a.validate_only:p.error('skip-source-probe is a validation-only option')
    if a.mode=='native' and a.mask_source!='global':p.error('Native mode uses the full strict edit-evidence contract; mask-source must remain global')
    if a.mode=='edited_shape' and a.mask_source!='global':p.error('Edited-shape mode requires mask-source global: the edited Global is authoritative')
    if a.mode=='continuous_field' and a.mask_source!='global':p.error('continuous_field requires the explicitly edited Global, not a component-union substitute')
    if a.field_cache and a.mode!='continuous_field':p.error('field-cache requires continuous_field mode')
    if bool(a.native_union_receipt)!=bool(a.native_union_source_pairs):p.error('native-union-receipt and native-union-source-pairs must be supplied together')
    if a.mode!='native' and (a.native_union_receipt or a.native_union_source_pairs):p.error('native union options require native mode')
    if a.volumes_file and a.stems:p.error('Use positional stems OR --volumes-file.')
    stems=a.stems
    if a.volumes_file:
        stems=[s.strip() for line in a.volumes_file.read_text().splitlines() if (s:=line.split('#',1)[0]).strip()]
        if not stems:p.error('Volume list is empty; no implicit default applied to an explicit list.')
    if not stems:p.error('Give at least one volume stem or --volumes-file.')
    if len(set(stems))!=len(stems):p.error('Duplicate volume stems.')
    if a.bundle_dir and len(stems)!=1:p.error('bundle-dir requires exactly one volume')
    if a.source_video and len(stems)!=1:p.error('source-video requires exactly one volume')
    if a.output_dir and len(stems)!=1:p.error('output-dir requires exactly one volume')
    if a.output_stem and len(stems)!=1:p.error('output-stem requires exactly one volume')
    # --root and --training-root locate the legacy layout only; explicit paths need neither.
    # An empty value, such as an unset shell variable, counts as absent and is named as empty.
    if not a.root and not a.source_video and not (a.validate_only and a.skip_source_probe):
        p.error(('root is empty; it is' if a.root=='' else 'root is')+' required without --source-video, '
                'except with --validate-only --skip-source-probe (the intensity is then ROOT/STEM.mkv)')
    legacy=[]
    if not a.training_root and not a.bundle_dir:
        legacy.append(('--bundle-dir','the bundle is then TRAINING_ROOT/GENERATION/STEM_Filtered'))
    if not a.training_root and not a.output_dir and not a.validate_only:
        legacy.append(('--output-dir when preparing','the output is then TRAINING_ROOT/GENERATION/STEM_Input'))
    if legacy:
        p.error(('training-root is empty; it is' if a.training_root=='' else 'training-root is')+' required without '
                +' and '.join(flag for flag,_ in legacy)+' ('+'; '.join(path for _,path in legacy)+')')
    if a.output_stem:safe_name(a.output_stem)
    for s in stems:safe_name(s)
    safe_name(a.generation)
    log_resolved=a.log_dir.resolve()
    base=training_base(a)
    for s in stems:
        bundle=bundle_path(a,s,base).resolve()
        # Validation without --output-dir or --training-root has no prepared-input directory to protect.
        output=output_path(a,s,base) if a.output_dir or base is not None else None
        protected=(bundle,) if output is None else (bundle,output)
        # No source: bundle-only validation (skip-source-probe) without --source-video or --root.
        source=(Path(a.source_video).resolve() if a.source_video else
                (legacy_root(a)/(s+'.mkv')).resolve() if a.root else None)
        if output is not None and (output==bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output)):
            p.error('Prepared input directory must be disjoint from the filtered bundle.')
        if source is not None and output is not None and (source==output or source.is_relative_to(output)):
            p.error('Prepared input directory must not contain the original intensity source.')
        for item in protected:
            if log_resolved==item or log_resolved.is_relative_to(item) or item.is_relative_to(log_resolved):
                p.error('Log directory must be disjoint from filtered bundles and prepared input directories.')
        if source is not None and (source==log_resolved or source.is_relative_to(log_resolved)):
            p.error('Log directory must not contain the original intensity source.')
        if a.field_cache:
            cache=a.field_cache.resolve()
            for item in protected:
                if cache==item or cache.is_relative_to(item) or item.is_relative_to(cache):
                    p.error('Field cache must be disjoint from input bundles and published label directories.')
            if source is not None and (source==cache or source.is_relative_to(cache)):
                p.error('Field cache must not contain the original intensity source.')
    if a.log_dir.exists() or a.log_dir.is_symlink():
        p.error(f'log-dir must name a new directory for each run, including after --validate-only: {a.log_dir}')
    a.log_dir.mkdir(parents=True,exist_ok=False)
    save_json(a.log_dir/'request.json',{'stems':stems,'mode':a.mode,'mask_source':a.mask_source,'generation':a.generation,
        'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}})
    try:
        outputs=[str(validate_only(a,s,a.log_dir) if a.validate_only else prepare(a,s,a.log_dir)) for s in stems]
        if a.smoke:smoke(a,outputs,a.log_dir)
        save_json(a.log_dir/'complete.json',{'status':'validated_only' if a.validate_only else 'complete','inputs':outputs})
    except BaseException as e:
        save_json(a.log_dir/'failure.json',{'status':'failed','error':f'{type(e).__name__}: {e}'})
        raise

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Prepare a PTA intensity/mask pair from edited preview NPZs, without training.

Dependencies: Python 3.10+, NumPy and ffprobe. Edited-shape reconstruction also
uses SciPy and the adjacent xta_edited_shape.py helper. Legacy preview upscaling
is separate from optional strict native evidence transfer and edited-shape mode.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
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
import tempfile
import uuid
import zipfile
from collections import OrderedDict
import numpy as np

VERSION = '1.0.0'
SCHEMA = 'pta-preview-input/1'
POLICY = 'preview-cubic-cell-center-threshold-0.5'
MIB = 1024**2
REFERENCE_RAS = np.diag([-1., -1., 1., 1.])

def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(4*MIB), b''): h.update(b)
    return h.hexdigest()

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

def load_bundle(bundle_dir, stem, mask_source='global'):
    edits_path = child(bundle_dir, 'edits.json')
    manifest_path = child(bundle_dir, 'source_manifest.json')
    e, m = read_json(edits_path), read_json(manifest_path)
    if e.get('format') != 'xta-slicer-edits-v1' or m.get('quality') != 'low_quality':
        raise ValueError('Expected a Slicer edit bundle and its low_quality source manifest.')
    if digest(manifest_path) != e.get('source_manifest_sha256'):
        raise ValueError('Source-manifest hash does not match edits.json.')
    low, full = shape3(e['preview_shape_tyx']), shape3(e['full_quality_shape_tyx'])
    if low != shape3(m['output_shape_tyx']) or full != shape3(m['full_quality_output_shape_tyx']):
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
        if not globals_[0].get('foreground_voxels',0):
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
    return low, full, selected, provenance

def insert_snapshot(target, path, entry, max_snapshot_bytes):
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
        a, extent, affine = z['mask'], z['extent_xyz'], z['ijk_to_ras']
        if a.dtype != np.uint8 or a.ndim != 3 or a.nbytes > max_snapshot_bytes:
            raise ValueError('Snapshot must be a bounded 3D uint8 binary mask.')
        count = 0
        for plane in a:
            if np.any(plane > 1): raise ValueError('Snapshot is not binary.')
            count += int(np.count_nonzero(plane))
        if count != entry.get('foreground_voxels'):
            raise ValueError('Snapshot foreground count differs from edits.json.')
        if a.size == 0:
            if count != 0: raise ValueError('Invalid empty snapshot.')
            return 0
        if extent.shape != (6,) or extent.dtype.kind not in 'iu':
            raise ValueError('Extent must have six integer XYZ entries.')
        if tuple((extent[1::2]-extent[::2]+1)[::-1]) != a.shape:
            raise ValueError('Snapshot extent/array shape mismatch.')
        if affine.shape != (4,4) or not np.isfinite(affine).all():
            raise ValueError('Invalid snapshot affine.')
        rel = REFERENCE_RAS @ affine
        if not np.allclose(rel[:3,:3],np.eye(3),rtol=0,atol=1e-5) or not np.allclose(rel[3],[0,0,0,1],rtol=0,atol=1e-7):
            raise ValueError('Edited lattice was rotated/resampled; expected XTA unit-LPS preview lattice.')
        offset = np.rint(rel[:3,3])
        if not np.allclose(offset,rel[:3,3],rtol=0,atol=1e-5) or np.max(np.abs(offset)) > 2**40:
            raise ValueError('Edited origin is not an integer preview-lattice offset.')
        start = (extent[::2]+offset.astype(np.int64))[::-1]
        lo, hi = np.maximum(start,0), np.minimum(start+np.array(a.shape),target.shape)
        inside_count = 0
        if np.all(hi > lo):
            src = tuple(slice(int(l-s),int(h-s)) for l,h,s in zip(lo,hi,start))
            dst = tuple(slice(int(l),int(h)) for l,h in zip(lo,hi))
            view=a[src]
            inside_count=int(np.count_nonzero(view))
            np.bitwise_or(target[dst],view,out=target[dst])
        if inside_count != count:
            raise ValueError('Edited foreground lies outside the declared preview FOV; no clipping allowed.')
        return count

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
    if tool.is_relative_to(bundle) or low==full:raise ValueError('Invalid native tool/source paths.')
    version=subprocess.check_output([args.python,str(tool),'--version'],text=True).strip()
    if version!='0.3.1':raise ValueError(f'This wrapper is qualified for native_filter 0.3.1, found {version!r}.')
    if digest(low)!=read_json(bundle/'edits.json')['source_manifest_sha256']:
        raise ValueError('Native low manifest is not the exact edit-bundle source manifest.')
    contract={'filter':str(tool),'filter_sha256':digest(tool),'filter_version':version,
              'low_manifest':str(low),'low_sha256':digest(low),'full_manifest':str(full),'full_sha256':digest(full)}
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
                if actual!=pair[kind+'_manifest_sha256']:
                    raise ValueError(f'Union {kind} source-manifest checksum mismatch: {path}')
                source_manifests.append({'path':str(path),'sha256':actual})
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


def edited_shape_quality(mask, low, full, foreground, contract):
    """Require the known writer's actual-byte audit before trusting its result.

    This checks the writer/receipt contract, not a second numerical audit. The
    in-process helper already rereads every native payload byte and checks its
    hash, binary values, edited centers, background cells and 6-components.
    """
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


def verify_compressed_identity(path, mask):
    """Detect a changed audited file without decompressing its payload again."""
    before=path.stat()
    actual=digest(path)
    after=path.stat()
    if (before.st_size!=mask['bytes'] or actual!=mask['sha256']
            or (before.st_size,before.st_mtime_ns,before.st_ino)!=
               (after.st_size,after.st_mtime_ns,after.st_ino)):
        raise ValueError('Audited edited-shape NRRD changed before publication.')


def output_path(args,stem,base):
    # Keep the final path unresolved so an existing symlink is still rejected.
    if args.output_dir:
        path=Path(args.output_dir).absolute()
        return path.parent.resolve()/path.name
    return base/(stem+'_Input')

def prepare(args, stem, log_dir):
    safe_name(stem)
    root=Path(args.root).resolve();base=Path(args.training_root).resolve()/safe_name(args.generation)
    bundle=base/(stem+'_Filtered'); output=output_path(args,stem,base)
    source=Path(args.source_video).resolve() if args.source_video else root/(stem+'.mkv')
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
        if math.prod(low)>args.max_preview_mib*MIB:raise ValueError('Preview exceeds configured union allocation budget.')
        source_stat=source_identity(source,hashed=args.intensity_mode=='copy')
        probe=probe_source(source,args.ffprobe)
        if tuple(probe['shape_tyx'])!=full:raise ValueError(f'Original intensity dimensions differ from bundle: {probe["shape_tyx"]} != {full}')
        native=native_contract(args,stem,bundle) if args.mode=='native' else None
        shape_module,shape_contract=edited_shape_contract() if args.mode=='edited_shape' else (None,None)
        policy=POLICY if args.mode=='upscale' else shape_contract['policy'] if shape_contract else 'strict-native-filter-0.3.1-auto'
        identity={'stem':stem,'mode':args.mode,'mask_source':args.mask_source,'policy':policy,
                  'preparation_algorithm_version':VERSION,'generation':args.generation,'bundle':bundle_prov,'intensity':source_stat,'native':native}
        if shape_contract:identity['edited_shape']=shape_contract
        if args.intensity_mode!='symlink':identity['intensity_mode']=args.intensity_mode
        if output.exists() or output.is_symlink():
            if output.is_symlink() or not output.is_dir():raise ValueError('Existing Input must be a regular directory.')
            receipt=read_json(output/'preparation.json')
            if receipt.get('format')!=SCHEMA or receipt.get('status')!='complete' or receipt.get('identity')!=identity:
                raise ValueError('Existing input provenance differs; nothing overwritten. Use a new generation or inspect it.')
            if args.mode=='native':native_quality(receipt.get('native_quality',{}))
            if shape_contract:
                edited_shape_quality(receipt['mask'],low,full,selected[0]['foreground_voxels'],shape_contract)
            if {p.name for p in output.iterdir()}!={stem+'.mkv',stem+'.nrrd','preparation.json'}:
                raise ValueError('Existing Input contains unexpected files; refuse ambiguous PTA discovery.')
            intensity=output/(stem+'.mkv')
            if args.intensity_mode=='symlink':
                if not intensity.is_symlink() or intensity.resolve()!=source.resolve():raise ValueError('Existing intensity link has changed.')
            elif (intensity.is_symlink() or not intensity.is_file() or intensity.stat().st_size!=source_stat['bytes']
                  or digest(intensity)!=source_stat['sha256'] or receipt.get('intensity_publication')!=
                  {'method':'copy','source':str(source),'bytes':source_stat['bytes'],'sha256':source_stat['sha256']}):
                raise ValueError('Existing copied intensity identity has changed.')
            validate_nrrd(output/(stem+'.nrrd'),full,receipt['mask'])
            if source_identity(source,hashed=args.intensity_mode=='copy')!=source_stat:
                raise ValueError('Inputs changed during existing-input verification.')
            _,_,_,after=load_bundle(bundle,stem,args.mask_source)
            if (after!=bundle_prov or (args.mode=='native' and native_contract(args,stem,bundle)!=native)
                    or (shape_contract and edited_shape_contract()[1]!=shape_contract)):
                raise ValueError('Inputs changed during existing-input verification.')
            save_json(log_dir/(stem+'.result.json'),{'status':'validated_existing','input':str(output),'preparation_sha256':digest(output/'preparation.json')})
            print(f'Validated existing input: {output}',flush=True);return output
        stage=Path(tempfile.mkdtemp(prefix='.'+output.name+'.prep-',dir=output.parent))
        count=None
        quality=None
        if args.mode in {'upscale','edited_shape'}:
            preview=np.zeros(low,np.uint8)
            for i,entry in enumerate(selected,1):
                insert_snapshot(preview,child(bundle,entry['file']),entry,args.max_snapshot_mib*MIB)
                print(f'{stem}: selected snapshot {i}/{len(selected)} {entry["name"]}',flush=True)
            count=int(np.count_nonzero(preview))
            if not count:raise ValueError('Selected preview is empty; no output published.')
            mask=(shape_module.write_edited_shape(stage/(stem+'.nrrd'),preview,full,
                    block_mib=args.block_mib,workers=args.workers) if shape_module else
                  write_upscaled(stage/(stem+'.nrrd'),preview,full,args.block_mib))
            del preview
            if shape_contract:
                edited_shape_quality(mask,low,full,count,shape_contract)
            else:
                validate_nrrd(stage/(stem+'.nrrd'),full,mask)
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
                shutil.move(str(native_out/(stem+'.nrrd')),str(stage/(stem+'.nrrd')))
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
        intensity_publication=publish_intensity(source,stage/(stem+'.mkv'),args.intensity_mode,source_stat)
        # Detect source and bundle changes before making output visible.
        _,_,_,after=load_bundle(bundle,stem,args.mask_source)
        if (after!=bundle_prov or source_identity(source,hashed=args.intensity_mode=='copy')!=source_stat
                or (args.mode=='native' and native_contract(args,stem,bundle)!=native)
                or (shape_contract and edited_shape_contract()[1]!=shape_contract)):
            raise ValueError('Inputs changed during preparation; output not published.')
        if shape_contract:verify_compressed_identity(stage/(stem+'.nrrd'),mask)
        receipt={'format':SCHEMA,'tool_version':VERSION,'status':'complete','created_utc':datetime.now(timezone.utc).isoformat(),
                 'identity':identity,'source_probe':probe,'mask':mask,'preview_union_foreground_voxels':count,
                 'algorithm':('Selected preview mask; separable Keys cubic a=-0.5, zero exterior, coordinate (j+.5)*L/H-.5; float32 output >=0.5' if args.mode=='upscale' else
                              shape_contract['policy'] if shape_contract else 'Existing strict native_filter 0.3.1 --auto; its evidence checks and resolved-success requirement apply'),
                 'source_intensity_hash':source_stat.get('sha256','not calculated; path/size/mtime and probed dimensions recorded'),
                 'intensity_publication':intensity_publication,
                 'native_component_recovery':args.mode=='native','original_source_nrrds_required':args.mode=='native',
                 'notes':['Default selects the updated nonempty Global. Components OR requires explicit mask-source selection.',
                          'Upscale does not apply largest-object filtering, smoothing or filling. Native mode retains its tool defaults.',
                          ('Historical source NRRD hashes in edits.json are provenance only; native/source layer bytes were not read.' if args.mode=='upscale' else 'Existing native tool reads and verifies original preview/native layer bytes; native diagnostics are retained in the run log directory.'),
                          'Output uses uncalibrated XTA source-grid unit LPS, not physical millimetre calibration.',
                          ('Preview upscale cannot recover boundaries removed by downsampling.' if args.mode=='upscale' else 'Native transfer is conditional on the unmodified deletion-evidence contract; no mixed-confidence bypass.')],
                 'helper_sha256':digest(Path(__file__))}
        if shape_contract:
            receipt.update(edited_shape_reconstruction=True,label_authority='edited_preview_global',
                           restored_native_detail=False)
            receipt['notes']=['The selected edited Global is authoritative for shape and separation.',
                              'Signed-distance reconstruction samples the source-grid cell centers; original prediction NRRDs are not read or restored.',
                              'Output uses uncalibrated XTA source-grid unit LPS, not physical millimetre calibration.']
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
    bundle=Path(args.bundle_dir).resolve() if args.bundle_dir else Path(args.training_root).resolve()/safe_name(args.generation)/(stem+'_Filtered')
    low,full,selected,provenance=load_bundle(bundle,stem,args.mask_source)
    if math.prod(low)>args.max_preview_mib*MIB:raise ValueError('Preview exceeds union allocation budget.')
    preview=np.zeros(low,np.uint8)
    for entry in selected:insert_snapshot(preview,child(bundle,entry['file']),entry,args.max_snapshot_mib*MIB)
    report={'status':'validated_only','bundle':str(bundle),'provenance':provenance,
            'selected_foreground_voxels':int(np.count_nonzero(preview)),'no_hq_output_or_input_directory_created':True}
    if not args.skip_source_probe:
        source=Path(args.source_video).resolve() if args.source_video else Path(args.root)/(stem+'.mkv')
        report['source_probe']=probe_source(source,args.ffprobe)
        if tuple(report['source_probe']['shape_tyx'])!=full:raise ValueError('Intensity dimensions do not match bundle.')
    else:report['source_probe']='explicitly skipped for bundle-only validation; not permitted for preparation'
    if args.mode=='native':report['native']=native_contract(args,stem,bundle)
    if args.mode=='edited_shape':report['edited_shape']=edited_shape_contract()[1]
    save_json(log_dir/(stem+'.result.json'),report)
    print(f'Validated bundle only: {bundle}; selected FG={report["selected_foreground_voxels"]}',flush=True)
    return str(bundle)

def smoke(args,inputs,log_dir):
    if not args.smoke_script or not Path(args.smoke_script).is_file():raise ValueError('--smoke requires an existing --smoke-script entrypoint.')
    command=[args.python,str(Path(args.smoke_script).resolve()),'--mode','pta','--task','segment',
             '--input',*inputs,'--output',str(log_dir/'pta-smoke'),'--imgsz',str(args.smoke_imgsz),
             '--background_percent','1','--channel_format','gray','--output_format','png',
             '--worker_backend','thread','--enable_cartesian','transverse',
             '--fraction',str(args.smoke_fraction),'--save','images','labels','summary']
    save_json(log_dir/'smoke-command.json',{'command':command,'training':False,'requested_gpu':False})
    env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']='';env['YOLO_TTA_TELEMETRY_DIR']=str(log_dir/'smoke-telemetry')
    with (log_dir/'smoke.log').open('w',encoding='utf-8') as f:
        subprocess.run(command,check=True,stdout=f,stderr=subprocess.STDOUT,env=env)
    if not (log_dir/'pta-smoke'/'manifest.json').is_file():raise ValueError('PTA smoke returned without manifest.json.')

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stems',nargs='*')
    p.add_argument('--volumes-file',type=Path,help='One stem per line; blank lines and # comments ignored')
    p.add_argument('--root',default='/project/community/sanghyus')
    p.add_argument('--source-video',type=Path,help='Explicit original intensity path for exactly one volume; default root/stem.mkv remains')
    p.add_argument('--output-dir',type=Path,help='Explicit prepared-input directory for exactly one volume; existing incompatible data is never overwritten')
    p.add_argument('--intensity-mode',choices=['symlink','copy'],default='symlink',help='Publish intensity as an original-source link (default) or a verified independent copy')
    p.add_argument('--training-root',default='/project/community/sanghyus/XTA_TRAINING')
    p.add_argument('--generation',default='v1')
    p.add_argument('--mode',choices=['upscale','native','edited_shape'],default='upscale')
    p.add_argument('--mask-source',choices=['global','components'],default='global')
    p.add_argument('--validate-only',action='store_true')
    p.add_argument('--bundle-dir',type=Path,help='Explicit bundle for single-volume validation only')
    p.add_argument('--skip-source-probe',action='store_true',help='Bundle-only validation; forbidden for preparation')
    p.add_argument('--native-filter')
    p.add_argument('--native-low-template',help='Explicit original low manifest path; {stem} substitution allowed')
    p.add_argument('--native-full-template',help='Explicit matching full manifest path; {stem} substitution allowed')
    p.add_argument('--native-union-receipt',type=Path,help='Original preview-union receipt; native mode only, paired with --native-union-source-pairs')
    p.add_argument('--native-union-source-pairs',type=Path,help='Explicit native union source-pair JSON; native mode only, paired with --native-union-receipt')
    p.add_argument('--python',default=os.sys.executable)
    p.add_argument('--workers',type=int,default=1)
    p.add_argument('--scratch-root',default=os.environ.get('SLURM_TMPDIR',tempfile.gettempdir()),help='Local fast storage for native intermediate arrays only')
    p.add_argument('--smoke',action='store_true',help='Run optional CPU PNG transverse PTA smoke after preparation; never training')
    p.add_argument('--smoke-script',help='Installed PTA CLI entrypoint, required only for --smoke')
    p.add_argument('--smoke-fraction',type=float,default=.001)
    p.add_argument('--smoke-imgsz',type=int,default=2048)
    p.add_argument('--log-dir',type=Path,required=True)
    p.add_argument('--ffprobe',default='ffprobe')
    p.add_argument('--max-preview-mib',type=int,default=512)
    p.add_argument('--max-snapshot-mib',type=int,default=512)
    p.add_argument('--block-mib',type=int,default=64)
    a=p.parse_args(argv)
    if min(a.max_preview_mib,a.max_snapshot_mib,a.block_mib)<=0:p.error('Memory budgets must be positive.')
    if a.workers<1:p.error('workers must be positive')
    if a.smoke and a.validate_only:p.error('smoke cannot be combined with validate-only')
    if not 0<a.smoke_fraction<=1 or a.smoke_imgsz<1:p.error('Invalid smoke sampling/image size')
    if (a.bundle_dir or a.skip_source_probe) and not a.validate_only:p.error('bundle-dir/skip-source-probe are validation-only options')
    if a.mode=='native' and a.mask_source!='global':p.error('Native mode uses the full strict edit-evidence contract; mask-source must remain global')
    if a.mode=='edited_shape' and a.mask_source!='global':p.error('Edited-shape mode requires mask-source global: the edited Global is authoritative')
    if bool(a.native_union_receipt)!=bool(a.native_union_source_pairs):p.error('native-union-receipt and native-union-source-pairs must be supplied together')
    if a.mode!='native' and (a.native_union_receipt or a.native_union_source_pairs):p.error('native union options require native mode')
    if a.volumes_file and a.stems:p.error('Use positional stems OR --volumes-file.')
    stems=a.stems
    if a.volumes_file:
        stems=[s.strip() for line in a.volumes_file.read_text().splitlines() if (s:=line.split('#',1)[0]).strip()]
        if not stems:p.error('Volume list is empty; no implicit default applied to an explicit list.')
    if not stems:stems=['F2_30_5_3_2026_8bit_Y']
    if len(set(stems))!=len(stems):p.error('Duplicate volume stems.')
    if a.bundle_dir and len(stems)!=1:p.error('bundle-dir requires exactly one volume')
    if a.source_video and len(stems)!=1:p.error('source-video requires exactly one volume')
    if a.output_dir and len(stems)!=1:p.error('output-dir requires exactly one volume')
    for s in stems:safe_name(s)
    safe_name(a.generation)
    log_resolved=a.log_dir.resolve()
    for s in stems:
        base=Path(a.training_root).resolve()/a.generation
        bundle=Path(a.bundle_dir).resolve() if a.bundle_dir else base/(s+'_Filtered')
        output=output_path(a,s,base)
        source=Path(a.source_video).resolve() if a.source_video else (Path(a.root)/(s+'.mkv')).resolve()
        if output==bundle or output.is_relative_to(bundle) or bundle.is_relative_to(output):
            p.error('Prepared input directory must be disjoint from the filtered bundle.')
        if source==output or source.is_relative_to(output):
            p.error('Prepared input directory must not contain the original intensity source.')
        for protected in (bundle,output):
            if log_resolved==protected or log_resolved.is_relative_to(protected) or protected.is_relative_to(log_resolved):
                p.error('Log directory must be disjoint from filtered bundles and prepared input directories.')
        if source==log_resolved or source.is_relative_to(log_resolved):
            p.error('Log directory must not contain the original intensity source.')
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

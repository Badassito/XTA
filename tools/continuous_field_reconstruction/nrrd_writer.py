"""Portable streamed binary NRRD publication for the fixed research upscaler.

The sampler owns reconstruction and valid-domain/outside-support policy. This
module owns complete ordered native bytes, full-grid coordinates, gzip readback,
hashes and atomic no-overwrite publication. It never infers hard edit authority
or promises coarse-center, topology or anatomical correctness.
"""
from __future__ import annotations

import gzip
import hashlib
import importlib
import copy
import math
import os
from pathlib import Path
import tempfile
import numpy as np

try:
    from isal.igzip import IGzipFile as _gzip_reader
except (ImportError, OSError):
    _gzip_reader = gzip.GzipFile

MIB=1024**2
READBACK_CHUNK=4*MIB


def _shape3(value):
    try:items=tuple(value)
    except TypeError as exc:raise ValueError('native_shape_tyx must contain three positive integers') from exc
    if len(items)!=3 or any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) or v<=0 for v in items):
        raise ValueError('native_shape_tyx must contain three positive integers')
    return tuple(int(v) for v in items)


def _parameters(block_mib,workers=1):
    if isinstance(block_mib,(bool,np.bool_)) or not isinstance(block_mib,(int,float,np.integer,np.floating)) or not math.isfinite(block_mib) or block_mib<=0:
        raise ValueError('block_mib must be positive and finite')
    if isinstance(workers,(bool,np.bool_)) or not isinstance(workers,(int,np.integer)) or workers<1:
        raise ValueError('workers must be a positive integer')


def _destination(destination):
    # Test the final lexical entry BEFORE resolving its parent. Otherwise a
    # dangling destination symlink could be followed and accidentally filled.
    raw=Path(destination).expanduser()
    if os.path.lexists(raw):raise FileExistsError(raw)
    raw.parent.mkdir(parents=True,exist_ok=True)
    result=raw.parent.resolve()/raw.name
    if os.path.lexists(result):raise FileExistsError(result)
    return result


def _header(shape):
    t,y,x=shape
    return ('NRRD0005\n# Fixed research upscaler: coarse samples are not implicit edit authority\n'
            'type: unsigned char\ndimension: 3\nspace: left-posterior-superior\n'
            f'sizes: {x} {y} {t}\n'
            'space directions: (1,0,0) (0,1,0) (0,0,1)\n'
            'kinds: domain domain domain\nencoding: gzip\nspace origin: (0,0,0)\n'
            'coordinate units:=uncalibrated native source-grid voxel-center indices\n\n').encode('ascii')


def _notify(progress,stage,completed,total,**extra):
    if progress is not None:
        progress({'stage':stage,'completed_voxels':int(completed),'total_voxels':int(total),**extra})


def _digest(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda:stream.read(READBACK_CHUNK),b''):result.update(data)
    return result.hexdigest()


def _readback(path,shape,expected_header,progress=None):
    total=math.prod(shape);done=0;foreground=0;payload=hashlib.sha256()
    _notify(progress,'readback',0,total)
    with Path(path).open('rb') as raw:
        if raw.read(len(expected_header))!=expected_header:
            raise ValueError('Written NRRD header, native shape or global origin differs from the contract')
        with _gzip_reader(fileobj=raw,mode='rb') as compressed:
            while done<total:
                needed=min(READBACK_CHUNK,total-done)
                data=compressed.read(needed)
                if len(data)!=needed:raise ValueError('Written NRRD payload is truncated')
                values=np.frombuffer(data,np.uint8)
                if values.max(initial=0)>1:raise ValueError('Written NRRD payload is not binary uint8')
                payload.update(data);foreground+=int(np.count_nonzero(values));done+=len(data)
                _notify(progress,'readback',done,total)
            # Forces gzip footer/CRC validation and rejects an extra payload or
            # concatenated member containing additional native voxels.
            if compressed.read(1):raise ValueError('Written NRRD exceeds its declared native shape')
    return {'foreground_voxels':foreground,'binary_payload_sha256':payload.hexdigest(),
            'native_voxels_read':done,'binary_values_verified':True,'gzip_readback_to_end':True,
            'header_and_global_origin_verified':True}


def _publish_exclusive(temporary,destination):
    if os.path.lexists(destination):raise FileExistsError(destination)
    if os.name=='nt':
        # Windows rename is atomic and refuses an existing destination.
        os.rename(temporary,destination)
    else:
        # POSIX rename may overwrite. A same-filesystem hard link atomically
        # publishes the complete inode and fails if any destination entry exists.
        # Filesystems lacking atomic no-clobber hard links fail rather than use
        # an observable partial-copy or overwrite fallback.
        os.link(temporary,destination)
        os.unlink(temporary)


def write_native_blocks(blocks,destination,*,native_shape_tyx,block_mib=128,progress=None):
    """Validate/publish complete ordered ``(t,y,uint8_rows)`` native blocks.

    ``rows`` is two-dimensional and spans the entire native X axis. No full
    native volume is allocated. ``progress`` receives event dictionaries for
    write/readback/validated; callback exceptions before publication abort cleanly.
    Returns a receipt, without writing surprise sidecars. Existing outputs,
    including dangling symlinks, are refused. A failed temporary is removed.
    """
    shape=_shape3(native_shape_tyx);_parameters(block_mib)
    if progress is not None and not callable(progress):raise ValueError('progress must be callable or None')
    path=_destination(destination);header=_header(shape);total=math.prod(shape)
    expected_t=expected_y=0;done=foreground=block_count=0;payload=hashlib.sha256();temporary=None
    iterator=None
    limit=max(shape[2],int(float(block_mib)*MIB)) # one full X row is unavoidable
    stage='write'
    try:
        iterator=iter(blocks)
        _notify(progress,'write',0,total)
        with tempfile.NamedTemporaryFile(mode='wb',prefix=path.name+'.',suffix='.partial',dir=path.parent,delete=False) as raw:
            temporary=Path(raw.name);raw.write(header)
            with gzip.GzipFile(filename='',fileobj=raw,mode='wb',compresslevel=1,mtime=0) as compressed:
                for item in iterator:
                    if not isinstance(item,(tuple,list)) or len(item)!=3:raise ValueError('Each native block must be (t,y,uint8_rows)')
                    t,y,rows=item
                    if any(isinstance(v,(bool,np.bool_)) or not isinstance(v,(int,np.integer)) for v in [t,y]):
                        raise ValueError('Native block indices must be integers')
                    if not 0<=t<shape[0] or t!=expected_t or y!=expected_y:
                        raise ValueError('Native blocks must cover the full volume in ordered t/y positions without gaps or overlaps')
                    a=np.asarray(rows)
                    if a.ndim!=2 or a.shape[1]!=shape[2] or not 1<=a.shape[0]<=shape[1]-y:
                        raise ValueError('Native row block shape does not match the remaining full-grid rows')
                    if a.dtype!=np.dtype('uint8'):raise ValueError('Native row blocks must have uint8 dtype')
                    if a.nbytes>limit:raise ValueError('Native row block exceeds the declared bounded buffer budget')
                    if a.max(initial=0)>1:raise ValueError('Native row block contains nonbinary values')
                    a=np.ascontiguousarray(a);data=memoryview(a).cast('B')
                    compressed.write(data);payload.update(data);done+=a.size
                    foreground+=int(np.count_nonzero(a));block_count+=1
                    expected_y+=a.shape[0]
                    if expected_y==shape[1]:expected_t+=1;expected_y=0
                    _notify(progress,'write',done,total,foreground_voxels=foreground)
                close=getattr(iterator,'close',None)
                if close is not None:close()
                iterator=None
            raw.flush();os.fsync(raw.fileno())
        if expected_t!=shape[0] or expected_y!=0 or done!=total:
            raise ValueError('Native block iterator ended before the declared full volume was complete')
        stage='readback'
        validation=_readback(temporary,shape,header,progress)
        if validation['foreground_voxels']!=foreground or validation['binary_payload_sha256']!=payload.hexdigest():
            raise ValueError('Written/readback native bytes differ from generated bytes or foreground count')
        compressed_sha=_digest(temporary);size=temporary.stat().st_size
        _notify(progress,'validated',total,total,foreground_voxels=foreground)
        stage='publish';_publish_exclusive(temporary,path);temporary=None
        return {'status':'complete','path':str(path),'shape_tyx':list(shape),'sizes_xyz':list(reversed(shape)),
                'space':'left-posterior-superior','space_origin_xyz':[0,0,0],
                'coordinate_units':'uncalibrated native source-grid voxel-center indices',
                'foreground_voxels':foreground,'binary_payload_sha256':payload.hexdigest(),
                'sha256':compressed_sha,'bytes':size,'encoding':'gzip','dtype':'uint8',
                'validation':validation,'native_blocks_written':block_count,
                'block_mib':float(block_mib),'native_buffer_limit_bytes':limit,
                'publication':'atomic exclusive same-volume publish after mandatory readback',
                'authority':'No coarse-center/topology guarantee or hard edit authority is inferred by the byte writer.'}
    except Exception as exc:
        # The enclosing CLI may persist this diagnostic in its chosen evidence
        # location; the portable library never creates unsolicited log sidecars.
        try:exc.upscale_write_context={'stage':stage,'native_voxels_written':done,'total_native_voxels':total,'destination':str(path)}
        except Exception:pass
        raise
    finally:
        try:
            # Settle generator/executor cleanup before the public wrapper closes
            # path-opened coefficient memmaps, including interruption/failure.
            if iterator is not None:
                close=getattr(iterator,'close',None)
                if close is not None:close()
        finally:
            if temporary is not None and os.path.lexists(temporary):temporary.unlink()


def _reference_shape(reference):
    if hasattr(reference,'native_shape_tyx'):return _shape3(reference.native_shape_tyx)
    if hasattr(reference,'native'):return _shape3(reference.native)
    if isinstance(reference,dict) and 'native_shape_tyx' in reference:return _shape3(reference['native_shape_tyx'])
    raise ValueError('Resolved reference must declare native_shape_tyx')


def write_upscaled_nrrd(reference_or_cache,destination,*,native_shape_tyx=None,block_mib=128,workers=1,progress=None):
    """Resolve a reference/cache, sample bounded blocks and publish one NRRD.

    Core integration requires ``radius_upscaler.resolve_reference(value,
    native_shape_tyx=...)`` and ``iter_native_blocks(reference,block_mib,workers)``.
    That core is responsible for mathematically justified outside-context zero
    handling. This writer never replaces missing/invalid scalar data by background.
    A caller-owned reference remains open; a reference opened from a path is closed.
    """
    _parameters(block_mib,workers)
    if native_shape_tyx is not None:native_shape_tyx=_shape3(native_shape_tyx)
    # Refuse the final entry before opening/reconstructing a potentially large cache.
    path=_destination(destination)
    radius_upscaler=importlib.import_module((__package__+'.' if __package__ else '')+'radius_upscaler')
    reference=radius_upscaler.resolve_reference(reference_or_cache,native_shape_tyx=native_shape_tyx)
    # Paths and builder-result dictionaries resolve to a newly opened reference;
    # passing an existing CoreFieldReference returns that same caller-owned object.
    own=reference is not reference_or_cache
    try:
        shape=_reference_shape(reference)
        if native_shape_tyx is not None and shape!=native_shape_tyx:
            raise ValueError('Resolved reference/native-shape override mismatch')
        initial_stats=dict(getattr(reference,'stats',{}))
        blocks=radius_upscaler.iter_native_blocks(reference,block_mib=block_mib,workers=int(workers))
        receipt=write_native_blocks(blocks,path,native_shape_tyx=shape,block_mib=block_mib,progress=progress)
        receipt['workers']=int(workers)
        receipt['algorithm']=getattr(radius_upscaler,'ALGORITHM',getattr(radius_upscaler,'POLICY','fixed radius-cap upscaler'))
        receipt['reference_type']=type(reference).__name__
        receipt['reference_provenance']=copy.deepcopy(getattr(reference,'provenance',None))
        final_stats=dict(getattr(reference,'stats',{}))
        receipt['sampler_statistics']={key:value-initial_stats.get(key,0)
            if isinstance(value,(int,float,np.integer,np.floating)) else value
            for key,value in final_stats.items()}
        receipt['sampler_statistics_scope']='This write; numeric deltas from reference counters at entry.'
        return receipt
    finally:
        if own and hasattr(reference,'close'):reference.close()

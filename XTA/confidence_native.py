"""Deferred native confidence pieces and explicit one-layer source conversion."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
import math
import hashlib
import json
from numbers import Integral
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .confidence_storage import BLOCK_SCHEMA, BLOCK_LAYOUT, PIECE_LAYOUT, checked_shape, plane_blocks, assembled_crop


def _copy_numeric_file(source,target,expected=None):
    digest=hashlib.sha256()
    with Path(source).open('rb') as src,Path(target).open('xb') as dst:
        while chunk:=src.read(1024*1024):
            if dst.write(chunk) != len(chunk):
                raise OSError('Incomplete native confidence piece write')
            digest.update(chunk)
    actual=digest.hexdigest()
    if expected is not None and actual!=expected:
        raise ValueError('Native confidence piece changed before/during persistent copy')
    return actual


def _native_piece_inputs(pieces, destination, native_shape, source_shape, model_name):
    """Snapshot immutable identities and validate their native-grid contracts."""
    from .confidence_evidence import ConfidenceEvidenceRef
    inputs = []
    for piece in pieces:
        original = piece['reference']
        directory = Path(original.path if isinstance(original, ConfidenceEvidenceRef) else original).resolve()
        if destination.is_relative_to(directory) or directory.is_relative_to(destination):
            raise ValueError('Confidence destination must be separate from every input piece')
        raw = (directory/'metadata.json').read_bytes()
        metadata = json.loads(raw)
        ref = ConfidenceEvidenceRef.open(directory)
        if ref.metadata != metadata or (isinstance(original, ConfidenceEvidenceRef) and original.metadata != metadata):
            raise ValueError('Native confidence piece metadata changed before copy')
        if ref.model_name != str(model_name):
            raise ValueError('Native confidence piece belongs to a different model')
        if ref.metadata.get('layout') not in (None, BLOCK_LAYOUT):
            raise ValueError('Native confidence pieces must contain direct numeric blocks')
        if ref.coordinate_space != 'native_view_processing':
            raise ValueError('Native confidence piece coordinates differ')
        if ref.source_shape is not None and ref.source_shape != source_shape:
            raise ValueError('Native confidence piece source geometry differs')
        offset = tuple(piece.get('offset_tyx', (0, 0, 0)))
        if (len(offset) != 3 or any(isinstance(v, bool) or not isinstance(v, Integral) for v in offset)
                or any(a < 0 or a+b > c for a,b,c in zip(offset, ref.storage_shape, native_shape))):
            raise ValueError('Native confidence piece exceeds its parent grid')
        count = metadata.get('known_voxels')
        if isinstance(count, bool) or not isinstance(count, Integral) or not 0 <= count <= math.prod(ref.storage_shape):
            raise ValueError('A direct confidence piece must declare a valid known-value count')
        files = ('scores.u8.zlib', 'index.bin' if metadata.get('layout') == BLOCK_LAYOUT else 'index.json')
        for name in ('metadata.json', *files):
            if not (directory/name).resolve().is_relative_to(directory):
                raise ValueError('Native confidence piece file escapes its directory')
        checksums = {}
        for name in files:
            checksum = metadata.get('payload_sha256' if name == 'scores.u8.zlib' else 'index_sha256')
            if name == 'index.json' and checksum is None:
                # Schema 1 did not store an index digest; snapshot it before copying.
                digest = hashlib.sha256()
                with (directory/name).open('rb') as stream:
                    while chunk := stream.read(1024*1024):
                        digest.update(chunk)
                checksum = digest.hexdigest()
            if (not isinstance(checksum, str) or len(checksum) != 64
                    or any(c not in '0123456789abcdef' for c in checksum)):
                raise ValueError('Native confidence piece checksum is missing or invalid')
            checksums[name] = checksum
        # Validate indices without decoding or constructing a dense score volume.
        with ref.native_reader():
            pass
        inputs.append((ref, tuple(map(int, offset)), hashlib.sha256(raw).hexdigest(), checksums))
    return inputs


def write_native_disjoint_leases(path, pieces, *, native_shape, source_shape, layer_key, model_name,
                                 provenance):
    """Persist full-frame D1 leases as one block payload without decoding."""
    from .confidence_consolidation import write_disjoint_leases
    return write_disjoint_leases(path, pieces, native_shape=native_shape, source_shape=source_shape,
        layer_key=layer_key, model_name=model_name, provenance=provenance)


def write_native_pieces(path, pieces, *, native_shape, source_shape, layer_key, model_name,
                        provenance, disjoint=False):
    """Publish validated numeric copies; metadata is the final completion marker.

    Only a new destination is accepted. Each attempt owns its private staging
    tree, so failed copies can be retried without modifying input or prior evidence.
    """
    from .confidence_evidence import ConfidenceEvidenceRef, SCORE_SEMANTICS, _write_json_atomic, _json_value
    destination = Path(path).resolve()
    native_shape,source_shape = checked_shape(native_shape),checked_shape(source_shape)
    if destination.exists():
        raise FileExistsError(f'Confidence destination already exists: {destination}')
    inputs = _native_piece_inputs(pieces, destination, native_shape, source_shape, model_name)
    destination.parent.mkdir(parents=True,exist_ok=True)
    destination.mkdir(exist_ok=False)
    owned = []
    try:
        staging = Path(tempfile.mkdtemp(prefix='.staging-', dir=destination)).resolve()
        owned.append(staging)
        (staging/'pieces').mkdir()
        records,contributions = [],0
        for number,(ref,offset,metadata_sha,checksums) in enumerate(inputs):
            name = f'pieces/{number:06d}'
            target = staging/name
            target.mkdir()
            copied={}
            for filename,expected in checksums.items():
                copied[filename]=_copy_numeric_file(ref.path/filename,target/filename,expected)
            _write_json_atomic(target/'metadata.json',ref.metadata)
            with ConfidenceEvidenceRef.open(target).native_reader():
                pass
            count = int(ref.metadata['known_voxels'])
            contributions += count
            records.append(dict(directory=name,offset_tyx=list(offset),stored_shape_tyx=list(ref.storage_shape),
                                layer_key=ref.layer_key,known_storage_voxels=count,copied_sha256=copied))
        for ref,_,metadata_sha,_ in inputs:
            if hashlib.sha256((ref.path/'metadata.json').read_bytes()).hexdigest() != metadata_sha:
                raise ValueError('Native confidence piece metadata changed during copy')
        metadata = dict(schema=BLOCK_SCHEMA,layout=PIECE_LAYOUT,coordinate_space='native_view_processing',
            stored_shape_tyx=list(native_shape),source_shape_tyx=list(source_shape),output_shape_tyx=list(source_shape),
            model_name=str(model_name),layer_key=str(layer_key),dtype='uint8',unknown='score_zero',
            score_semantics=SCORE_SEMANTICS,quantization='uint8/255; zero is unknown',pieces=records,
            reduction='maximum',pieces_disjoint=bool(disjoint),known_voxels=contributions if disjoint else None,
            piece_contract='disjoint_frame_leases' if disjoint else 'overlapping_native_crops',
            known_contributions=contributions,known_voxels_coordinate_space='native_view_processing',
            count_semantics='unique native known voxels' if disjoint else 'piece contributions may overlap; unique total not computed',
            provenance=_json_value(provenance))
        _write_json_atomic(staging/'metadata.json',metadata)
        with ConfidenceEvidenceRef.open(staging).native_reader():
            pass
        for name in ('pieces', 'metadata.json'):
            target = destination/name
            (staging/name).replace(target)
            owned.append(target)
        result = ConfidenceEvidenceRef.open(destination)
        staging.rmdir()
        return result
    except BaseException as error:
        for target in reversed(owned):
            try:
                # Every removable path belongs to this attempt and stays inside its new directory.
                if target.parent != destination or target.resolve().parent != destination:
                    raise RuntimeError('Confidence copy cleanup escaped its owned destination')
                if target.is_dir():
                    shutil.rmtree(target)
                else:
                    target.unlink(missing_ok=True)
            except (OSError, RuntimeError) as cleanup_error:
                if hasattr(error, 'add_note'):
                    error.add_note(f'Confidence copy cleanup failed for {target}: {cleanup_error}')
        try:
            destination.rmdir()
        except OSError:
            pass
        raise


class NativePieceReader:
    def __init__(self,reference):
        from .confidence_evidence import ConfidenceEvidenceRef
        self.reference,self.shape = reference,reference.storage_shape
        self.pieces=[]
        self.readers=OrderedDict()
        self._closed=False
        for record in reference.metadata.get('pieces',()):
            path=(reference.path/str(record['directory'])).resolve()
            if not path.is_relative_to(reference.path.resolve()) or path==reference.path.resolve():
                raise ValueError('Native confidence piece escapes its parent directory')
            ref=ConfidenceEvidenceRef.open(path)
            offset=tuple(map(int,record['offset_tyx']))
            if (ref.model_name!=reference.model_name or ref.layer_key!=record['layer_key']
                    or ref.storage_shape!=tuple(map(int,record['stored_shape_tyx']))
                    or len(offset)!=3 or any(a<0 or a+b>c for a,b,c in zip(offset,ref.storage_shape,self.shape))):
                raise ValueError('Native confidence piece identity or grid is inconsistent')
            self.pieces.append((ref,offset))
        if reference.metadata.get('piece_contract')=='disjoint_frame_leases':
            expected=0
            for ref,offset in sorted(self.pieces,key=lambda piece:piece[1][0]):
                if offset!=(expected,0,0) or ref.storage_shape[1:]!=self.shape[1:]:
                    raise ValueError('Stored D1 confidence leases have gaps, overlaps, or inconsistent planes')
                expected+=ref.storage_shape[0]
            if expected!=self.shape[0]:
                raise ValueError('Stored D1 confidence leases do not cover their native view')

    def iter_crops(self,z):
        if self._closed:
            raise RuntimeError('Native confidence piece reader is closed')
        z=int(z)
        if not 0<=z<self.shape[0]:
            raise IndexError('Native confidence slice is outside its grid')
        for number,(ref,offset) in enumerate(self.pieces):
            if self._closed:
                raise RuntimeError('Native confidence piece reader is closed')
            dz,dy,dx=offset
            if not dz<=z<dz+ref.storage_shape[0]:
                continue
            reader=self.readers.pop(number,None)
            if reader is None:
                reader=ref.native_reader()
            self.readers[number]=reader
            while len(self.readers)>32:
                _,old=self.readers.popitem(last=False)
                old.close()
            for y0,y1,x0,x1,crop in reader.iter_crops(z-dz):
                yield y0+dy,y1+dy,x0+dx,x1+dx,crop

    def close(self):
        self._closed=True
        for reader in getattr(self,'readers',{}).values():
            reader.close()
        if hasattr(self,'readers'):
            self.readers.clear()

    def __del__(self):
        self.close()


class ProjectedScoreReader:
    def __init__(self,shape,read_slice,memory_bytes):
        self.shape,self.read_slice,self.memory_bytes=tuple(shape),read_slice,int(memory_bytes)
        self._closed=False

    def iter_crops(self,z):
        if self._closed:
            raise RuntimeError('Explicit confidence source reader is closed')
        z=int(z)
        if not 0<=z<self.shape[0]:
            raise IndexError('Projected confidence slice is outside the source grid')
        value=self.read_slice(z)
        if value is not None:
            yield from plane_blocks(value)

    def __call__(self,z0,z1):
        if self._closed:
            raise RuntimeError('Explicit confidence source reader is closed')
        z0,z1=int(z0),int(z1)
        if not 0<=z0<=z1<=self.shape[0]:
            raise IndexError('Projected confidence slab is outside the source grid')
        if (z1-z0)*self.shape[1]*self.shape[2]*2>self.memory_bytes//2:
            raise MemoryError('Requested confidence slab exceeds the explicit conversion buffer budget')
        scores=np.zeros((z1-z0,*self.shape[1:]),np.uint8)
        for z in range(z0,z1):
            for y0,y1,x0,x1,crop in self.iter_crops(z):
                target=scores[z-z0,y0:y1,x0:x1]
                np.maximum(target,crop,out=target)
        return scores,scores>0

    def read_crop(self,z):
        if self._closed:
            raise RuntimeError('Explicit confidence source reader is closed')
        z=int(z)
        if not 0<=z<self.shape[0]:
            raise IndexError('Projected confidence slice is outside the source grid')
        value=self.read_slice(z)
        if value is None:
            return None
        # The projection already owns one plane. Keeping every block and its
        # Python record before assembling a bbox can exceed the remaining
        # budget, particularly for a million-row, one-column source grid.
        height,width=value.shape
        y0,y1,x0,x1=height,0,width,0
        flat=value.reshape(-1)
        for first in range(0,flat.size,16384):
            positions=np.flatnonzero(flat[first:first+16384])
            if not positions.size:
                continue
            positions+=first
            rows,columns=positions//width,positions%width
            y0,y1=min(y0,int(rows[0])),max(y1,int(rows[-1])+1)
            x0,x1=min(x0,int(columns.min())),max(x1,int(columns.max())+1)
        if y0==height:
            return None
        # This single result copy owns its bytes even when the bbox is the
        # entire plane or only one contiguous row of it.
        return y0,y1,x0,x1,np.array(value[y0:y1,x0:x1],dtype=np.uint8,copy=True,order='C')

    def close(self):
        self._closed=True
        self.read_slice=None


@contextmanager
def source_reader(reference,workspace=None,*,memory_mib=512,max_staging_mib=32768):
    """Explicitly project one native layer with a checked disk staging limit.

    The buffer budget excludes OS-managed file cache. The native mmap is disk
    staging, never an implicit source-reader allocation during collection.
    """
    if reference.coordinate_space=='source':
        with reference.reader() as reader:
            yield reader
        return
    if workspace is None:
        raise ValueError('Native confidence source conversion requires an explicit workspace')
    source=reference.source_shape
    if source is None:
        raise ValueError('Native confidence fragment lacks a complete source-grid descriptor')
    memory_bytes=int(float(memory_mib)*1024**2)
    disk_bytes=int(float(max_staging_mib)*1024**2)
    if memory_bytes<=0 or disk_bytes<=0:
        raise ValueError('Confidence conversion budgets must be positive')
    native=reference.storage_shape
    view_record=reference.metadata.get('provenance',{}).get('view')
    if not isinstance(view_record,dict):
        raise ValueError('Native confidence lacks its full view geometry')
    from .geometry import ViewInfo
    from .confidence_projection import (score_projection_reader, score_projection_workspace,
                                       score_projection_staging_shape)
    from .runtime import close_memmap_array_without_flush
    fields_available={field.name for field in fields(ViewInfo)}
    view=ViewInfo(**{k:v for k,v in view_record.items() if k in fields_available})
    if native[0]!=int(view.num_slices):
        raise ValueError('Native confidence frame count differs from its view geometry')
    if reference.metadata.get('known_voxels')==0 or reference.metadata.get('known_contributions')==0:
        with reference.native_reader():
            pass
        reader=ProjectedScoreReader(source,lambda z:None,memory_bytes)
        try:
            yield reader
        finally:
            reader.close()
        return
    # Half belongs to returned scores/known masks and crop assembly. The other
    # half is propagated into the backend, which chooses bounded strip sizes
    # after accounting for its own geometry and one output plane.
    score_projection_workspace(native,view,source,memory_bytes//2)
    # Decode happens before projection. Numeric payload, zlib output/input and
    # bounded index validation must also fit even when a producer uses one
    # large block. The normal 128-pixel format needs much less than this bound.
    decode_bytes=3*math.prod(native[1:])+2*1024**2
    if decode_bytes>memory_bytes:
        raise MemoryError(f'Native confidence decoding needs up to {decode_bytes} '
                          f'workspace bytes; limit is {memory_bytes}')
    staging_shape=score_projection_staging_shape(native,view,source)
    extra=math.prod(staging_shape) if staging_shape is not None else 0
    need=math.prod(native)+extra
    if need>disk_bytes:
        raise MemoryError(f'Native confidence conversion needs up to {need} staging bytes; limit is {disk_bytes}')
    work=Path(workspace).resolve()
    work.mkdir(parents=True,exist_ok=True)
    if shutil.disk_usage(work).free < need + 64*1024**2:
        raise OSError(f'Native confidence conversion needs {need} staging bytes plus 64 MiB free-disk reserve')
    with tempfile.TemporaryDirectory(prefix='confidence-source-',dir=work) as temporary:
        directory=Path(temporary).resolve()
        if not directory.is_relative_to(work):
            raise RuntimeError('Confidence staging escaped its explicit workspace')
        path=directory/'native.u8.dat'
        values=np.memmap(path,mode='w+',dtype=np.uint8,shape=native)
        try:
            with reference.native_reader() as reader:
                for z in range(native[0]):
                    for y0,y1,x0,x1,crop in reader.iter_crops(z):
                        target=values[z,y0:y1,x0:x1]
                        np.maximum(target,crop,out=target)
                    if (z+1)%16==0:
                        values.flush()
            values.flush()
            # The last decoded crop is owned bytes, not mmap storage. Retire
            # it before lending the same workspace budget to the projector.
            crop=target=None
            del reader
            with score_projection_reader(values,view,source,directory/'projection',
                                         memory_bytes=memory_bytes//2) as read_slice:
                projected=ProjectedScoreReader(source,read_slice,memory_bytes)
                try:
                    yield projected
                finally:
                    projected.close()
        finally:
            close_memmap_array_without_flush(values)


__all__=['write_native_pieces','write_native_disjoint_leases','NativePieceReader','ProjectedScoreReader','source_reader']

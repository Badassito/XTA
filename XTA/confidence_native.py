"""Deferred native confidence pieces and explicit one-layer source conversion."""
from __future__ import annotations

from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import fields
import math
import hashlib
from pathlib import Path
import shutil
import tempfile

import numpy as np

from .confidence_storage import BLOCK_SCHEMA, BLOCK_LAYOUT, PIECE_LAYOUT, checked_shape, plane_blocks, assembled_crop


def _copy_numeric_file(source,target,expected=None):
    digest=hashlib.sha256()
    with Path(source).open('rb') as src,Path(target).open('wb') as dst:
        while chunk:=src.read(1024*1024):
            dst.write(chunk)
            digest.update(chunk)
    actual=digest.hexdigest()
    if expected is not None and actual!=expected:
        raise ValueError('Native confidence piece changed before/during persistent copy')
    return actual


def write_native_pieces(path, pieces, *, native_shape, source_shape, layer_key, model_name,
                        provenance, disjoint=False):
    """Copy immutable numeric pieces once; never decompress or project them."""
    from .confidence_evidence import ConfidenceEvidenceRef, SCORE_SEMANTICS, _write_json_atomic, _json_value
    destination = Path(path)
    native_shape,source_shape = checked_shape(native_shape),checked_shape(source_shape)
    if (destination/'metadata.json').exists():
        raise FileExistsError(f'Confidence evidence already exists: {destination}')
    destination.mkdir(parents=True,exist_ok=True)
    records,contributions = [],0
    for number,piece in enumerate(pieces):
        ref = piece['reference']
        if not isinstance(ref,ConfidenceEvidenceRef):
            ref = ConfidenceEvidenceRef.open(ref)
        if ref.model_name != str(model_name):
            raise ValueError('Native confidence piece belongs to a different model')
        if ref.metadata.get('layout') not in (None,BLOCK_LAYOUT):
            raise ValueError('Native confidence pieces must contain direct numeric blocks')
        offset = tuple(map(int,piece.get('offset_tyx',(0,0,0))))
        if len(offset)!=3 or any(a<0 or a+b>c for a,b,c in zip(offset,ref.storage_shape,native_shape)):
            raise ValueError('Native confidence piece exceeds its parent grid')
        # Validate index/size contracts before making a persistent reference.
        with ref.native_reader():
            pass
        name = f'pieces/{number:06d}'
        target = destination/name
        target.mkdir(parents=True,exist_ok=False)
        payload_files = ('scores.u8.zlib','index.bin') if ref.metadata.get('layout') == BLOCK_LAYOUT else ('scores.u8.zlib','index.json')
        copied={}
        for filename in payload_files:
            expected=ref.metadata.get('payload_sha256' if filename=='scores.u8.zlib' else 'index_sha256')
            copied[filename]=_copy_numeric_file(ref.path/filename,target/filename,expected)
        _write_json_atomic(target/'metadata.json',ref.metadata)
        count = ref.metadata.get('known_voxels')
        if count is None:
            raise ValueError('A direct confidence piece must declare its known-value count')
        contributions += int(count)
        records.append(dict(directory=name,offset_tyx=list(offset),stored_shape_tyx=list(ref.storage_shape),
                            layer_key=ref.layer_key,known_storage_voxels=int(count),copied_sha256=copied))
    metadata = dict(schema=BLOCK_SCHEMA,layout=PIECE_LAYOUT,coordinate_space='native_view_processing',
        stored_shape_tyx=list(native_shape),source_shape_tyx=list(source_shape),output_shape_tyx=list(source_shape),
        model_name=str(model_name),layer_key=str(layer_key),dtype='uint8',unknown='score_zero',
        score_semantics=SCORE_SEMANTICS,quantization='uint8/255; zero is unknown',pieces=records,
        reduction='maximum',pieces_disjoint=bool(disjoint),known_voxels=contributions if disjoint else None,
        piece_contract='disjoint_frame_leases' if disjoint else 'overlapping_native_crops',
        known_contributions=contributions,known_voxels_coordinate_space='native_view_processing',
        count_semantics='unique native known voxels' if disjoint else 'piece contributions may overlap; unique total not computed',
        provenance=_json_value(provenance))
    _write_json_atomic(destination/'metadata.json',metadata)
    return ConfidenceEvidenceRef.open(destination)


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
        return assembled_crop(self.iter_crops(z))

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
    from .geometry import ViewInfo,is_tilted_view,is_tilted_azimuthal_view
    from .confidence_projection import score_projection_reader
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
    plane=max(math.prod(source[1:]),math.prod(native[1:]))
    if plane*8>memory_bytes:
        raise MemoryError('Confidence conversion cannot fit one XY working plane in its buffer budget')
    extra=max(math.prod(source),math.prod(native)) if is_tilted_view(view) or is_tilted_azimuthal_view(view) else 0
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
            with score_projection_reader(values,view,source,directory/'projection') as read_slice:
                projected=ProjectedScoreReader(source,read_slice,memory_bytes)
                try:
                    yield projected
                finally:
                    projected.close()
        finally:
            close_memmap_array_without_flush(values)


__all__=['write_native_pieces','NativePieceReader','ProjectedScoreReader','source_reader']

"""Bounded nogil masked-cell copy and independent zlib compression batches.

No library is installed or built. Linux first resolves Python's zlib extension,
then the system zlib loader name. Windows keeps the portable encoder unless an
explicit qualification override supplies a library. Every candidate must pass
bounded ABI/error-code and Python-decode probes before scientific execution.
"""
from __future__ import annotations

import ctypes
import ctypes.util
import os
import threading
import zlib

import numpy as np
from numba import types
from numba.extending import intrinsic

from ._deps import _numba
from .confidence_capture import COPY_BATCH_MAX_BYTES,COPY_BATCH_MAX_CELLS


_LOCK=threading.Lock()
_CACHE={}
_MAX_BLOCK=128


@intrinsic
def _void_pointer(typingctx,address):
    signature=types.voidptr(address)
    def codegen(context,builder,signature,args):
        return builder.inttoptr(args[0],context.get_value_type(types.voidptr))
    return signature,codegen


def _make_kernel(compressor,ulong_dtype):
    @_numba.njit(nogil=True,cache=False)
    def encode(mask,scores,z,bounds,cell_bytes,output_bytes,block):
        scratch=np.empty(cell_bytes,np.uint8)
        output=np.empty(output_bytes,np.uint8)
        offsets=np.empty(bounds.shape[0]+1,np.int64)
        length=np.empty(1,ulong_dtype)
        offsets[0]=0
        used=0
        for item in range(bounds.shape[0]):
            y0,y1,x0,x1=bounds[item,:4]
            if (not (0<=y0<y1<=mask.shape[1] and 0<=x0<x1<=mask.shape[2])
                    or y1-y0>block or x1-x0>block or (y1-y0)*(x1-x0)>cell_bytes):
                raise ValueError('Native confidence crop exceeds its bounded cell')
            at=0
            for row in range(y0,y1):
                for column in range(x0,x1):
                    scratch[at]=scores[z,row,column] if mask[z,row,column]!=0 else 0
                    at+=1
            length[0]=output_bytes-used
            status=compressor(_void_pointer(output.ctypes.data+np.uintp(used)),
                _void_pointer(length.ctypes.data),_void_pointer(scratch.ctypes.data),at,3)
            if status!=0:
                return output,offsets,item,int(status)
            used+=int(length[0])
            offsets[item+1]=used
        return output,offsets,bounds.shape[0],0
    return encode


class NativeConfidenceCodec:
    """Retain the library and direct C function pointer for one nogil kernel."""
    def __init__(self,library,*,source):
        self._library=library
        self.source=str(source)
        version=library.zlibVersion
        version.argtypes=[]
        version.restype=ctypes.c_char_p
        self.version=version().decode('ascii')
        if not self.version or len(self.version)>64:
            raise ValueError('Native zlib returned an invalid runtime version')
        self._bound=library.compressBound
        self._bound.argtypes=[ctypes.c_ulong]
        self._bound.restype=ctypes.c_ulong
        compressor=library.compress2
        compressor.argtypes=[ctypes.c_void_p,ctypes.c_void_p,ctypes.c_void_p,
                             ctypes.c_ulong,ctypes.c_int]
        compressor.restype=ctypes.c_int
        self._compressor=compressor
        self._bounds_cache={}
        self._kernel=_make_kernel(compressor,np.uint32 if ctypes.sizeof(ctypes.c_ulong)==4 else np.uint64)
        self._probe()

    def _probe(self):
        fixtures=(b'',b'\0',bytes(range(256)),b'\0'*16384,
                  bytes((index*37+index//11)%256 for index in range(16384)))
        guard=16
        for raw in fixtures:
            bound=int(self._bound(len(raw)))
            extra=(len(raw)>>12)+(len(raw)>>14)+(len(raw)>>25)+13
            if not 1<=bound<=len(raw)+max(64,extra):
                raise ValueError('Native zlib compression bound exceeds its workspace contract')
            self._bounds_cache[len(raw)]=bound
            source=ctypes.create_string_buffer(raw,max(1,len(raw)))
            output=(ctypes.c_ubyte*(bound+guard))(*([167]*(bound+guard)))
            length=ctypes.c_ulong(bound)
            status=self._compressor(output,ctypes.byref(length),source,len(raw),3)
            if status!=0 or not 0<=length.value<=bound:
                raise ValueError('Native zlib failed a bounded compression probe')
            if bytes(output[bound:])!=bytes([167])*guard:
                raise ValueError('Native zlib wrote outside its declared output capacity')
            if zlib.decompress(bytes(output[:length.value]))!=raw:
                raise ValueError('Native zlib probe does not round-trip through Python zlib')
        source=ctypes.create_string_buffer(bytes(range(256)))
        output=(ctypes.c_ubyte*17)(*([167]*17))
        length=ctypes.c_ulong(1)
        status=self._compressor(output,ctypes.byref(length),source,256,3)
        if status!=-5 or length.value>1 or bytes(output[1:])!=bytes([167])*16:
            raise ValueError('Native zlib did not respect its bounded buffer-error contract')
        length=ctypes.c_ulong(1)
        if self._compressor(output,ctypes.byref(length),source,256,99)!=-2:
            raise ValueError('Native zlib did not reject an invalid compression level')

    def supports(self,block_size):
        return 1<=int(block_size)<=_MAX_BLOCK

    def encode_batch(self,mask,scores,z,bounds,block_size):
        block=int(block_size)
        if (not self.supports(block) or not isinstance(mask,np.ndarray)
                or not isinstance(scores,np.ndarray) or mask.ndim!=3 or mask.shape!=scores.shape
                or mask.dtype not in (np.dtype(np.uint8),np.dtype(bool)) or scores.dtype!=np.uint8
                or not 0<=int(z)<mask.shape[0] or not isinstance(bounds,np.ndarray)
                or bounds.dtype!=np.int64 or bounds.ndim!=2 or bounds.shape[1]!=5
                or bounds.shape[0]>COPY_BATCH_MAX_CELLS):
            raise ValueError('Native confidence batch exceeds its source/control bounds')
        cell_bytes=min(block,mask.shape[1])*min(block,mask.shape[2])
        if len(bounds)*cell_bytes>COPY_BATCH_MAX_BYTES:
            raise MemoryError('Native confidence batch exceeds its admitted raw-byte limit')
        bound=self._bounds_cache.get(cell_bytes)
        if bound is None:
            bound=int(self._bound(cell_bytes))
            extra=(cell_bytes>>12)+(cell_bytes>>14)+(cell_bytes>>25)+13
            if not 1<=bound<=cell_bytes+max(64,extra):
                raise RuntimeError('Native zlib bound exceeds its admitted codec workspace')
            self._bounds_cache[cell_bytes]=bound
        output_bytes=max(1,len(bounds))*bound
        output,offsets,count,status=self._kernel(mask,scores,int(z),bounds,cell_bytes,output_bytes,block)
        if status or count!=len(bounds):
            raise RuntimeError(f'Native confidence compression failed: status={status}, cells={count}/{len(bounds)}')
        return output,offsets

    def warm(self):
        mask=np.asarray([[[1,1,0],[1,0,1]]],np.uint8)
        scores=np.asarray([[[17,29,41],[53,67,79]]],np.uint8)
        mask.setflags(write=False)
        scores.setflags(write=False)
        bounds=np.asarray([[0,2,0,3,4]],np.int64)
        output,offsets=self.encode_batch(mask,scores,0,bounds,128)
        if zlib.decompress(bytes(output[:int(offsets[-1])]))!=bytes([17,29,0,53,0,79]):
            raise ValueError('Native nogil compression kernel failed its qualification probe')


def _load_backend(key):
    mode,qualification,path=key
    base=dict(backend='python_zlib',codec_version=zlib.ZLIB_RUNTIME_VERSION,
              python_zlib_version=zlib.ZLIB_RUNTIME_VERSION)
    if mode in ('0','false','off','python'):
        return None,{**base,'fallback_reason':'disabled_by_configuration'}
    candidates=[]
    if qualification=='1' and path:
        candidates.append((path,'explicit_qualification'))
    else:
        extension=getattr(zlib,'__file__',None)
        if extension:candidates.append((extension,'python_zlib_extension'))
        if os.name!='nt':
            shared=ctypes.util.find_library('z')
            if shared and shared!=extension:candidates.append((shared,'system_zlib'))
    errors=[]
    for filename,source in candidates:
        try:
            codec=NativeConfidenceCodec(ctypes.CDLL(filename),source=source)
            codec.warm()
            return codec,dict(backend='native_compress2',codec_version=codec.version,
                python_zlib_version=zlib.ZLIB_RUNTIME_VERSION,library_source=source,fallback_reason=None)
        except Exception as exc:
            errors.append(f'{source}:{type(exc).__name__}:{str(exc)[:160]}')
    reason=';'.join(errors) if errors else 'native_zlib_library_not_available'
    return None,{**base,'fallback_reason':reason}


def native_confidence_codec():
    """Return a probed backend and its explicit backend/version/fallback receipt."""
    key=(os.environ.get('YOLO_TTA_CONFIDENCE_NATIVE_CODEC','auto').strip().lower(),
         os.environ.get('XTA_QUALIFY_NATIVE_ZLIB',''),
         os.environ.get('XTA_TEST_NATIVE_ZLIB_LIBRARY',''))
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key]=_load_backend(key)
            _,receipt=_CACHE[key]
            print('Confidence cell codec: '+', '.join(f'{name}={value}' for name,value in receipt.items()),flush=True)
        codec,receipt=_CACHE[key]
    return codec,dict(receipt)

"""Exact, bounded CUDA execution of prepared destination-owned native pulls.

The caller owns GPU scheduling and must hold its stage/residency lease for the
entire session. Angular ownership stays in the NumPy-prepared plan; CUDA never
recomputes trigonometry. A session snapshots its immutable input in bounded
uploads, optionally packing categorical nonzero values to one bit each.
"""
from __future__ import annotations

import math
import operator
import threading

import numpy as np

from .native_encoded_cuda import (NativeEncodedStaging, NativePullEncodedBlock,
                                  estimate_native_pull_encoded_output)


class NativePullCudaUnavailable(MemoryError):
    """The exact CUDA implementation cannot fit the declared device credit."""
    def __init__(self,message,*,reason_code='unavailable',details=None):
        super().__init__(message)
        self.reason_code=str(reason_code)
        self.details=dict(details or {})


class NativePullCudaUnsafeFailure(BaseException):
    """Device work failed; retain this session and its GPU lease until settled."""
    def __init__(self, message, session):
        super().__init__(message)
        self.session = session


_KERNEL = r'''
__device__ __forceinline__ int source_pixel(const unsigned char* source,
 const long long* records, long long f, long long row, long long column,
 long long sh, long long sw, int packed, unsigned long long* reads) {
#if ENCODED_SOURCE
  const long long* record=records+6*f;
  long long y0=record[2], x0=record[3], y1=record[4], x1=record[5];
  if (!record[0] || row<y0 || row>=y1 || column<x0 || column>=x1) return 0;
  long long width=x1-x0, local_column=column-x0;
  long long stride=packed ? (width+7)/8 : width;
  long long address=record[1]+(row-y0)*stride+(packed ? local_column/8 : local_column);
  int pixel=source[address];
  ++*reads;
  return packed ? ((pixel>>(local_column&7))&1) : pixel;
#else
  long long address=(f*sh+row)*sw+column;
  return packed ? ((source[address>>3]>>(address&7))&1) : source[address];
#endif
}
extern "C" __global__ void native_pull(
 const unsigned char* source, unsigned char* output, long long first, long long count,
 const long long* integer, const double* c, const long long* bbox,
 const KEY_TYPE* keys, int keys_are_plane, unsigned long long invalid,
 const int* rows, long long sf, long long sh, long long sw,
 int packed, int scalar_max, const long long* records, unsigned long long* totals) {
  long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  unsigned long long visited = 0, addresses = 0, input_reads = 0;
  int value = 0;
  if (i < count) {
    long long wt=integer[2], wh=integer[3], ww=integer[4];
    long long ot=integer[5], oh=integer[6], ow=integer[7];
    long long nh=integer[8], nw=integer[9];
    int base=(int)integer[0], vertical=(int)integer[1];
    long long at=first+i, z=at/(oh*ow), y=(at/ow)%oh, x=at%ow;
    if (bbox[0]<=z && z<bbox[3] && bbox[1]<=y && y<bbox[4] && bbox[2]<=x && x<bbox[5]) {
      visited = 1;
      if (!integer[10]) {
        long long z0=ot<wt ? z*wt/ot : 0, z1=ot<wt ? ((z+1)*wt+ot-1)/ot : 1;
        long long y0=oh<wh ? y*wh/oh : 0, y1=oh<wh ? ((y+1)*wh+oh-1)/oh : 1;
        long long x0=ow<ww ? x*ww/ow : 0, x1=ow<ww ? ((x+1)*ww+ow-1)/ow : 1;
        for (long long iz=z0; iz<z1; ++iz) {
          double cz=ot<wt ? (double)iz : ((double)z+.5)*c[9]-.5;
          for (long long iy=y0; iy<y1; ++iy) {
            double cy=oh<wh ? (double)iy : ((double)y+.5)*c[10]-.5;
            for (long long ix=x0; ix<x1; ++ix) {
              double cx=ow<ww ? (double)ix : ((double)x+.5)*c[11]-.5;
              double nv=base==0 ? cy : cz, nu=base!=2 ? cx : cy;
              double stack=base==0 ? cz : (base==1 ? cy : cx);
              double axis=vertical ? nv : nu;
              double frame=stack-c[0]*(axis-c[2])-c[1];
              if (frame<-.5 || frame>=(double)sf-.5 || nv<-.5 || nv>=(double)nh-.5 || nu<-.5 || nu>=(double)nw-.5) continue;
              double row=c[6]*nu+c[7]*nv+c[8], column=c[3]*nu+c[4]*nv+c[5];
              long long f=(long long)floor(frame+.5), r=(long long)floor(row+.5), col=(long long)floor(column+.5);
              r=r<0 ? 0 : (r>=sh ? sh-1 : r); col=col<0 ? 0 : (col>=sw ? sw-1 : col);
              int pixel=source_pixel(source,records,f,r,col,sh,sw,packed,&input_reads);
              ++addresses;
              if (scalar_max) value=value>pixel ? value : pixel; else if (pixel) value=1;
            }
          }
        }
      } else {
        long long plane=base==0 ? y*ow+x : (base==1 ? z*ow+x : z*oh+y);
        unsigned long long key=(unsigned long long)keys[keys_are_plane ? plane : i];
        if (key!=invalid) {
          double cz=((double)z+.5)*c[9]-.5, cy=((double)y+.5)*c[10]-.5, cx=((double)x+.5)*c[11]-.5;
          double nv=base==0 ? cy : cz, nu=base!=2 ? cx : cy;
          double stack=base==0 ? cz : (base==1 ? cy : cx), axis=vertical ? nv : nu;
          double center=stack-c[0]*(axis-c[2]), local_frame=center-c[1];
          long long stack_size=base==0 ? wt : (base==1 ? wh : ww);
          long long output_stack=base==0 ? ot : (base==1 ? oh : ow);
          long long dest=base==0 ? z : (base==1 ? y : x), lower=0, upper=0;
          if (nh>output_stack) {
            double shift=center-stack-c[1];
            lower=(long long)floor((double)(dest*nh)/(double)output_stack+shift*(double)nh/(double)stack_size);
            upper=(long long)ceil((double)((dest+1)*nh)/(double)output_stack+shift*(double)nh/(double)stack_size);
            lower=lower<0 ? 0 : lower; upper=upper>nh ? nh : upper;
          } else if (local_frame>=-.5 && local_frame<(double)stack_size-.5) {
            double row=(local_frame+.5)*(double)nh/(double)stack_size-.5;
            lower=(long long)nearbyint(row); lower=lower<0 ? 0 : (lower>=nh ? nh-1 : lower); upper=lower+1;
          }
          long long angle=(long long)(key/(unsigned long long)sw), col=(long long)(key%(unsigned long long)sw);
          for (long long r=lower; r<upper; ++r) {
            int pixel=source_pixel(source,records,angle,(long long)rows[r],col,sh,sw,packed,&input_reads);
            ++addresses;
            if (scalar_max) value=value>pixel ? value : pixel; else if (pixel) value=1;
          }
        }
      }
    }
    output[i]=(unsigned char)value;
  }
  __shared__ unsigned long long visits[256], reads[256];
  visits[threadIdx.x]=visited; reads[threadIdx.x]=addresses;
#if ENCODED_SOURCE
  __shared__ unsigned long long byte_reads[256];
  byte_reads[threadIdx.x]=input_reads;
#endif
  __syncthreads();
  for (int step=128; step; step>>=1) {
    if (threadIdx.x<step) {
      visits[threadIdx.x]+=visits[threadIdx.x+step]; reads[threadIdx.x]+=reads[threadIdx.x+step];
#if ENCODED_SOURCE
      byte_reads[threadIdx.x]+=byte_reads[threadIdx.x+step];
#endif
    }
    __syncthreads();
  }
  if (!threadIdx.x) {
    atomicAdd(totals,visits[0]); atomicAdd(totals+1,reads[0]);
#if ENCODED_SOURCE
    atomicAdd(totals+2,byte_reads[0]);
#endif
  }
}
'''


def estimate_native_pull_cuda_bytes(source_shape, plan, *, max_output_voxels, packed_input,
                                    encoded_nbytes=None, encoded_output_block_depth=0):
    """Numerical residency only; caller reserves CUDA context/compiler overhead."""
    source_bytes = (int(encoded_nbytes) if encoded_nbytes is not None else
                    (math.prod(source_shape) + 7) // 8 if packed_input else math.prod(source_shape))
    cached = plan.backend == 'compiled_azimuthal_cached'
    angular = bool(plan.integers[10])
    max_output_voxels = min(int(max_output_voxels), int(plan.max_strip_voxels))
    encoded = estimate_native_pull_encoded_output(plan, block_depth=encoded_output_block_depth,
        encoded_input=encoded_nbytes is not None)
    key_bytes = plan.keys.nbytes if cached else (max_output_voxels * plan.keys.dtype.itemsize if angular else 0)
    # CuPy's private pool rounds each nonempty allocation to 512-byte units.
    # Charge that physical ownership, including the small control allocations.
    buffers=(source_bytes,key_bytes,max(max_output_voxels, encoded['dense_output_bytes']),plan.integers.nbytes,
             plan.coefficients.nbytes,plan.row_lookup.nbytes,6*8,3*8,
             int(source_shape[0])*6*8 if encoded_nbytes is not None else 0)
    return sum((int(size)+511)//512*512 for size in buffers) + encoded['device_encoder_bytes']


def _encoded_records(encoded,index,source_shape,packed):
    """Validate offsets and make only a small numeric frame index, never pixels."""
    required=('kind','offset','payload_size','y0','x0','y1','x1','payload_nbytes')
    if (not isinstance(index,np.ndarray) or index.shape!=(source_shape[0],)
            or index.dtype.names is None or any(name not in index.dtype.names for name in required)):
        raise ValueError('Encoded CUDA input requires a bound crop index')
    for record in index:
        kind=int(record['kind'])
        if not kind:
            continue
        if kind!=1:
            raise ValueError('Encoded CUDA crop kind must be zero or one')
        y0,x0,y1,x1=(int(record[name]) for name in ('y0','x0','y1','x1'))
        if not (0<=y0<y1<=source_shape[1] and 0<=x0<x1<=source_shape[2]):
            raise ValueError('Encoded CUDA crop exceeds source geometry')
        height,width=y1-y0,x1-x0
        stride=(width+7)//8 if packed else width
        size,offset=int(record['payload_size']),int(record['offset'])
        if (int(record['payload_nbytes'])!=height*width or size!=height*stride
                or not 0<=offset<=offset+size<=encoded.size):
            raise ValueError('Encoded CUDA crop payload exceeds its bound byte array')
    records=np.zeros((source_shape[0],6),np.int64)
    for field,name in enumerate(('kind','offset','y0','x0','y1','x1')):
        records[:,field]=index[name]
    # Empty records never dereference their ignored offsets/bounds. Normalize
    # those fields before unsigned-to-signed conversion can affect a pointer.
    for first in range(0, records.shape[0], 4096):
        chunk = records[first:first + 4096]
        chunk[chunk[:, 0] == 0] = 0
    return records


class NativePullCudaSession:
    """Exclusive CUDA scratch and source snapshot; caller supplies GPU leases.

    ``pull_flat_into`` is serialized so multiple CPU plane workers can share a
    session safely. The caller must close it before releasing GPU residency.
    No device failure is converted to partially valid output or CPU fallback.
    """
    def __init__(self, source, plan, *, device_index, max_device_bytes,
                 max_output_voxels, scalar_max=False, packed_input=None,
                 max_upload_bytes=8 * 1024**2, _encoded_index=None,
                 encoded_output_block_depth=0, encoded_output_slots=2, max_host_bytes=None):
        self.encoded_input=_encoded_index is not None
        if (not isinstance(source, np.ndarray) or source.dtype != np.uint8
                or (self.encoded_input and (source.ndim!=1 or not source.flags.c_contiguous))
                or (not self.encoded_input and source.shape!=plan.source_shape)):
            raise ValueError('CUDA native pull requires the bound uint8 source geometry')
        if math.prod(plan.source_shape) > np.iinfo(np.int64).max or source.size>np.iinfo(np.int64).max:
            raise ValueError('CUDA native pull source exceeds int64 addressing')
        self.scalar_max = bool(scalar_max)
        if self.encoded_input and self.scalar_max:
            raise NativePullCudaUnavailable('Encoded CUDA native pull supports binary OR only',
                                            reason_code='encoded_numeric_max')
        self.packed_input = not self.scalar_max if packed_input is None else bool(packed_input)
        if self.scalar_max and self.packed_input:
            raise ValueError('Numeric max cannot consume packed categorical input')
        if encoded_output_block_depth and self.scalar_max:
            raise NativePullCudaUnavailable('Packed native output requires binary OR',
                                             reason_code='encoded_output_numeric_max')
        self.max_output_voxels = min(int(max_output_voxels), int(plan.max_strip_voxels))
        upload_bytes = int(max_upload_bytes) // 8 * 8
        if self.max_output_voxels <= 0 or upload_bytes < 8:
            raise ValueError('CUDA native pull requires positive bounded upload/output credit')
        self._encoded_spec = estimate_native_pull_encoded_output(plan,
            block_depth=encoded_output_block_depth, slots=encoded_output_slots,
            max_upload_bytes=upload_bytes, encoded_input=self.encoded_input)
        self.max_encoded_block_depth = self._encoded_spec['block_depth']
        self.max_encoded_payload_bytes = self._encoded_spec['max_payload_bytes']
        self.encoded_host_bytes = self._encoded_spec['host_bytes']
        if self.max_encoded_block_depth and max_host_bytes is None:
            raise NativePullCudaUnavailable('Packed native output requires explicit host credit',
                                             reason_code='host_credit')
        if max_host_bytes is not None and self.encoded_host_bytes > int(max_host_bytes):
            raise NativePullCudaUnavailable('Native staging exceeds its declared host credit',
                reason_code='host_credit', details=dict(required_host_bytes=self.encoded_host_bytes,
                                                        allowed_host_bytes=int(max_host_bytes)))
        self.device_bytes = estimate_native_pull_cuda_bytes(plan.source_shape, plan,
            max_output_voxels=self.max_output_voxels, packed_input=self.packed_input,
            encoded_nbytes=source.size if self.encoded_input else None,
            encoded_output_block_depth=self.max_encoded_block_depth)
        if self.device_bytes > int(max_device_bytes):
            raise NativePullCudaUnavailable('CUDA native source and bounded scratch exceed device credit',
                reason_code='device_credit',details=dict(required_device_bytes=self.device_bytes,
                    allowed_device_bytes=int(max_device_bytes),encoded_input=self.encoded_input,
                    source_payload_bytes=source.nbytes))
        records_host=(_encoded_records(source,_encoded_index,plan.source_shape,self.packed_input)
                      if self.encoded_input else np.empty(0,np.int64))
        try:
            import cupy as cp
        except Exception as exc:
            raise NativePullCudaUnavailable(f'CUDA native pull runtime unavailable: {exc}',
                reason_code='runtime_import',details=dict(error_type=type(exc).__name__)) from exc
        self.cp, self.device_index, self.plan = cp, int(device_index), plan
        self._source_owner = source
        self._index_owner = _encoded_index
        self._lock, self._closed = threading.Lock(), False
        self._pool = cp.cuda.MemoryPool()
        self._arrays = []
        self._pending_host = []
        self._staging = NativeEncodedStaging()
        self.stream = None
        try:
            with cp.cuda.Device(self.device_index), cp.cuda.using_allocator(self._pool.malloc):
                self.stream = cp.cuda.Stream(non_blocking=True)
                with self.stream:
                    self._staging.initialize(cp, self.stream, self._encoded_spec,
                        plan.output_shape, self._arrays)
                    def allocate(shape, dtype):
                        array = cp.empty(shape, dtype)
                        self._arrays.append(array)
                        return array
                    def upload(host):
                        array = allocate(host.shape, host.dtype)
                        self._staging.upload(array, host)
                        return array
                    self._source = allocate((source.size+7)//8 if self.packed_input and not self.encoded_input else source.size, cp.uint8)
                    flat_source = source.reshape(-1) if source.flags.c_contiguous else source.flat
                    for first in range(0, source.size, upload_bytes):
                        host = flat_source[first:min(source.size, first+upload_bytes)]
                        if self.packed_input and not self.encoded_input:
                            host = np.packbits(host != 0, bitorder='little')
                            self._staging.upload(self._source[first//8:first//8+host.size], host)
                        else:
                            self._staging.upload(self._source[first:first+host.size], host)
                    self._integer = upload(plan.integers)
                    self._coefficient = upload(plan.coefficients)
                    self._rows = upload(plan.row_lookup)
                    self._bbox = allocate(6, cp.int64)
                    self._output = allocate(max(self.max_output_voxels,
                        self._encoded_spec['dense_output_bytes']), cp.uint8)
                    self._totals = allocate(3, cp.uint64)
                    self._records = upload(records_host)
                    self._keys_are_plane = plan.backend == 'compiled_azimuthal_cached'
                    key_count = (plan.keys.size if self._keys_are_plane else
                                 self.max_output_voxels if plan.integers[10] else 0)
                    self._keys = allocate(key_count, plan.keys.dtype)
                    if self._keys_are_plane:
                        self._staging.upload(self._keys, plan.keys.reshape(-1))
                    key_type = 'unsigned int' if plan.keys.dtype == np.uint32 else 'unsigned long long'
                    kernel_source=('#define ENCODED_SOURCE '+str(int(self.encoded_input))+'\n'
                                   +_KERNEL.replace('KEY_TYPE',key_type))
                    self._kernel = cp.RawKernel(kernel_source, 'native_pull',
                                                options=('--std=c++11', '--fmad=false'))
                    self._kernel.compile()
                self.stream.synchronize()
        except BaseException as exc:
            # The constructor has never published output. It may decline only
            # after all submitted uploads and private allocations are settled.
            self.close()
            if isinstance(exc, Exception):
                raise NativePullCudaUnavailable(f'CUDA native initialization declined: {exc}',
                    reason_code='initialization',details=dict(error_type=type(exc).__name__,
                        required_device_bytes=self.device_bytes)) from exc
            raise

    @classmethod
    def from_encoded(cls, encoded, index, packed_input, plan, **kwargs):
        """Borrow validated raw/row-packed crop bytes without native expansion.

        The caller must retain the payload mapping and index until close. Input
        offsets, crop bounds and exact byte lengths are revalidated before any
        upload. Numeric MAX is deliberately unavailable for this binary route.
        """
        return cls(encoded,plan,packed_input=bool(packed_input),_encoded_index=index,**kwargs)

    def pull_flat_into(self, output_flat, *, first_flat, destination_bbox_tyx=None):
        if (not isinstance(output_flat, np.ndarray) or output_flat.dtype != np.uint8
                or output_flat.ndim != 1 or not output_flat.flags.writeable or not output_flat.flags.c_contiguous):
            raise ValueError('CUDA native output must be a writable contiguous uint8 flat span')
        first, count = int(first_flat), output_flat.size
        if not 0 <= first <= first+count <= math.prod(self.plan.output_shape):
            raise ValueError('CUDA native output span exceeds its bound geometry')
        if count > self.max_output_voxels:
            raise NativePullCudaUnavailable('CUDA native output span exceeds admitted scratch',
                reason_code='output_credit',details=dict(required_voxels=count,
                                                       allowed_voxels=self.max_output_voxels))
        if (np.may_share_memory(output_flat, self._source_owner)
                or (self._index_owner is not None and np.may_share_memory(output_flat,self._index_owner))):
            raise ValueError('CUDA native output cannot alias its borrowed source')
        bbox = ((0, 0, 0, *self.plan.output_shape) if destination_bbox_tyx is None
                else tuple(map(int, destination_bbox_tyx)))
        if len(bbox) != 6 or any(not 0 <= bbox[a] <= bbox[a+3] <= self.plan.output_shape[a] for a in range(3)):
            raise ValueError('CUDA native bbox exceeds output geometry')
        with self._lock:
            if self._closed:
                raise RuntimeError('CUDA native pull session has been closed')
            cp = self.cp
            bbox_host = np.asarray(bbox, np.int64)
            self._pending_host = [bbox_host]
            try:
                with cp.cuda.Device(self.device_index), cp.cuda.using_allocator(self._pool.malloc), self.stream:
                    self._staging.upload(self._bbox, bbox_host)
                    if self.plan.integers[10] and not self._keys_are_plane and count:
                        from .projection_coverage_cpu import _angular_keys
                        indices = np.arange(first, first+count, dtype=np.int64)
                        oh, ow = self.plan.output_shape[1:]
                        base = int(self.plan.integers[0])
                        vertical = ((indices//ow)%oh) if base == 0 else indices//(oh*ow)
                        horizontal = (indices%ow) if base != 2 else (indices//ow)%oh
                        keys = _angular_keys(self.plan, vertical, horizontal)
                        self._staging.upload(self._keys[:count], keys)
                    self._totals.fill(0)
                    if count:
                        self._kernel(((count+255)//256,), (256,),
                            (self._source, self._output, np.int64(first), np.int64(count),
                             self._integer, self._coefficient, self._bbox, self._keys,
                             np.int32(self._keys_are_plane), self.plan.invalid_key, self._rows,
                             *map(np.int64, self.plan.source_shape), np.int32(self.packed_input),
                             np.int32(self.scalar_max), self._records,self._totals), stream=self.stream)
                        self._staging.download_dense(self._output[:count], output_flat)
                    totals = self._staging.download_totals(self._totals)
                    self._pending_host.clear()
            except BaseException as exc:
                raise NativePullCudaUnsafeFailure('CUDA native pull failed during device execution', self) from exc
            return dict(backend=('cuda_native_encoded_packed' if self.packed_input else 'cuda_native_encoded_raw')
                                if self.encoded_input else 'cuda_native_packed' if self.packed_input else 'cuda_native_uint8',
                        kernel_calls=int(bool(count)), destination_voxels_visited=int(totals[0]),
                        contribution_addresses=int(totals[1]), device_bytes=self.device_bytes,
                        indexed_input_byte_reads=int(totals[2] if self.encoded_input else totals[1]),
                         dense_d2h_bytes=count,
                        persistent_bytes=self.plan.persistent_bytes,
                        temporary_strip_bytes=self.plan.temporary_strip_bytes)

    def project_encoded(self, first_z, count, *, destination_bbox_tyx=None):
        """Project and row-pack a bounded block before any mask D2H transfer.

        Returned bytes are independently owned and immutable. The caller funds
        and enforces ``encoded_output_slots`` outstanding copied payloads.
        Uncached angular plans prepare only their admitted CPU key strips.
        """
        first_z, count = operator.index(first_z), operator.index(count)
        if not self.max_encoded_block_depth:
            raise NativePullCudaUnavailable('Packed output was not admitted for this session',
                                             reason_code='encoded_output_disabled')
        if not 0 <= count <= self.max_encoded_block_depth or not 0 <= first_z <= first_z + count <= self.plan.output_shape[0]:
            raise ValueError('Native encoded block exceeds its admitted destination depth')
        bbox = ((0, 0, 0, *self.plan.output_shape) if destination_bbox_tyx is None
                else tuple(map(int, destination_bbox_tyx)))
        if len(bbox) != 6 or any(not 0 <= bbox[a] <= bbox[a+3] <= self.plan.output_shape[a] for a in range(3)):
            raise ValueError('CUDA native bbox exceeds output geometry')
        with self._lock:
            if self._closed:
                raise RuntimeError('CUDA native pull session has been closed')
            cp = self.cp
            voxels = count * math.prod(self.plan.output_shape[1:])
            first = first_z * math.prod(self.plan.output_shape[1:])
            calls = 0
            try:
                with cp.cuda.Device(self.device_index), cp.cuda.using_allocator(self._pool.malloc), self.stream:
                    self._staging.upload(self._bbox, np.asarray(bbox, np.int64))
                    self._totals.fill(0)
                    for offset in range(0, voxels, self.max_output_voxels):
                        length = min(self.max_output_voxels, voxels - offset)
                        at = first + offset
                        if self.plan.integers[10] and not self._keys_are_plane:
                            from .projection_coverage_cpu import _angular_keys
                            indices = np.arange(at, at + length, dtype=np.int64)
                            oh, ow = self.plan.output_shape[1:]
                            base = int(self.plan.integers[0])
                            vertical = ((indices // ow) % oh) if base == 0 else indices // (oh * ow)
                            horizontal = indices % ow if base != 2 else (indices // ow) % oh
                            keys = _angular_keys(self.plan, vertical, horizontal)
                            self._staging.upload(self._keys[:length], keys)
                        self._kernel(((length + 255) // 256,), (256,),
                            (self._source, self._output[offset:offset + length], np.int64(at), np.int64(length),
                             self._integer, self._coefficient, self._bbox, self._keys,
                             np.int32(self._keys_are_plane), self.plan.invalid_key, self._rows,
                             *map(np.int64, self.plan.source_shape), np.int32(self.packed_input),
                             np.int32(self.scalar_max), self._records, self._totals), stream=self.stream)
                        calls += 1
                    if count:
                        block, metadata_bytes = self._staging.encode(self._output, first_z, count)
                    else:
                        from .cylindrical_cuda_projection import RadialEncodedBlock
                        payload = np.empty(0, np.uint8)
                        payload.flags.writeable = False
                        block, metadata_bytes = RadialEncodedBlock(first_z, (), payload, True), 0
                    totals = self._staging.download_totals(self._totals)
            except BaseException as exc:
                raise NativePullCudaUnsafeFailure('CUDA native packed output failed during device execution', self) from exc
            stats = dict(backend='cuda_native_device_rowpacked', kernel_calls=calls,
                destination_voxels_visited=totals[0], contribution_addresses=totals[1],
                indexed_input_byte_reads=totals[2] if self.encoded_input else totals[1],
                device_bytes=self.device_bytes, encoded_host_bytes=self.encoded_host_bytes,
                dense_d2h_bytes=0, payload_d2h_bytes=block.payload.nbytes,
                metadata_d2h_bytes=metadata_bytes + 24, encoded_blocks=int(bool(count)),
                encoded_planes=count, projected_foreground=sum(record.foreground for record in block.records),
                persistent_bytes=self.plan.persistent_bytes,
                temporary_strip_bytes=self.plan.temporary_strip_bytes)
            return NativePullEncodedBlock(block.first_z, block.records, block.payload, True, stats)

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                with self.cp.cuda.Device(self.device_index):
                    if self.stream is not None:
                        self.stream.synchronize()
                    # An exception traceback may retain a failed kernel's
                    # argument tuple. After fencing, explicitly retire these
                    # exclusively owned allocations so those stale references
                    # cannot keep residency alive after the lease is released.
                    for array in self._arrays:
                        if array.nbytes:
                            array.data.mem.free()
                    self._arrays.clear()
                    self._pending_host.clear()
                    for name in ('_source', '_integer', '_coefficient', '_rows', '_bbox', '_output', '_totals', '_keys','_records'):
                        setattr(self, name, None)
                    self._pool.free_all_blocks()
                    self._staging.close()
            except BaseException as exc:
                # Retain ownership on failed synchronization. The scheduler
                # must quarantine this residency instead of releasing its GPU.
                raise NativePullCudaUnsafeFailure('CUDA native pull cleanup could not settle residency', self) from exc
            self._closed = True
            self._source_owner = None
            self._index_owner = None

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()

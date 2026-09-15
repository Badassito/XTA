"""CPU-only contract, state and lifetime checks; these do not qualify CUDA kernels."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import tilted_azimuthal_projection_cuda as backend


def fixture(base=0):
    _, points, first_stack, expected = backend._preflight_case(base)
    source = np.ones((3,5,7), np.uint8)
    source[2,:,6] = 0
    arrays = dict(points=points, stack_map=np.repeat(first_stack,2,axis=0),
        row_offsets=np.asarray([0,3,4],np.int64), rows=np.asarray([0,3,4,1],np.int32))
    for array in arrays.values():
        array.flags.writeable = False
    plan = SimpleNamespace(source_shape=source.shape, output_shape=expected.shape,
                          base_id=base, frame_count=2, **arrays)
    return source, plan, expected


def contract(source, plan):
    return backend._validate_tilted_azimuthal_contract(source,plan,170,42)


@pytest.mark.parametrize('base', [0,1,2])
def test_contract_preserves_exact_row_packing_and_bounded_source_upload(base):
    source,plan,_ = fixture(base)
    value = contract(source,plan)
    assert value.band_rows == 2 and value.source_band_bytes == 42 < source.nbytes
    assert value.packed_bytes == 45 and value.packed_words == 12
    assert value.max_block_depth == 2
    assert list(backend._frame_row_bands(value,0)) == [(0,2,0,1),(2,2,1,2),(4,1,2,3)]
    assert list(backend._frame_row_bands(value,1)) == [(0,2,3,4)]


def test_backend_accepts_authoritative_readonly_cpu_plans_for_every_base_axis():
    from XTA.config import AzimuthalViewRequest,TiltedViewGroup
    from XTA.geometry import view_processing_volume_shape
    from XTA.tilted_azimuthal_projection import build_tilted_azimuthal_plan
    from XTA.unification.runtime import compile_physical_views
    for base in ('transverse','sagittal','coronal'):
        views=compile_physical_views(t_dim=7,height=9,width=11,cartesian_views=(),
            azimuthal_requests=(AzimuthalViewRequest('tilted_'+base,30.),),
            tilted_groups=(TiltedViewGroup((base,),(30.,),('vertical','horizontal')),),
            azimuthal_native_raster=16).views
        for view in views:
            if view.family!='azimuthal':continue
            source=np.zeros(view_processing_volume_shape(view,16),np.uint8)
            plan=build_tilted_azimuthal_plan(source,view,(7,9,11))
            value=backend._validate_tilted_azimuthal_contract(source,plan,1024,1024)
            assert value.frame_count==plan.frame_count
            assert value.base_id==('transverse','sagittal','coronal').index(base)


@pytest.mark.parametrize('fault', ['source_dtype','source_stride','source_shape','output_shape',
    'base','point_dtype','point_stride','point_mutable','point_width','azimuth','u','axis',
    'fixed_a','fixed_b','stack_low','stack_high','frames','csr_start','csr_end','csr_order','row_index'])
def test_invalid_contract_is_rejected_before_cuda_initialization(fault):
    source,plan,_ = fixture()
    if fault == 'source_dtype': source=source.astype(np.float32)
    elif fault == 'source_stride': source=source[:,:,::-1]
    elif fault == 'source_shape': plan.source_shape=(3,4,7)
    elif fault == 'output_shape': plan.output_shape=(3,0,17)
    elif fault == 'base': plan.base_id=3
    elif fault == 'point_dtype': plan.points=plan.points.astype(np.int64)
    elif fault == 'point_stride': plan.points=plan.points[::-1]
    elif fault == 'point_mutable': plan.points=plan.points.copy()
    elif fault == 'point_width': plan.points=plan.points[:,:4].copy()
    elif fault in ('azimuth','u','axis','fixed_a','fixed_b'):
        points=plan.points.copy()
        column=('azimuth','u','axis','fixed_a','fixed_b').index(fault)
        points[0,column]=999
        points.flags.writeable=False
        plan.points=points
    elif fault.startswith('stack_'):
        stack=plan.stack_map.copy();stack[0,0]=-2 if fault=='stack_low' else 3
        stack.flags.writeable=False;plan.stack_map=stack
    elif fault=='frames': plan.frame_count=3
    elif fault.startswith('csr_'):
        values={'csr_start':[1,3,4],'csr_end':[0,3,5],'csr_order':[0,5,4]}[fault]
        offsets=np.asarray(values,np.int64);offsets.flags.writeable=False;plan.row_offsets=offsets
    else:
        rows=np.asarray([0,3,5,1],np.int32);rows.flags.writeable=False;plan.rows=rows
    with mock.patch.object(backend.TiltedAzimuthalCudaProjector,'_initialize_cuda',side_effect=AssertionError('CUDA initialized')):
        with pytest.raises(ValueError):
            backend.TiltedAzimuthalCudaProjector(source,plan,block_bytes=170,upload_bytes=42)


def test_unsupported_budgets_decline_before_any_device_allocation():
    source,plan,_=fixture()
    with pytest.raises(backend.TiltedAzimuthalCudaProjectionUnavailable,match='processing row'):
        backend._validate_tilted_azimuthal_contract(source,plan,170,20)
    with pytest.raises(backend.TiltedAzimuthalCudaProjectionUnavailable,match='output plane'):
        backend._validate_tilted_azimuthal_contract(source,plan,84,42)


def test_cpu_prefix_requires_exact_bytes_and_a_valid_frame_watermark():
    source,plan,expected=fixture()
    value=contract(source,plan)
    prefix=np.packbits(expected,axis=2,bitorder='big')
    assert backend._validate_initial_packed(prefix,1,value) is prefix
    for initial,first in ((None,1),(prefix,-1),(prefix,3),(prefix.astype(np.int32),1),
                          (prefix[:,:,:2],1),(prefix[:,:,::-1],1)):
        with pytest.raises(ValueError):backend._validate_initial_packed(initial,first,value)


@pytest.mark.parametrize('target',['source','prefix','points','stack_map','row_offsets','rows'])
def test_array_like_inputs_are_rejected_without_materializing_them(target):
    source,plan,_=fixture()
    class ForbiddenArrayConversion:
        def __array__(self,*args,**kwargs):
            raise AssertionError('Implicit array conversion could materialize a whole source volume')
    value=ForbiddenArrayConversion()
    keywords={}
    if target=='source':
        source=value
    elif target=='prefix':
        keywords.update(initial_packed=value,first_frame=1)
    else:
        setattr(plan,target,value)
    with mock.patch.object(backend.TiltedAzimuthalCudaProjector,'_initialize_cuda',
                           side_effect=AssertionError('CUDA initialized before validation')):
        with pytest.raises(ValueError,match='NumPy array'):
            backend.TiltedAzimuthalCudaProjector(source,plan,block_bytes=170,upload_bytes=42,**keywords)


def test_word_masks_match_cpu_big_endian_bytes_across_row_and_word_boundaries():
    for base in range(3):
        _,_,_,expected=backend._preflight_case(base)
        packed=np.packbits(expected,axis=2,bitorder='big')
        words=np.zeros((packed.size+3)//4,np.uint32)
        for t,y,x in np.argwhere(expected):
            byte=(int(t)*expected.shape[1]+int(y))*packed.shape[2]+(int(x)>>3)
            words[byte>>2] |= np.uint32(1 << ((byte&3)*8+7-(int(x)&7)))
        np.testing.assert_array_equal(words.view(np.uint8)[:packed.size],packed.reshape(-1))
        assert not np.any(words.view(np.uint8)[packed.size:])


class CpuStream:
    fail=False
    fences=0
    def __enter__(self):return self
    def __exit__(self,*args):pass
    def synchronize(self):
        self.fences+=1
        if self.fail:raise RuntimeError('injected stream fence failure')


def simulated_projector(source,plan,**kwargs):
    """Exercise production control flow with a NumPy stand-in, never CUDA."""
    def initialize(self,initial):
        self._cp=SimpleNamespace(cuda=SimpleNamespace(Device=lambda _:nullcontext(),
                                using_allocator=lambda _:nullcontext()))
        self._pool=SimpleNamespace(malloc=None,free_all_blocks=mock.Mock())
        self._pinned_pool=SimpleNamespace(free_all_blocks=mock.Mock())
        self._stream=CpuStream()
        self._bits_gpu=np.zeros(self.contract.packed_words,np.uint32)
        if initial is not None:
            self._bits_gpu.view(np.uint8)[:initial.size]=initial.reshape(-1)
        self._arrays={name:array.copy() for name,array in self.contract.arrays.items()}
        self._source_gpu=np.empty(self.contract.source_band_bytes,np.uint8)
        self.uploaded=[]
        def band(first,count):
            self.uploaded.append((first,count))
            block=source[:,first:first+count,:].copy()
            assert block.nbytes <= self.contract.source_band_bytes
            self._source_gpu[:block.size]=block.reshape(-1)
        self._ensure_source_band=band
        def scatter(grid,block,args,stream=None):
            src,points,stack,rows,words,n,begin,end,sw,bfirst,brows,frame,axis,base,oh,ow=args
            for az,u,which,a,b in points[:int(n)]:
                mapped=int(stack[int(frame),int(which)])
                if mapped<0:continue
                if not any(src[(int(az)*int(brows)+int(row)-int(bfirst))*int(sw)+int(u)]
                           for row in rows[int(begin):int(end)]):continue
                t,y,x=((mapped,int(a),int(b)),(int(a),mapped,int(b)),(int(a),int(b),mapped))[int(base)]
                byte=(t*int(oh)+y)*((int(ow)+7)//8)+(x>>3)
                words[byte>>2] |= np.uint32(1 << ((byte&3)*8+7-(x&7)))
        self._scatter_kernel=scatter
        self._record=lambda *args:None
        self._elapsed=lambda *args:0.
        def decode(first,count):
            shape=self.contract.output_shape
            packed=self._bits_gpu.view(np.uint8)[:self.packed_bytes].reshape(*shape[:2],(shape[2]+7)//8)
            return np.unpackbits(packed[first:first+count],axis=2,count=shape[2],bitorder='big').copy()
        self._run_block=decode
    with mock.patch.object(backend.TiltedAzimuthalCudaProjector,'_initialize_cuda',initialize):
        return backend.TiltedAzimuthalCudaProjector(source,plan,block_bytes=170,upload_bytes=42,**kwargs)


@pytest.mark.parametrize('base',[0,1,2])
def test_accumulation_can_span_bands_and_publication_waits_for_every_frame(base):
    source,plan,expected=fixture(base)
    original=source.copy()
    projector=simulated_projector(source,plan)
    with pytest.raises(RuntimeError,match='all input frames'):projector.project(0,1)
    projector.accumulate(0,1)
    assert projector.accumulated_frames==1
    with pytest.raises(ValueError,match='watermark'):projector.accumulate(0,1)
    with pytest.raises(RuntimeError,match='all input frames'):projector.project_encoded(0,1)
    projector.accumulate(1,2)
    actual=np.concatenate((projector.project(0,2),projector.project(2,1)))
    np.testing.assert_array_equal(actual,expected)
    assert projector.uploaded==[(0,2),(2,2),(4,1),(0,2)]
    actual[:]=99
    np.testing.assert_array_equal(projector.project(0,2),expected[:2])
    with pytest.raises(RuntimeError,match='published'):projector.accumulate(2,2)
    projector.close();projector.close()
    np.testing.assert_array_equal(source,original)


def test_cpu_prefix_is_loaded_without_endian_conversion_and_final_padding_stays_zero():
    source,plan,expected=fixture()
    prefix=np.packbits(expected,axis=2,bitorder='big')
    with simulated_projector(source,plan,initial_packed=prefix,first_frame=1) as projector:
        np.testing.assert_array_equal(projector._bits_gpu.view(np.uint8)[:prefix.size],prefix.reshape(-1))
        assert not np.any(projector._bits_gpu.view(np.uint8)[prefix.size:])
        projector.accumulate(1,2)
        np.testing.assert_array_equal(projector.project(0,2),expected[:2])


def test_failed_stream_fence_preserves_source_and_every_owner_until_a_later_safe_close():
    source,plan,_=fixture()
    projector=simulated_projector(source,plan)
    pool,stream,bits=projector._pool,projector._stream,projector._bits_gpu
    stream.fail=True
    with pytest.raises(backend.TiltedAzimuthalCudaProjectionUnsafeFailure) as caught:
        projector.accumulate(0,1)
    assert caught.value.projector is projector and projector._source is source
    assert projector.accumulated_frames==0 and projector._bits_gpu is bits
    with pytest.raises(backend.TiltedAzimuthalCudaProjectionUnsafeFailure):projector.close()
    pool.free_all_blocks.assert_not_called()
    stream.fail=False
    projector.close()
    assert projector._source is None and projector._bits_gpu is None
    pool.free_all_blocks.assert_called_once()


def test_launch_failure_cannot_advance_or_retry_a_partially_written_accumulator():
    source,plan,_=fixture()
    projector=simulated_projector(source,plan)
    projector._scatter_kernel=mock.Mock(side_effect=RuntimeError('launch failure'))
    with pytest.raises(RuntimeError,match='launch failure'):projector.accumulate(0,1)
    assert projector.accumulated_frames==0
    with pytest.raises(RuntimeError,match='failed'):projector.accumulate(0,1)
    projector.close()


@pytest.mark.parametrize('unsafe',[False,True])
def test_startup_failure_releases_only_fenced_resources(unsafe):
    source,plan,_=fixture()
    owners=[]
    def failing_startup(self,initial):
        owners.append(self)
        self._cp=SimpleNamespace(cuda=SimpleNamespace(Device=lambda _:nullcontext()))
        self._stream=CpuStream()
        self._stream.fail=unsafe
        self._pool=SimpleNamespace(free_all_blocks=mock.Mock())
        self._pinned_pool=SimpleNamespace(free_all_blocks=mock.Mock())
        raise RuntimeError('injected startup failure')
    expected=(backend.TiltedAzimuthalCudaProjectionUnsafeFailure if unsafe
              else backend.TiltedAzimuthalCudaProjectionUnavailable)
    with mock.patch.object(backend.TiltedAzimuthalCudaProjector,'_initialize_cuda',failing_startup):
        with pytest.raises(expected) as caught:
            backend.TiltedAzimuthalCudaProjector(source,plan,block_bytes=170,upload_bytes=42)
    owner=owners[0]
    if unsafe:
        assert caught.value.projector is owner and owner._source is source
        owner._pool.free_all_blocks.assert_not_called()
        owner._stream.fail=False
        owner.close()
    assert owner._closed and owner._source is None
    owner._pool.free_all_blocks.assert_called_once()

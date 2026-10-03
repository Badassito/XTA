"""Independent capture-resource, ordering and borrowed-source lifecycle checks."""
from contextlib import contextmanager
import gc
import threading
import tracemalloc

import numpy as np
import pytest

from XTA import confidence_storage as storage
from XTA.confidence_capture import (plan_confidence_capture,confidence_capture_resources,
    current_confidence_capture_plan)
from XTA.confidence_evidence import _MaskedNativeScoreReader


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES','-1')


def arrays(shape=(4,263,279)):
    scores=np.random.default_rng(10798).integers(0,256,shape,dtype=np.uint8)
    mask=np.zeros(shape,np.uint8)
    mask[0,7:251,129:270]=2
    mask[1,130:259,3:141]=255
    mask[2,0:20,0:20]=1
    scores[0,130:141,137:150]=0
    active=np.any(mask,axis=(1,2))
    boxes=np.full((shape[0],4),-1,np.int32)
    for z in np.flatnonzero(active):
        y,x=np.nonzero(mask[z]);boxes[z]=(y.min(),y.max()+1,x.min(),x.max()+1)
    return mask,scores,active,boxes


@pytest.mark.parametrize('block',[8,128])
def test_nonbinary_mask_offset_global_grid_is_byte_exact_and_inputs_are_borrowed(tmp_path,monkeypatch,block):
    mask,scores,active,boxes=arrays()
    original_mask,original_scores=mask.copy(),scores.copy()
    plan=plan_confidence_capture(scores.shape,3,workspace_bytes=64*1024**2,block_size=block)
    options=dict(layer_key='k',model_name='m',provenance={},block_size=block,
        coordinate_space='native_view_processing',source_shape=scores.shape)
    reference=storage.write_blocks(tmp_path/'reference',scores.shape,
        lambda z:np.where(mask[z]!=0,scores[z],np.uint8(0)),**options)
    with confidence_capture_resources(plan):reader=_MaskedNativeScoreReader(mask,scores,active,boxes)
    assert np.shares_memory(reader.mask,mask) and np.shares_memory(reader.scores,scores)
    assert not reader.mask.flags.writeable and not reader.scores.flags.writeable
    monkeypatch.setattr(reader,'iter_crops',lambda *a:pytest.fail('Compiled capture fell back to hull copy'))
    metrics={}
    actual=storage.write_blocks(tmp_path/'actual',scores.shape,reader,metrics=metrics,**options)
    assert actual==reference
    assert actual['known_voxels']==int(np.count_nonzero((mask!=0)&(scores!=0)))
    for name in ('metadata.json','scores.u8.zlib','index.bin'):
        assert (tmp_path/'actual'/name).read_bytes()==(tmp_path/'reference'/name).read_bytes()
    assert metrics['capture_backend']=='compiled_masked_cells_ordered_frames'
    np.testing.assert_array_equal(mask,original_mask)
    np.testing.assert_array_equal(scores,original_scores)


@pytest.mark.parametrize('layout',['ndarray-bool','ndarray-int','list-bool','list-int'])
def test_valid_long_thin_reader_initialization_fits_its_charged_peak(layout):
    # All inputs are allocated outside the trace; no encoding or JIT occurs.
    shape=(60000,1,1)
    mask=np.ones(shape,np.uint8);scores=np.full(shape,173,np.uint8)
    active=np.ones(shape[0],bool)
    boxes=np.tile(np.array([0,1,0,1],np.int32),(shape[0],1))
    if layout=='ndarray-int':
        active=active.astype(np.int64)
    elif layout=='list-bool':
        active,boxes=active.tolist(),boxes.tolist()
    elif layout=='list-int':
        active,boxes=[1]*shape[0],boxes.tolist()
    plan=plan_confidence_capture(shape,1,workspace_bytes=16*1024**2)
    gc.collect();tracemalloc.start()
    try:
        with confidence_capture_resources(plan):reader=_MaskedNativeScoreReader(mask,scores,active,boxes)
        _,peak=tracemalloc.get_traced_memory()
    finally:tracemalloc.stop()
    assert peak<=plan.workspace_bytes,(layout,peak,plan.workspace_bytes)
    assert reader.known_z_bounds==(0,shape[0])


def test_small_credit_refuses_long_thin_initializer_before_copies():
    with pytest.raises(MemoryError):
        plan_confidence_capture((60000,1,1),1,workspace_bytes=4*1024**2)


def test_invalid_active_shape_is_rejected_before_owned_array_copy(monkeypatch):
    mask,scores,active,boxes=arrays()
    plan=plan_confidence_capture(scores.shape,1,workspace_bytes=16*1024**2)
    invalid=np.broadcast_to(np.bool_(True),(scores.shape[0],1000000))
    original_array=np.array
    def guarded(value,*args,**kwargs):
        if value is invalid:pytest.fail('Invalid metadata was copied before validating shape')
        return original_array(value,*args,**kwargs)
    monkeypatch.setattr(np,'array',guarded)
    with confidence_capture_resources(plan),pytest.raises(ValueError):
        _MaskedNativeScoreReader(mask,scores,invalid,boxes)


def test_unknown_reader_encode_method_cannot_select_parallel_typed_fast_path(tmp_path):
    values=np.zeros((3,17,19),np.uint8);values[:,3:7,5:11]=173
    calls=[]
    class Generic:
        known_z_bounds=(0,3)
        def __call__(self,z):calls.append(z);return values[z]
        def encode_frame(self,*args):pytest.fail('Unknown callback entered typed compiled path')
    plan=plan_confidence_capture(values.shape,3,workspace_bytes=16*1024**2)
    metrics={}
    with confidence_capture_resources(plan):
        actual=storage.write_blocks(tmp_path/'generic',values.shape,Generic(),
            layer_key='k',model_name='m',provenance={},metrics=metrics)
    assert calls==[0,1,2] and 'capture_workers' not in metrics
    assert actual['known_voxels']==72


def test_nested_task_local_context_restores_outer_plan_on_error():
    outer=plan_confidence_capture((4,17,19),3,workspace_bytes=16*1024**2)
    inner=plan_confidence_capture((4,17,19),1,workspace_bytes=8*1024**2)
    assert current_confidence_capture_plan() is None
    with confidence_capture_resources(outer):
        with pytest.raises(RuntimeError):
            with confidence_capture_resources(inner):
                assert current_confidence_capture_plan() is inner
                raise RuntimeError('nested scope failure')
        assert current_confidence_capture_plan() is outer
    assert current_confidence_capture_plan() is None


def test_explicit_encoder_grid_refuses_before_any_fused_scan(monkeypatch):
    from XTA import confidence_capture_cpu
    mask,scores,active,boxes=arrays()
    plan=plan_confidence_capture(scores.shape,2,workspace_bytes=16*1024**2,block_size=128)
    with confidence_capture_resources(plan):reader=_MaskedNativeScoreReader(mask,scores,active,boxes)
    monkeypatch.setattr(confidence_capture_cpu,'masked_cell_bounds',
        lambda *a,**k:pytest.fail('Changed encoder grid entered fused scan'))
    with pytest.raises(ValueError):reader.encode_frame(0,8)


def test_ordered_queue_retains_at_most_worker_window_plus_one_consumer():
    lock=threading.Lock()
    counts={'live':0,'peak':0}
    class Frame:
        def __init__(self,z):
            self.z=z
            with lock:
                counts['live']+=1;counts['peak']=max(counts['peak'],counts['live'])
        def __del__(self):
            with lock:counts['live']-=1
    class Reader:
        def encode_frame(self,z,block):return Frame(z)
    workers=3
    iterator=storage._ordered_encoded_frames(Reader(),0,19,128,workers)
    previous=None
    observed=[]
    try:
        for z,frame in iterator:
            # Holding the previous caller result deliberately exercises the
            # separately charged consumer slot while the next future returns.
            assert frame.z==z
            observed.append(z)
            previous=frame
        assert observed==list(range(19))
    finally:iterator.close()
    previous=frame=None
    gc.collect()
    assert counts['peak']<=workers+1 and counts['live']==0


@pytest.mark.parametrize('capacity',[False,True],ids=['worker-error','capacity-error'])
def test_error_or_capacity_return_waits_for_other_borrowed_source_reads(tmp_path,monkeypatch,capacity):
    shape=(3,17,19)
    mask=np.ones(shape,np.uint8);scores=np.full(shape,173,np.uint8)
    active=np.ones(shape[0],bool);boxes=np.tile(np.array([0,17,0,19],np.int32),(3,1))
    plan=plan_confidence_capture(shape,2,workspace_bytes=16*1024**2)
    with confidence_capture_resources(plan):reader=_MaskedNativeScoreReader(mask,scores,active,boxes)
    real=reader.encode_frame
    started=threading.Event();release=threading.Event();finished=threading.Event()
    errors=[];running=[]
    lock=threading.Lock()
    def wrapped(z,block):
        with lock:running.append(z)
        try:
            if z==0:
                assert started.wait(3)
                if not capacity:raise OSError('first worker failed')
            else:
                started.set();assert release.wait(3)
            return real(z,block)
        finally:
            with lock:running.remove(z)
    monkeypatch.setattr(reader,'encode_frame',wrapped)
    def write():
        try:
            storage.write_blocks(tmp_path/'failed',shape,reader,layer_key='k',model_name='m',
                provenance={},max_numeric_bytes=1 if capacity else None)
        except BaseException as error:errors.append(error)
        finally:finished.set()
    thread=threading.Thread(target=write)
    thread.start()
    try:
        assert started.wait(3)
        assert not finished.wait(.05)  # The unrelated source read is still held.
        release.set();thread.join(5)
        assert not thread.is_alive() and not running and len(errors)==1
        assert isinstance(errors[0],storage.ConfidenceStageLimit if capacity else OSError)
        assert not list((tmp_path/'failed').iterdir())
        assert np.all(mask==1) and np.all(scores==173)
    finally:
        release.set();thread.join(5)

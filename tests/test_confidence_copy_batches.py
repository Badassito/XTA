"""Portable copy batches keep confidence cell coordinates and zlib bytes exact."""
import math
import threading
import weakref
import zlib

import numpy as np
import pytest

from XTA import confidence_capture_cpu as cpu
from XTA.confidence_capture import (COPY_BATCH_MAX_BYTES,COPY_BATCH_MAX_CELLS,
    confidence_capture_resources,plan_confidence_capture,_CODEC_BYTES)
from XTA.confidence_evidence import _MaskedNativeScoreReader


@pytest.fixture(autouse=True)
def portable_codec(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_CONFIDENCE_NATIVE_CODEC','python')


def _legacy(mask,scores,z,block):
    bounds=cpu.masked_cell_bounds(mask,scores,z,0,mask.shape[1],0,mask.shape[2],block)
    return [(int(a),int(b),int(c),int(d),int(known),
        zlib.compress(cpu.copy_masked_cell(mask,scores,z,a,b,c,d).tobytes(),level=3))
        for a,b,c,d,known in bounds]


def _reader(mask,scores,block,stop=None):
    active=np.ones(mask.shape[0],bool)
    boxes=np.tile(np.asarray((0,mask.shape[1],0,mask.shape[2]),np.int64),(mask.shape[0],1))
    plan=plan_confidence_capture(mask.shape,3,workspace_bytes=32*1024**2,block_size=block)
    with confidence_capture_resources(plan):
        return _MaskedNativeScoreReader(mask,scores,active,boxes,stop=stop),plan


@pytest.mark.parametrize('block',[8,128])
@pytest.mark.parametrize('dtype',[np.uint8,np.bool_])
@pytest.mark.parametrize('layout',['contiguous','fortran','negative-stride','readonly'])
def test_all_typed_layouts_preserve_legacy_compressed_cells(block,dtype,layout):
    rng=np.random.default_rng(160591)
    mask=(rng.integers(0,5,(3,257,263),dtype=np.uint8)!=0).astype(dtype)
    scores=rng.integers(0,256,mask.shape,dtype=np.uint8)
    mask[0]=0
    scores[1,:,71:]=0
    if layout=='fortran':mask,scores=np.asfortranarray(mask),np.asfortranarray(scores)
    elif layout=='negative-stride':mask,scores=mask[:,::-1,::-1],scores[:,::-1,::-1]
    elif layout=='readonly':mask.setflags(write=False);scores.setflags(write=False)
    original_mask,original_scores=mask.copy(),scores.copy()
    reader,plan=_reader(mask,scores,block)
    for z in range(mask.shape[0]):
        expected=_legacy(mask,scores,z,block)
        records,metrics=reader.encode_frame(z,block)
        assert records==expected
        assert metrics['copy_kernel_calls']==math.ceil(len(expected)/plan.copy_batch_cells)
        assert metrics['copy_batch_peak_bytes']<=plan.copy_batch_bytes
    np.testing.assert_array_equal(mask,original_mask)
    np.testing.assert_array_equal(scores,original_scores)


def test_canonical_grid_uses_one_nogil_copy_per_sixteen_cells_and_memoryviews(monkeypatch):
    mask=np.ones((1,513,515),np.uint8)
    scores=np.random.default_rng(160591).integers(1,256,mask.shape,dtype=np.uint8)
    expected=_legacy(mask,scores,0,128)
    reader,plan=_reader(mask,scores,128)
    original_copy=cpu.copy_masked_cells_batch
    original_compress=zlib.compress
    copied=[]
    def batch(*args):
        result=original_copy(*args)
        copied.append((args[3].shape[0],result[0].nbytes+result[1].nbytes))
        return result
    def compress(data,*args,**kwargs):
        assert isinstance(data,memoryview)
        return original_compress(data,*args,**kwargs)
    monkeypatch.setattr(cpu,'copy_masked_cells_batch',batch)
    monkeypatch.setattr(cpu,'copy_masked_cell',lambda *a:pytest.fail('Canonical frame used per-cell copy'))
    monkeypatch.setattr(zlib,'compress',compress)
    actual,metrics=reader.encode_frame(0,128)
    assert actual==expected
    assert len(copied)==metrics['copy_kernel_calls']==2
    assert copied[0][0]==16 and all(size<=plan.copy_batch_bytes for _,size in copied)


def test_cells_larger_than_copy_credit_retain_bounded_single_cell_fallback(monkeypatch):
    mask=np.ones((1,513,513),np.uint8)
    scores=np.random.default_rng(772).integers(0,256,mask.shape,dtype=np.uint8)
    expected=_legacy(mask,scores,0,1024)
    reader,plan=_reader(mask,scores,1024)
    assert plan.copy_batch_cells==plan.copy_batch_bytes==0
    monkeypatch.setattr(cpu,'copy_masked_cells_batch',lambda *a:pytest.fail('Oversized copy entered batch'))
    actual,metrics=reader.encode_frame(0,1024)
    assert actual==expected and metrics['copy_kernel_calls']==1
    assert metrics['copy_batch_peak_bytes']==0


def test_copy_batch_credit_is_an_additional_explicit_worker_charge():
    plan=plan_confidence_capture((111,2048,2048),32,workspace_bytes=64*1024**2)
    assert plan.copy_batch_cells==COPY_BATCH_MAX_CELLS==16
    assert plan.copy_batch_bytes==16*(128**2+64)+17*8+8
    assert plan.worker_bytes==plan.consumer_bytes+_CODEC_BYTES+2*128**2+plan.copy_batch_bytes
    assert plan.workspace_bytes<=plan.workspace_limit_bytes


@pytest.mark.parametrize('invalid',['count','bytes','bbox'])
def test_batch_refuses_outside_control_or_byte_credit(invalid):
    mask=np.ones((1,513,515),np.uint8)
    scores=np.full_like(mask,173)
    if invalid=='count':bounds=np.tile(np.array((0,1,0,1,1),np.int64),(17,1))
    elif invalid=='bytes':bounds=np.tile(np.array((0,513,0,515,1),np.int64),(2,1))
    else:bounds=np.array([[0,mask.shape[1]+1,0,1,1]],np.int64)
    with pytest.raises((ValueError,MemoryError)):
        cpu.copy_masked_cells_batch(mask,scores,0,bounds,COPY_BATCH_MAX_BYTES)


def test_cancelled_batch_releases_numeric_slab_before_next_source_read(monkeypatch):
    mask=np.ones((1,513,515),np.uint8)
    scores=np.full_like(mask,173)
    stop=threading.Event()
    reader,_=_reader(mask,scores,128,stop=stop)
    original_copy=cpu.copy_masked_cells_batch
    refs=[]
    def copy_and_cancel(*args):
        raw,offsets=original_copy(*args)
        refs.append(weakref.ref(raw))
        stop.set()
        return raw,offsets
    monkeypatch.setattr(cpu,'copy_masked_cells_batch',copy_and_cancel)
    monkeypatch.setattr(zlib,'compress',lambda *a,**k:pytest.fail('Cancelled batch compressed a cell'))
    with pytest.raises(RuntimeError,match='Confidence capture cancelled'):
        reader.encode_frame(0,128)
    assert len(refs)==1 and refs[0]() is None


def test_encoder_error_traceback_cannot_pin_a_copy_slab(monkeypatch):
    mask=np.ones((1,513,515),np.uint8)
    scores=np.full_like(mask,173)
    reader,_=_reader(mask,scores,128)
    original=cpu.copy_masked_cells_batch
    refs=[]
    views=[]
    def track(*args):
        result=original(*args)
        refs.append(weakref.ref(result[0]))
        return result
    def fail(data,*args,**kwargs):
        views.append(data)
        raise OSError('injected compression failure')
    monkeypatch.setattr(cpu,'copy_masked_cells_batch',track)
    monkeypatch.setattr(zlib,'compress',fail)
    with pytest.raises(OSError,match='injected compression failure') as caught:
        reader.encode_frame(0,128)
    assert caught.value.__traceback__ is not None
    assert len(refs)==1 and refs[0]() is None
    with pytest.raises(ValueError):
        _=views[0].obj

"""Real native zlib ABI qualification with strict portable fallbacks."""
import ctypes
import os
import threading
import weakref
import zlib

import numpy as np
import pytest

from XTA import confidence_native_codec as native,confidence_capture_cpu as cpu
from XTA.confidence_capture import confidence_capture_resources,plan_confidence_capture
from XTA.confidence_evidence import _MaskedNativeScoreReader


def test_disabled_native_codec_has_an_explicit_fallback(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_CONFIDENCE_NATIVE_CODEC','python')
    codec,receipt=native.native_confidence_codec()
    assert codec is None and receipt['backend']=='python_zlib'
    assert receipt['fallback_reason']=='disabled_by_configuration'


def test_no_candidate_never_searches_a_windows_git_installation(monkeypatch):
    monkeypatch.setattr(zlib,'__file__',None,raising=False)
    monkeypatch.setattr(ctypes.util,'find_library',lambda name:None)
    monkeypatch.setattr(ctypes,'CDLL',lambda *args:pytest.fail('No candidate should be loaded'))
    codec,receipt=native._load_backend(('auto','',''))
    assert codec is None and receipt['fallback_reason']=='native_zlib_library_not_available'


def test_unavailable_exports_produce_a_loud_portable_reason(monkeypatch):
    monkeypatch.setattr(ctypes,'CDLL',lambda path:object())
    codec,receipt=native._load_backend(('auto','1','explicit-test-library'))
    assert codec is None and receipt['backend']=='python_zlib'
    assert 'explicit_qualification:AttributeError' in receipt['fallback_reason']


@pytest.fixture(scope='module')
def real_codec():
    path=os.environ.get('XTA_TEST_NATIVE_ZLIB_LIBRARY')
    if path:
        return native.NativeConfidenceCodec(ctypes.CDLL(path),source='explicit_qualification')
    codec,_=native.native_confidence_codec()
    if codec is None:
        pytest.skip('No probed native library; explicit qualification DLL is optional')
    return codec


@pytest.mark.parametrize('dtype',[np.uint8,np.bool_])
@pytest.mark.parametrize('layout',['contiguous','fortran','negative-stride','readonly'])
@pytest.mark.parametrize('block',[8,128])
def test_actual_native_batch_has_exact_decoded_cells_and_local_python_bytes(real_codec,dtype,layout,block):
    rng=np.random.default_rng(160591)
    mask=(rng.integers(0,5,(4,257,263),dtype=np.uint8)!=0).astype(dtype)
    scores=rng.integers(0,256,mask.shape,dtype=np.uint8)
    mask[0]=0;scores[1]=173;scores[2,:,129:]=0
    if layout=='fortran':mask,scores=np.asfortranarray(mask),np.asfortranarray(scores)
    elif layout=='negative-stride':mask,scores=mask[:,::-1,::-1],scores[:,::-1,::-1]
    elif layout=='readonly':mask.setflags(write=False);scores.setflags(write=False)
    for z in range(mask.shape[0]):
        bounds=cpu.masked_cell_bounds(mask,scores,z,0,mask.shape[1],0,mask.shape[2],block)
        for first in range(0,len(bounds),16):
            selected=bounds[first:first+16]
            encoded,offsets=real_codec.encode_batch(mask,scores,z,selected,block)
            for i,(a,b,c,d,_) in enumerate(selected):
                raw=cpu.copy_masked_cell(mask,scores,z,a,b,c,d).tobytes()
                actual=encoded[int(offsets[i]):int(offsets[i+1])].tobytes()
                assert zlib.decompress(actual)==raw
                # Both qualified local runtimes (native 1.3.1/Python 1.3.2)
                # currently emit identical bytes. Format compatibility, rather
                # than universal cross-version byte identity, is the contract.
                assert actual==zlib.compress(raw,3)
    assert real_codec._kernel.targetoptions['nogil']
    assert real_codec._kernel.nopython_signatures


def _reader(mask,scores):
    plan=plan_confidence_capture(mask.shape,2,workspace_bytes=32*1024**2)
    boxes=np.tile(np.asarray((0,mask.shape[1],0,mask.shape[2]),np.int64),(mask.shape[0],1))
    with confidence_capture_resources(plan):
        reader=_MaskedNativeScoreReader(mask,scores,np.ones(mask.shape[0],bool),boxes)
    return reader,plan


def test_actual_reader_native_route_never_calls_python_cell_compress(real_codec,monkeypatch):
    mask=np.ones((1,513,515),np.uint8)
    scores=np.random.default_rng(160591).integers(1,256,mask.shape,dtype=np.uint8)
    bounds=cpu.masked_cell_bounds(mask,scores,0,0,513,0,515,128)
    expected=[(int(a),int(b),int(c),int(d),int(k),
        zlib.compress(cpu.copy_masked_cell(mask,scores,0,a,b,c,d).tobytes(),3)) for a,b,c,d,k in bounds]
    reader,plan=_reader(mask,scores)
    receipt=dict(backend='native_compress2',codec_version=real_codec.version,
                 python_zlib_version=zlib.ZLIB_RUNTIME_VERSION,fallback_reason=None)
    monkeypatch.setattr(native,'native_confidence_codec',lambda:(real_codec,receipt))
    monkeypatch.setattr(zlib,'compress',lambda *a,**k:pytest.fail('Native batch reentered Python zlib.compress'))
    records,metrics=reader.encode_frame(0,128)
    assert records==expected
    assert metrics['codec_backend']=='native_compress2'
    assert metrics['native_compression_batches']==metrics['copy_kernel_calls']==2
    assert metrics['copy_batch_peak_bytes']<=plan.copy_batch_bytes


def test_native_cancellation_releases_compressed_slab(real_codec,monkeypatch):
    mask=np.ones((1,513,515),np.uint8);scores=np.full_like(mask,173)
    reader,_=_reader(mask,scores)
    stop=threading.Event();reader.stop=stop
    original=real_codec.encode_batch
    refs=[]
    def encode(*args):
        result=original(*args)
        refs.append(weakref.ref(result[0]));stop.set()
        return result
    monkeypatch.setattr(real_codec,'encode_batch',encode)
    monkeypatch.setattr(native,'native_confidence_codec',lambda:(real_codec,dict(
        backend='native_compress2',codec_version=real_codec.version,fallback_reason=None)))
    with pytest.raises(RuntimeError,match='Confidence capture cancelled'):
        reader.encode_frame(0,128)
    assert len(refs)==1 and refs[0]() is None


@pytest.mark.parametrize('invalid',['control','bbox','raw-credit'])
def test_native_batch_rejects_outside_its_preflight_credit(real_codec,invalid):
    mask=np.ones((1,513,515),np.uint8);scores=np.full_like(mask,173)
    if invalid=='control':bounds=np.tile(np.array((0,1,0,1,1),np.int64),(17,1))
    elif invalid=='bbox':bounds=np.array([[-1,5,0,5,25]],np.int64)
    else:bounds=np.array([[0,129,0,128,1]],np.int64)
    with pytest.raises((ValueError,MemoryError)):
        real_codec.encode_batch(mask,scores,0,bounds,128)

"""Frame diagnostics preserve legacy bytes and settle borrowed readers on error."""
from concurrent.futures import ThreadPoolExecutor
import gc
from pathlib import Path
import threading
import weakref

import numpy as np
import pytest

from XTA import confidence_storage as storage
from XTA.confidence_capture import confidence_capture_resources,plan_confidence_capture
from XTA.confidence_evidence import _MaskedNativeScoreReader
from XTA.native_pull_scheduling import configure_native_pull_scheduler,shutdown_native_pull_scheduler


def test_frame_wall_and_thread_cpu_timing_separate_wait_from_consumption(monkeypatch):
    wall,cpu=[0.],[0.]
    monkeypatch.setattr(storage.time,'perf_counter',lambda:wall[0])
    monkeypatch.setattr(storage.time,'thread_time',lambda:cpu[0])
    def source():
        wall[0]+=5;cpu[0]+=2
        yield 'frame'
        wall[0]+=3;cpu[0]+=1
    timing=dict(frame_wait_seconds=0.,frame_wait_thread_cpu_seconds=0.,
                frame_consume_seconds=0.,frame_consume_thread_cpu_seconds=0.)
    frames=storage._measured_frames(source(),timing)
    assert next(frames)=='frame'
    wall[0]+=7;cpu[0]+=4
    with pytest.raises(StopIteration):next(frames)
    assert timing==dict(frame_wait_seconds=8.,frame_wait_thread_cpu_seconds=3.,
                        frame_consume_seconds=7.,frame_consume_thread_cpu_seconds=4.)


def test_new_frame_metrics_do_not_change_legacy_files(tmp_path):
    shape=(2,9,11)
    scores=np.random.default_rng(160756).integers(0,256,shape,np.uint8)
    options=dict(layer_key='k',model_name='m',provenance={})
    plain=storage.write_blocks(tmp_path/'plain',shape,lambda z:scores[z],**options)
    metrics={}
    measured=storage.write_blocks(tmp_path/'measured',shape,lambda z:scores[z],metrics=metrics,**options)
    assert measured==plain
    for name in ('index.bin','scores.u8.zlib','metadata.json'):
        assert (tmp_path/'plain'/name).read_bytes()==(tmp_path/'measured'/name).read_bytes()
    for name in ('frame_wait_seconds','frame_wait_thread_cpu_seconds',
                 'frame_consume_seconds','frame_consume_thread_cpu_seconds'):
        assert metrics[name]>=0


@pytest.mark.parametrize('failure_kind',['limit','write'])
def test_legacy_failure_joins_readers_and_failed_future_releases_borrowed_alias(tmp_path,monkeypatch,failure_kind):
    shape=(4,64,65)
    scores=np.full(shape,173,np.uint8);mask=np.ones(shape,np.uint8)
    plan=plan_confidence_capture(shape,2,block_size=32)
    boxes=np.tile(np.asarray((0,64,0,65),np.int64),(shape[0],1))
    with confidence_capture_resources(plan):
        reader=_MaskedNativeScoreReader(mask,scores,np.ones(shape[0],bool),boxes)
    entered,release=threading.Event(),threading.Event()
    original=_MaskedNativeScoreReader.encode_frame
    def encode(self,z,block):
        if z==0:assert entered.wait(5)
        if z==1:
            entered.set()
            assert release.wait(5)
        return original(self,z,block)
    monkeypatch.setattr(_MaskedNativeScoreReader,'encode_frame',encode)
    if failure_kind=='write':
        original_open=Path.open
        class FailedPayload:
            def __init__(self,stream):self.stream=stream
            def __enter__(self):return self
            def __exit__(self,*args):return self.stream.__exit__(*args)
            def write(self,value):
                value=None
                raise OSError('payload write failed')
        def open_file(path,mode='r',*args,**kwargs):
            stream=original_open(path,mode,*args,**kwargs)
            return FailedPayload(stream) if mode=='wb' and path.name=='scores.u8.zlib.partial' else stream
        monkeypatch.setattr(Path,'open',open_file)
    scheduler=configure_native_pull_scheduler(workers=2,max_workers=2)
    owners=[reader];reader_ref=weakref.ref(reader);reader=None
    metrics={}
    was_enabled=gc.isenabled();gc.disable()
    try:
        with ThreadPoolExecutor(max_workers=1) as pool:
            failure=pool.submit(lambda:storage.write_blocks(tmp_path/'failed',shape,owners[0],
                layer_key='k',model_name='m',provenance={},block_size=32,metrics=metrics,
                max_numeric_bytes=1 if failure_kind=='limit' else None))
            assert entered.wait(5)
            assert not failure.done()
            release.set()
            with pytest.raises(storage.ConfidenceStageLimit if failure_kind=='limit' else OSError):
                failure.result(timeout=5)
            owners.clear()
            assert reader_ref() is None
            assert scheduler.snapshot()['active_operations']==0
            assert scheduler.snapshot()['retained_operations']==0
        assert not (tmp_path/'failed/metadata.json').exists()
        assert not list((tmp_path/'failed').glob('*.partial'))
        assert metrics['frame_consume_seconds']>=0
    finally:
        release.set();shutdown_native_pull_scheduler(scheduler)
        if was_enabled:gc.enable()

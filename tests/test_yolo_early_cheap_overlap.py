"""Prepared cheap and curved tasks overlap without the removed Python early-owner policy."""
from XTA.native_scheduler import GPU
from tests.native_scheduler_support import NativeRun

def test_busy_curved_owner_does_not_block_another_workers_ready_cartesian():
    with NativeRun(workers=2) as run:
        run.declare(1,tasks=2,exclusive=GPU)
        run.prepare(1)
        run.submit(1,1,descriptor={'family':'radial','slice_start':0,'slice_count':2})
        run.submit(2,1,descriptor={'family':'radial','slice_start':2,'slice_count':2})
        first,_=run.take(0)
        run.receipt(first,'installed')
        run.declare(2)
        run.prepare(2)
        run.submit(3,2,descriptor={'family':'orthogonal','slice_start':0,'slice_count':3})
        cheap,descriptor=run.take(1)
        assert descriptor['family']=='orthogonal'
        run.finish(cheap)
        run.published(2)
        run.receipt(first,'compute')
        following,descriptor=run.take(0)
        assert descriptor['slice_start']==2
        run.receipt(first,'final',payload=b'curved-first')
        run.finish(following)
        run.published(1)

def test_final_without_early_compute_returns_one_window_and_keeps_payload_credit():
    with NativeRun() as run:
        run.declare(1,tasks=2,bytes=20)
        run.prepare(1)
        run.submit(1,1,retained_bytes=7)
        run.submit(2,1,retained_bytes=7)
        first,_=run.take()
        run.receipt(first,'installed')
        run.receipt(first,'final',payload=b'first')
        second,_=run.take()
        assert second.task_id==2
        assert run.owner.snapshot()['bytes_used']==34
        run.receipt(first,'compute')
        run.finish(second)
        run.published(1)
        run.consume_events()
        run.retire(first)
        run.retire(second,parent=True)
        assert run.owner.snapshot()['bytes_used']==0

def test_ready_descriptor_on_blocked_device_does_not_hide_another_eligible_task():
    with NativeRun(workers=2) as run:
        stage=run.owner.try_acquire_stage(0)
        run.declare(1,tasks=2)
        run.prepare(1)
        run.submit(1,1,required_worker=0,descriptor={'family':'radial'})
        run.submit(2,1,required_worker=1,descriptor={'family':'orthogonal'})
        cheap,_=run.take(1)
        assert cheap.task_id==2
        assert run.take(0,timeout=.03) is None
        run.finish(cheap)
        run.owner.release_stage(0,stage)
        curved,_=run.take(0)
        assert curved.task_id==1
        run.finish(curved)
        run.published(1)

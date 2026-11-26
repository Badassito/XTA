"""Spherical parent residency and exact range ownership through the native scheduler.

The removed Python cache-locality heuristic is not reconstructed in this test.
"""
from XTA.native_scheduler import GPU
from tests.native_scheduler_support import NativeRun

def test_distinct_spherical_parents_keep_distinct_resident_actors():
    with NativeRun(workers=2,tickets=2) as run:
        for parent in (1,2):
            run.declare(parent,tasks=2,exclusive=GPU)
            run.prepare(parent)
            for offset in (0,3):
                run.submit(parent*10+offset,parent,descriptor={
                    'family':'spherical','face':parent-1,'slice_start':offset,'slice_count':3})
        seen=[]
        for worker,parent in ((0,1),(1,2)):
            for _ in range(2):
                lease,desc=run.take(worker)
                assert lease.parent_id==parent
                seen.append((parent,desc['slice_start'],desc['slice_count']))
                run.finish(lease)
            run.published(parent)
        assert sorted(seen)==[(1,0,3),(1,3,3),(2,0,3),(2,3,3)]
        assert run.owner.snapshot()['final']==4

def test_spherical_parent_does_not_release_residency_on_compute_alone():
    with NativeRun(workers=1,tickets=2,preparation_window=2) as run:
        run.declare(1,exclusive=GPU)
        run.prepare(1)
        run.declare(2,exclusive=GPU)
        assert run.owner.wait_preparation(.03) is None
        run.submit(1,1,descriptor={'family':'spherical'})
        lease,_=run.take()
        run.receipt(lease,'installed')
        run.receipt(lease,'compute')
        assert run.owner.wait_preparation(.03) is None
        run.receipt(lease,'final',payload=b'done')
        assert run.owner.wait_task_final(1,lease.generation,2)
        run.prepare(2)
        run.submit(2,2,descriptor={'family':'spherical'})
        assert run.take()[0].parent_id==2

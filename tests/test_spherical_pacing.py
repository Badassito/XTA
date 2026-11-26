"""Native parent ticket pacing preserves all requested Spherical trajectories."""
from XTA.native_scheduler import GPU
from tests.native_scheduler_support import NativeRun

def test_bounded_parent_frontier_completes_every_spherical_trajectory():
    with NativeRun(workers=1,tickets=1,capacity=100) as run:
        for parent in range(1,7):
            run.declare(parent,tasks=2,bytes=10,exclusive=GPU)
        observed=[]
        for parent in range(1,7):
            run.prepare(parent)
            assert run.owner.wait_preparation(0) is None
            leases=[]
            for offset in (0,4):
                run.submit(parent*10+offset,parent,descriptor={
                    'family':'spherical','trajectory':parent,'slice_start':offset,'slice_count':4})
            for _ in range(2):
                lease,descriptor=run.take()
                observed.append(descriptor)
                leases.append(lease)
                run.finish(lease)
            run.published(parent)
            run.consume_events()
            for lease in leases:
                run.retire(lease)
            run.owner.retire_parent(parent,run.parents[parent])
        assert len(observed)==12
        assert sum(d['slice_count'] for d in observed)==48
        assert {d['trajectory'] for d in observed}==set(range(1,7))
        assert run.owner.snapshot()['bytes_used']==0
        assert run.owner.snapshot()['blocked_parents']==0

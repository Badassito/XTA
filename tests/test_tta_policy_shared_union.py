"""Native policy groups preserve separate real shared result maps."""
import numpy as np
import pytest
from tests.native_detector_test_support import DetectorRun
from tests.native_policy_test_support import PolicyWorkspaces, policy_task
from tests.native_scheduler_cases import finish_actor

def shared_group(group,task_id,start):
    task=policy_task(group,ratio=3,task_id=task_id,slice_start=start,mode='direct_union')
    for member in (task,*task['augmentation_pass_tasks']):
        member.update(processing_shape=(4,5,5),union_num_slices=4,slice_count=2)
    return task

def test_later_chunks_reuse_each_pass_parent_and_preserve_prior_windows(tmp_path):
    work=PolicyWorkspaces(tmp_path,confidence=True)
    try:
        with DetectorRun(tmp_path,state=work.state,inputs=work.inputs) as run:
            actor=run.worker(0)
            run.start_transport()
            run.declare(shared_group(0,1,0),shared_group(0,2,2))
            run.drain_until(lambda: run.native.snapshot()['dispatched']>=1)
            paths=[]
            for _ in range(2):
                task=run.take(0)
                members=(task,*task['augmentation_pass_tasks'])
                paths.append([(m['result_mask_path'],m['result_conf_path']) for m in members])
                assert len(set(paths[-1]))==3
                for index,member in enumerate(members):
                    for field,value in [('result_mask_path',1),('result_conf_path',200+index)]:
                        array=np.memmap(member[field],mode='r+',dtype=np.uint8,shape=(4,5,5))
                        array[task['slice_start']:task['slice_start']+task['slice_count'],index,:]=value
                        array.flush()
                        array._mmap.close()
                finish_actor(actor,task)
            run.drain_until(lambda: len(run.results)==2)
            assert paths[0]==paths[1]
            assert run.native.snapshot()['bytes_used']==600
            for index,(mask,score) in enumerate(work.arrays.values()):
                expected=np.zeros((4,5,5),np.uint8)
                expected[:,index,:]=1
                assert np.array_equal(mask,expected)
                assert np.array_equal(score,expected*(200+index))
            for key in list(work.arrays):
                work.retire(key)
            run.drain_until(lambda: not run.scheduler.publication_outstanding())
            assert run.native.snapshot()['bytes_used']==0
    finally:
        work.close()

def test_shared_policy_mode_rejects_mixed_file_ownership(tmp_path):
    work=PolicyWorkspaces(tmp_path,confidence=True)
    task=shared_group(0,1,0)
    task['augmentation_pass_tasks'][0]['result_mode']='file'
    try:
        with DetectorRun(tmp_path,state=work.state,inputs=work.inputs) as run:
            run.worker(0)
            with pytest.raises(RuntimeError,match='one result mode'):
                run.declare(task)
            assert not work.arrays and run.native.snapshot()['tickets_used']==0
    finally:
        work.close()

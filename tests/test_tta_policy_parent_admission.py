"""Native atomic policy-group funding, independent outputs and retirement."""
import pytest
from XTA.tta_scheduler import TtaSchedulerCallbacks
from tests.native_detector_test_support import DetectorRun
from tests.native_policy_test_support import PolicyWorkspaces, policy_task
from tests.native_scheduler_cases import finish_actor


def consume_policy_files(run, work):
    def consume(task, stats):
        work.consume_file_results(task)
        run.results.append((task, stats))
    run.scheduler.bind_result_callbacks(
        TtaSchedulerCallbacks(consume, consume, lambda: None, lambda: None))

@pytest.mark.parametrize('ratio', [1, 2, 4])
@pytest.mark.parametrize('confidence', [False, True])
def test_each_policy_pass_is_charged_and_keeps_its_output_identity(tmp_path, ratio, confidence):
    work = PolicyWorkspaces(tmp_path, confidence=confidence)
    task = policy_task(0, ratio=ratio, size=100, task_id=1, root=tmp_path)
    baseline_bytes = ratio * 100 * (2 if confidence else 1)
    result_bytes = ratio * 100 * 2  # Every declared pass has both file result channels.
    original_paths = [(m['result_mask_path'],m['result_conf_path'])
                      for m in (task,*task.get('augmentation_pass_tasks',()))]
    try:
        with DetectorRun(tmp_path, state=work.state, inputs=work.inputs) as run:
            actor=run.worker(0)
            consume_policy_files(run, work)
            run.start_transport()
            run.declare(task)
            run.drain_until(lambda: run.native.snapshot()['dispatched']==1)
            received=run.take(0)
            members=(received,*received.get('augmentation_pass_tasks',()))
            assert len(members)==ratio
            assert [(m['result_mask_path'],m['result_conf_path']) for m in members]==original_paths
            assert run.native.snapshot()['bytes_used'] == baseline_bytes + result_bytes
            assert len(work.arrays)==ratio and len({m['view'].name for m in members})==ratio
            assert work.write_file_results(received) == result_bytes
            finish_actor(actor,received)
            assert run.native.snapshot()['bytes_used'] == baseline_bytes + result_bytes
            run.drain_until(lambda: len(run.results)==1 and
                            run.native.snapshot()['bytes_used'] == baseline_bytes)
            assert work.consumed_file_bytes == result_bytes and not work.file_results
            assert run.native.snapshot()['bytes_used'] == baseline_bytes
            assert all(bool(arrays[0].all()) for arrays in work.arrays.values())
            assert run.scheduler.publication_outstanding()
            for key in list(work.arrays):
                work.retire(key)
            run.drain_until(lambda: not run.scheduler.publication_outstanding())
            assert run.native.snapshot()['bytes_used']==0
    finally:
        work.close()

@pytest.mark.parametrize('malform', ['unmarked','duplicate','mixed_mode'])
def test_invalid_sibling_group_refuses_before_workspace_or_native_funding(tmp_path, malform):
    work=PolicyWorkspaces(tmp_path)
    task=policy_task(0,ratio=2,task_id=1)
    other=task['augmentation_pass_tasks'][0]
    if malform=='unmarked':
        other['bounded_parent_admission']=False
    elif malform=='duplicate':
        other['view']=task['view']
    else:
        other['result_mode']='direct_union'
    try:
        with DetectorRun(tmp_path,state=work.state,inputs=work.inputs) as run:
            run.worker(0)
            with pytest.raises(RuntimeError,match='Policy parent'):
                run.declare(task)
            assert not work.arrays and not work.requested
            snapshot=run.native.snapshot()
            assert snapshot['dispatched']==snapshot['tickets_used']==snapshot['bytes_used']==0
    finally:
        work.close()

@pytest.mark.parametrize('capacity', [250, 800])
def test_group_over_total_cap_is_preflight_refusal_not_partial_admission(tmp_path, capacity):
    work=PolicyWorkspaces(tmp_path,capacity=capacity)
    try:
        with DetectorRun(tmp_path,state=work.state,inputs=work.inputs) as run:
            run.worker(0)
            with pytest.raises(ValueError,match='needs'):
                # At 800 bytes the 400-byte baseline fits, but baseline plus
                # the 800-byte actor file-result escrow does not.
                run.declare(policy_task(0,ratio=4,size=100,task_id=1,root=tmp_path))
            assert not work.arrays and run.native.snapshot()['bytes_used']==0
    finally:
        work.close()

def test_partial_component_retirement_returns_only_actual_pass_bytes(tmp_path):
    work=PolicyWorkspaces(tmp_path,capacity=1200)
    try:
        with DetectorRun(tmp_path,state=work.state,inputs=work.inputs) as run:
            actor=run.worker(0)
            consume_policy_files(run, work)
            run.start_transport()
            run.declare(policy_task(0,ratio=4,size=100,task_id=1,root=tmp_path))
            run.drain_until(lambda: run.native.snapshot()['dispatched']==1)
            received = run.take(0)
            assert run.native.snapshot()['bytes_used'] == 1200
            assert work.write_file_results(received) == 800
            finish_actor(actor,received)
            run.drain_until(lambda: len(run.results)==1 and
                            run.native.snapshot()['bytes_used'] == 400)
            assert work.consumed_file_bytes == 800 and not work.file_results
            assert run.native.snapshot()['bytes_used'] == 400
            for remaining,key in zip((300,200,100,0),list(work.arrays)):
                work.retire(key)
                run.scheduler.service_preparation()
                assert run.native.snapshot()['bytes_used']==remaining
            run.drain_until(lambda: not run.scheduler.publication_outstanding())
    finally:
        work.close()

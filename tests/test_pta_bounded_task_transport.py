"""Workers receive one bounded task, never every candidate in the dataset phase."""
import pickle
import queue
from concurrent.futures import Future
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta_workers as workers
from XTA import pta_gpu_classification as classifier


class CannotPickleTask:
    def __reduce__(self):
        raise AssertionError('phase metadata tried to pickle dataset tasks')


@pytest.fixture
def prep():
    volume_block=workers.SharedBlock(8)
    mask_block=workers.SharedBlock(8)
    prepared=SimpleNamespace(src=SimpleNamespace(stem='sample'),
        plans=[SimpleNamespace(tag='Transverse',tile_layout=())],
        volume_for_render=volume_block.ndarray((2,2,2)),mask_for_render=mask_block.ndarray((2,2,2)),
        volume_render_block=volume_block,mask_render_block=mask_block,
        semantic_coverage_for_render=None,semantic_coverage_render_block=None,semantic_foreground=False)
    try:
        yield prepared
    finally:
        prepared.volume_for_render=prepared.mask_for_render=None
        volume_block.release()
        mask_block.release()


def pool():
    result=object.__new__(workers.PersistentRenderPool)
    result.backend='process'
    result.results=queue.Queue()
    result.gpu_device_ids=(0,)
    result._pool=mock.Mock()
    return result


def test_phase_payload_is_independent_of_task_count_and_candidate_size(prep):
    p=pool()
    handles=[]
    try:
        handles.append(p.install_phase(gen=1,prep=prep,tasks=[CannotPickleTask()]))
        handles.append(p.install_phase(gen=1,prep=prep,tasks=[CannotPickleTask()]*10000))
        assert handles[0].payload_nbytes==handles[1].payload_nbytes
        assert handles[0].payload_nbytes<4096
        for handle in handles:
            payload=pickle.loads(bytes(handle.payload_block.shm.buf[:handle.payload_nbytes]))
            assert 'tasks' not in payload
            assert payload['plans'][0].tag=='Transverse'
    finally:
        for handle in handles:
            handle.payload_block.release()


def test_render_submission_sends_each_actual_task_and_preserves_callbacks(prep):
    p=pool()
    tasks=[SimpleNamespace(candidate=i) for i in range(3)]
    handle=p.install_phase(gen=2,prep=prep,tasks=tasks)
    try:
        assert p.submit_phase(handle,meta=(2,'A'))==3
        for index,call in enumerate(p._pool.apply_async.call_args_list):
            assert call.args[0] is workers._render_task_entry
            assert call.args[1]==((handle.payload_name,handle.payload_nbytes,tasks[index]),)
            call.kwargs['callback']((index,{}, {},{}))
            assert p.results.get_nowait()==((2,'A'),(index,{}, {},{}),None)
        error=RuntimeError('task failed')
        call.kwargs['error_callback'](error)
        assert p.results.get_nowait()==((2,'A'),None,error)
    finally:
        handle.payload_block.release()


def test_process_entry_consumes_supplied_task_without_task_list():
    task=SimpleNamespace(only='this task')
    payload={'plans':['plan']}
    arrays=(np.zeros((1,1,1),np.uint8),np.zeros((1,1,1),np.uint8),None)
    with mock.patch.object(workers,'_worker_load_payload',return_value=payload), \
         mock.patch.object(workers,'_worker_volume_arrays',return_value=arrays), \
         mock.patch.object(workers,'execute_render_task',return_value='done') as execute:
        assert workers._render_task_entry(('metadata',8,task))=='done'
    assert execute.call_args.args==(arrays[0],arrays[1],payload['plans'],task,None)


def test_classification_transport_sends_bounded_task_objects_and_releases_metadata(prep):
    p=pool()
    transmitted=[]
    def dispatch(entry,jobs,*,chunksize):
        assert entry is workers._classification_task_entry and chunksize==1
        replies=[]
        for name,nbytes,task in jobs:
            transmitted.append(task)
            assert isinstance(task,classifier.GpuClassificationTask)
            assert len(task.frames)<=64
            measured=classifier.classification_stats()
            measured['gpu_jobs']=len(task.frames)
            replies.append(({('Transverse',i,'full',0):True for _,i,_ in task.frames},[],measured))
        return iter(replies)
    p._pool.imap_unordered=dispatch
    with mock.patch.object(p,'install_phase',wraps=p.install_phase) as install:
        truth,remaining,stats=p.classify_foregrounds(prep,gen=1,jobs=[(0,i) for i in range(129)])
    assert len(transmitted)==3 and len(truth)==129
    assert remaining==[] and stats['gpu_jobs']==129
    assert install.call_count==1
    # The shared payload has been released; individual worker tasks retain no sibling tasks.
    assert all(not hasattr(task,'tasks') for task in transmitted)


def test_thread_phase_retains_direct_task_references(prep):
    p=pool()
    p.backend='thread'
    tasks=[SimpleNamespace(candidate=1)]
    handle=p.install_phase(gen=1,prep=prep,tasks=tasks)
    assert handle.payload_block is None
    assert handle.payload['tasks'][0] is tasks[0]
    assert handle.payload['volume'] is prep.volume_for_render


def test_cancelled_queued_thread_task_has_no_completion_payload(prep):
    p=pool()
    p.backend='thread'
    p._pool=None
    p._executor=mock.Mock()
    dispatched=mock.Mock()
    p._executor.submit.return_value=dispatched
    handle=p.install_phase(gen=1,prep=prep,tasks=[object()])
    assert p.submit_phase(handle,meta=(1,'A'))==1
    callback=dispatched.add_done_callback.call_args.args[0]
    cancelled=Future()
    assert cancelled.cancel()
    callback(cancelled)
    assert p.results.empty()

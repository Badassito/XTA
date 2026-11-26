"""Scientific task descriptors and actor replies used by native facade tests."""
from tests.native_detector_test_support import detector_view


def detector_task(task_id, *, view=None, start=0, count=1, mode='file', cpu=False, **fields):
    return dict(task_id=task_id, kind='fullframe', model_name='model',
                view=detector_view() if view is None else view, result_mode=mode,
                slice_start=start, slice_count=count, out_size=8,
                gpu_eligible=not cpu, cpu_eligible=cpu, **fields)


def finish_actor(actor, task, *, index=0, cpu=False, final_first=False, stats=None):
    actor.installed(task['task_id'])
    if not final_first:
        actor.compute_done(task['task_id'])
    actor.put(dict(type='result', task_id=task['task_id'], ok=True,
                   worker_kind='cpu' if cpu else 'gpu',
                   **({'cpu_index': index} if cpu else {'gpu_index': index}),
                   stats=dict(stats or {}, frames=task['slice_count'])))
    if final_first:
        actor.compute_done(task['task_id'])

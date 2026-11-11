"""A pipelined volume classifies on the CUDA owners behind the rendering volume."""
import queue
import threading
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace

from XTA import pta_workers
from XTA.pta_gpu_classification import classification_stats


def test_classification_queues_behind_each_owners_render_tasks(monkeypatch):
    """Uses the stdlib Pool task queue shared by the process pool (threads stand in for owners)."""
    events, lock, release_first = [], threading.Lock(), threading.Event()

    def record(kind, generation, index):
        with lock:
            events.append((threading.get_ident(), kind, generation, index))

    def render(job):
        name, _nbytes, task = job
        assert name == "gen0"
        record("render", 0, task)
        if task == 0:
            assert release_first.wait(10), "classification waited for every owner to drain"
        return 1, {}, {}, {}, pta_workers.DeferredForegroundCounts()

    def classify(job):
        name, _nbytes, task = job
        assert name == "gen1"
        record("classify", 1, task.frames[0][1])
        stats = classification_stats()
        stats["gpu_jobs"] = len(task.frames)
        stats["gpu_queries"] = len(task.frames)
        return {("view", frame, "full", 0): bool(frame % 2) for _plan, frame, _items in task.frames}, [], stats

    monkeypatch.setattr(pta_workers, "_render_task_entry", render)
    monkeypatch.setattr(pta_workers, "_classification_task_entry", classify)
    pool = object.__new__(pta_workers.PersistentRenderPool)
    pool.backend, pool.workers, pool.gpu_device_ids = "process", 2, (0, 1)
    pool.results, pool._executor, pool._pool = queue.Queue(), None, ThreadPool(2)
    pool.install_phase = lambda *, gen, prep, tasks: pta_workers.RenderPhaseHandle(
        f"gen{gen}", 1, None, list(tasks), None)
    try:
        handle = pool.install_phase(gen=0, prep=None, tasks=list(range(6)))
        progress = pta_workers.VolumeRenderProgress(stem="rendering", warnings=pta_workers.WarningLog())
        progress.pending_a = pool.submit_phase(handle, meta=(0, "A"))
        prep = SimpleNamespace(plans=[SimpleNamespace(tag="view", tile_layout=())],
            semantic_foreground=False, src=SimpleNamespace(stem="next"))
        jobs = [(0, frame) for frame in range(130)]
        # One owner is still rendering task 0; the other drains tasks 1-5 and
        # then classifies every job. Nothing falls back to CPU.
        truth, remaining, stats = pool.classify_foregrounds(prep, gen=1, jobs=jobs)
        assert remaining == [] and stats["gpu_jobs"] == 130
        assert truth == {("view", frame, "full", 0): bool(frame % 2) for frame in range(130)}
        assert not release_first.is_set() and progress.pending_a == 6
        release_first.set()
        pta_workers.drain_render_results(pool, {0: progress}, until=lambda: progress.pending_a == 0)
        assert progress.written == 6
    finally:
        release_first.set()
        pool._pool.close()
        pool._pool.join()
    by_owner = {}
    for owner, kind, generation, index in events:
        by_owner.setdefault(owner, []).append((kind, generation, index))
    blocked = next(owner for owner, items in by_owner.items() if items[0] == ("render", 0, 0))
    assert by_owner.pop(blocked) == [("render", 0, 0)]
    (free,) = by_owner.values()
    # FIFO: the free owner finished every queued render task before any classification.
    assert free == [("render", 0, index) for index in range(1, 6)] + [
        ("classify", 1, start) for start in (0, 64, 128)]

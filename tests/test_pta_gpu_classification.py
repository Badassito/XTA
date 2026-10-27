"""Bounded CUDA-classifier protocol without initializing CUDA in CPU tests."""
from contextlib import nullcontext
from types import SimpleNamespace
from unittest import mock
import queue

import numpy as np
import pytest

from XTA import pta_gpu_classification as classifier, pta_workers
from XTA import pta_cuda_masks
from XTA.pta_publication import mask_has_yolo_polygon


def plan(*, tiles=(), size=8):
    matrix = np.asarray(((1, 0, 0), (0, 1, 0)), np.float32)
    aff = SimpleNamespace(M_src_to_out=matrix, M_out_to_src=matrix, out_h=size, out_w=size)
    return SimpleNamespace(tag='view', view=SimpleNamespace(shared_view=object()), aff=aff,
        tile_layout=tuple(SimpleNamespace(tile_tag=key, shared_job=aff, out_h=size, out_w=size)
                          for key in tiles))


@pytest.fixture
def cpu_runtime(monkeypatch):
    torch = pytest.importorskip('torch')
    stream = SimpleNamespace(wait_event=mock.Mock(), synchronize=mock.Mock())
    owner = SimpleNamespace(stream=stream)
    fake = SimpleNamespace(cuda=SimpleNamespace(stream=lambda _: nullcontext(), device=lambda _: nullcontext()),
                           stack=torch.stack, count_nonzero=torch.count_nonzero)
    monkeypatch.setattr(pta_cuda_masks, '_owner_for_runtime', lambda _: owner)
    monkeypatch.setattr(pta_cuda_masks, '_family', lambda v: 'cartesian' if v is not None else None)
    return dict(torch=fake, device_id=0), owner, torch


@pytest.mark.parametrize('kind', ['empty', 'point', 'line', 'square', 'sixteen_fragmented'])
def test_segment_count16_and_tiny_polygon_decisions_match_existing_helper(monkeypatch, cpu_runtime, kind):
    runtime, owner, torch = cpu_runtime
    mask = np.zeros((8, 8), np.uint8)
    if kind == 'point': mask[2, 3] = 1
    if kind == 'line': mask[2, :] = 1
    if kind == 'square': mask[2:5, 2:5] = 1
    if kind == 'sixteen_fragmented': mask[::2, ::2] = 1
    project = mock.Mock(return_value=(torch.from_numpy(mask), None, object()))
    monkeypatch.setattr(pta_cuda_masks, 'render_gpu_categorical_item', project)
    actual, remaining, stats = classifier.classify_task(runtime, np.zeros((1, 8, 8), np.uint8),
        [plan()], classifier.GpuClassificationTask(((0, 0, None),)), identity='source')
    assert actual == {('view', 0, 'full', 0): mask_has_yolo_polygon(mask)}
    assert remaining == [] and stats['gpu_jobs'] == stats['gpu_queries'] == stats['count_batches'] == 1
    assert stats['sparse_polygon_checks'] == int(0 < np.count_nonzero(mask) < 16)
    owner.stream.synchronize.assert_called_once()


def test_semantic_uses_independent_projected_intersection(monkeypatch, cpu_runtime):
    runtime, _, torch = cpu_runtime
    foreground = torch.tensor([[1, 1], [0, 0]], dtype=torch.uint8)
    known = torch.tensor([[0, 1], [0, 0]], dtype=torch.uint8)
    project = mock.Mock(return_value=(foreground, known, object()))
    monkeypatch.setattr(pta_cuda_masks, 'render_gpu_categorical_item', project)
    coverage = np.zeros((1, 2, 2), np.uint8)
    result, _, stats = classifier.classify_task(runtime, coverage, [plan(size=2)],
        classifier.GpuClassificationTask(((0, 0, None),), True), identity='source', coverage=coverage)
    assert result == {('view', 0, 'full', 0): True}
    assert project.call_args.args[2] is coverage
    assert stats['sparse_polygon_checks'] == 0


def test_selected_tiles_and_byte_bounded_count_batches(monkeypatch, cpu_runtime):
    runtime, _, torch = cpu_runtime
    monkeypatch.setattr(classifier, 'MAX_PROJECTED_BYTES', 32)
    project = mock.Mock(return_value=(torch.ones((4, 4), dtype=torch.uint8), None, object()))
    monkeypatch.setattr(pta_cuda_masks, 'render_gpu_categorical_item', project)
    result, _, stats = classifier.classify_task(runtime, np.zeros((1, 4, 4), np.uint8),
        [plan(tiles=('a', 'b', 'c'), size=4)],
        classifier.GpuClassificationTask(((0, 0, ('a', 'b', 'c')),)), identity='source')
    assert set(result) == {('view', 0, key, 0) for key in ('a', 'b', 'c')}
    assert [call.args[5] for call in project.call_args_list] == ['a', 'b', 'c']
    assert stats['count_batches'] == 2


def test_unsupported_tile_declines_whole_frame_before_any_projection(monkeypatch, cpu_runtime):
    runtime, _, _ = cpu_runtime
    p = plan(tiles=('legacy',))
    p.tile_layout[0].shared_job = None
    project = mock.Mock(side_effect=AssertionError('unsupported frame launched'))
    monkeypatch.setattr(pta_cuda_masks, 'render_gpu_categorical_item', project)
    result, remaining, stats = classifier.classify_task(runtime, np.zeros((1, 8, 8), np.uint8),
        [p], classifier.GpuClassificationTask(((0, 0, None),)), identity='source')
    assert result == {} and remaining == [(0, 0)] and stats['cpu_jobs'] == 1
    project.assert_not_called()


def test_late_prelaunch_decline_does_not_return_partial_job_keys(monkeypatch, cpu_runtime):
    runtime, _, torch = cpu_runtime
    monkeypatch.setattr(pta_cuda_masks, 'render_gpu_categorical_item', mock.Mock(side_effect=[
        (torch.ones((8, 8), dtype=torch.uint8), None, object()), None]))
    result, remaining, stats = classifier.classify_task(runtime, np.zeros((1, 8, 8), np.uint8),
        [plan(tiles=('a',))], classifier.GpuClassificationTask(((0, 0, None),)), identity='source')
    assert result == {} and remaining == [(0, 0)] and stats['gpu_jobs'] == 0


def test_launched_kernel_failure_propagates_and_fences(monkeypatch, cpu_runtime):
    runtime, owner, _ = cpu_runtime
    monkeypatch.setattr(pta_cuda_masks, 'render_gpu_categorical_item',
                        mock.Mock(side_effect=RuntimeError('kernel failed')))
    with pytest.raises(RuntimeError, match='kernel failed'):
        classifier.classify_task(runtime, np.zeros((1, 8, 8), np.uint8), [plan()],
            classifier.GpuClassificationTask(((0, 0, None),)), identity='source')
    owner.stream.synchronize.assert_called_once()


def test_task_packing_counts_selected_items_and_preserves_frame_atomicity():
    plans = [plan(tiles=('a', 'b'))]
    jobs = [(0, 0), (0, 1), (0, 2)]
    tasks = classifier.build_classification_tasks(plans, jobs, max_queries=4)
    assert len(tasks) == 3
    assert [frame[:2] for task in tasks for frame in task.frames] == jobs
    selected = {('view', index): {'a'} for index in range(3)}
    tasks = classifier.build_classification_tasks(plans, jobs, selected_items=selected, max_queries=4)
    assert len(tasks) == 1 and len(tasks[0].frames) == 3
    with pytest.raises(ValueError, match='Unknown classification'):
        classifier.item_keys(plans[0], ['invalid'])


def test_pool_dedicated_results_settle_errors_before_payload_release():
    p = object.__new__(pta_workers.PersistentRenderPool)
    p.gpu_device_ids = (0,)
    p.results = queue.Queue()
    p.results.put('unrelated rendering callback')
    consumed = []
    response = ({}, [], classifier.classification_stats())
    events = [RuntimeError('worker failed'), response]

    class Results:
        def __next__(self):
            value = events[len(consumed)]
            consumed.append(value)
            if isinstance(value, BaseException): raise value
            return value

    pool = mock.Mock()
    pool.imap_unordered.return_value = Results()
    p._pool = pool
    block = SimpleNamespace(release=lambda: consumed.append('released'))
    p.install_phase = mock.Mock(return_value=SimpleNamespace(
        payload_name='payload', payload_nbytes=10, payload_block=block))
    prep = SimpleNamespace(plans=[plan()], semantic_foreground=False, src=SimpleNamespace(stem='sample'))
    with pytest.raises(RuntimeError, match='GPU foreground classification failed'):
        p.classify_foregrounds(prep, gen=2, jobs=[(0, i) for i in range(65)])
    assert consumed == [events[0], response, 'released']
    assert p.results.get_nowait() == 'unrelated rendering callback'
    assert pool.imap_unordered.call_args.kwargs == {'chunksize': 1}


def test_non_gpu_pool_declines_without_installing_payload():
    p = object.__new__(pta_workers.PersistentRenderPool)
    p._pool, p.gpu_device_ids = None, ()
    p.install_phase = mock.Mock(side_effect=AssertionError('CPU pool created CUDA tasks'))
    assert p.classify_foregrounds(None, gen=0, jobs=[(0, 2)])[:2] == ({}, [(0, 2)])


@pytest.mark.parametrize('failure', ['interrupt', 'aggregation'])
def test_parent_cancellation_or_aggregation_failure_joins_before_payload_release(failure):
    p = object.__new__(pta_workers.PersistentRenderPool)
    p.gpu_device_ids = (0,)
    events = []
    class Results:
        def __next__(self):
            if failure == 'interrupt':
                raise KeyboardInterrupt()
            return {}, [], {}  # Malformed counters fail parent aggregation.
    p._pool = SimpleNamespace(imap_unordered=lambda *a, **k: Results())
    p.close = lambda **kwargs: events.append(('terminate_join', kwargs))
    p.install_phase = lambda **kwargs: SimpleNamespace(
        payload_name='p', payload_nbytes=1,
        payload_block=SimpleNamespace(release=lambda: events.append('release')))
    prep = SimpleNamespace(plans=[plan()], semantic_foreground=False, src=SimpleNamespace(stem='sample'))
    with pytest.raises(KeyboardInterrupt if failure == 'interrupt' else KeyError):
        p.classify_foregrounds(prep, gen=0, jobs=[(0, 0)])
    assert events == [('terminate_join', {'terminate': True}), 'release']


def test_pool_progress_counts_completed_or_declined_frames_and_parent_stays_cuda_cold():
    p = object.__new__(pta_workers.PersistentRenderPool)
    p.gpu_device_ids = (0,)
    first, second = classifier.classification_stats(), classifier.classification_stats()
    first['gpu_jobs'], second['cpu_jobs'] = 64, 1
    p._pool = SimpleNamespace(imap_unordered=lambda *a, **kw: iter([
        ({('view', i, 'full', 0): True for i in range(64)}, [], first),
        ({}, [(0, 64)], second)]))
    block = SimpleNamespace(release=mock.Mock())
    p.install_phase = mock.Mock(return_value=SimpleNamespace(
        payload_name='p', payload_nbytes=1, payload_block=block))
    progress = mock.Mock()
    context = mock.MagicMock()
    context.__enter__.return_value = progress
    prep = SimpleNamespace(plans=[plan()], semantic_foreground=False, src=SimpleNamespace(stem='sample'))
    with (mock.patch.object(pta_workers, 'tqdm', return_value=context) as bar,
          mock.patch.object(pta_workers, '_gpu_runtime_for_worker', side_effect=AssertionError('parent CUDA'))):
        truth, remaining, stats = p.classify_foregrounds(prep, gen=0, jobs=[(0, i) for i in range(65)])
    assert len(truth) == 64 and remaining == [(0, 64)]
    assert stats['gpu_jobs'] == 64 and stats['cpu_jobs'] == 1
    assert progress.update.call_args_list == [mock.call(64), mock.call(1)]
    assert bar.call_args.kwargs['total'] == 65
    assert bar.call_args.kwargs['desc'].endswith('sample [CUDA]')
    context.__exit__.assert_called_once()
    block.release.assert_called_once()

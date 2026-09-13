"""Policy projection groups retain independent outputs under bounded parent leases."""
from __future__ import annotations

import contextlib
import ast
import io
import queue
import tempfile
import unittest
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.test_tta_scheduler_boundary import _bind_callbacks, _scheduler, _state
from XTA.geometry import ViewInfo
from XTA.interpolation import _DirectUnionBackingLease


def policy_task(group, *, ratio=4, size=100, task_id=0, slice_start=0):
    members = []
    for index in range(ratio):
        view = ViewInfo(name=f'spherical_{group}__policy_{index}', family='spherical',
                        num_slices=4, src_h=5, src_w=5, pad_mode='clamp')
        members.append(dict(task_id=task_id, kind='fullframe', result_mode='file',
                            model_name='model', view=view, processing_shape=(1, 1, size),
                            bounded_parent_admission=True, disable_runtime_split=True,
                            gpu_eligible=True, slice_count=1, slice_start=slice_start,
                            result_mask_path=f'chunk-{task_id}-{index}.mask',
                            result_conf_path=f'chunk-{task_id}-{index}.conf'))
    members[0]['augmentation_pass_tasks'] = members[1:]
    return members[0]


class PolicyParentAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.directory = Path(self.temporary.name)
        self.state = _state()
        self.admitted = []

    def scheduler(self, *, size=100, **overrides):
        def ensure(model, view):
            key = (model, view.name)
            if key in self.state.direct_union_backing_leases:
                return
            self.admitted.append(key)
            self.state.baseline_union_paths[key] = self.directory / f'{view.name}.parent'
            self.state.direct_union_backing_leases[key] = _DirectUnionBackingLease(key, size)
            self.state.direct_union_inference_views.add(key)
            self.state.direct_union_inference_bytes[key] = size
        options = dict(ensure_baseline_workspaces=ensure,
                       direct_union_inference_view_limit=2,
                       direct_union_inference_byte_limit=800,
                       direct_union_total_dense_byte_limit=800)
        options.update(overrides)
        return _scheduler(self.directory, state=self.state, input_overrides=options)

    def handoff(self, key):
        lease = self.state.direct_union_backing_leases[key]
        lease.transition('inference', 'postprocess')
        self.state.direct_union_inference_views.remove(key)
        self.state.direct_union_inference_bytes.pop(key)
        self.state.direct_union_postprocess_views.add(key)
        self.state.direct_union_postprocess_bytes[key] = lease.nbytes

    def retire(self, key):
        self.state.direct_union_backing_leases.pop(key).release('postprocess')
        self.state.direct_union_postprocess_views.remove(key)
        self.state.direct_union_postprocess_bytes.pop(key)

    def test_admission_charges_all_passes_without_redirecting_worker_files(self):
        scheduler = self.scheduler()
        task = policy_task(0)
        passes = (task, *task['augmentation_pass_tasks'])
        paths = [(member['result_mask_path'], member['result_conf_path']) for member in passes]
        self.assertEqual(scheduler.direct_union_task_bytes(task), 400)
        scheduler.activate_direct_union_task(task)
        scheduler.activate_direct_union_task(task)
        self.assertEqual(len(self.admitted), 4)
        self.assertEqual(sum(self.state.direct_union_inference_bytes.values()), 400)
        self.assertEqual([(member['result_mask_path'], member['result_conf_path']) for member in passes], paths)
        self.assertTrue(all(member['result_mode'] == 'file' for member in passes))
        self.assertEqual(len({self.state.baseline_union_paths[key] for key in self.admitted}), 4)

    def test_view_slots_count_groups_while_byte_limits_count_every_pass(self):
        scheduler = self.scheduler()
        first, second, third = (policy_task(index) for index in range(3))
        scheduler.activate_direct_union_task(first)
        self.assertTrue(scheduler.direct_union_task_admissible(second))
        scheduler.activate_direct_union_task(second)
        self.assertEqual(len(self.state.direct_union_inference_views), 8)
        self.assertFalse(scheduler.direct_union_task_admissible(third))
        self.assertTrue(scheduler.direct_union_task_admissible(first))

    def test_sibling_postprocess_and_tile_lifetimes_keep_their_byte_credit(self):
        scheduler = self.scheduler(direct_union_total_dense_byte_limit=400)
        first, second = policy_task(0), policy_task(1)
        scheduler.activate_direct_union_task(first)
        keys = list(self.admitted)
        for key in keys:
            self.handoff(key)
        # Neither inference completion nor retirement of only the base frees the
        # three policy canvases still owned by publication/tile consolidation.
        self.assertFalse(scheduler.direct_union_task_admissible(second))
        for key in keys[:-1]:
            self.retire(key)
            self.assertFalse(scheduler.direct_union_task_admissible(second))
        self.retire(keys[-1])
        self.assertTrue(scheduler.direct_union_task_admissible(second))
        scheduler.activate_direct_union_task(second)

    def test_partial_group_handoff_cannot_reopen_any_sibling(self):
        scheduler = self.scheduler()
        task = policy_task(0)
        scheduler.activate_direct_union_task(task)
        self.handoff(self.admitted[-1])
        with self.assertRaisesRegex(RuntimeError, 'postprocess-owned'):
            scheduler.direct_union_task_admissible(task)

    def test_oversize_inference_group_runs_alone_but_cannot_exceed_total_limit(self):
        scheduler = self.scheduler(direct_union_inference_byte_limit=100,
                                   direct_union_total_dense_byte_limit=600)
        first, second = policy_task(0), policy_task(1)
        self.assertTrue(scheduler.direct_union_task_admissible(first))
        scheduler.activate_direct_union_task(first)
        self.assertFalse(scheduler.direct_union_task_admissible(second))
        oversized = policy_task(2, ratio=7)
        with self.assertRaisesRegex(RuntimeError, 'exceeding.*bounded parent dense limit'):
            scheduler.direct_union_task_admissible(oversized)

    def test_unmarked_file_mode_is_not_silently_admitted(self):
        scheduler = self.scheduler()
        task = policy_task(0, ratio=1)
        task.pop('bounded_parent_admission')
        self.assertIsNone(scheduler.direct_union_task_key(task))
        self.assertTrue(scheduler.direct_union_task_admissible(task))
        scheduler.activate_direct_union_task(task)
        self.assertFalse(self.admitted)

    def test_missing_sibling_marker_is_rejected_before_any_allocation(self):
        scheduler = self.scheduler()
        task = policy_task(0)
        task['augmentation_pass_tasks'][-1].pop('bounded_parent_admission')
        with self.assertRaisesRegex(RuntimeError, 'independently marked'):
            scheduler.activate_direct_union_task(task)
        self.assertFalse(self.admitted)

    def test_confidence_and_tile_category_canvases_are_charged_per_pass(self):
        scheduler = self.scheduler(min_conf=0.1, dense_tiling_active=True, nrrd_layers_needed=True)
        self.assertEqual(scheduler.direct_union_task_bytes(policy_task(0)), 4 * 100 * 5)

    def test_policy_spherical_groups_keep_projection_cache_locality(self):
        scheduler = self.scheduler()
        task = policy_task(0)
        self.assertEqual(scheduler.spherical_locality_parent(task), ('model', task['view'].name))
        task['kind'] = 'tile'
        self.assertIsNone(scheduler.spherical_locality_parent(task))
        self.assertIsNone(scheduler.direct_union_task_key(task))

    def test_many_groups_complete_on_four_workers_with_retirement_backpressure(self):
        scheduler = self.scheduler(gpu_device_count=4, direct_union_inference_view_limit=4)
        task_id = 0
        for group in range(12):
            for chunk in range(4):
                task = policy_task(group, task_id=task_id, slice_start=chunk)
                self.state.gpu_worker_tasks_by_id[task_id] = task
                self.state.gpu_worker_pending_task_ids.append(task_id)
                for member in (task, *task['augmentation_pass_tasks']):
                    key = ('model', member['view'].name)
                    self.state.fullframe_remaining[key] = self.state.fullframe_remaining.get(key, 0) + 1
                task_id += 1
        self.state.gpu_worker_total_tasks = task_id
        self.state.gpu_worker_next_dynamic_task_id = task_id
        self.state.gpu_task_queues.update({worker: queue.Queue() for worker in range(4)})
        workers_used, retirements = set(), []
        peak_bytes = 0

        def complete(task, _stats):
            for member in (*task['augmentation_pass_tasks'], task):
                key = ('model', member['view'].name)
                self.state.fullframe_remaining[key] -= 1
                if self.state.fullframe_remaining[key] == 0:
                    self.handoff(key)
                    # The real pipeline refills synchronously after each sibling
                    # hands off, before the base finishes its own result callback.
                    scheduler.dispatch_gpu_worker_inference_window()

        _bind_callbacks(scheduler, fullframe=mock.Mock(side_effect=complete))
        with contextlib.redirect_stdout(io.StringIO()):
            scheduler.dispatch_gpu_worker_inference_window()
            iterations = 0
            while self.state.gpu_worker_pending_task_ids or any(not q.empty() for q in self.state.gpu_task_queues.values()):
                iterations += 1
                self.assertLess(iterations, 200)
                progressed = False
                for worker, worker_queue in self.state.gpu_task_queues.items():
                    if worker_queue.empty():
                        continue
                    task = worker_queue.get_nowait()
                    workers_used.add(worker)
                    scheduler.process_one_worker_result(dict(type='result', gpu_index=worker,
                        task_id=task['task_id'], ok=True, stats={}))
                    progressed = True
                    live = sum(self.state.direct_union_inference_bytes.values()) + sum(self.state.direct_union_postprocess_bytes.values())
                    peak_bytes = max(peak_bytes, live)
                    self.assertLessEqual(live, 800)
                if not progressed:
                    self.assertTrue(self.state.direct_union_postprocess_views)
                    self.assertFalse(self.state.direct_union_inference_views)
                    for key in list(self.state.direct_union_postprocess_views):
                        self.retire(key)
                        retirements.append(key)
                    scheduler.dispatch_gpu_worker_inference_window()
            for key in list(self.state.direct_union_postprocess_views):
                self.retire(key)
                retirements.append(key)
        self.assertEqual(workers_used, set(range(4)))
        self.assertEqual(peak_bytes, 800)
        self.assertEqual(self.state.gpu_worker_results_collected, 48)
        self.assertEqual(len(self.admitted), 48)
        self.assertEqual(set(retirements), set(self.admitted))
        self.assertFalse(self.state.direct_union_backing_leases)
        self.assertTrue(all(value == 0 for value in self.state.fullframe_remaining.values()))
        self.assertEqual(scheduler.process_quiescence_issues(), {})


class PolicyParentWorkspaceIntegrationTests(unittest.TestCase):
    """Run production allocation, file-result consumption, and handoff on tiny data."""

    def test_file_policy_parents_survive_chunk_cleanup_and_handoff_independently(self):
        from tests.test_terminal_component_refs import _function
        from XTA import pipeline, runtime

        state = _state()
        shape = (4, 5, 6)
        task = policy_task(0)
        members = (task, *task['augmentation_pass_tasks'])
        keys = {('model', member['view'].name) for member in members}
        submitted = []

        def submit(callback):
            # Hold the actual postprocess closure and its arrays without running
            # image processing; this isolates the production ownership boundary.
            submitted.append(callback)
            return Future()

        with tempfile.TemporaryDirectory() as td, contextlib.redirect_stdout(io.StringIO()):
            root = Path(td)
            namespace = dict(vars(pipeline))
            namespace.update(temp_dir=root, worker_direct_union_active=False,
                policy_settings=SimpleNamespace(enabled=True), bounded_policy_parent_keys=keys,
                args=SimpleNamespace(imgsz=8, min_conf=1., interpolation_distance=0),
                dense_tiling_active=False, nrrd_layers_needed=True,
                baseline_union_by_model_view={}, baseline_confmap_by_model_view={},
                baseline_slice_locks_by_model_view={},
                baseline_union_paths=state.baseline_union_paths,
                baseline_confmap_paths=state.baseline_confmap_paths,
                direct_union_backing_leases=state.direct_union_backing_leases,
                direct_union_inference_views=state.direct_union_inference_views,
                direct_union_postprocess_views=state.direct_union_postprocess_views,
                direct_union_inference_bytes=state.direct_union_inference_bytes,
                direct_union_postprocess_bytes=state.direct_union_postprocess_bytes,
                view_processing_volume_shape=lambda *_args: shape,
                fullframe_remaining={key: 2 for key in keys},
                pending_azimuthal_padding_by_parent={},
                view_processing_submitted=set(), d1_view_shadow_path_by_parent={},
                view_device_hole_filled_slices={}, view_slice_meta={},
                input_T=4, input_H=5, input_W=6,
                parent_postprocess_executor=SimpleNamespace(submit=submit),
                view_processing_futures={}, gpu_worker_pending_task_ids=[],
                slice_postprocess_workers=1, keep_temp_artifacts=False)
            source = Path(pipeline.__file__).read_text(encoding='utf-8')
            ensure = _function(source, '_ensure_baseline_workspaces', namespace)
            _function(source, '_merge_pending_azimuthal_padding_for_parent', namespace)
            _function(source, '_submit_view_prepare', namespace)
            _function(source, '_finalize_fullframe_view_after_worker', namespace)
            handler = next(node for node in ast.walk(ast.parse(source))
                           if isinstance(node, ast.FunctionDef) and node.name == '_handle_fullframe_worker_result')
            start = next(index for index, node in enumerate(handler.body)
                         if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call)
                         and isinstance(node.value.func, ast.Name)
                         and node.value.func.id == '_ensure_baseline_workspaces')
            consume = compile(ast.fix_missing_locations(ast.Module(body=handler.body[start:], type_ignores=[])),
                              '<actual file-mode fullframe result consumption>', 'exec')
            scheduler = _scheduler(root, state=state, input_overrides={
                'ensure_baseline_workspaces': ensure, 'min_conf': 1.,
                'direct_union_inference_byte_limit': 4096,
                'direct_union_total_dense_byte_limit': 4096})
            for member in members:
                member['processing_shape'] = shape
            parent_masks = parent_confmaps = None
            try:
                with mock.patch.object(runtime, 'should_use_in_memory_workspace', return_value=False), \
                        mock.patch.object(runtime, 'memfd_workspace_enabled', return_value=False):
                    scheduler.activate_direct_union_task(task)
                self.assertEqual(set(state.direct_union_backing_leases), keys)
                self.assertEqual(sum(state.direct_union_inference_bytes.values()), 4 * 2 * int(np.prod(shape)))
                parent_masks = dict(namespace['baseline_union_by_model_view'])
                parent_confmaps = dict(namespace['baseline_confmap_by_model_view'])
                parent_paths = dict(state.baseline_union_paths)
                expected_masks = {key: np.zeros(shape, np.uint8) for key in keys}
                expected_conf = {key: np.zeros(shape, np.uint8) for key in keys}
                for chunk in range(2):
                    # Repeated dispatch must reuse all four existing parents.
                    scheduler.activate_direct_union_task(task)
                    for index in (1, 2, 3, 0):
                        member = members[index]
                        key = ('model', member['view'].name)
                        start_slice = 2 * chunk
                        member.update(slice_start=start_slice, slice_count=2,
                            result_mask_path=str(root / f'chunk-{chunk}-pass-{index}.mask'),
                            result_conf_path=str(root / f'chunk-{chunk}-pass-{index}.conf'))
                        mask = np.zeros((2, 5, 6), np.uint8)
                        conf = np.zeros_like(mask)
                        mask[:, index, chunk] = 1
                        conf[:, index, chunk] = 20 + index
                        mask.tofile(member['result_mask_path'])
                        conf.tofile(member['result_conf_path'])
                        expected_masks[key][start_slice:start_slice + 2] = mask
                        expected_conf[key][start_slice:start_slice + 2] = conf
                        namespace.update(task=member, view=member['view'], model_name_s='model')
                        exec(consume, namespace)
                        self.assertFalse(Path(member['result_mask_path']).exists())
                        self.assertFalse(Path(member['result_conf_path']).exists())
                        self.assertTrue(parent_paths[key].exists())
                        self.assertFalse(parent_masks[key]._mmap.closed)
                        self.assertFalse(parent_confmaps[key]._mmap.closed)
                        np.testing.assert_array_equal(parent_masks[key], expected_masks[key])
                        np.testing.assert_array_equal(parent_confmaps[key], expected_conf[key])
                        self.assertEqual(state.direct_union_backing_leases[key].phase,
                                         'inference' if chunk == 0 else 'postprocess')
                        if chunk == 1 and index != 0:
                            self.assertEqual(state.direct_union_backing_leases[('model', members[0]['view'].name)].phase,
                                             'inference')
                self.assertEqual(len(submitted), 4)
                self.assertFalse(state.direct_union_inference_views)
                self.assertEqual(state.direct_union_postprocess_views, keys)
                self.assertEqual(sum(state.direct_union_postprocess_bytes.values()), 4 * 2 * int(np.prod(shape)))
                self.assertFalse(namespace['baseline_union_by_model_view'])
                self.assertTrue(all(value == 0 for value in namespace['fullframe_remaining'].values()))
                self.assertEqual(len({id(value) for value in parent_masks.values()}), 4)
            finally:
                for mapping in (parent_masks, parent_confmaps):
                    for value in (mapping or {}).values():
                        runtime.close_memmap_array_without_flush(value)


if __name__ == '__main__':
    unittest.main()

"""Production-shaped native parent admission without allocating production data."""
from __future__ import annotations

import ast
import contextlib
import gc
import io
import os
import queue
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from tests.native_detector_test_support import detector_state
from XTA.native_scheduler import PreparedWorkScheduler
from tests.test_terminal_component_refs import _function
from XTA import pipeline, publication_memory, runtime, workers
from XTA.geometry import ViewInfo
from XTA.interpolation import _DirectUnionBackingLease

GIB = 1024 ** 3
SHAPE = (1212, 3072, 3072)
PARENT_BYTES = 1212 * 3072 * 3072


def native_tasks(count=120, mode='direct_union'):
    tasks = []
    for parent in range(count):
        view = ViewInfo(name=f'spherical_{parent}__tta_a0', family='spherical',
                        num_slices=1212, src_h=3072, src_w=3072, pad_mode='clamp')
        for chunk in range(4):
            tasks.append(dict(task_id=len(tasks), kind='fullframe', model_name='model',
                              view=view, result_mode=mode, processing_shape=SHAPE,
                              slice_start=chunk * 303, slice_count=303,
                              union_num_slices=1212, gpu_eligible=True,
                              disable_runtime_split=True))
    return tasks


class SphericalHostAdmissionTests(unittest.TestCase):
    def test_d1_activation_preserves_explicit_shared_union_capability(self):
        tree = ast.parse(Path(pipeline.__file__).read_text(encoding='utf-8'))
        main = next(node for node in tree.body if isinstance(node, ast.FunctionDef)
                    and node.name == '_main_impl')
        block = next(node for node in main.body if isinstance(node, ast.If)
                     and isinstance(node.test, ast.Name)
                     and node.test.id == 'v1613_d1_owner_active')
        code = compile(ast.fix_missing_locations(ast.Module(body=[block], type_ignores=[])),
                       '<D1 activation>', 'exec')
        for enabled in (False, True):
            env = dict(v1613_d1_owner_active=True, gpu_worker_direct_union_active=enabled,
                       legacy_d1_model_eligible=True)
            with contextlib.redirect_stdout(io.StringIO()):
                exec(code, env)
            self.assertIs(env['gpu_worker_direct_union_active'], enabled)

    def test_120_native_parents_finish_on_four_workers_with_retirement_backpressure(self):
        from tests.native_scheduler_support import NativeRun
        import time

        admitted, retired, completed = {}, set(), {}
        seen, used_workers = set(), set()
        peak_inference = peak_bytes = pressure_stops = 0
        with NativeRun(capacity=256 * GIB, tickets=24, inference_capacity=128 * GIB,
                       preparation_window=4, workers=4, window=4) as run:
            for parent in range(120):
                run.declare(parent, tasks=4, bytes=PARENT_BYTES)
            deadline = time.monotonic() + 30
            while len(seen) < 480:
                self.assertLess(time.monotonic(), deadline, run.owner.snapshot())
                progressed = False
                while (ready := run.owner.wait_preparation(0)) is not None:
                    parent, generation = ready
                    admitted[parent] = run.parents[parent] = generation
                    for chunk in range(4):
                        run.submit(parent * 4 + chunk, parent, descriptor={
                            'family': 'spherical', 'view': parent,
                            'slice_start': chunk * 303, 'slice_count': 303})
                    progressed = True
                for worker in range(4):
                    packet = run.take(worker, timeout=0)
                    if packet is None:
                        continue
                    lease, task = packet
                    identity = (lease.parent_id, task['slice_start'], task['slice_count'])
                    self.assertNotIn(identity, seen)
                    seen.add(identity); used_workers.add(worker)
                    run.finish(lease)
                    progressed = True
                for _kind, lease, _payload in run.consume_events():
                    if _kind == 'final':
                        run.owner.retire_task(lease.task_id, lease.generation)
                while (published := run.owner.wait_publication(0)) is not None:
                    parent, generation = published
                    self.assertEqual(admitted[parent], generation)
                    run.owner.adopt_postprocess(parent, generation)
                    completed[parent] = generation
                    progressed = True
                state = run.owner.snapshot()
                peak_inference = max(peak_inference, state['preparing_parents'])
                peak_bytes = max(peak_bytes, state['bytes_used'])
                self.assertLessEqual(state['preparing_parents'], 4)
                self.assertLessEqual(state['bytes_used'], 256 * GIB)
                if not progressed:
                    if (state['blocked_parents'] and not state['active'] and not state['preparing_parents']
                            and state['bytes_used'] + PARENT_BYTES > 256 * GIB):
                        self.assertTrue(completed)
                        pressure_stops += 1
                        for parent, generation in completed.items():
                            run.owner.retire_parent(parent, generation)
                            retired.add(parent)
                        completed.clear()
                    else:
                        time.sleep(.001)
            # The last final can precede its publication notification.
            while len(completed) + len(retired) < 120:
                published = run.owner.wait_publication(2)
                self.assertIsNotNone(published)
                parent, generation = published
                run.owner.adopt_postprocess(parent, generation)
                completed[parent] = generation
            for _kind, lease, _payload in run.consume_events():
                if _kind == 'final':
                    run.owner.retire_task(lease.task_id, lease.generation)
            for parent, generation in completed.items():
                run.owner.retire_parent(parent, generation)
                retired.add(parent)
            self.assertGreater(pressure_stops, 0)
            self.assertGreater(peak_bytes, 128 * GIB)
            self.assertLessEqual(peak_inference, 4)
            self.assertEqual(used_workers, {0, 1, 2, 3})
            self.assertEqual(run.owner.snapshot()['final'], 480)
            self.assertEqual(len(admitted), 120)
            self.assertEqual(retired, set(admitted))
            self.assertEqual(seen, {(parent, chunk * 303, 303)
                                   for parent in range(120) for chunk in range(4)})
            self.assertEqual(run.owner.snapshot()['bytes_used'], 0)
            self.assertEqual(run.owner.snapshot()['tickets_used'], 0)

    def test_native_dense_window_reduces_publication_budget_without_spherical_grants(self):
        reserve = publication_memory.native_fullframe_dense_reserve(native_tasks(), total_dense_limit=256 * GIB)
        self.assertEqual(reserve, 256 * GIB)
        shape = (1931, 3064, 3022)
        plain = publication_memory.retained_payload_plan([shape] * 50, 975 * GIB, 4, 12, 256 * 1024 ** 2)
        bounded = publication_memory.retained_payload_plan([shape] * 50, 975 * GIB, 4, 12, 256 * 1024 ** 2,
                                                            native_dense_reserve_bytes=reserve)
        self.assertEqual(len(bounded['grants']), 50)
        self.assertEqual(bounded['reserve_bytes'] - plain['reserve_bytes'], reserve)
        self.assertEqual(plain['budget_bytes'] - bounded['budget_bytes'], reserve // 2)

    def test_explicit_file_optout_allows_small_work_and_refuses_unbounded_native_parents(self):
        with self.assertRaisesRegex(RuntimeError, 'File-mode.*1278.3 GiB'):
            publication_memory.native_fullframe_dense_reserve(native_tasks(mode='file'), total_dense_limit=256 * GIB)
        self.assertEqual(publication_memory.native_fullframe_dense_reserve(native_tasks(2, 'file'),
                                                                          total_dense_limit=256 * GIB), 2 * PARENT_BYTES)
        # A single oversized geometry retains the established emergency lane.
        self.assertEqual(publication_memory.native_fullframe_dense_reserve(native_tasks(1, 'file'),
                                                                          total_dense_limit=1), PARENT_BYTES)

    def test_hybrid_parents_reserve_potential_native_backings_but_d1_does_not(self):
        tasks = native_tasks(2, runtime.HYBRID_DEFERRED_RESULT_MODE) + native_tasks(1, 'd1_owner')
        self.assertEqual(publication_memory.native_fullframe_dense_reserve(tasks,
                                                                          total_dense_limit=256 * GIB), 2 * PARENT_BYTES)


class SharedNativeWorkspaceTests(unittest.TestCase):
    """Execute actual parent allocation and worker open/close blocks on tiny data."""

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.state = detector_state()
        self.view = ViewInfo(name='spherical_test__tta_a0', family='spherical',
                             num_slices=4, src_h=5, src_w=6, pad_mode='clamp')
        self.shape = (4, 5, 6)
        self.namespace = dict(vars(pipeline))
        self.namespace.update(temp_dir=self.root, worker_direct_union_active=True,
            sam_context=None,sam_parent_staging=None,
            cpu_input_prefetch=None, sam_cpu_futures={'source': None},
            input_volume_rgb=np.zeros(self.shape, np.uint8),
            parent_transient_admission=SimpleNamespace(in_use=0),
            gray_capture_reserve_bytes=0, cpu_input_pinned_reserve_bytes=0,
            _sam_parents_ready=lambda:True,
            sam_cpu_max_detector_parent=0,
            interpolation_settings=SimpleNamespace(backend='sdf',extrapolation_enabled=False),
            policy_settings=SimpleNamespace(enabled=False),
            bounded_policy_parent_keys=set(),
            args=SimpleNamespace(imgsz=8, min_conf=1., interpolation_distance=0), dense_tiling_active=False,
            nrrd_layers_needed=False, component_layers_needed=False,
            baseline_union_by_model_view={},
            baseline_confmap_by_model_view={}, baseline_slice_locks_by_model_view={},
            view_slice_meta={},
            baseline_union_paths=self.state.baseline_union_paths,
            baseline_confmap_paths=self.state.baseline_confmap_paths,
            direct_union_backing_leases=self.state.direct_union_backing_leases,
            direct_union_inference_views=self.state.direct_union_inference_views,
            direct_union_postprocess_views=self.state.direct_union_postprocess_views,
            direct_union_inference_bytes=self.state.direct_union_inference_bytes,
            direct_union_postprocess_bytes=self.state.direct_union_postprocess_bytes,
            view_processing_volume_shape=lambda *_args: self.shape)
        self.namespace['view_prepare_leases'] = pipeline.ViewPrepareLeaseState(
            self.state.direct_union_backing_leases, self.state.direct_union_inference_views,
            self.state.direct_union_inference_bytes, self.state.direct_union_postprocess_views,
            self.state.direct_union_postprocess_bytes)
        source = Path(pipeline.__file__).read_text(encoding='utf-8')
        for helper in ('_sam_parent_requires_staging', '_detector_future_ram_rights',
                       '_plan_baseline_workspace', '_discard_baseline_bundles',
                       '_build_baseline_workspaces', '_adopt_baseline_workspaces'):
            _function(source, helper, self.namespace)
        self.ensure = _function(source, '_ensure_baseline_workspaces', self.namespace)
        tree = ast.parse(Path(workers.__file__).read_text(encoding='utf-8'))
        fn = next(node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)
                  and node.name == 'run_prediction_volume_in_worker')
        outer = next(node for node in ast.walk(fn) if isinstance(node, ast.Try) and node.finalbody
                     and isinstance(node.body[0], ast.If)
                     and "task.get('result_mode', 'file')" in ast.unparse(node.body[0].test)
                     and any(isinstance(child, ast.Name) and child.id == 'DeviceOnlyRadialTarget'
                             for child in ast.walk(node.body[0])))
        self.worker_open = compile(ast.fix_missing_locations(ast.Module(body=[outer.body[0]], type_ignores=[])),
                                   '<worker shared union open>', 'exec')
        self.worker_close = compile(ast.fix_missing_locations(ast.Module(body=outer.finalbody, type_ignores=[])),
                                    '<worker shared union close>', 'exec')

    def close_parent(self):
        for name in ('baseline_union_by_model_view', 'baseline_confmap_by_model_view'):
            mapping = self.namespace[name]
            for value in mapping.values():
                path = Path(str(value.filename)) if isinstance(value, np.memmap) else None
                if path is not None and str(path).startswith('/proc/'):
                    path = None
                runtime.close_memmap_array_without_flush(value, unlink_path=path)
            mapping.clear()
        value = None
        gc.collect()

    def run_shared_worker_windows(self, memfd):
        owner = PreparedWorkScheduler(77431, 2 * int(np.prod(self.shape)), 1)
        self.addCleanup(owner.close)
        self.addCleanup(self.close_parent)
        owner.declare_parent(1, 0, 2 * int(np.prod(self.shape)))
        self.assertEqual(owner.wait_preparation(2)[0], 1)
        with mock.patch.object(runtime, 'should_use_in_memory_workspace', return_value=memfd), \
                mock.patch.object(runtime, 'memfd_workspace_enabled', return_value=memfd), \
                contextlib.redirect_stdout(io.StringIO()):
            self.ensure('model', self.view)
            for start, fail in ((0, False), (2, True)):
                task = dict(kind='fullframe', result_mode='direct_union', model_name='model',
                            view=self.view, processing_shape=self.shape, union_num_slices=4)
                key = ('model', self.view.name)
                task.update(result_mask_path=str(self.state.baseline_union_paths[key]),
                            result_conf_path=str(self.state.baseline_confmap_paths[key]))
                env = dict(vars(workers))
                env.update(task=task, slice_count=2, slice_offset=start, processing_h=5, processing_w=6,
                    result_shape=(2, 5, 6), result_mask=None, result_conf=None, result_mask_full=None,
                    result_conf_full=None, azimuthal_padding_mask=None, azimuthal_padding_conf=None,
                    source_mm=None, source=None, prepared_input_mm=None,
                    startup_source_started=None, startup_source_prepared=False,
                    startup_device='cpu')
                try:
                    exec(self.worker_open, env)
                    env['result_mask'][:] = start + 1
                    env['result_conf'][:] = start + 2
                    if fail:
                        raise RuntimeError('inference failed after native write')
                except RuntimeError as error:
                    self.assertEqual(str(error), 'inference failed after native write')
                finally:
                    exec(self.worker_close, env)
                self.assertFalse(env['result_mask_full']._mmap.closed)
                self.assertFalse(env['result_conf_full']._mmap.closed)
                self.assertEqual(int(env['result_mask_full'][start, 0, 0]), start + 1)
                key = ('model', self.view.name)
                parent = self.namespace['baseline_union_by_model_view'][key]
                self.assertFalse(parent._mmap.closed)
                self.assertEqual(self.state.direct_union_backing_leases[key].phase, 'inference')
            expected = np.ones(self.shape, np.uint8)
            expected[2:] = 3
            np.testing.assert_array_equal(parent, expected)
            self.assertEqual(self.state.direct_union_inference_bytes[key], 2 * int(np.prod(self.shape)))
            if memfd:
                self.assertTrue(str(self.state.baseline_union_paths[key]).startswith('/proc/'))
            else:
                self.assertTrue(self.state.baseline_union_paths[key].is_file())

    def test_shared_path_windows_and_worker_failure_preserve_parent_owner(self):
        self.run_shared_worker_windows(False)

    @unittest.skipUnless(hasattr(os, 'memfd_create'), 'Linux memfd unavailable')
    def test_shared_memfd_windows_and_worker_failure_preserve_parent_owner(self):
        self.run_shared_worker_windows(True)

    def test_partial_parent_allocation_failure_closes_mapping_before_any_lease_exists(self):
        allocations = []

        def allocate(**kwargs):
            if allocations:
                raise RuntimeError('confidence allocation failed')
            value = runtime.allocate_workspace_array(**kwargs)
            allocations.append(value)
            return value

        self.namespace['allocate_workspace_array'] = allocate
        with mock.patch.object(runtime, 'should_use_in_memory_workspace', return_value=False), \
                contextlib.redirect_stdout(io.StringIO()), \
                self.assertRaisesRegex(RuntimeError, 'confidence allocation failed'):
            self.ensure('model', self.view)
        self.assertFalse(allocations[0]._mmap.closed)
        self.assertEqual(int(allocations[0].sum()), 0)
        self.assertFalse(self.state.direct_union_backing_leases)
        self.assertFalse(self.state.baseline_union_paths)
        failed_path = Path(str(allocations[0].filename))
        self.namespace.pop('allocate_workspace_array')
        allocations.clear()
        gc.collect()
        runtime.wait_for_retired_memmap_unlinks(path=failed_path)
        self.assertFalse(list(self.root.rglob('*.dat')))


if __name__ == '__main__':
    unittest.main()

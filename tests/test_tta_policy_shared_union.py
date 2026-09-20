"""Independent policy shared unions retain atomic bounded group admission."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from tests.test_tta_policy_parent_admission import policy_task
from tests.test_tta_scheduler_boundary import _scheduler, _state
from XTA.interpolation import _DirectUnionBackingLease


SHAPE = (4, 5, 5)
PARENT_BYTES = int(np.prod(SHAPE)) * 2


def shared_group(index, *, slice_start=0, ratio=4):
    task = policy_task(index, ratio=ratio)
    for member in (task, *task['augmentation_pass_tasks']):
        member.update(result_mode='direct_union', processing_shape=SHAPE,
                      union_num_slices=SHAPE[0], slice_start=slice_start, slice_count=2)
    return task


class SharedPolicyUnionSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.state = _state()
        self.masks = {}
        self.confidence = {}
        self.addCleanup(self.close_arrays)

    def close_arrays(self):
        for mapping in (self.masks, self.confidence):
            for array in mapping.values():
                array._mmap.close()

    def ensure(self, model, view):
        key = (model, view.name)
        if key in self.state.direct_union_backing_leases:
            return
        for suffix, arrays, paths in (
            ('mask', self.masks, self.state.baseline_union_paths),
            ('conf', self.confidence, self.state.baseline_confmap_paths),
        ):
            path = self.root / f'{view.name}.{suffix}'
            arrays[key] = np.memmap(path, mode='w+', dtype=np.uint8, shape=SHAPE)
            arrays[key][...] = 0
            paths[key] = path
        self.state.direct_union_backing_leases[key] = _DirectUnionBackingLease(key, PARENT_BYTES)
        self.state.direct_union_inference_views.add(key)
        self.state.direct_union_inference_bytes[key] = PARENT_BYTES

    def scheduler(self, **overrides):
        options = dict(ensure_baseline_workspaces=self.ensure, min_conf=0.1,
                       direct_union_inference_view_limit=2,
                       direct_union_inference_byte_limit=8 * PARENT_BYTES,
                       direct_union_total_dense_byte_limit=8 * PARENT_BYTES)
        options.update(overrides)
        return _scheduler(self.root, state=self.state, input_overrides=options)

    def handoff(self, key):
        lease = self.state.direct_union_backing_leases[key]
        lease.transition('inference', 'postprocess')
        self.state.direct_union_inference_views.remove(key)
        self.state.direct_union_inference_bytes.pop(key)
        self.state.direct_union_postprocess_views.add(key)
        self.state.direct_union_postprocess_bytes[key] = lease.nbytes

    def test_every_shared_pass_receives_its_own_full_parent_targets(self):
        scheduler = self.scheduler()
        task = shared_group(0)
        scheduler.activate_direct_union_task(task)
        members = (task, *task['augmentation_pass_tasks'])
        mask_paths, conf_paths = set(), set()
        for member in members:
            key = ('model', member['view'].name)
            self.assertEqual(member['result_mask_path'], str(self.state.baseline_union_paths[key]))
            self.assertEqual(member['result_conf_path'], str(self.state.baseline_confmap_paths[key]))
            self.assertEqual(member['result_mode'], 'direct_union')
            mask_paths.add(member['result_mask_path'])
            conf_paths.add(member['result_conf_path'])
        self.assertEqual(len(mask_paths), 4)
        self.assertEqual(len(conf_paths), 4)
        self.assertFalse(mask_paths & conf_paths)
        self.assertEqual(scheduler.direct_union_task_bytes(task), 4 * PARENT_BYTES)
        self.assertEqual(sum(self.state.direct_union_inference_bytes.values()), 4 * PARENT_BYTES)

    def test_keep_temp_still_binds_every_policy_parent_independently(self):
        scheduler = self.scheduler(keep_temp_artifacts=True, direct_union_sparse_retirement_active=False)
        task = shared_group(0)
        scheduler.activate_direct_union_task(task)
        members = (task, *task['augmentation_pass_tasks'])
        paths = {member['result_mask_path'] for member in members}
        self.assertEqual(len(paths), 4)
        self.assertEqual(len(self.state.direct_union_admission_group_by_parent), 4)

    def test_cpu_and_gpu_can_claim_disjoint_policy_leases_of_one_parent_group(self):
        scheduler = self.scheduler()
        first, second = shared_group(0), shared_group(0, slice_start=2)
        first.update(task_id=0, cpu_eligible=True, gpu_eligible=True)
        second.update(task_id=1, cpu_eligible=True, gpu_eligible=True)
        self.state.gpu_worker_tasks_by_id.update({0: first, 1: second})
        self.state.gpu_worker_pending_task_ids.extend((0, 1))
        claimed = scheduler.pop_cpu_worker_pending_task_id()
        self.assertIn(claimed, (0, 1))
        cpu_task = self.state.gpu_worker_tasks_by_id[claimed]
        scheduler.activate_direct_union_task(cpu_task)
        selection = scheduler.pop_gpu_worker_pending_task_id(candidate_workers=(0,))
        self.assertIsNotNone(selection)
        gpu_task = self.state.gpu_worker_tasks_by_id[selection[0]]
        scheduler.activate_direct_union_task(gpu_task)
        self.assertNotEqual(cpu_task['slice_start'], gpu_task['slice_start'])
        self.assertEqual([m['result_mask_path'] for m in (cpu_task, *cpu_task['augmentation_pass_tasks'])],
                         [m['result_mask_path'] for m in (gpu_task, *gpu_task['augmentation_pass_tasks'])])
        self.assertEqual(len(self.masks), 4)

    def test_later_chunks_preserve_prior_windows_and_other_passes(self):
        scheduler = self.scheduler()
        for start in (0, 2):
            task = shared_group(0, slice_start=start)
            scheduler.activate_direct_union_task(task)
            for index, member in enumerate((task, *task['augmentation_pass_tasks'])):
                key = ('model', member['view'].name)
                for field, parents, offset in (
                    ('result_mask_path', self.masks, 1),
                    ('result_conf_path', self.confidence, 10),
                ):
                    worker_map = np.memmap(member[field], mode='r+', dtype=np.uint8, shape=SHAPE)
                    try:
                        worker_map[start:start + 2] = index + start + offset
                    finally:
                        worker_map._mmap.close()
                    self.assertFalse(parents[key]._mmap.closed)
                    expected = np.zeros(SHAPE, np.uint8)
                    expected[:2] = index + offset
                    if start == 2:
                        expected[2:] = index + 2 + offset
                    np.testing.assert_array_equal(parents[key], expected)
        self.assertEqual(len(self.state.direct_union_backing_leases), 4)
        self.assertEqual(len({id(array) for array in self.masks.values()}), 4)

    def test_no_shared_targets_are_bound_before_all_allocations_succeed(self):
        attempts = []

        def fail_third(model, view):
            attempts.append(view.name)
            if len(attempts) == 3:
                raise RuntimeError('injected parent allocation failure')
            self.ensure(model, view)

        scheduler = self.scheduler(ensure_baseline_workspaces=fail_third)
        task = shared_group(0)
        members = (task, *task['augmentation_pass_tasks'])
        initial_paths = [(member['result_mask_path'], member['result_conf_path']) for member in members]
        with self.assertRaisesRegex(RuntimeError, 'injected parent allocation failure'):
            scheduler.activate_direct_union_task(task)
        self.assertEqual([(member['result_mask_path'], member['result_conf_path']) for member in members], initial_paths)
        self.assertEqual(len(self.state.direct_union_backing_leases), 2)
        self.assertFalse(self.state.gpu_worker_dispatched_tasks)

    def test_shared_group_cannot_use_single_parent_oversize_escape(self):
        scheduler = self.scheduler(direct_union_total_dense_byte_limit=3 * PARENT_BYTES)
        with self.assertRaisesRegex(RuntimeError, 'exceeding.*bounded parent dense limit'):
            scheduler.activate_direct_union_task(shared_group(0))
        self.assertFalse(self.state.direct_union_backing_leases)

    def test_all_sibling_bytes_remain_reserved_through_individual_retirement(self):
        scheduler = self.scheduler(direct_union_total_dense_byte_limit=4 * PARENT_BYTES)
        first, following = shared_group(0), shared_group(1)
        scheduler.activate_direct_union_task(first)
        keys = list(self.state.direct_union_inference_views)
        for key in keys:
            self.handoff(key)
        self.assertFalse(scheduler.direct_union_task_admissible(following))
        for key in keys[:-1]:
            self.state.direct_union_backing_leases.pop(key).release('postprocess')
            self.state.direct_union_postprocess_views.remove(key)
            self.state.direct_union_postprocess_bytes.pop(key)
            self.assertFalse(scheduler.direct_union_task_admissible(following))
        key = keys[-1]
        self.state.direct_union_backing_leases.pop(key).release('postprocess')
        self.state.direct_union_postprocess_views.remove(key)
        self.state.direct_union_postprocess_bytes.pop(key)
        self.assertTrue(scheduler.direct_union_task_admissible(following))
        scheduler.activate_direct_union_task(following)

    def test_partial_postprocess_handoff_cannot_accept_another_shared_chunk(self):
        scheduler = self.scheduler()
        task = shared_group(0)
        scheduler.activate_direct_union_task(task)
        self.handoff(('model', task['augmentation_pass_tasks'][0]['view'].name))
        with self.assertRaisesRegex(RuntimeError, 'postprocess-owned'):
            scheduler.activate_direct_union_task(shared_group(0, slice_start=2))

    def test_mixed_file_and_shared_group_fails_before_allocation(self):
        scheduler = self.scheduler()
        for base_mode in ('file', 'direct_union'):
            task = shared_group(0)
            task['result_mode'] = base_mode
            task['augmentation_pass_tasks'][0]['result_mode'] = 'file' if base_mode == 'direct_union' else 'direct_union'
            with self.subTest(base_mode=base_mode), self.assertRaisesRegex(RuntimeError, 'one result mode'):
                scheduler.activate_direct_union_task(task)
        self.assertFalse(self.state.direct_union_backing_leases)

    def test_marked_group_rejects_affine_only_owner_modes(self):
        scheduler = self.scheduler()
        for mode in ('d1_owner', 'hybrid_deferred'):
            for which in ('base', 'sibling'):
                task = shared_group(0)
                member = task if which == 'base' else task['augmentation_pass_tasks'][0]
                member['result_mode'] = mode
                with self.subTest(mode=mode, which=which), self.assertRaisesRegex(RuntimeError, 'cannot use result mode'):
                    scheduler.activate_direct_union_task(task)
        self.assertFalse(self.state.direct_union_backing_leases)


if __name__ == '__main__':
    unittest.main()

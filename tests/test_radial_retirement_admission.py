"""Dense-parent backpressure opens idle Radial GPUs without releasing busy workers."""
from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest import mock

from tests.test_tta_policy_parent_admission import policy_task
from tests.test_tta_scheduler_boundary import _bind_callbacks, _scheduler, _state, _view
from XTA import backprojection as bp
from XTA.interpolation import _DirectUnionBackingLease


class RadialSchedulerRetirementAdmissionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        for patch in (
            mock.patch.object(bp, 'main_process_gpu_stage_inference_priority_enabled', return_value=True),
            mock.patch.object(bp, 'main_process_gpu_stage_inference_overlap_enabled', return_value=False),
            mock.patch.object(bp, 'v1613_d1_backprojection_overlap_enabled', return_value=True),
            mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool', return_value=None),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.coordinator = bp._MainProcessGpuStageCoordinator()
        self.coordinator.configure_workers([0, 1, 2, 3])
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            device_count=lambda: 4, mem_get_info=mock.Mock(return_value=(1000, 2000))),
            device=lambda value: value)
        self.state = _state()
        self.finish_inference = mock.Mock(wraps=self.coordinator.finish_inference)
        self.scheduler = _scheduler(Path(self.temporary.name), state=self.state,
            input_overrides=dict(gpu_device_count=4, direct_union_inference_view_limit=4,
                direct_union_inference_byte_limit=400, direct_union_total_dense_byte_limit=400),
            operation_overrides=dict(
                _set_main_process_gpu_pending_inference=self.coordinator.set_pending_inference_backlog,
                _main_process_gpu_stage_begin_inference=self.coordinator.begin_inference,
                _main_process_gpu_stage_finish_inference=self.finish_inference,
                _main_process_gpu_stage_can_dispatch_inference=self.coordinator.can_dispatch_inference))
        _bind_callbacks(self.scheduler)
        pending = policy_task(1, ratio=4, size=100, task_id=0)
        self.state.gpu_worker_tasks_by_id[0] = pending
        self.state.gpu_worker_pending_task_ids.append(0)
        self.state.gpu_worker_total_tasks = 1
        for index in range(4):
            key = ('model', f'completed-parent-{index}')
            lease = _DirectUnionBackingLease(key, 100)
            lease.transition('inference', 'postprocess')
            self.state.direct_union_backing_leases[key] = lease
            self.state.direct_union_postprocess_views.add(key)
            self.state.direct_union_postprocess_bytes[key] = 100
        self.purpose = 'Radial source projection completed-parent'

    def acquire(self, device):
        lease = self.coordinator.try_acquire_specific_stage(self.torch, device, self.purpose)
        if lease is not None:
            self.addCleanup(lease.release)
        return lease

    def begin_task(self, task_id, worker):
        self.assertTrue(self.coordinator.begin_inference(worker))
        self.state.gpu_worker_tasks_by_id[task_id] = dict(task_id=task_id, kind='fullframe',
            result_mode='file', model_name='model', view=_view(), slice_count=1,
            gpu_eligible=True, disable_runtime_split=True)
        self.state.gpu_worker_total_tasks += 1
        self.state.gpu_worker_dispatched_tasks += 1
        self.state.gpu_worker_dispatched_by_id[worker] = self.state.gpu_worker_dispatched_by_id.get(worker, 0) + 1

    def release_compute(self, task_id, worker):
        self.scheduler.process_one_worker_result(dict(type='compute_released',
            task_id=task_id, gpu_index=worker, stats={}))

    def finish_result(self, task_id, worker, *, ok=True):
        self.scheduler.process_one_worker_result(dict(type='result', task_id=task_id,
            gpu_index=worker, ok=ok, stats={}, error='injected worker failure'))

    def test_dense_cap_publishes_no_admissible_backlog_and_opens_idle_radial_stage(self):
        self.scheduler.publish_gpu_worker_admissible_backlog()
        self.assertTrue(self.state.gpu_worker_pending_task_ids)
        snapshot = self.coordinator.snapshot()
        self.assertFalse(snapshot['pending_inference_backlog'])
        self.assertTrue(snapshot['inference_priority_active'])
        self.assertFalse(snapshot['inference_inflight'])
        for fallback_overlap in (False, True):
            with self.subTest(fallback_overlap=fallback_overlap), mock.patch.object(
                bp, 'v1613_d1_backprojection_overlap_enabled', return_value=fallback_overlap,
            ):
                lease = self.acquire(0)
                self.assertIsNotNone(lease)
                lease.release()

    def test_cluster_shaped_57_parent_255_gib_backlog_opens_idle_projection(self):
        gib = 1024 ** 3
        self.scheduler.inputs = replace(self.scheduler.inputs,
            direct_union_inference_byte_limit=128 * gib,
            direct_union_total_dense_byte_limit=256 * gib)
        self.scheduler.operations = replace(self.scheduler.operations,
            _set_main_process_gpu_spherical_retirement_pressure=self.coordinator.set_spherical_retirement_pressure)
        self.state.direct_union_backing_leases.clear()
        self.state.direct_union_postprocess_views.clear()
        self.state.direct_union_postprocess_bytes.clear()
        for index in range(57):
            key = ('model', f'completed-parent-{index}')
            need = (4 if index < 56 else 31) * gib
            lease = _DirectUnionBackingLease(key, need)
            lease.transition('inference', 'postprocess')
            self.state.direct_union_backing_leases[key] = lease
            self.state.direct_union_postprocess_views.add(key)
            self.state.direct_union_postprocess_bytes[key] = need
        pending = self.state.gpu_worker_tasks_by_id[0]
        for member in (pending, *pending['augmentation_pass_tasks']):
            member['processing_shape'] = (1024, 1024, 1024)
        self.scheduler.publish_gpu_worker_admissible_backlog()
        snapshot = self.coordinator.snapshot()
        self.assertEqual(len(self.state.direct_union_postprocess_views), 57)
        self.assertEqual(sum(self.state.direct_union_postprocess_bytes.values()), 255 * gib)
        self.assertEqual(snapshot['inference_inflight'], {})
        self.assertEqual(snapshot['stage_leases'], {})
        self.assertTrue(snapshot['inference_priority_active'])
        self.assertTrue(snapshot['spherical_retirement_pressure'])
        self.assertFalse(snapshot['pending_inference_backlog'])
        self.assertIsNotNone(self.acquire(0))

    def test_retiring_parent_bytes_restores_inference_priority_before_global_drain(self):
        self.scheduler.publish_gpu_worker_admissible_backlog()
        for key in list(self.state.direct_union_postprocess_views):
            self.state.direct_union_backing_leases.pop(key).release('postprocess')
            self.state.direct_union_postprocess_views.remove(key)
            self.state.direct_union_postprocess_bytes.pop(key)
        self.scheduler.publish_gpu_worker_admissible_backlog()
        self.assertTrue(self.coordinator.snapshot()['pending_inference_backlog'])
        self.assertIsNone(self.acquire(0))
        self.assertEqual(self.state.gpu_worker_results_collected, 0)
        self.torch.cuda.mem_get_info.assert_not_called()

    def test_queued_worker_leases_and_result_lag_keep_exact_per_device_fences(self):
        self.begin_task(10, 0)
        self.begin_task(11, 0)
        self.begin_task(12, 1)
        self.scheduler.publish_gpu_worker_admissible_backlog()
        self.assertEqual(self.coordinator.snapshot()['inference_inflight'], {0: 2, 1: 1})
        self.assertIsNone(self.acquire(0))
        self.assertIsNone(self.acquire(1))
        idle = self.acquire(2)
        self.assertIsNotNone(idle)
        idle.release()

        self.release_compute(10, 0)
        self.release_compute(10, 0)
        self.assertEqual(self.coordinator.snapshot()['inference_inflight'], {0: 1, 1: 1})
        self.assertEqual(self.scheduler.gpu_worker_inflight(0), 1)
        self.assertEqual(self.state.gpu_worker_results_collected, 0)
        self.assertEqual(self.finish_inference.call_count, 1)
        self.assertIsNone(self.acquire(0))

        self.finish_result(10, 0)
        self.assertEqual(self.finish_inference.call_count, 1)
        self.assertEqual(self.coordinator.snapshot()['inference_inflight'], {0: 1, 1: 1})
        self.release_compute(11, 0)
        self.assertEqual(self.scheduler.gpu_worker_inflight(0), 0)
        self.assertEqual(self.coordinator.snapshot()['inference_inflight'], {1: 1})
        # Task 11 publication has not arrived, but its GPU compute lease is free.
        self.assertEqual(self.state.gpu_worker_results_collected, 1)
        lease = self.acquire(0)
        self.assertIsNotNone(lease)
        self.assertFalse(self.coordinator.begin_inference(0))
        self.assertIsNone(self.acquire(1))
        self.finish_result(11, 0)
        self.assertEqual(self.finish_inference.call_count, 2)
        self.assertEqual(self.coordinator.snapshot()['inference_inflight'], {1: 1})
        lease.release()

        # A synchronous result without a preceding compute-release message also
        # returns exactly one device credit before its parent result is handled.
        self.finish_result(12, 1)
        self.assertEqual(self.finish_inference.call_count, 3)
        self.assertFalse(self.coordinator.snapshot()['inference_inflight'])
        self.assertEqual(self.state.gpu_worker_results_collected, 3)
        self.assertIsNotNone(self.acquire(1))

    def test_radial_stage_stays_exclusive_under_overlap_override(self):
        self.begin_task(10, 0)
        self.scheduler.publish_gpu_worker_admissible_backlog()
        with mock.patch.object(bp, 'main_process_gpu_stage_inference_overlap_enabled', return_value=True):
            self.assertIsNone(self.acquire(0))
            self.release_compute(10, 0)
            lease = self.acquire(0)
            self.assertIsNotNone(lease)
            self.assertFalse(self.coordinator.can_dispatch_inference(0))
            self.assertFalse(self.coordinator.begin_inference(0))
            lease.release()
            self.assertTrue(self.coordinator.begin_inference(0))
            self.coordinator.finish_inference(0)

    def test_terminal_asset_ack_and_active_auxiliary_work_remain_authoritative(self):
        self.scheduler.publish_gpu_worker_admissible_backlog()
        self.coordinator.set_inference_asset_retirement_pending(True)
        self.assertIsNone(self.acquire(0))
        self.coordinator.set_inference_asset_retirement_pending(False)
        with mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool',
                               return_value=SimpleNamespace(
                                   claim_worker_for_stage=lambda _device: None,
                                   release_stage_claim=lambda _device, _token: None)):
            self.assertIsNone(self.acquire(0))
        self.assertIsNotNone(self.acquire(0))

    def test_failed_result_clears_only_its_own_compute_credit_before_raising(self):
        self.begin_task(10, 0)
        self.begin_task(11, 1)
        self.scheduler.publish_gpu_worker_admissible_backlog()
        with self.assertRaisesRegex(RuntimeError, 'injected worker failure'):
            self.finish_result(10, 0, ok=False)
        self.assertEqual(self.coordinator.snapshot()['inference_inflight'], {1: 1})
        self.assertEqual(self.finish_inference.call_count, 1)
        self.assertEqual(self.scheduler.gpu_worker_inflight(0), 0)
        self.assertEqual(self.scheduler.gpu_worker_inflight(1), 1)


class MixedNativeRetirementQueueTests(unittest.TestCase):
    def setUp(self):
        for patch in (
            mock.patch.object(bp, 'main_process_gpu_stage_inference_priority_enabled', return_value=True),
            mock.patch.object(bp, 'main_process_gpu_stage_inference_overlap_enabled', return_value=False),
            mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool', return_value=None),
            mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT': '1',
                                           'YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT': '1'}),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            device_count=lambda: 4, mem_get_info=lambda _device: (1000, 2000)),
            device=lambda value: value)

    def coordinator(self, *, pressure=True):
        coordinator = bp._MainProcessGpuStageCoordinator()
        coordinator.configure_workers([0, 1, 2, 3])
        coordinator.set_pending_inference_backlog(True)
        coordinator.set_spherical_retirement_pressure(pressure)
        for worker in range(4):
            coordinator.begin_inference(worker)
            coordinator.begin_inference(worker)
        return coordinator

    def acquire(self, coordinator, purpose):
        lease = coordinator.try_acquire_stage(self.torch, purpose)
        if lease is not None:
            self.addCleanup(lease.release)
        return lease

    def test_radial_and_spherical_share_fifo_refresh_and_two_turn_burst_cap(self):
        for first_family, second_family in (('Radial', 'Spherical'), ('Spherical', 'Radial')):
            with self.subTest(first_family=first_family), mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
                coordinator = self.coordinator()
                oldest = f'{first_family} source projection oldest'
                middle = f'{second_family} source projection middle'
                newest = f'{first_family} source projection newest'
                for purpose in (oldest, middle, newest):
                    self.assertIsNone(self.acquire(coordinator, purpose))
                    clock.return_value += 1.
                clock.return_value = 15.
                self.assertIsNone(self.acquire(coordinator, oldest))
                self.assertEqual(coordinator.snapshot()['spherical_retirement_reserved_device'], 0)
                coordinator.finish_inference(0)
                coordinator.finish_inference(0)
                self.assertIsNone(self.acquire(coordinator, newest))
                first = self.acquire(coordinator, oldest)
                self.assertEqual(first.device_index, 0)
                self.assertIsNone(self.acquire(coordinator, middle))
                first.release()
                self.assertFalse(coordinator.begin_inference(0))
                self.assertTrue(all(coordinator.can_dispatch_inference(worker) for worker in (1, 2, 3)))
                self.assertIsNone(self.acquire(coordinator, newest))
                second = self.acquire(coordinator, middle)
                self.assertEqual(second.device_index, 0)
                second.release()
                self.assertEqual(coordinator.snapshot()['spherical_retirement_handoffs'], 1)
                self.assertEqual(coordinator.snapshot()['spherical_retirement_pressure_acquisitions'], 2)
                # Changing projection families does not reset the two-turn cap.
                self.assertIsNone(self.acquire(coordinator, newest))
                self.assertEqual(coordinator.snapshot()['spherical_retirement_reserved_device'], 1)
                self.assertTrue(coordinator.begin_inference(0))
                coordinator.finish_inference(0)

    def test_aged_radial_reader_hands_off_to_spherical_without_memory_pressure(self):
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            coordinator = self.coordinator(pressure=False)
            radial = 'Radial source projection aged'
            spherical = 'Spherical source projection aged'
            self.assertIsNone(self.acquire(coordinator, radial))
            clock.return_value = 11.
            self.assertIsNone(self.acquire(coordinator, spherical))
            clock.return_value = 35.
            self.assertIsNone(self.acquire(coordinator, spherical))
            self.assertIsNone(self.acquire(coordinator, radial))
            clock.return_value = 40.
            self.assertIsNone(self.acquire(coordinator, radial))
            coordinator.finish_inference(0)
            coordinator.finish_inference(0)
            first = self.acquire(coordinator, radial)
            self.assertEqual(first.device_index, 0)
            clock.return_value = 41.
            self.assertIsNone(self.acquire(coordinator, spherical))
            first.release()
            second = self.acquire(coordinator, spherical)
            self.assertEqual(second.device_index, 0)
            self.assertEqual(coordinator.snapshot()['spherical_retirement_aged_acquisitions'], 2)
            self.assertEqual(coordinator.snapshot()['spherical_retirement_pressure_acquisitions'], 0)
            second.release()

    def test_cancelled_radial_cpu_reader_yields_fifo_to_spherical(self):
        coordinator = self.coordinator()
        radial = 'Radial source projection CPU finished'
        spherical = 'Spherical source projection pending'
        self.assertIsNone(self.acquire(coordinator, radial))
        self.assertIsNone(self.acquire(coordinator, spherical))
        coordinator.cancel_spherical_retirement_request(radial)
        coordinator.finish_inference(0)
        coordinator.finish_inference(0)
        lease = self.acquire(coordinator, spherical)
        self.assertEqual(lease.device_index, 0)
        self.assertEqual(coordinator.snapshot()['spherical_retirement_request_count'], 0)
        lease.release()


if __name__ == '__main__':
    unittest.main()

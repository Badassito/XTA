"""Host-only race checks for CUDA stage admission and auxiliary ownership."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA import backprojection as bp
from XTA import runtime


class StageAdmissionConcurrencyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.coordinator = bp._MainProcessGpuStageCoordinator()
        self.coordinator.configure_workers([0, 1])
        self.torch = SimpleNamespace(
            cuda=SimpleNamespace(device_count=lambda: 2,
                                 mem_get_info=lambda device: (2000 if device == 'cuda:0' else 1000, 4000)),
            device=lambda device: device,
        )
        self.pool = runtime._GpuWorkerAuxInterpolationPool({0: queue.SimpleQueue(), 1: queue.SimpleQueue()})
        patch = mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool', return_value=self.pool)
        patch.start()
        self.addCleanup(patch.stop)
        patch = mock.patch.object(bp, 'main_process_gpu_stage_inference_priority_enabled', return_value=False)
        patch.start()
        self.addCleanup(patch.stop)
        self.coordinator.set_inference_priority_active(False)

    def _start(self, target):
        result = []
        errors = []

        def run():
            try:
                result.append(target())
            except BaseException as exc:
                errors.append(exc)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, result, errors

    def test_blocked_cuda_query_does_not_hold_dispatch_lock(self) -> None:
        entered = threading.Event()
        resume = threading.Event()

        def slow_query(device):
            if device == 'cuda:0':
                entered.set()
                self.assertTrue(resume.wait(5))
            return (2000 if device == 'cuda:0' else 1000, 4000)

        self.torch.cuda.mem_get_info = slow_query
        thread, result, errors = self._start(lambda: self.coordinator.try_acquire_stage(self.torch, 'output'))
        self.assertTrue(entered.wait(2))
        self.assertTrue(self.coordinator.can_dispatch_inference(1))
        self.assertTrue(self.coordinator.begin_inference(1))
        self.coordinator.finish_inference(1)
        # Even the queried device may dispatch before its provisional reservation.
        self.assertTrue(self.coordinator.begin_inference(0))
        resume.set()
        thread.join(3)
        self.assertFalse(errors)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].device_index, 1)
        self.coordinator.finish_inference(0)
        result[0].release()

    def test_blocked_auxiliary_claim_excludes_only_reserved_device(self) -> None:
        entered = threading.Event()
        resume = threading.Event()
        original_claim = self.pool.claim_worker_for_stage

        def slow_claim(device):
            if device == 0:
                entered.set()
                self.assertTrue(resume.wait(5))
            return original_claim(device)

        with mock.patch.object(self.pool, 'claim_worker_for_stage', side_effect=slow_claim):
            thread, result, errors = self._start(
                lambda: self.coordinator.try_acquire_stage(self.torch, 'output'))
            self.assertTrue(entered.wait(2))
            self.assertEqual(self.coordinator.snapshot()['provisional_stage_devices'], [0])
            self.assertFalse(self.coordinator.can_dispatch_inference(0))
            self.assertFalse(self.coordinator.begin_inference(0))
            self.assertTrue(self.coordinator.can_dispatch_inference(1))
            self.assertTrue(self.coordinator.begin_inference(1))
            self.coordinator.finish_inference(1)
            resume.set()
            thread.join(3)
        self.assertFalse(errors)
        self.assertEqual(result[0].device_index, 0)
        result[0].release()
        self.assertTrue(self.coordinator.can_dispatch_inference(0))

    def test_auxiliary_enable_cannot_reopen_a_stage_device(self) -> None:
        self.assertTrue(self.pool.enable_worker(0))
        lease = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        self.assertIsNotNone(lease)
        self.assertFalse(self.pool.enable_worker(0))
        self.assertTrue(self.pool.revoke_worker(0))
        self.assertIsNone(self.pool.try_submit({'input': 'unused'}))
        lease.release()
        self.assertTrue(self.pool.enable_worker(0))

    def test_opt_in_stage_overlap_allows_inference_but_not_auxiliary_reuse(self) -> None:
        with mock.patch.object(bp, 'main_process_gpu_stage_inference_overlap_enabled', return_value=True):
            stage = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
            self.assertIsNotNone(stage)
            self.assertTrue(self.coordinator.can_dispatch_inference(0))
            self.assertTrue(self.pool.revoke_worker(0))
            self.assertTrue(self.coordinator.begin_inference(0))
            self.assertFalse(self.pool.enable_worker(0))
            self.assertIsNone(self.pool.try_submit({'input': 'unused'}))
            self.coordinator.finish_inference(0)
            stage.release()

    def test_opt_in_overlap_still_fences_provisional_and_retirement_stages(self) -> None:
        entered = threading.Event()
        resume = threading.Event()
        original_claim = self.pool.claim_worker_for_stage

        def slow_claim(device):
            entered.set()
            self.assertTrue(resume.wait(5))
            return original_claim(device)

        with mock.patch.object(bp, 'main_process_gpu_stage_inference_overlap_enabled', return_value=True):
            with mock.patch.object(self.pool, 'claim_worker_for_stage', side_effect=slow_claim):
                thread, result, errors = self._start(
                    lambda: self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output'))
                self.assertTrue(entered.wait(2))
                self.assertFalse(self.coordinator.can_dispatch_inference(0))
                self.assertFalse(self.coordinator.begin_inference(0))
                resume.set()
                thread.join(3)
            self.assertFalse(errors)
            result[0].release()
            retirement = self.coordinator.try_acquire_specific_stage(
                self.torch, 0, 'Spherical source projection sample')
            self.assertIsNotNone(retirement)
            self.assertFalse(self.coordinator.can_dispatch_inference(0))
            self.assertFalse(self.coordinator.begin_inference(0))
            retirement.release()

    def test_reset_during_external_claim_discards_old_attempt(self) -> None:
        entered = threading.Event()
        resume = threading.Event()
        original_claim = self.pool.claim_worker_for_stage

        def slow_claim(device):
            entered.set()
            self.assertTrue(resume.wait(5))
            return original_claim(device)

        with mock.patch.object(self.pool, 'claim_worker_for_stage', side_effect=slow_claim):
            thread, result, errors = self._start(
                lambda: self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output'))
            self.assertTrue(entered.wait(2))
            self.coordinator.configure_workers([0, 1])
            resume.set()
            thread.join(3)
        self.assertFalse(errors)
        self.assertEqual(result, [None])
        self.assertEqual(self.coordinator.snapshot()['stage_leases'], {})
        self.assertTrue(self.pool.enable_worker(0))
        self.assertTrue(self.coordinator.begin_inference(0))
        self.coordinator.finish_inference(0)

    def test_stale_release_cannot_remove_new_owner_with_same_purpose(self) -> None:
        old = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        self.coordinator.configure_workers([0, 1])
        new = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        self.assertIsNotNone(new)
        old.release()
        self.assertEqual(self.coordinator.snapshot()['stage_leases'], {0: 'output'})
        self.assertFalse(self.pool.enable_worker(0))
        new.release()
        self.assertEqual(self.coordinator.snapshot()['stage_leases'], {})

    def test_busy_auxiliary_candidate_falls_back_without_retirement_accounting(self) -> None:
        self.assertTrue(self.pool.enable_worker(0))
        handle = self.pool.try_submit({'input': 'small'})
        self.assertIsNotNone(handle)
        purpose = 'Spherical source projection sample'
        lease = self.coordinator.try_acquire_stage(self.torch, purpose)
        self.assertIsNotNone(lease)
        self.assertEqual(lease.device_index, 1)
        self.assertEqual(self.coordinator._spherical_retirement_burst_counts.get(0, 0), 0)
        self.assertEqual(self.coordinator._spherical_retirement_burst_counts.get(1, 0), 1)
        lease.release()
        self.pool.complete(handle['task_id'], 0, True, {}, None)

    def test_revoke_only_pool_is_refused_without_running_revoke(self) -> None:
        legacy_pool = SimpleNamespace(revoke_worker=mock.Mock(return_value=True))
        with mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool', return_value=legacy_pool):
            self.assertIsNone(self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output'))
        legacy_pool.revoke_worker.assert_not_called()
        self.assertEqual(self.coordinator.snapshot()['provisional_stage_devices'], [])
        self.assertTrue(self.coordinator.begin_inference(0))
        self.coordinator.finish_inference(0)

    def test_auxiliary_getter_failure_releases_provisional_reservation(self) -> None:
        with mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool',
                               side_effect=RuntimeError('pool lookup failed')):
            with self.assertRaisesRegex(RuntimeError, 'pool lookup failed'):
                self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        self.assertEqual(self.coordinator.snapshot()['provisional_stage_devices'], [])
        self.assertTrue(self.coordinator.begin_inference(0))
        self.coordinator.finish_inference(0)

    def test_auxiliary_release_failure_still_releases_provisional_reservation(self) -> None:
        coordinator = self.coordinator

        def claim(_device):
            coordinator.set_inference_asset_retirement_pending(True)
            return object()

        failed_pool = SimpleNamespace(
            claim_worker_for_stage=claim,
            release_stage_claim=mock.Mock(side_effect=RuntimeError('release failed')),
        )
        with mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool', return_value=failed_pool):
            with self.assertRaisesRegex(RuntimeError, 'release failed'):
                coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        failed_pool.release_stage_claim.assert_called_once()
        self.assertEqual(coordinator.snapshot()['provisional_stage_devices'], [])
        coordinator.set_inference_asset_retirement_pending(False)
        self.assertTrue(coordinator.begin_inference(0))
        coordinator.finish_inference(0)


if __name__ == '__main__':
    unittest.main()

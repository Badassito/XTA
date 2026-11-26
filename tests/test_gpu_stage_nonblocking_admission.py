"""Real native stage admission races with CUDA probes and auxiliary actors."""
from __future__ import annotations

import queue
import threading
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA import backprojection as bp
from XTA import runtime
from tests.native_stage_test_support import native_coordinator


class StageAdmissionConcurrencyTests(unittest.TestCase):
    def setUp(self):
        self.coordinator = native_coordinator(self)
        self.coordinator.configure_workers((0, 1))
        self.coordinator.set_inference_priority_active(False)
        self.torch = SimpleNamespace(cuda=SimpleNamespace(device_count=lambda: 2,
            mem_get_info=lambda device: (2000 if device == 'cuda:0' else 1000, 4000)),
            device=lambda device: device)
        self.pool = runtime._GpuWorkerAuxInterpolationPool({0: queue.SimpleQueue(), 1: queue.SimpleQueue()})

    def _start(self, target):
        result, errors = [], []
        def run():
            try:
                result.append(target())
            except BaseException as error:
                errors.append(error)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread, result, errors

    def _probe_gate(self):
        entered, resume = threading.Event(), threading.Event()
        def query(_device):
            entered.set()
            if not resume.wait(5):
                raise RuntimeError('test CUDA probe was not released')
            return 2000, 4000
        self.torch.cuda.mem_get_info = query
        self.addCleanup(resume.set)
        return entered, resume

    def test_failed_coordinator_release_remains_owned_and_can_retry(self):
        lease = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        with mock.patch.object(self.coordinator, 'release_stage', side_effect=RuntimeError('before ledger update')):
            with self.assertRaisesRegex(RuntimeError, 'before ledger update'):
                lease.release()
        self.assertFalse(lease._released)
        self.assertFalse(self.coordinator.can_dispatch_inference(0))
        self.assertFalse(self.pool.enable_worker(0))
        lease.release()
        lease.release()
        self.assertTrue(lease._released)
        self.assertTrue(self.coordinator.can_dispatch_inference(0))
        self.assertTrue(self.pool.enable_worker(0))

    def test_blocked_cuda_ranking_query_does_not_hold_native_dispatch_lock(self):
        entered, resume = self._probe_gate()
        thread, result, errors = self._start(lambda: self.coordinator.try_acquire_stage(self.torch, 'output'))
        self.assertTrue(entered.wait(2))
        self.assertEqual(self.coordinator.snapshot()['provisional_stage_devices'], [])
        self.assertTrue(self.coordinator.begin_inference(0))
        self.assertTrue(self.coordinator.begin_inference(1))
        self.coordinator.finish_inference(1)
        resume.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertFalse(errors)
        self.assertEqual(result[0].device_index, 1)
        self.coordinator.finish_inference(0)
        result[0].release()

    def test_provisional_query_excludes_only_its_device(self):
        self.assertTrue(self.coordinator.begin_inference(1))
        entered, resume = self._probe_gate()
        thread, result, errors = self._start(lambda: self.coordinator.try_acquire_stage(self.torch, 'output'))
        self.assertTrue(entered.wait(2))
        self.assertEqual(self.coordinator.snapshot()['provisional_stage_devices'], [0])
        self.assertFalse(self.coordinator.begin_inference(0))
        self.assertFalse(self.pool.enable_worker(0))
        self.assertTrue(self.coordinator.begin_inference(1))
        self.coordinator.finish_inference(1)
        resume.set()
        thread.join(3)
        self.assertFalse(errors)
        self.assertEqual(result[0].device_index, 0)
        result[0].release()
        self.coordinator.finish_inference(1)

    def test_auxiliary_enable_cannot_reopen_a_stage_device(self):
        self.assertTrue(self.pool.enable_worker(0))
        lease = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        self.assertIsNotNone(lease)
        self.assertFalse(self.pool.enable_worker(0))
        self.assertTrue(self.pool.revoke_worker(0))
        self.assertIsNone(self.pool.try_submit({'input': 'unused'}))
        lease.release()
        self.assertTrue(self.pool.enable_worker(0))

    def test_stage_excludes_inference_and_auxiliary_reuse(self):
        stage = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        self.assertFalse(self.coordinator.can_dispatch_inference(0))
        self.assertFalse(self.coordinator.begin_inference(0))
        self.assertFalse(self.pool.enable_worker(0))
        self.assertIsNone(self.pool.try_submit({'input': 'unused'}))
        stage.release()
        self.assertTrue(self.coordinator.can_dispatch_inference(0))

    def test_reconfiguration_cannot_erase_an_inflight_provisional_probe(self):
        self.coordinator.begin_inference(1)
        entered, resume = self._probe_gate()
        thread, result, errors = self._start(lambda: self.coordinator.try_acquire_stage(self.torch, 'output'))
        self.assertTrue(entered.wait(2))
        epoch = self.coordinator.current_epoch()
        for operation in (lambda: self.coordinator.configure_workers((0, 1)), self.coordinator.reset):
            with self.assertRaisesRegex(RuntimeError, 'live owners'):
                operation()
        self.assertEqual(self.coordinator.current_epoch(), epoch)
        self.assertEqual(self.coordinator.snapshot()['provisional_stage_devices'], [0])
        resume.set()
        thread.join(3)
        self.assertFalse(errors)
        result[0].release()
        self.coordinator.finish_inference(1)

    def test_stale_release_cannot_remove_new_owner_with_same_purpose(self):
        old = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        old_token = old._token
        old.release()
        self.coordinator.configure_workers((0, 1))
        self.coordinator.set_inference_priority_active(False)
        new = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        self.coordinator.release_stage(0, 'output', token=old_token)
        self.assertEqual(self.coordinator.snapshot()['stage_leases'], {0: 'output'})
        self.assertFalse(self.pool.enable_worker(0))
        new.release()

    def test_busy_auxiliary_candidate_is_not_stolen(self):
        self.assertTrue(self.pool.enable_worker(0))
        handle = self.pool.try_submit({'input': 'small'})
        self.assertIsNotNone(handle)
        lease = self.coordinator.try_acquire_stage(self.torch, 'Spherical source projection sample')
        self.assertEqual(lease.device_index, 1)
        self.assertEqual(self.coordinator.snapshot()['auxiliary_inflight'], {0: handle['task_id']})
        lease.release()
        self.pool.complete(handle['task_id'], 0, True, {}, None)
        self.assertEqual(self.coordinator.snapshot()['auxiliary_inflight'], {})
        lease = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        self.assertIsNotNone(lease)
        lease.release()

    def test_native_admission_does_not_call_a_legacy_python_pool_getter(self):
        with mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool',
                               side_effect=AssertionError('Python eligibility callback')) as getter:
            lease = self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
            self.assertIsNotNone(lease)
            lease.release()
        getter.assert_not_called()

    def test_handle_construction_failure_returns_committed_native_claim(self):
        with mock.patch.object(self.coordinator, '_stage_factory', side_effect=RuntimeError('handle failed')):
            with self.assertRaisesRegex(RuntimeError, 'handle failed'):
                self.coordinator.try_acquire_specific_stage(self.torch, 0, 'output')
        snapshot = self.coordinator.snapshot()
        self.assertEqual(snapshot['stage_leases'], {})
        self.assertEqual(snapshot['provisional_stage_devices'], [])
        self.assertTrue(self.coordinator.begin_inference(0))
        self.coordinator.finish_inference(0)

    def test_retirement_fence_changed_during_probe_refuses_commit_and_releases_reservation(self):
        self.coordinator.begin_inference(1)
        entered, resume = self._probe_gate()
        thread, result, errors = self._start(lambda: self.coordinator.try_acquire_stage(self.torch, 'output'))
        self.assertTrue(entered.wait(2))
        self.coordinator.set_inference_asset_retirement_pending(True)
        resume.set()
        thread.join(3)
        self.assertFalse(errors)
        self.assertEqual(result, [None])
        self.assertEqual(self.coordinator.snapshot()['provisional_stage_devices'], [])
        self.assertEqual(self.coordinator.snapshot()['stage_leases'], {})
        self.coordinator.set_inference_asset_retirement_pending(False)
        self.coordinator.finish_inference(1)
        self.assertTrue(self.coordinator.begin_inference(0))
        self.coordinator.finish_inference(0)


if __name__ == '__main__':
    unittest.main()

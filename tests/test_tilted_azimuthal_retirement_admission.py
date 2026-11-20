"""Tilted Azimuthal (native destination pull) shares exclusive bounded retirement with Radial/Spherical."""
from __future__ import annotations

from itertools import permutations
from types import SimpleNamespace
import unittest
from unittest import mock

from XTA import backprojection as bp


class TiltedAzimuthalRetirementAdmissionTests(unittest.TestCase):
    def setUp(self):
        for patch in (
            mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool', return_value=None),
            mock.patch.dict('os.environ', {'YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT': '1',
                                           'YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT': '1'}),
        ):
            patch.start()
            self.addCleanup(patch.stop)
        self.torch = SimpleNamespace(cuda=SimpleNamespace(
            device_count=lambda: 4, mem_get_info=lambda _device: (1000, 2000)),
            device=lambda token: token)

    def coordinator(self, *, backlog, pressure, queued=0):
        result = bp._MainProcessGpuStageCoordinator()
        result.configure_workers([0, 1, 2, 3])
        result.set_pending_inference_backlog(backlog)
        result.set_spherical_retirement_pressure(pressure)
        for worker in range(4):
            for _ in range(queued):
                self.assertTrue(result.begin_inference(worker))
        return result

    def acquire(self, coordinator, purpose, device=None):
        lease = (coordinator.try_acquire_stage(self.torch, purpose) if device is None
                 else coordinator.try_acquire_specific_stage(self.torch, device, purpose))
        if lease is not None:
            self.addCleanup(lease.release)
        return lease

    def test_idle_tilted_projection_never_steals_busy_worker(self):
        coordinator = self.coordinator(backlog=False, pressure=True, queued=2)
        purpose = bp._native_pull_cuda_stage_purpose(SimpleNamespace(name='tilted_azimuthal_parent'))
        self.assertIsNone(self.acquire(coordinator, purpose))
        coordinator.finish_inference(0)
        self.assertIsNone(self.acquire(coordinator, purpose, 0))
        coordinator.finish_inference(0)
        lease = self.acquire(coordinator, purpose, 0)
        self.assertIsNotNone(lease)
        self.assertFalse(coordinator.can_dispatch_inference(0))
        self.assertFalse(coordinator.begin_inference(0))
        self.assertIsNone(self.acquire(coordinator, purpose, 1))
        lease.release()
        coordinator.set_inference_asset_retirement_pending(True)
        self.assertIsNone(self.acquire(coordinator, purpose, 0))
        coordinator.set_inference_asset_retirement_pending(False)
        with mock.patch.object(bp, 'gpu_worker_aux_interpolation_pool',
                               return_value=SimpleNamespace(
                                   claim_worker_for_stage=lambda _device: None,
                                   release_stage_claim=lambda _device, _token: None)):
            self.assertIsNone(self.acquire(coordinator, purpose, 0))

    def test_all_three_families_share_fifo_and_one_two_turn_burst(self):
        for families in permutations(('Native', 'Radial', 'Spherical')):
            with self.subTest(order=families), mock.patch.object(bp.time, 'monotonic', return_value=10.):
                coordinator = self.coordinator(backlog=True, pressure=True, queued=2)
                names = [f'{family} source projection {index}' for index, family in enumerate(families)]
                for name in names:
                    self.assertIsNone(self.acquire(coordinator, name))
                coordinator.finish_inference(0)
                coordinator.finish_inference(0)
                self.assertIsNone(self.acquire(coordinator, names[2]))
                first = self.acquire(coordinator, names[0])
                self.assertEqual(first.device_index, 0)
                self.assertIsNone(self.acquire(coordinator, names[1]))
                first.release()
                self.assertFalse(coordinator.begin_inference(0))
                self.assertTrue(all(coordinator.can_dispatch_inference(worker) for worker in (1, 2, 3)))
                second = self.acquire(coordinator, names[1])
                self.assertEqual(second.device_index, 0)
                second.release()
                self.assertEqual(coordinator.snapshot()['spherical_retirement_handoffs'], 1)
                self.assertEqual(coordinator.snapshot()['spherical_retirement_pressure_acquisitions'], 2)
                self.assertIsNone(self.acquire(coordinator, names[2]))
                self.assertEqual(coordinator.snapshot()['spherical_retirement_reserved_device'], 1)
                self.assertTrue(coordinator.begin_inference(0))
                coordinator.finish_inference(0)

    def test_tilted_reader_ages_then_hands_off_to_radial_below_pressure(self):
        with mock.patch.object(bp.time, 'monotonic', return_value=10.) as clock:
            coordinator = self.coordinator(backlog=True, pressure=False, queued=2)
            names = ['Native source projection aged', 'Radial source projection aged',
                     'Spherical source projection aged']
            for name in names:
                self.assertIsNone(self.acquire(coordinator, name))
            clock.return_value = 35.
            for name in reversed(names):
                self.assertIsNone(self.acquire(coordinator, name))
            clock.return_value = 40.
            self.assertIsNone(self.acquire(coordinator, names[2]))
            coordinator.finish_inference(0)
            coordinator.finish_inference(0)
            first = self.acquire(coordinator, names[0])
            self.assertEqual(first.device_index, 0)
            first.release()
            second = self.acquire(coordinator, names[1])
            self.assertEqual(second.device_index, 0)
            second.release()
            self.assertEqual(coordinator.snapshot()['spherical_retirement_aged_acquisitions'], 2)
            self.assertIsNone(self.acquire(coordinator, names[2]))
            self.assertTrue(coordinator.begin_inference(0))

    def test_classifier_requires_exact_source_projection_prefixes(self):
        classifier = bp._MainProcessGpuStageCoordinator._is_source_projection_retirement
        for purpose in ('Native source projection parent', 'Radial source projection parent',
                        'Spherical source projection parent'):
            with self.subTest(purpose=purpose):
                self.assertTrue(classifier(purpose))
        # 'Tilted Azimuthal source projection' is a retired family label: Tilted
        # Azimuthal now projects through native destination pull.
        for purpose in ('Tilted Azimuthal source projection parent', 'Native source projections parent',
                        'Native inference parent', 'Azimuthal source projection parent',
                        'NRRD Tilted Azimuthal output', 'SAM image preparation',
                        'v streaming Azimuthal backprojection'):
            with self.subTest(purpose=purpose):
                self.assertFalse(classifier(purpose))


if __name__ == '__main__':
    unittest.main()

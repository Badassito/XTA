from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import os
import types
import sys
import threading
import unittest
from unittest import mock

import pytest

from XTA.lta_cpu import (
    bind_worker_cpu_environment, configure_worker_runtime_threads, resolve_worker_cpu_budget,
    bounded_lta_host_threads,
)


class LtaCpuBudgetTests(unittest.TestCase):
    def test_wide_affinity_does_not_override_slurm_allocation(self):
        budget = resolve_worker_cpu_budget(4, affinity_count=128, cpu_count=256, environ={
            "SLURM_CPUS_PER_TASK": "16", "SLURM_CPUS_ON_NODE": "128",
            "SLURM_JOB_CPUS_PER_NODE": "128(x2)",
        })
        self.assertEqual(budget["effective_cpu_count"], 16)
        self.assertEqual(budget["threads_per_worker"], 3)
        self.assertLessEqual(4 * budget["threads_per_worker"] + 1, 16)

    def test_affinity_and_explicit_lower_thread_limit_both_apply(self):
        budget = resolve_worker_cpu_budget(4, affinity_count=8, cpu_count=128, environ={
            "SLURM_CPUS_PER_TASK": "64", "OMP_NUM_THREADS": "1",
        })
        self.assertEqual(budget["effective_cpu_count"], 8)
        self.assertEqual(budget["threads_per_worker"], 1)
        self.assertEqual(resolve_worker_cpu_budget(4, cpu_count=128, environ={})["threads_per_worker"], 4)

    def test_insufficient_cpu_allocation_is_reported_without_zero_thread_pools(self):
        budget = resolve_worker_cpu_budget(4, cpu_count=128, environ={"SLURM_CPUS_PER_TASK": "1"})
        self.assertTrue(budget["minimum_worker_threads_exceed_allocation"])
        self.assertEqual(budget["threads_per_worker"], 1)

    def test_runtime_settings_report_actual_limits(self):
        torch = types.SimpleNamespace(
            set_num_threads=mock.Mock(), get_num_threads=lambda: 2,
            set_num_interop_threads=mock.Mock(), get_num_interop_threads=lambda: 1,
        )
        cv2 = types.SimpleNamespace(setNumThreads=mock.Mock(), getNumThreads=lambda: 2)
        with mock.patch.dict(os.environ, {}, clear=True):
            bind_worker_cpu_environment({"threads_per_worker": 2})
            receipt = configure_worker_runtime_threads(torch, cv2)
            self.assertEqual(os.environ["OMP_NUM_THREADS"], "2")
            self.assertEqual(os.environ["OPENBLAS_NUM_THREADS"], "2")
            self.assertEqual(os.environ["OMP_WAIT_POLICY"], "PASSIVE")
            self.assertEqual(receipt["torch_num_threads"], 2)
            self.assertEqual(receipt["opencv_num_threads"], 2)
            torch.set_num_threads.assert_called_once_with(2)
            cv2.setNumThreads.assert_called_once_with(2)

    def test_host_native_pools_are_bounded_and_restored_after_failure(self):
        counts = {"cv2": 64, "torch": 128}
        cv2 = types.SimpleNamespace(
            getNumThreads=lambda: counts['cv2'],
            setNumThreads=lambda value: counts.update(cv2=value),
        )
        torch = types.SimpleNamespace(
            get_num_threads=lambda: counts['torch'],
            set_num_threads=lambda value: counts.update(torch=value),
        )
        @bounded_lta_host_threads
        def failed_run():
            self.assertEqual(counts, {"cv2": 1, "torch": 128})
            raise ValueError('failed run')
        before = dict(os.environ)
        with mock.patch.dict(sys.modules, {'cv2': cv2, 'torch': torch}):
            with self.assertRaisesRegex(ValueError, 'failed run'):
                failed_run()
        self.assertEqual(counts, {"cv2": 64, "torch": 128})
        self.assertEqual(dict(os.environ), before)

    def test_overlapping_host_scopes_restore_native_pools_after_last_exit(self):
        current = [8]
        entered = [threading.Event() for _ in range(2)]
        release = [threading.Event() for _ in range(2)]
        cv2 = types.SimpleNamespace(
            getNumThreads=lambda: current[0],
            setNumThreads=lambda value: current.__setitem__(0, value),
        )
        @bounded_lta_host_threads
        def run(index):
            entered[index].set()
            release[index].wait(5)

        with mock.patch.dict(sys.modules, {'cv2': cv2}):
            threads = [threading.Thread(target=run, args=(index,)) for index in range(2)]
            try:
                for index, thread in enumerate(threads):
                    thread.start()
                    self.assertTrue(entered[index].wait(5))
                release[0].set()
                threads[0].join(5)
                self.assertEqual(current[0], 1)
            finally:
                for event in release:
                    event.set()
                for thread in threads:
                    if thread.ident is not None:
                        thread.join(5)
            self.assertEqual(current[0], 8)

    def test_overlapping_host_scopes_preserve_real_torch_thread_limits(self):
        torch = pytest.importorskip("torch")
        if "ATen parallel backend: OpenMP" not in torch.__config__.parallel_info():
            pytest.skip("requires PyTorch's OpenMP backend")
        previous = torch.get_num_threads()
        entered = [threading.Event(), threading.Event()]
        release = [threading.Event(), threading.Event()]

        @bounded_lta_host_threads
        def run(index):
            entered[index].set()
            self.assertTrue(release[index].wait(10))
            return torch.get_num_threads()

        def fresh_thread_limit():
            with ThreadPoolExecutor(max_workers=1) as executor:
                return executor.submit(torch.get_num_threads).result()

        torch.set_num_threads(3)
        try:
            with ThreadPoolExecutor(max_workers=2) as executor:
                try:
                    first = executor.submit(run, 0)
                    self.assertTrue(entered[0].wait(10))
                    second = executor.submit(run, 1)
                    self.assertTrue(entered[1].wait(10))
                    self.assertEqual(fresh_thread_limit(), 3)
                    release[0].set()
                    self.assertEqual(first.result(timeout=10), 3)
                finally:
                    for event in release:
                        event.set()
                self.assertEqual(second.result(timeout=10), 3)
            self.assertEqual(fresh_thread_limit(), 3)
        finally:
            torch.set_num_threads(previous)

    def test_failed_host_scope_start_rolls_back_existing_native_limit(self):
        cv2 = types.SimpleNamespace(getNumThreads=lambda: 8, setNumThreads=mock.Mock(
            side_effect=[RuntimeError('native setup failed'), None]))
        function = mock.Mock()
        with mock.patch.dict(sys.modules, {'cv2': cv2}):
            with self.assertRaisesRegex(RuntimeError, 'native setup failed'):
                bounded_lta_host_threads(function)()
        function.assert_not_called()
        self.assertEqual(cv2.setNumThreads.call_args_list, [mock.call(1), mock.call(8)])


if __name__ == '__main__':
    unittest.main()

import gc
import unittest
from unittest import mock

from XTA import runtime as _xta_runtime  # noqa: F401  (import side effects before measuring)
from XTA import sam_tracker_runtime as runtime


class SamWorkerGcFreezeTests(unittest.TestCase):
    def tearDown(self):
        gc.unfreeze()

    def test_freeze_moves_startup_heap_out_of_collection(self):
        gc.unfreeze()
        before = gc.get_freeze_count()
        startup_heap = [[index] for index in range(64)]
        with mock.patch.dict('os.environ', {}, clear=False):
            receipt = runtime._freeze_worker_startup_heap()
        self.assertEqual(receipt['schema'], 'xta.sam_worker_gc_freeze/1')
        self.assertTrue(receipt['enabled'])
        self.assertGreaterEqual(receipt['frozen_objects'], before + len(startup_heap))
        self.assertGreaterEqual(receipt['seconds'], 0.0)
        del startup_heap

    def test_objects_created_after_freeze_are_still_collected(self):
        runtime._freeze_worker_startup_heap()

        class Node:
            pass

        first, second = Node(), Node()
        first.peer, second.peer = second, first
        import weakref
        witness = weakref.ref(first)
        del first, second
        gc.collect()
        self.assertIsNone(witness())

    def test_disabled_by_environment(self):
        with mock.patch.dict('os.environ', {'YOLO_TTA_SAM_GC_FREEZE': '0'}),                 mock.patch('gc.freeze') as freeze, mock.patch('gc.collect') as collect:
            receipt = runtime._freeze_worker_startup_heap()
        self.assertEqual(receipt, dict(schema='xta.sam_worker_gc_freeze/1', enabled=False))
        freeze.assert_not_called()
        collect.assert_not_called()

    def test_enabled_collects_before_freezing(self):
        calls = []
        with mock.patch('gc.collect', side_effect=lambda *a: calls.append('collect')),                 mock.patch('gc.freeze', side_effect=lambda: calls.append('freeze')):
            runtime._freeze_worker_startup_heap()
        self.assertEqual(calls, ['collect', 'freeze'])

    def test_receipt_is_per_worker_startup_state_not_scope_identity(self):
        # Scope identity compares manifest sam_runtime across workers; per-worker
        # startup values must be moved to worker_startup_receipt.
        import inspect
        source = inspect.getsource(runtime._execute_interpolation_tracker_task)
        self.assertIn("'resident_cpu_bytes', 'gc_freeze',", source)


if __name__ == '__main__':
    unittest.main()

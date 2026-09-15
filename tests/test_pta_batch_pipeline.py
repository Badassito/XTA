"""CPU-only scheduling and lifetime tests for overlapping PTA publication."""

from __future__ import annotations

import gc
import threading
import unittest
import weakref
from concurrent.futures import ThreadPoolExecutor

from XTA.pta_batch_pipeline import OrderedBatchPipeline


class _FatalPublication(BaseException):
    pass


class _Payload:
    pass


class OrderedBatchPipelineTests(unittest.TestCase):
    def test_two_slots_allow_one_batch_ahead_of_blocked_consumer(self):
        started = threading.Event()
        release = threading.Event()
        allocated_second = threading.Event()
        attempted_third = threading.Event()
        allocated_third = threading.Event()
        merged = []
        producer_thread = []

        def consume(value):
            if value == 0:
                started.set()
                self.assertTrue(release.wait(5))
            return value

        def merged_result(value):
            self.assertEqual(threading.get_ident(), producer_thread[0])
            merged.append(value)

        def produce():
            producer_thread.append(threading.get_ident())
            with OrderedBatchPipeline(consume, on_result=merged_result) as pipeline:
                for value in range(3):
                    if value == 2:
                        attempted_third.set()
                    with pipeline.reserve(10) as slot:
                        if value == 1:
                            allocated_second.set()
                        if value == 2:
                            allocated_third.set()
                        slot.submit(value)

        with ThreadPoolExecutor(max_workers=1) as producer:
            result = producer.submit(produce)
            try:
                self.assertTrue(started.wait(5))
                self.assertTrue(allocated_second.wait(5))
                self.assertTrue(attempted_third.wait(5))
                self.assertFalse(allocated_third.is_set())
            finally:
                release.set()
            result.result(timeout=5)
        self.assertEqual(merged, [0, 1, 2])

    def test_byte_limit_blocks_second_batch_even_with_a_free_slot(self):
        started = threading.Event()
        release = threading.Event()
        attempted = threading.Event()
        admitted = threading.Event()

        def consume(value):
            if value == 0:
                started.set()
                self.assertTrue(release.wait(5))
            return value

        def produce():
            with OrderedBatchPipeline(consume, max_pending_bytes=100) as pipeline:
                with pipeline.reserve(60) as slot:
                    slot.submit(0)
                attempted.set()
                with pipeline.reserve(60) as slot:
                    admitted.set()
                    slot.submit(1)

        with ThreadPoolExecutor(max_workers=1) as producer:
            result = producer.submit(produce)
            try:
                self.assertTrue(started.wait(5))
                self.assertTrue(attempted.wait(5))
                self.assertFalse(admitted.is_set())
            finally:
                release.set()
            result.result(timeout=5)

    def test_single_oversize_batch_is_exclusive_including_reservations(self):
        with OrderedBatchPipeline(lambda value: value, max_pending_bytes=100) as pipeline:
            with pipeline.reserve(150) as oversized:
                self.assertEqual(pipeline.pending_count, 1)
                self.assertEqual(pipeline.pending_bytes, 150)
                with self.assertRaisesRegex(RuntimeError, "Unused batch reservations"):
                    pipeline.reserve(1)
                oversized.submit(7)
            pipeline.drain()
            self.assertEqual(pipeline.pending_count, 0)
            with pipeline.reserve(1):
                with self.assertRaisesRegex(RuntimeError, "Unused batch reservations"):
                    pipeline.reserve(150)

    def test_unused_reservation_cancels_on_producer_error(self):
        with OrderedBatchPipeline(lambda value: value) as pipeline:
            with self.assertRaisesRegex(ValueError, "producer"):
                with pipeline.reserve(30):
                    self.assertEqual(pipeline.pending_count, 1)
                    raise ValueError("producer")
            self.assertEqual(pipeline.pending_count, 0)
            self.assertEqual(pipeline.pending_bytes, 0)
            with pipeline.reserve(20) as slot:
                slot.submit(1)
            pipeline.drain()
            slot.cancel()
            self.assertEqual(pipeline.pending_bytes, 0)
            with self.assertRaisesRegex(RuntimeError, "no longer active"):
                slot.submit(2)

    def test_get_ready_merges_once_and_drain_can_be_reused(self):
        merged = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            with OrderedBatchPipeline(lambda value: value, executor=executor, on_result=merged.append) as pipeline:
                with pipeline.reserve() as slot:
                    slot.submit(1)
                executor.submit(lambda: None).result(timeout=5)
                self.assertEqual(pipeline.get_ready(), [1])
                self.assertEqual(pipeline.get_ready(), [])
                self.assertEqual(pipeline.drain(), [])
                with pipeline.reserve() as slot:
                    slot.submit(2)
                self.assertEqual(pipeline.drain(), [2])
            self.assertEqual(executor.submit(lambda: 3).result(timeout=5), 3)
        self.assertEqual(merged, [1, 2])

    def test_reserve_collects_ready_results_through_callback(self):
        merged = []
        with ThreadPoolExecutor(max_workers=1) as executor:
            with OrderedBatchPipeline(lambda value: value, executor=executor, on_result=merged.append) as pipeline:
                with pipeline.reserve() as slot:
                    slot.submit(1)
                executor.submit(lambda: None).result(timeout=5)
                with pipeline.reserve() as slot:
                    self.assertEqual(merged, [1])
                    slot.submit(2)
        self.assertEqual(merged, [1, 2])

    def test_consumer_failure_drains_queued_work_before_raising(self):
        started = threading.Event()
        release = threading.Event()
        retired = []
        first_error = ValueError("first batch")

        def consume(value):
            if value == 0:
                started.set()
                self.assertTrue(release.wait(5))
            retired.append(value)
            if value == 0:
                raise first_error
            raise RuntimeError("second batch")

        pipeline = OrderedBatchPipeline(consume)
        with pipeline.reserve() as slot:
            slot.submit(0)
        self.assertTrue(started.wait(5))
        with pipeline.reserve() as slot:
            slot.submit(1)
        release.set()
        with self.assertRaises(ValueError) as caught:
            pipeline.drain()
        self.assertIs(caught.exception, first_error)
        self.assertEqual(retired, [0, 1])
        self.assertEqual(pipeline.pending_count, 0)
        self.assertEqual(pipeline.pending_bytes, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            pipeline.reserve()
        self.assertEqual(pipeline.close(), [])

    def test_base_exception_also_drains_and_external_executor_survives(self):
        first_started = threading.Event()
        release = threading.Event()
        retired = []
        fatal = _FatalPublication("unsafe device owners retained by consumer")

        def consume(value):
            if value == 0:
                first_started.set()
                self.assertTrue(release.wait(5))
                raise fatal
            retired.append(value)
            return value

        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline = OrderedBatchPipeline(consume, executor=executor)
            with pipeline.reserve() as slot:
                slot.submit(0)
            self.assertTrue(first_started.wait(5))
            with pipeline.reserve() as slot:
                slot.submit(1)
            release.set()
            with self.assertRaises(_FatalPublication) as caught:
                pipeline.close()
            self.assertIs(caught.exception, fatal)
            self.assertEqual(retired, [1])
            self.assertEqual(executor.submit(lambda: 5).result(timeout=5), 5)

    def test_callback_failure_never_recounts_completed_outputs(self):
        started = threading.Event()
        release = threading.Event()
        callbacks = []
        retired = []
        first_error = ValueError("merge failed")

        def consume(value):
            if value == 0:
                started.set()
                self.assertTrue(release.wait(5))
            retired.append(value)
            return value

        def merge(value):
            callbacks.append(value)
            if value == 0:
                raise first_error

        pipeline = OrderedBatchPipeline(consume, on_result=merge)
        with pipeline.reserve() as slot:
            slot.submit(0)
        self.assertTrue(started.wait(5))
        with pipeline.reserve() as slot:
            slot.submit(1)
        release.set()
        with self.assertRaises(ValueError) as caught:
            pipeline.close()
        self.assertIs(caught.exception, first_error)
        self.assertEqual(callbacks, [0, 1])
        self.assertEqual(retired, [0, 1])
        pipeline.close()
        self.assertEqual(callbacks, [0, 1])

    def test_completed_error_is_seen_before_next_output_reservation(self):
        def consume(value):
            raise ValueError("already failed")

        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline = OrderedBatchPipeline(consume, executor=executor)
            with pipeline.reserve() as slot:
                slot.submit(0)
            executor.submit(lambda: None).result(timeout=5)
            with self.assertRaisesRegex(ValueError, "already failed"):
                pipeline.reserve()
            self.assertEqual(pipeline.pending_count, 0)

    def test_payload_remains_owned_until_consumer_finishes(self):
        started = threading.Event()
        release = threading.Event()

        def consume(payload):
            started.set()
            self.assertTrue(release.wait(5))
            return 1

        with OrderedBatchPipeline(consume) as pipeline:
            with pipeline.reserve() as slot:
                payload = _Payload()
                reference = weakref.ref(payload)
                slot.submit(payload)
                del payload
            try:
                self.assertTrue(started.wait(5))
                gc.collect()
                self.assertIsNotNone(reference())
            finally:
                release.set()
            pipeline.drain()
        gc.collect()
        self.assertIsNone(reference())

    def test_producer_exception_still_drains_and_preserves_both_errors(self):
        retired = []
        producer_error = ValueError("producer error")
        consumer_error = RuntimeError("publication error")

        def consume(value):
            retired.append(value)
            raise consumer_error

        with self.assertRaises(ValueError) as caught:
            with OrderedBatchPipeline(consume) as pipeline:
                with pipeline.reserve() as slot:
                    slot.submit(1)
                raise producer_error
        self.assertIs(caught.exception, producer_error)
        self.assertIs(caught.exception.__cause__, consumer_error)
        self.assertEqual(retired, [1])

    def test_close_returns_unused_reservations_and_is_idempotent(self):
        pipeline = OrderedBatchPipeline(lambda value: value)
        reservation = pipeline.reserve(40)
        pipeline.close()
        pipeline.close()
        reservation.cancel()
        self.assertEqual(pipeline.pending_count, 0)
        self.assertEqual(pipeline.pending_bytes, 0)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            reservation.submit(1)

    def test_executor_submission_failure_cancels_unused_capacity(self):
        with ThreadPoolExecutor(max_workers=1) as executor:
            pipeline = OrderedBatchPipeline(lambda value: value, executor=executor)
            slot = pipeline.reserve(40)
            executor.shutdown(wait=True)
            payload = _Payload()
            with self.assertRaisesRegex(RuntimeError, "cannot schedule new futures"):
                slot.submit(payload)
            # No consumer accepted this object: caller still has sole ownership.
            self.assertEqual(pipeline.pending_count, 0)
            self.assertEqual(pipeline.pending_bytes, 0)
            slot.cancel()
            with self.assertRaisesRegex(RuntimeError, "closed"):
                pipeline.reserve()

    def test_nested_queue_transfers_collector_only_after_producer_drains(self):
        publications = []
        results = []
        with ThreadPoolExecutor(max_workers=1) as gpu_executor:
            with ThreadPoolExecutor(max_workers=1) as file_executor:
                host = OrderedBatchPipeline(
                    lambda value: publications.append(value), executor=file_executor,
                )

                def consume_gpu(value):
                    with host.reserve(10) as slot:
                        slot.submit(value)
                    return value

                with OrderedBatchPipeline(
                    consume_gpu, executor=gpu_executor, on_result=results.append,
                ) as gpu:
                    for value in range(8):
                        with gpu.reserve(10) as slot:
                            slot.submit(value)
                host.close()
        self.assertEqual(publications, list(range(8)))
        self.assertEqual(results, list(range(8)))

    def test_resize_reserves_count_before_encode_and_waits_for_older_bytes(self):
        started = threading.Event()
        release = threading.Event()
        encoded = threading.Event()
        resized = threading.Event()
        observed = []
        merged = []

        def consume(value):
            if value == 0:
                started.set()
                self.assertTrue(release.wait(5))
            return value

        def produce():
            with OrderedBatchPipeline(
                consume, max_pending_bytes=100, on_result=merged.append,
            ) as pipeline:
                with pipeline.reserve(60) as slot:
                    slot.submit(0)
                with pipeline.reserve(0) as slot:
                    # Encoding is allowed only after its count reservation.
                    observed.append((pipeline.pending_count, pipeline.pending_bytes))
                    encoded.set()
                    slot.resize(70)
                    observed.append((pipeline.pending_count, pipeline.pending_bytes))
                    resized.set()
                    slot.submit(1)
                pipeline.drain()
                observed.append((pipeline.pending_count, pipeline.pending_bytes))

        with ThreadPoolExecutor(max_workers=1) as producer:
            result = producer.submit(produce)
            try:
                self.assertTrue(started.wait(5))
                self.assertTrue(encoded.wait(5))
                self.assertFalse(resized.is_set())
            finally:
                release.set()
            result.result(timeout=5)
        self.assertEqual(observed, [(2, 60), (1, 70), (0, 0)])
        self.assertEqual(merged, [0, 1])

    def test_resize_oversized_batch_is_exclusive_and_does_not_wait_on_self(self):
        with OrderedBatchPipeline(lambda value: value, max_pending_bytes=100) as pipeline:
            with pipeline.reserve(0) as slot:
                slot.resize(150)
                self.assertEqual(pipeline.pending_count, 1)
                self.assertEqual(pipeline.pending_bytes, 150)
                slot.resize(200)
                self.assertEqual(pipeline.pending_bytes, 200)
                with self.assertRaisesRegex(RuntimeError, "Unused batch reservations"):
                    pipeline.reserve(1)
                slot.submit(1)
            pipeline.drain()
            self.assertEqual(pipeline.pending_bytes, 0)

    def test_resize_cannot_bypass_another_unused_reservation(self):
        with OrderedBatchPipeline(lambda value: value, max_pending_bytes=100) as pipeline:
            with pipeline.reserve(40) as first:
                with pipeline.reserve(0) as second:
                    with self.assertRaisesRegex(RuntimeError, "Other unused batch reservations"):
                        second.resize(150)
                    self.assertEqual(second.byte_count, 0)
                    self.assertEqual(pipeline.pending_bytes, 40)
                    first.cancel()
                    second.resize(150)
                    self.assertEqual(pipeline.pending_count, 1)
                    self.assertEqual(pipeline.pending_bytes, 150)

    def test_resize_shrink_and_cancel_release_exact_bytes(self):
        with OrderedBatchPipeline(lambda value: value, max_pending_bytes=100) as pipeline:
            with pipeline.reserve(50) as first:
                with pipeline.reserve(40) as second:
                    first.resize(20)
                    self.assertEqual(pipeline.pending_bytes, 60)
                    second.resize(0)
                    self.assertEqual(pipeline.pending_bytes, 20)
                self.assertEqual(pipeline.pending_bytes, 20)
                first.resize(90)
            self.assertEqual(pipeline.pending_bytes, 0)
            self.assertEqual(pipeline.pending_count, 0)

    def test_resize_rejects_invalid_token_or_size_without_changing_accounting(self):
        with OrderedBatchPipeline(lambda value: value) as pipeline:
            with pipeline.reserve(20) as slot:
                with self.assertRaises(ValueError):
                    slot.resize(-1)
                with self.assertRaises(TypeError):
                    slot.resize(0.5)
                self.assertEqual(pipeline.pending_bytes, 20)
                slot.submit(1)
                with self.assertRaisesRegex(RuntimeError, "no longer active"):
                    slot.resize(5)
            with pipeline.reserve(10) as canceled:
                canceled.cancel()
                with self.assertRaisesRegex(RuntimeError, "no longer active"):
                    canceled.resize(5)
        with self.assertRaisesRegex(RuntimeError, "closed"):
            slot.resize(5)

    def test_resize_callback_error_poisons_queue_and_drains_admitted_work(self):
        started = threading.Event()
        release = threading.Event()
        attempting_resize = threading.Event()
        consumed = []
        callbacks = []
        callback_error = ValueError("merge during resize failed")
        state = []

        def consume(value):
            if value == 0:
                started.set()
                self.assertTrue(release.wait(5))
            consumed.append(value)
            return value

        def merge(value):
            callbacks.append(value)
            if value == 0:
                raise callback_error

        def produce():
            pipeline = OrderedBatchPipeline(
                consume, capacity=3, max_pending_bytes=100, on_result=merge,
            )
            for value in (0, 1):
                with pipeline.reserve(40) as slot:
                    slot.submit(value)
            with pipeline.reserve(0) as slot:
                attempting_resize.set()
                try:
                    slot.resize(80)
                finally:
                    state.append((pipeline.pending_count, pipeline.pending_bytes))

        with ThreadPoolExecutor(max_workers=1) as producer:
            result = producer.submit(produce)
            try:
                self.assertTrue(started.wait(5))
                self.assertTrue(attempting_resize.wait(5))
            finally:
                release.set()
            with self.assertRaises(ValueError) as caught:
                result.result(timeout=5)
        self.assertIs(caught.exception, callback_error)
        self.assertEqual(consumed, [0, 1])
        self.assertEqual(callbacks, [0, 1])
        self.assertEqual(state, [(0, 0)])

    def test_rejects_invalid_limits_and_byte_estimates(self):
        for capacity in (0, -1):
            with self.assertRaises(ValueError):
                OrderedBatchPipeline(lambda value: value, capacity=capacity)
        for limit in (0, -1):
            with self.assertRaises(ValueError):
                OrderedBatchPipeline(lambda value: value, max_pending_bytes=limit)
        with OrderedBatchPipeline(lambda value: value) as pipeline:
            with self.assertRaises(ValueError):
                pipeline.reserve(-1)
            with self.assertRaises(TypeError):
                pipeline.reserve(1.5)


if __name__ == "__main__":
    unittest.main()

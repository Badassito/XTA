"""Native worker/owner transactions using the production actor wire encoder."""
import os
import socket
import threading
import unittest

from XTA.native_scheduler import CPU, GPU, PreparedWorkScheduler, recv_dispatch, send_receipt


class PreparedSchedulerTests(unittest.TestCase):
    def setUp(self):
        self.owner = PreparedWorkScheduler(127, 100, 3)
        self.addCleanup(self.owner.close)

    def actor(self, worker=0, backend=CPU, window=1, device=-1):
        local, remote = socket.socketpair()
        with local:
            self.owner.register_worker(worker, 5, os.getpid(), backend, window, local, device)
        remote.settimeout(3)
        self.addCleanup(remote.close)
        return remote

    def test_delayed_consumers_and_final_before_compute(self):
        actor = self.actor()
        self.owner.declare_parent(40, 2, 50)
        parent = self.owner.wait_preparation(2)
        self.owner.declare_parent(41, 1, 60)
        self.assertIsNone(self.owner.wait_preparation(0))
        self.owner.submit_prepared(0, 40, b"projection one", CPU, retained_bytes=15)
        self.owner.submit_prepared(1, 40, b"projection two", CPU, retained_bytes=15)
        first, first_data = recv_dispatch(actor)
        self.assertEqual(first_data, b"projection one")
        send_receipt(actor, first, "installed")
        send_receipt(actor, first, "final", payload=b"result one")
        # The actor receives the next prepared task before Python reads any science event.
        second, second_data = recv_dispatch(actor)
        self.assertEqual(second_data, b"projection two")
        send_receipt(actor, first, "compute")
        send_receipt(actor, second, "installed")
        send_receipt(actor, second, "final", payload=b"result two")
        self.assertEqual(self.owner.wait_publication(2), parent)
        self.assertEqual(self.owner.snapshot()["bytes_used"], 80)
        self.assertEqual(self.owner.wait_event(2), ("final", first, b"result one"))
        self.assertEqual(self.owner.wait_event(2), ("final", second, b"result two"))
        with self.assertRaisesRegex(RuntimeError, "retirement"):
            self.owner.retire_parent(*parent)
        self.owner.retire_task(0, first.generation)
        self.owner.retire_task(1, second.generation)
        self.assertIsNone(self.owner.wait_preparation(0))
        self.owner.retire_parent(*parent)
        self.assertEqual(self.owner.wait_preparation(2)[0], 41)

    def test_preparation_window_and_residency_follow_verified_final(self):
        self.owner.configure_capacity(100, 2, 70, 1)
        actor = self.actor(4, GPU, device=0)
        for parent_id in (10, 11):
            self.owner.declare_parent(parent_id, 1, 40, exclusive_worker_kind=GPU)
        parent = self.owner.wait_preparation(2)
        self.assertEqual(parent[0], 10)
        self.owner.submit_prepared(100, 10, b"resident", GPU)
        lease, _ = recv_dispatch(actor)
        send_receipt(actor, lease, "installed")
        send_receipt(actor, lease, "compute", payload=b"compute complete")
        self.assertEqual(self.owner.wait_event(2)[0], "compute")
        self.assertIsNone(self.owner.wait_preparation(0))
        send_receipt(actor, lease, "final")
        second = self.owner.wait_preparation(2)
        self.assertEqual(second[0], 11)
        self.assertEqual(self.owner.snapshot()["bytes_used"], 80)
        self.owner.retire_parent(*parent)

    def test_control_reply_bound_to_native_request(self):
        actor = self.actor()
        generation = self.owner.send_control(0, b"release model", expect_reply=True)
        lease, data = recv_dispatch(actor)
        self.assertEqual(data, b"release model")
        self.assertTrue(lease.control)
        self.assertEqual(lease.generation, generation)
        send_receipt(actor, lease, "control", payload=b"release proof")
        self.assertEqual(self.owner.wait_event(2), ("control", lease, b"release proof"))
        self.owner.send_control(0, b"stop", stop=True)
        self.assertEqual(recv_dispatch(actor), (None, b"stop"))
        actor.close()
        self.assertFalse(self.owner.snapshot()["failed"])

    def test_stage_release_wakes_dispatch_without_python_pump(self):
        actor = self.actor(3, GPU, device=8)
        gate = self.owner.try_acquire_stage(8)
        self.assertIsNotNone(gate)
        self.owner.declare_parent(2, 1, 15)
        self.owner.wait_preparation(2)
        self.owner.submit_prepared(50, 2, b"render", GPU)
        self.assertEqual(self.owner.snapshot()["active"], 0)
        self.owner.release_stage(8, gate)
        lease, _ = recv_dispatch(actor)
        self.assertIsNone(self.owner.try_acquire_stage(8))
        send_receipt(actor, lease, "installed")
        send_receipt(actor, lease, "final")
        self.assertTrue(self.owner.wait_task_final(50, lease.generation, 2))
        self.assertIsNotNone(self.owner.try_acquire_stage(8))

    def test_cancel_wakes_waiters(self):
        failures = []
        def await_ticket():
            try:
                self.owner.wait_publication()
            except RuntimeError as error:
                failures.append(str(error))
        thread = threading.Thread(target=await_ticket)
        thread.start()
        self.owner.cancel("workspace preparation failed")
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertIn("workspace preparation failed", failures[0])

    def test_close_interrupts_silent_actor_with_open_peer(self):
        actor = self.actor()
        thread = threading.Thread(target=self.owner.close)
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive(), "native shutdown waited for actor to exit")
        with self.assertRaises(EOFError):
            recv_dispatch(actor)

    def test_device_fence_retirement_remains_possible_after_fault(self):
        gate = self.owner.try_acquire_stage(0)
        self.assertIsNotNone(gate)
        self.owner.cancel("actor failed")
        with self.assertRaisesRegex(RuntimeError, "actor failed"):
            self.owner.try_acquire_stage(1)
        self.owner.release_stage(0, gate)
        self.assertEqual(self.owner.snapshot()["stage_claims"], 0)

    def test_host_only_parent_and_verified_component_retirement(self):
        self.owner.configure_capacity(100, 2, 40, 2)
        self.owner.declare_parent(8, 0, 60)
        first = self.owner.wait_preparation(2)
        self.owner.declare_parent(9, 0, 50)
        self.assertIsNone(self.owner.wait_preparation(0))
        self.owner.host_prepared(*first)
        self.assertEqual(self.owner.wait_publication(2), first)
        self.assertIsNone(self.owner.wait_preparation(0))
        self.owner.set_parent_remaining_bytes(*first, 20)
        second = self.owner.wait_preparation(2)
        self.assertEqual(second[0], 9)
        with self.assertRaisesRegex(RuntimeError, "component retirement"):
            self.owner.set_parent_remaining_bytes(*first, 20)
        self.assertEqual(self.owner.snapshot()["bytes_used"], 70)
        self.owner.retire_parent(*first)
        self.owner.host_prepared(*second)
        self.owner.retire_parent(*second)
        self.assertEqual(self.owner.snapshot()["bytes_used"], 0)

    def test_guarded_sam_restore_owns_actual_oversize_until_retirement(self):
        self.owner.declare_parent(10, 0, 25)
        previous = self.owner.wait_preparation(2)
        self.owner.host_prepared(*previous)
        with self.assertRaisesRegex(RuntimeError, "exceeds"):
            self.owner.declare_parent(11, 0, 130)
        with self.assertRaisesRegex(RuntimeError, "zero-task"):
            self.owner.declare_parent(11, 1, 130, allow_oversize_host=True)
        self.owner.declare_parent(11, 0, 130, allow_oversize_host=True)
        self.assertIsNone(self.owner.wait_preparation(0))
        self.owner.retire_parent(*previous)
        oversized = self.owner.wait_preparation(2)
        self.assertEqual(oversized[0], 11)
        self.owner.declare_parent(12, 0, 0)
        self.owner.host_prepared(*oversized)
        self.assertEqual(self.owner.snapshot()["bytes_used"], 130)
        self.assertEqual(self.owner.snapshot()["oversize_host_bytes"], 30)
        self.assertIsNone(self.owner.wait_preparation(0))
        self.assertFalse(self.owner.try_resize_capacity(90, 50))
        self.owner.set_parent_remaining_bytes(*oversized, 101)
        self.assertIsNone(self.owner.wait_preparation(0))
        self.owner.set_parent_remaining_bytes(*oversized, 100)
        self.assertEqual(self.owner.wait_preparation(2)[0], 12)
        self.assertEqual(self.owner.snapshot()["bytes_used"], 100)

    def test_runtime_partition_respects_live_and_declared_work(self):
        self.owner.configure_capacity(100, 1, 80, 1)
        self.owner.declare_parent(11, 0, 40)
        first = self.owner.wait_preparation(2)
        self.owner.host_prepared(*first)
        self.owner.declare_parent(12, 0, 60)
        self.assertFalse(self.owner.try_resize_capacity(30, 20))
        self.assertFalse(self.owner.try_resize_capacity(50, 40))
        self.assertEqual(self.owner.snapshot()["byte_capacity"], 100)
        self.assertTrue(self.owner.try_resize_capacity(60, 45, pressure_byte_capacity=80))
        after = self.owner.snapshot()
        self.assertEqual((after["byte_capacity"], after["inference_byte_capacity"], after["bytes_used"],
                          after["pressure_byte_capacity"]), (60, 45, 40, 80))
        self.assertIsNone(self.owner.wait_preparation(0))
        self.owner.retire_parent(*first)
        self.assertEqual(self.owner.wait_preparation(2)[0], 12)
        self.owner.declare_parent(13, 1, 1)
        self.assertFalse(self.owner.try_resize_capacity(60, 45))

    def test_cpu_restore_headroom_remains_free_until_a_real_grant(self):
        generation = self.owner.declare_parent(41, 0, 75, admission_reserve=30)
        self.assertIsNone(self.owner.wait_preparation(0))
        self.assertEqual(self.owner.snapshot()["bytes_used"], 0)
        self.owner.declare_parent(42, 0, 25)
        cheap = self.owner.wait_preparation(2)
        self.assertEqual(cheap[0], 42)
        self.assertTrue(self.owner.update_host_reserve(41, generation, 10))
        self.assertIsNone(self.owner.wait_preparation(0))
        self.assertTrue(self.owner.update_host_reserve(41, generation, 0))
        self.assertEqual(self.owner.wait_preparation(2), (41, generation))
        self.assertFalse(self.owner.update_host_reserve(41, generation, 30))
        self.assertEqual(self.owner.snapshot()["bytes_used"], 100)

    def test_planned_mutable_handle_rejected_before_funding(self):
        self.actor()
        self.owner.declare_parent(48, 1, 17, descriptor_only=True)
        for role in (1, 2):
            with self.assertRaisesRegex(RuntimeError, "immutable source handles"):
                self.owner.submit_planned(6, 48, b"payload", CPU, handles=[(0, "unused", role)])
        self.assertEqual(self.owner.snapshot()["bytes_used"], 0)
        self.assertEqual(self.owner.snapshot()["dispatched"], 0)

    def test_semantic_adoption_controls_pressure_with_distinct_result_escrow(self):
        actor = self.actor()
        self.owner.configure_capacity(220, 2, 210, 2, pressure_byte_capacity=256)
        self.owner.declare_parent(7, 1, 210, retirement_pressure_bytes=192)
        parent = self.owner.wait_preparation(2)
        self.owner.submit_prepared(14, 7, b"lease", CPU)
        lease, _ = recv_dispatch(actor)
        send_receipt(actor, lease, "installed")
        send_receipt(actor, lease, "final")
        self.assertEqual(self.owner.wait_publication(2), parent)
        self.assertEqual(self.owner.snapshot()["retained_parent_bytes"], 0)
        self.owner.adopt_postprocess(*parent)
        self.assertTrue(self.owner.snapshot()["retirement_pressure"])
        self.assertEqual(self.owner.snapshot()["retained_parent_bytes"], 192)
        self.owner.set_parent_remaining_bytes(*parent, 191)
        self.assertTrue(self.owner.snapshot()["retirement_pressure"])
        self.owner.set_parent_remaining_bytes(*parent, 161)
        self.assertTrue(self.owner.snapshot()["retirement_pressure"])
        self.owner.set_parent_remaining_bytes(*parent, 160)
        self.assertFalse(self.owner.snapshot()["retirement_pressure"])

    def test_immutable_parent_refills_after_final_without_python_preparation(self):
        actor = self.actor(1, GPU, window=1, device=0)
        self.owner.configure_capacity(100, 3, 60, 1)
        generation = self.owner.declare_parent(1, 1, 30, descriptor_only=True, exclusive_worker_kind=GPU)
        self.owner.submit_planned(100, 1, b"first", GPU)
        first, _ = recv_dispatch(actor)
        second_generation = self.owner.declare_parent(2, 2, 40, descriptor_only=True, exclusive_worker_kind=GPU)
        self.owner.submit_planned(101, 2, b"second", GPU)
        self.owner.submit_planned(102, 2, b"third", GPU)
        # The next parent awaits the resident worker, without acquiring its bytes.
        self.assertEqual(self.owner.snapshot()["bytes_used"], 30)
        send_receipt(actor, first, "installed")
        send_receipt(actor, first, "final")
        # Only the actor runs now. No controller drain, preparation callback,
        # publication request or Python refill is needed to get the second parent.
        second, data = recv_dispatch(actor)
        self.assertEqual((second.task_id, data), (101, b"second"))
        send_receipt(actor, second, "installed")
        send_receipt(actor, second, "final")
        third, _ = recv_dispatch(actor)
        send_receipt(actor, third, "installed")
        send_receipt(actor, third, "final")
        self.assertTrue(self.owner.wait_task_final(102, third.generation, 2))
        self.assertEqual(self.owner.snapshot()["bytes_used"], 70)
        self.assertIsNone(self.owner.wait_preparation(0))
        self.assertEqual(self.owner.wait_publication(2), (1, generation))
        self.assertEqual(self.owner.wait_publication(2), (2, second_generation))

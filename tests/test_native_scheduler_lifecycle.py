"""Receipt, publication and actor-process checks against the production native scheduler."""
from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import ExitStack, closing
import ctypes
from dataclasses import replace
from itertools import product
import mmap
import multiprocessing
import os
from pathlib import Path
import pickle
import queue
import select
import socket
import struct
import tempfile
import threading
import time
import unittest

from XTA.native_scheduler import CPU, GPU, PreparedWorkScheduler, recv_dispatch, send_receipt
from XTA.tta_worker_transport import DetectorActorPort


def _actor_advanced_while_controller_held_gil(stream, timeout_ms):
    # PyDLL intentionally holds this process's GIL. poll/WSAPoll distinguishes
    # actual actor progress from Python event loops that resume after the wait.
    windows = os.name == "nt"

    class PollDescriptor(ctypes.Structure):
        _fields_ = [
            ("fd", ctypes.c_size_t if windows else ctypes.c_int),
            ("events", ctypes.c_short),
            ("revents", ctypes.c_short),
        ]

    library = ctypes.PyDLL("ws2_32" if windows else None)
    poll = library.WSAPoll if windows else library.poll
    poll.argtypes = [
        ctypes.POINTER(PollDescriptor),
        ctypes.c_ulong,
        ctypes.c_int,
    ]
    poll.restype = ctypes.c_int
    read_event = 0x100 if windows else 1
    descriptor = PollDescriptor(stream.fileno(), read_event, 0)
    stream.sendall(b"G")
    entered = time.monotonic_ns()
    ready = poll(ctypes.byref(descriptor), 1, timeout_ms)
    returned = time.monotonic_ns()
    if ready != 1 or not descriptor.revents & read_event:
        return False
    with stream.makefile("rb") as reader:
        first_receipt_sent, next_command_received = struct.unpack("<QQ", reader.read(16))
    return entered <= first_receipt_sent <= next_command_received <= returned


def _write_result(port, task):
    with open(task["result_mask_path"], "r+b") as backing, mmap.mmap(backing.fileno(), 0) as view:
        port.installed(task["task_id"])
        start = task["slice_start"]
        stop = start + task["slice_count"]
        view[start:stop] = bytes([task["task_id"]]) * (stop - start)
    port.put(dict(type="result", ok=True, worker_kind="cpu", cpu_index=0,
                  task_id=task["task_id"], stats={"written": stop - start}))


def _detector_actor(stream, witness):
    # Only model execution is synthetic. Dispatch, ACK, result, compute refill,
    # process transport, publication, and stop all use the production interfaces.
    with stream, witness:
        port = DetectorActorPort(stream, queue.SimpleQueue())
        first = port.get()
        _write_result(port, first)
        second = port.get()  # Final freed the slot before compute arrived.
        port.compute_done(first["task_id"])
        witness.sendall(b"R")
        if witness.recv(1) != b"G":
            return
        time.sleep(0.05)  # Controller has entered PyDLL's OS wait.
        first_receipt = time.monotonic_ns()
        _write_result(port, second)
        third = port.get()
        next_dispatch = time.monotonic_ns()
        port.compute_done(second["task_id"])  # Must not free the new lease's slot.
        _write_result(port, third)
        port.compute_done(third["task_id"])
        witness.sendall(struct.pack("<QQ", first_receipt, next_dispatch))
        fourth = port.get()
        _write_result(port, fourth)
        port.compute_done(fourth["task_id"])
        if port.get() is not None:
            raise AssertionError("Expected native actor stop after the fourth task")


def _stop_process(process):
    if process.is_alive():
        process.terminate()
    process.join(5)
    process.close()


def _planned_detector_actor(stream, witness):
    with stream, witness:
        port = DetectorActorPort(stream, queue.SimpleQueue())
        persistent_sources = {}
        for expected_id in (1, 2, 3):
            task = port.get()
            received = time.monotonic_ns()
            if task["task_id"] != expected_id:
                raise AssertionError(f"Unexpected actor dispatch {task['task_id']} before {expected_id}")
            if port.materialize(task, persistent_sources):
                raise AssertionError("Planned source fixture received a transient writable mapping")
            if Path(task["source_volume_path"]).read_bytes() != b"shared decoded voxels":
                raise AssertionError("Planned task did not receive its shared source")
            if expected_id == 2 and Path(task["native_resize"]["path"]).read_bytes() != b"second parent source":
                raise AssertionError("Newly funded parent did not install its own source")
            port.installed(expected_id)
            if task["result_mode"] == "file":
                Path(task["result_mask_path"]).write_bytes(bytes([expected_id]) * task["processing_shape"][-1])
            if expected_id == 1:
                witness.sendall(b"A")
                if witness.recv(1) != b"G":
                    return
                time.sleep(0.05)  # Let the controller enter its GIL-held OS wait.
                first_final = time.monotonic_ns()
            port.put(dict(type="result", task_id=expected_id, ok=True, worker_kind="gpu", gpu_index=0,
                          stats={"frames": 1, "mode": task["result_mode"], "sources": len(persistent_sources)}))
            port.compute_done(expected_id)
            if expected_id == 2:
                witness.sendall(struct.pack("<QQ", first_final, received))
        if port.get() is not None:
            raise AssertionError("Expected actor stop after complete immutable inventory")
        for descriptor in persistent_sources.values():
            os.close(descriptor)


class NativeReceiptTests(unittest.TestCase):
    def _worker(self, stack, owner, worker_id, *, backend=CPU, window=1):
        source, actor = socket.socketpair()
        stack.enter_context(actor)
        actor.settimeout(3)
        with source:
            owner.register_worker(worker_id, 11, os.getpid(), backend, window, source,
                                  device=worker_id if backend == GPU else -1)
        return actor

    def test_invalid_receipt_cannot_change_any_credit_or_publish(self):
        for violation in ("old_epoch", "other_endpoint", "wrong_generation", "missing_install"):
            with self.subTest(violation=violation), ExitStack() as stack:
                owner = stack.enter_context(PreparedWorkScheduler(100, 32, 1))
                actor = self._worker(stack, owner, 7)
                other = self._worker(stack, owner, 8)
                owner.declare_parent(10, 2, 12)
                self.assertEqual(owner.wait_preparation(3)[0], 10)
                for task_id in (41, 42):
                    owner.submit_prepared(task_id, 10, b"descriptor", CPU, required_worker=7, retained_bytes=3)
                lease, _ = recv_dispatch(actor)
                if violation == "missing_install":
                    send_receipt(actor, lease, "final")
                elif violation == "old_epoch":
                    send_receipt(actor, replace(lease, epoch=lease.epoch - 1), "installed")
                elif violation == "other_endpoint":
                    send_receipt(other, lease, "installed")
                else:
                    send_receipt(actor, lease, "installed")
                    send_receipt(actor, replace(lease, generation=lease.generation + 1), "final")
                with self.assertRaisesRegex(RuntimeError, "receipt|ACK"):
                    owner.wait_publication(3)
                counters = owner.snapshot()
                expected = dict(bytes_used=15, tickets_used=1, active=1, final=0, dispatched=1,
                                ready=1, failed=1, installed=int(violation == "wrong_generation"))
                self.assertEqual({key: counters[key] for key in expected}, expected)

    def test_repeated_final_cannot_return_successor_slot_or_bytes(self):
        with ExitStack() as stack:
            owner = stack.enter_context(PreparedWorkScheduler(101, 64, 1))
            actor = self._worker(stack, owner, 1)
            owner.declare_parent(1, 3, 10)
            owner.wait_preparation(3)
            for task_id in (1, 2, 3):
                owner.submit_prepared(task_id, 1, b"descriptor", CPU, retained_bytes=5)
            first, _ = recv_dispatch(actor)
            send_receipt(actor, first, "installed")
            send_receipt(actor, first, "final")
            second, _ = recv_dispatch(actor)
            send_receipt(actor, second, "installed")
            send_receipt(actor, first, "compute")
            send_receipt(actor, first, "final")
            with self.assertRaisesRegex(RuntimeError, "duplicate final"):
                owner.wait_publication(3)
            counters = owner.snapshot()
            expected = dict(dispatched=2, installed=2, final=1, active=1, ready=1, bytes_used=20)
            self.assertEqual({key: counters[key] for key in expected}, expected)

    def test_worker_exit_wakes_waiter_with_outstanding_owned_bytes(self):
        with ExitStack() as stack:
            owner = stack.enter_context(PreparedWorkScheduler(102, 24, 1))
            actor = self._worker(stack, owner, 2)
            owner.declare_parent(1, 1, 12)
            owner.wait_preparation(3)
            owner.submit_prepared(22, 1, b"descriptor", CPU, retained_bytes=4)
            lease, _ = recv_dispatch(actor)
            send_receipt(actor, lease, "installed")
            actor.close()
            with self.assertRaisesRegex(RuntimeError, "worker 2"):
                owner.wait_task_final(22, lease.generation, 3)
            snapshot = owner.snapshot()
            self.assertEqual((snapshot["bytes_used"], snapshot["active"], snapshot["final"]), (16, 1, 0))

    def test_scientific_sender_mismatch_is_rejected_before_native_refill(self):
        with ExitStack() as stack:
            owner = stack.enter_context(PreparedWorkScheduler(103, 24, 1))
            # The production CPU wire namespace starts above the CUDA indices.
            actor = self._worker(stack, owner, 2**32)
            port = DetectorActorPort(actor, queue.SimpleQueue())
            owner.declare_parent(1, 2, 8)
            owner.wait_preparation(3)
            for task_id in (51, 52):
                owner.submit_prepared(task_id, 1, pickle.dumps({"task_id": task_id}), CPU)
            self.assertEqual(port.get()["task_id"], 51)
            port.installed(51)
            try:
                port.put(dict(type="result", task_id=51, worker_kind="cpu", cpu_index=1, ok=True, stats={}))
            except RuntimeError:
                pass  # The actor may also report its terminal failure to its caller.
            with self.assertRaisesRegex(RuntimeError, "worker|sender|identity|actor"):
                owner.wait_publication(2)
            counters = owner.snapshot()
            expected = dict(dispatched=1, installed=1, active=1, final=0, ready=1, tickets_used=1)
            self.assertEqual({key: counters[key] for key in expected}, expected)

    def test_d1_parent_reserves_actor_even_between_compute_and_final(self):
        with ExitStack() as stack:
            owner = stack.enter_context(PreparedWorkScheduler(104, 32, 2))
            actor = self._worker(stack, owner, 0, backend=GPU, window=2)
            owner.declare_parent(10, 1, 8, exclusive_worker_kind=GPU)
            self.assertEqual(owner.wait_preparation(2)[0], 10)
            owner.declare_parent(11, 1, 8)
            self.assertEqual(owner.wait_preparation(2)[0], 11)
            # Ordinary work is ready first while the exclusive parent is preparing.
            owner.submit_prepared(21, 11, b"direct_union", GPU)
            self.assertFalse(select.select([actor], [], [], 0.05)[0])
            owner.submit_prepared(20, 10, b"d1_owner", GPU)
            resident, data = recv_dispatch(actor)
            self.assertEqual(data, b"d1_owner")
            send_receipt(actor, resident, "installed")
            send_receipt(actor, resident, "compute", payload=b"computed")
            self.assertEqual(owner.wait_event(2), ("compute", resident, b"computed"))
            self.assertFalse(select.select([actor], [], [], 0.05)[0])
            send_receipt(actor, resident, "final")
            ordinary, data = recv_dispatch(actor)
            self.assertEqual((ordinary.parent_id, data), (11, b"direct_union"))
            send_receipt(actor, ordinary, "installed")
            send_receipt(actor, ordinary, "compute", payload=b"ordinary computed")
            self.assertEqual(owner.wait_event(2), ("compute", ordinary, b"ordinary computed"))
            owner.retire_parent(*owner.wait_publication(2))
            owner.declare_parent(12, 1, 8, exclusive_worker_kind=GPU)
            self.assertIsNone(owner.wait_preparation(0.05))
            send_receipt(actor, ordinary, "final")
            self.assertEqual(owner.wait_preparation(2)[0], 12)


class NativeDetectorIntegrationTests(unittest.TestCase):
    def test_unfunded_immutable_parent_dispatches_without_controller_preparation(self):
        from tests.native_detector_test_support import DetectorRun, detector_inputs, detector_view
        from tests.native_scheduler_cases import detector_task
        from XTA import runtime
        from XTA.tta_scheduler import TtaSchedulerCallbacks

        # Both successor forms must cross an actual native funding transition.
        modes = (("d1_owner", "file"), ("file", "d1_owner"))
        sources = ("disk", "memfd") if hasattr(os, "memfd_create") else ("disk",)
        for (first_mode, second_mode), source_kind in product(modes, sources):
            with self.subTest(successor=second_mode, source=source_kind), ExitStack() as stack:
                root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
                shared, fresh = root / "shared-source", root / "second-source"
                resize_ready = root / "resize.ready"
                for source, contents in ((shared, b"shared decoded voxels"), (fresh, b"second parent source")):
                    if source_kind == "memfd":
                        fd = os.memfd_create(source.name, os.MFD_CLOEXEC)
                        runtime._register_memfd_owner(source, fd, "immutable actor source")
                        stack.callback(runtime._release_memfd_owner_key, source)
                        os.write(fd, contents)
                    else:
                        source.write_bytes(contents)
                resize_ready.touch()
                inputs = detector_inputs(root, direct_union_inference_byte_limit=4,
                                         direct_union_total_dense_byte_limit=4, direct_union_inference_view_limit=1)
                run = stack.enter_context(DetectorRun(root, inputs=inputs))
                host, child = socket.socketpair()
                stack.enter_context(host)
                host.settimeout(15)
                with run.scheduler.open_worker("gpu", 0) as endpoint, child:
                    process = multiprocessing.get_context("spawn").Process(target=_planned_detector_actor,
                                                                            args=(endpoint, child))
                    process.start()
                    stack.callback(_stop_process, process)
                    run.scheduler.register_worker("gpu", 0, process.pid)

                def assemble(task, stats):
                    self.assertEqual(stats["mode"], task["result_mode"])
                    expected_sources = min(task["task_id"], 2) if source_kind == "memfd" else 0
                    self.assertEqual(stats["sources"], expected_sources)
                    if task["result_mode"] == "file":
                        path = Path(task["result_mask_path"])
                        self.assertEqual(path.read_bytes(), bytes([task["task_id"]]) * task["processing_shape"][-1])
                        path.unlink()
                    run.results.append(task["task_id"])

                run.scheduler.bind_result_callbacks(TtaSchedulerCallbacks(assemble, None, lambda: None, lambda: None))
                tasks = []
                for task_id, mode, width in ((1, first_mode, 2), (2, second_mode, 2), (3, "file", 4)):
                    view = detector_view(name=f"v{task_id}", num_slices=1, src_h=1, src_w=width)
                    path = str(root / f"{task_id}.mask") if mode == "file" else None
                    task = detector_task(task_id, view=view, mode=mode, processing_shape=(1, 1, width),
                                         result_mask_path=path, job_id=task_id, union_num_slices=1,
                                         source_volume_path=str(shared))
                    if task_id == 2:
                        task["native_resize"] = {"path": str(fresh), "sentinel": str(resize_ready)}
                    tasks.append(task)
                run.declare(*tasks, ticket_limit=3)
                if source_kind == "memfd":
                    # Only the native owner may hold the unfunded successor's source.
                    # Its actor cannot reach either registry alias as an ordinary file.
                    self.assertTrue(runtime._release_memfd_owner_key(shared))
                    self.assertTrue(runtime._release_memfd_owner_key(fresh))
                    self.assertFalse(shared.exists() or fresh.exists())
                # No Python preparation or result port has been started. Only A is funded.
                self.assertEqual(host.recv(1), b"A")
                snapshot = run.native.snapshot()
                self.assertEqual((snapshot["dispatched"], snapshot["blocked_parents"], snapshot["tickets_used"]),
                                 (1, 2, 1))
                self.assertTrue(_actor_advanced_while_controller_held_gil(host, 5000),
                                "Unfunded successor did not advance during the GIL-held interval")
                deadline = time.monotonic() + 3
                while run.native.snapshot()["final"] < 2:
                    self.assertLess(time.monotonic(), deadline)
                    time.sleep(0.005)
                # C needs four bytes. Its preparation slot and a ticket are free, but
                # one two-byte result still exists until actual scientific assembly.
                snapshot = run.native.snapshot()
                self.assertEqual((snapshot["dispatched"], snapshot["blocked_parents"], snapshot["tickets_used"],
                                  snapshot["bytes_used"]), (2, 1, 2, 2))
                self.assertFalse((root / "3.mask").exists())
                run.start_transport()
                run.drain_until(lambda: not run.scheduler.publication_outstanding(), timeout=10)
                self.assertEqual(run.results, [1, 2, 3])
                self.assertEqual(run.native.snapshot()["bytes_used"], 0)
                run.state.gpu_task_queues[0].put(None)
                process.join(5)
                self.assertEqual(process.exitcode, 0)

    @staticmethod
    def _adapter(native, root, *, allocate=None):
        from tests.native_detector_test_support import detector_inputs, detector_state
        from XTA.tta_scheduler import TtaScheduler

        inputs = detector_inputs(root, imgsz=1, min_conf=0.5, direct_union_inference_byte_limit=8,
                                 direct_union_total_dense_byte_limit=8, gpu_worker_process_active=False,
                                 gpu_device_count=0, request_baseline_workspaces=allocate)
        return TtaScheduler(native=native, state=detector_state(), inputs=inputs)

    def test_publication_failure_stops_owner_with_other_work_still_in_flight(self):
        from tests.native_detector_test_support import DetectorRun, detector_view
        from tests.native_scheduler_cases import detector_task, finish_actor
        from XTA.tta_scheduler import TtaSchedulerCallbacks

        publication = Future()
        with tempfile.TemporaryDirectory() as directory, DetectorRun(Path(directory)) as run:
            actor = run.worker(0, kind="cpu")
            run.start_transport()

            def consume(task, stats):
                run.results.append((task, stats))
                run.scheduler.track_publication(("model", task["view"].name), publication)

            run.scheduler.bind_result_callbacks(TtaSchedulerCallbacks(consume, None, lambda: None, lambda: None))
            first = detector_task(1, cpu=True, view=detector_view(name="first"))
            second = detector_task(2, cpu=True, view=detector_view(name="second"))
            run.declare(first, second, ticket_limit=2)
            run.drain_until(lambda: run.native.snapshot()["dispatched"] == 1)
            finish_actor(actor, run.take(0, kind="cpu"), cpu=True)
            run.drain_until(lambda: len(run.results) == 1)
            in_flight = run.take(0, kind="cpu")
            actor.installed(in_flight["task_id"])
            publication.set_exception(OSError("publication destination unavailable"))
            # No frontend service after the publisher delivers its terminal failure.
            with self.assertRaisesRegex(RuntimeError, "publication destination unavailable"):
                run.native.wait_preparation(2)
            counts = run.native.snapshot()
            self.assertEqual((counts["failed"], counts["active"], counts["final"], counts["tickets_used"]),
                             (1, 1, 1, 2))

    def test_invalid_inventory_refuses_entire_run_before_funding_valid_sibling(self):
        from XTA.geometry import ViewInfo

        for violation in ("task_id", "resident_gpu_parent_on_cpu"):
            with self.subTest(violation=violation), ExitStack() as stack:
                native = stack.enter_context(PreparedWorkScheduler(990, 32, 1))
                scheduler = self._adapter(native, Path(stack.enter_context(tempfile.TemporaryDirectory())))
                stack.enter_context(scheduler.open_worker("cpu", 0))
                scheduler.register_worker("cpu", 0, os.getpid(), 1)
                for task_id in (20, 21):
                    scheduler.state.gpu_worker_tasks_by_id[task_id] = dict(
                        task_id=task_id, model_name="detector", kind="fullframe", result_mode="file",
                        cpu_eligible=True, gpu_eligible=False, slice_start=0, slice_count=2,
                        view=ViewInfo(name=str(task_id), num_slices=2, src_h=1, src_w=1, pad_mode="clamp"),
                    )
                task = scheduler.state.gpu_worker_tasks_by_id[21]
                if violation == "task_id":
                    task["task_id"] = 99
                else:
                    task["result_mode"] = "d1_owner"
                with self.assertRaises((ValueError, RuntimeError)):
                    scheduler.declare_inventory(ticket_limit=1)
                self.assertIsNone(native.wait_preparation(0.05))
                counts = native.snapshot()
                self.assertEqual((counts["tickets_used"], counts["dispatched"], counts["blocked_parents"]), (0, 0, 0))

    def test_tta_actor_refills_while_controller_holds_gil_and_publisher_keeps_live_mapping(self):
        from XTA.geometry import ViewInfo
        from XTA.interpolation import _DirectUnionBackingLease
        from XTA.tta_scheduler import TtaSchedulerCallbacks

        with ExitStack() as stack:
            root = Path(stack.enter_context(tempfile.TemporaryDirectory()))
            native = stack.enter_context(PreparedWorkScheduler(987, 32, 2))
            mappings = {}
            workspace_requested = threading.Event()
            workspace_enabled = threading.Event()

            def allocate(task):
                workspace_requested.set()
                if not workspace_enabled.is_set():
                    return False
                key = ("detector", task["view"].name)
                path = root / (key[1] + ".mask")
                path.write_bytes(bytes(2))
                with path.open("r+b") as backing:
                    handle = stack.enter_context(closing(mmap.mmap(backing.fileno(), 0)))
                state.baseline_union_paths[key] = path
                confidence = root / (key[1] + ".confidence")
                confidence.write_bytes(bytes(2))
                with confidence.open("r+b") as backing:
                    confidence_handle = stack.enter_context(closing(mmap.mmap(backing.fileno(), 0)))
                state.baseline_confmap_paths[key] = confidence
                mappings[key] = handle, confidence_handle
                state.direct_union_backing_leases[key] = _DirectUnionBackingLease(key, len(handle) + len(confidence_handle))
                return True

            scheduler = self._adapter(native, root, allocate=allocate)
            state = scheduler.state
            for task_id, view_name, start, count in [(1, "first", 0, 2), (2, "second", 0, 1),
                                                     (3, "second", 1, 1), (4, "third", 0, 2)]:
                view = ViewInfo(name=view_name, num_slices=2, src_h=1, src_w=1, pad_mode="clamp")
                state.gpu_worker_tasks_by_id[task_id] = dict(
                    task_id=task_id, model_name="detector", kind="fullframe", result_mode="direct_union",
                    cpu_eligible=True, gpu_eligible=False, processing_shape=(2, 1, 1), view=view,
                    slice_start=start, slice_count=count,
                )
            state.gpu_worker_total_tasks = 4
            published = threading.Event()
            publisher_release = threading.Event()
            publisher = stack.enter_context(ThreadPoolExecutor(1))
            stack.callback(publisher_release.set)

            def publish():
                published.set()
                if not publisher_release.wait(30):
                    raise TimeoutError("Publication fixture did not retire")
                return bytes(mappings["detector", "first"][0])

            assembled = []
            futures = []

            def consume(task, stats):
                self.assertEqual(stats["written"], task["slice_count"])
                assembled.append(task["task_id"])
                key = ("detector", task["view"].name)
                if task["task_id"] == 1:
                    future = publisher.submit(publish)
                    scheduler.track_publication(key, future)
                    futures.append(future)

            scheduler.bind_result_callbacks(TtaSchedulerCallbacks(consume, None, lambda: None, lambda: None))
            host, actor_witness = socket.socketpair()
            stack.enter_context(host)
            host.settimeout(10)
            with scheduler.open_worker("cpu", 0) as endpoint, actor_witness:
                actor = multiprocessing.get_context("spawn").Process(target=_detector_actor,
                                                                      args=(endpoint, actor_witness))
                actor.start()
                stack.callback(_stop_process, actor)
                scheduler.register_worker("cpu", 0, actor.pid, 1)
            threads = []

            def stop_ports():
                state.push_drain_stop.set()
                for thread in threads:
                    thread.join(2)
                    self.assertFalse(thread.is_alive())

            stack.callback(stop_ports)
            scheduler.configure_result_transport(push_drain_active=True, track_thread=lambda t, _e: threads.append(t))
            scheduler.declare_inventory(ticket_limit=2)

            def service_until(condition):
                deadline = time.monotonic() + 10
                while not condition():
                    self.assertLess(time.monotonic(), deadline, f"Detector failed to progress: {native.snapshot()}")
                    scheduler.drain_process_inference_results()
                    if not condition():
                        scheduler.wait_for_one_process_result(0.05)

            service_until(workspace_requested.is_set)
            self.assertEqual(native.snapshot()["dispatched"], 0)
            self.assertFalse(mappings)
            workspace_enabled.set()
            service_until(lambda: 1 in assembled)
            self.assertTrue(published.wait(2))
            self.assertEqual(host.recv(1), b"R")
            self.assertTrue(_actor_advanced_while_controller_held_gil(host, 5000),
                            "Actor did not refill during the GIL-held interval")
            service_until(lambda: len(assembled) == 3)
            self.assertEqual(native.snapshot()["blocked_parents"], 1)
            self.assertEqual(native.snapshot()["bytes_used"], 8)
            self.assertEqual(mappings["detector", "second"][0][:], b"\x02\x03")

            def release_mapping(name, owner=None):
                key = ("detector", name)
                for mapping in mappings.pop(key):
                    mapping.close()
                if owner is None:
                    owner = state.direct_union_backing_leases.pop(key)
                owner.release("inference")

            publisher_release.set()
            self.assertEqual(futures[0].result(3), b"\x01\x01")
            scheduler.drain_process_inference_results()
            self.assertEqual(native.snapshot()["blocked_parents"], 1)
            self.assertEqual(native.snapshot()["bytes_used"], 8)
            # Registry removal can precede the actual last-use notification.
            # The captured owner and its mapping remain live across that interval.
            detached = state.direct_union_backing_leases.pop(("detector", "first"))
            scheduler.drain_process_inference_results()
            self.assertEqual(mappings["detector", "first"][0][:], b"\x01\x01")
            self.assertEqual(native.snapshot()["bytes_used"], 8)
            self.assertEqual(native.snapshot()["blocked_parents"], 1)
            release_mapping("first", detached)
            service_until(lambda: len(assembled) == 4)
            self.assertEqual(assembled, [1, 2, 3, 4])
            self.assertEqual(mappings["detector", "third"][0][:], b"\x04\x04")
            self.assertEqual(native.snapshot()["bytes_used"], 8)
            release_mapping("second")
            release_mapping("third")
            service_until(lambda: not scheduler.publication_outstanding())
            self.assertEqual(native.snapshot()["bytes_used"], 0)
            self.assertEqual(native.snapshot()["tickets_used"], 0)
            self.assertEqual(scheduler.process_quiescence_issues(), {})
            state.cpu_task_queues[0].put(None)
            actor.join(5)
            self.assertEqual(actor.exitcode, 0)


if __name__ == "__main__":
    unittest.main()

"""Scheduler-safe serial submission and cancellation ownership contracts."""
from concurrent.futures import thread as thread_pool
import gc
import threading
import weakref

import pytest

from XTA.tta_background import SchedulerBackgroundExecutor


def _gate(entered, release):
    entered.set()
    if not release.wait(3):
        raise TimeoutError('test did not release running background work')
    return 'settled'


def test_submit_returns_while_process_global_thread_pool_lock_is_held():
    executor = SchedulerBackgroundExecutor(thread_name='background-lock-test')
    submit_now, returned = threading.Event(), threading.Event()
    result = []

    def submit():
        submit_now.wait(3)
        result.append(executor.submit(lambda: 42))
        returned.set()

    caller = threading.Thread(target=submit, daemon=True)
    caller.start()
    try:
        with thread_pool._global_shutdown_lock:
            submit_now.set()
            assert returned.wait(.75), 'scheduler submission waited for ThreadPool startup lock'
            assert result[0].result(timeout=3) == 42
    finally:
        submit_now.set()
        caller.join(3)
        executor.shutdown()
    assert not caller.is_alive()


def test_fifo_and_job_exceptions_settle_without_losing_later_work():
    executor = SchedulerBackgroundExecutor(thread_name='background-fifo-test')
    entered, release = threading.Event(), threading.Event()
    order = []

    def work(number, error=None):
        order.append(number)
        if error is not None:
            raise error
        return number

    try:
        running = executor.submit(_gate, entered, release)
        assert entered.wait(3)
        first = executor.submit(work, 1)
        failed = executor.submit(work, 2, ValueError('ordinary job failure'))
        exited = executor.submit(work, 3, SystemExit('job exit remains a Future failure'))
        final = executor.submit(work, 4)
        release.set()
        assert running.result(timeout=3) == 'settled'
        assert first.result(timeout=3) == 1
        with pytest.raises(ValueError, match='ordinary job failure'):
            failed.result(timeout=3)
        with pytest.raises(SystemExit, match='Future failure'):
            exited.result(timeout=3)
        assert final.result(timeout=3) == 4
        assert order == [1, 2, 3, 4]
    finally:
        release.set()
        executor.shutdown()


def test_shutdown_cancellation_callbacks_can_resubmit_and_close_without_owner_lock():
    executor = SchedulerBackgroundExecutor(thread_name='background-cancel-callback-test')
    entered, release = threading.Event(), threading.Event()
    calls, rejected = [], []
    try:
        running = executor.submit(_gate, entered, release)
        assert entered.wait(3)
        cancelled = executor.submit(lambda: calls.append('must not execute'))
        releases_running = executor.submit(lambda: calls.append('must not execute either'))

        def cancelled_callback(future):
            assert future.cancelled()
            try:
                executor.submit(lambda: calls.append('late submit'))
            except RuntimeError:
                rejected.append(True)
            executor.shutdown(wait=False)

        cancelled.add_done_callback(cancelled_callback)
        releases_running.add_done_callback(lambda future: release.set())
        executor.shutdown(wait=False, cancel_futures=True)
        assert cancelled.cancelled() and releases_running.cancelled()
        assert rejected == [True]
        assert running.result(timeout=3) == 'settled'
        executor.shutdown(wait=True, cancel_futures=True)
        assert calls == []
        assert not executor._thread.is_alive()
    finally:
        release.set()
        executor.shutdown()


def test_nested_waiting_shutdown_does_not_join_before_later_cancel_callbacks_run():
    executor = SchedulerBackgroundExecutor(thread_name='background-nested-shutdown-test')
    entered, release, stopped = threading.Event(), threading.Event(), threading.Event()
    errors = []
    running = executor.submit(_gate, entered, release)
    assert entered.wait(3)
    first = executor.submit(lambda: 'never started')
    second = executor.submit(lambda: 'never started either')
    first.add_done_callback(lambda future: executor.shutdown(wait=True))
    second.add_done_callback(lambda future: release.set())

    def shutdown():
        try:
            executor.shutdown(wait=True, cancel_futures=True)
        except BaseException as error:
            errors.append(error)
        finally:
            stopped.set()

    caller = threading.Thread(target=shutdown, daemon=True)
    caller.start()
    try:
        assert stopped.wait(.75), 'nested join prevented a later cancellation from releasing running work'
        assert errors == []
        assert first.cancelled() and second.cancelled()
        assert running.result(timeout=3) == 'settled'
        assert not executor._thread.is_alive()
    finally:
        release.set()
        caller.join(3)
        executor.shutdown()


def test_completion_callback_can_shutdown_its_own_worker_then_fifo_drains():
    executor = SchedulerBackgroundExecutor(thread_name='background-self-shutdown-test')
    entered, release, callback_done = threading.Event(), threading.Event(), threading.Event()
    errors = []
    first = executor.submit(_gate, entered, release)
    assert entered.wait(3)

    def callback(future):
        try:
            executor.shutdown(wait=True)
        except BaseException as error:
            errors.append(error)
        finally:
            callback_done.set()

    first.add_done_callback(callback)
    queued = executor.submit(lambda: 'queued work remains owned')
    try:
        release.set()
        assert callback_done.wait(3)
        assert errors == [], 'worker cleanup attempted to join itself'
        assert queued.result(timeout=3) == 'queued work remains owned'
        executor.shutdown(wait=True)
        assert not executor._thread.is_alive()
    finally:
        release.set()
        executor.shutdown()


def test_repeated_nonwaiting_then_cancelling_shutdown_removes_queued_work():
    executor = SchedulerBackgroundExecutor(thread_name='background-repeated-shutdown-test')
    entered, release = threading.Event(), threading.Event()
    ran = []
    try:
        running = executor.submit(_gate, entered, release)
        assert entered.wait(3)
        queued = executor.submit(lambda: ran.append(True))
        executor.shutdown(wait=False)
        with pytest.raises(RuntimeError, match='after shutdown'):
            executor.submit(lambda: None)
        executor.shutdown(wait=False, cancel_futures=True)
        assert queued.cancelled()
        release.set()
        assert running.result(timeout=3) == 'settled'
        executor.shutdown(wait=True)
        executor.shutdown(wait=True, cancel_futures=True)
        assert ran == [] and not executor._thread.is_alive()
    finally:
        release.set()
        executor.shutdown()


def test_idle_worker_drops_completed_args_kwargs_and_future():
    class Payload:
        pass

    executor = SchedulerBackgroundExecutor(thread_name='background-idle-owner-test')
    released_argument, released_keyword, released_future = [threading.Event() for _ in range(3)]
    argument, keyword = Payload(), Payload()
    weakref.finalize(argument, released_argument.set)
    weakref.finalize(keyword, released_keyword.set)

    def consume(value, *, option):
        return None

    try:
        future = executor.submit(consume, argument, option=keyword)
        weakref.finalize(future, released_future.set)
        assert future.result(timeout=3) is None
        del future, argument, keyword
        gc.collect()
        assert released_argument.wait(3)
        assert released_keyword.wait(3)
        assert released_future.wait(3)
    finally:
        executor.shutdown()

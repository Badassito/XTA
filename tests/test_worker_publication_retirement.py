"""Completed publication callbacks release mappings without a cyclic-GC pass."""

from concurrent.futures import Future, ThreadPoolExecutor
import gc
import os
import weakref

import numpy as np
import pytest

from XTA import workers


@pytest.fixture
def no_cyclic_gc():
    enabled = gc.isenabled()
    gc.disable()
    try:
        yield
    finally:
        if enabled:
            gc.enable()


def publication(root):
    future = Future()
    window = root[1:3]
    owner = workers._DeferredGpuWorkerTaskResult({'before': True}, future, (window, root))
    observations = []

    def completed(done, retained_owner=owner):
        try:
            observations.append(retained_owner.finish())
        except BaseException as error:
            # Keep the transport's scalar error representation, as production
            # does; storing traceback objects would deliberately retain inputs.
            observations.append((type(error), str(error)))

    future.add_done_callback(completed)
    return future, weakref.ref(owner), observations


@pytest.mark.parametrize('failure', [False, True])
@pytest.mark.parametrize('close_fails', [False, True])
def test_completed_callback_drops_owned_mapping_and_future_cycle_without_gc(
        tmp_path, monkeypatch, no_cyclic_gc, failure, close_fails):
    root = np.memmap(tmp_path/'parent-mask.u8', dtype=np.uint8, mode='w+', shape=(4, 8, 8))
    root[:] = 1
    root_reference, mmap_reference = weakref.ref(root), weakref.ref(root._mmap)
    future, owner_reference, observations = publication(root)
    if close_fails:
        def fail_close(_mapping):
            raise OSError('retirement cleanup failed')
        monkeypatch.setattr(workers, 'close_memmap_array', fail_close)
    del root
    assert root_reference() is not None  # Pending D2H still owns the backing.
    if failure:
        future.set_exception(ValueError('D2H failed'))
        assert observations == [(ValueError, 'D2H failed')]
    else:
        future.set_result({'after': True})
        assert observations == [{'before': True, 'after': True}]
    assert owner_reference().memmaps == ()
    assert owner_reference().flush_future is None
    assert root_reference() is None
    assert mmap_reference() is None
    future_reference = weakref.ref(future)
    del future
    if not failure:
        assert future_reference() is None
        assert owner_reference() is None
    # Future.result() preserves a raised exception's traceback, which can
    # retain that small Future until cyclic GC. Its mapped payloads above must
    # already be dead in both success and failure; do not erase diagnostics.


def test_pending_writes_keep_mappings_until_publication_future_has_settled(tmp_path, no_cyclic_gc):
    root = np.memmap(tmp_path/'pending-parent.u8', dtype=np.uint8, mode='w+', shape=(4, 8, 8))
    reference = weakref.ref(root._mmap)
    future = Future()
    owner = workers._DeferredGpuWorkerTaskResult({}, future, (root[1:3], root))
    del root
    with ThreadPoolExecutor(max_workers=1) as pool:
        finished = pool.submit(owner.finish)
        assert not finished.done()
        assert reference() is not None
        assert owner.memmaps and owner.flush_future is future
        future.set_result({'published': True})
        assert finished.result(timeout=5) == {'published': True}
    assert owner.memmaps == () and owner.flush_future is None
    assert reference() is None


def test_outside_numpy_view_stays_readable_after_publication_detaches_its_aliases(tmp_path, no_cyclic_gc):
    root = np.memmap(tmp_path/'parent-confidence.u8', dtype=np.uint8, mode='w+', shape=(4, 8, 8))
    root[:] = 19
    borrowed = np.asarray(root[2:])
    reference = weakref.ref(root._mmap)
    future, _owner, _observations = publication(root)
    del root
    future.set_result({})
    assert reference() is not None
    assert np.all(borrowed == 19)
    del borrowed
    assert reference() is None


@pytest.mark.skipif(not hasattr(os, 'memfd_create'), reason='Linux memfd ownership regression')
def test_finished_worker_drops_last_actual_memfd_mapping_without_gc(no_cyclic_gc):
    descriptor = os.memfd_create('xta-publication-retirement-test', flags=getattr(os, 'MFD_CLOEXEC', 0))
    try:
        os.ftruncate(descriptor, 4096)
        root = np.memmap(f'/proc/self/fd/{descriptor}', dtype=np.uint8, mode='r+', shape=(4, 32, 32))
        root[:] = 1
        reference = weakref.ref(root._mmap)
        future, _owner, _observations = publication(root)
        del root
    finally:
        os.close(descriptor)  # The parent owner has already retired its fd.
    assert reference() is not None
    future.set_result({})
    assert reference() is None

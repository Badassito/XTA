"""Private detector-canvas birth with scheduler-owned publication and credit."""
from concurrent.futures import CancelledError
from dataclasses import dataclass
import threading
import weakref

from .tta_background import SchedulerBackgroundExecutor


class WorkspaceBirthFailure(RuntimeError):
    def __init__(self, error, owner_refs=()):
        super().__init__(str(error))
        self.owner_refs = tuple(owner_refs)


@dataclass
class _Preparation:
    plans: tuple
    future: object = None


class DetectorWorkspacePreparation:
    """One bounded, prestarted allocator; callbacks publish only on the caller thread."""
    def __init__(self, *, build, refresh, adopt, discard, rollback, wake):
        self.build, self.refresh, self.adopt = build, refresh, adopt
        self.discard, self.rollback, self.wake = discard, rollback, wake
        self.executor = SchedulerBackgroundExecutor(thread_name='detector-workspace-birth')
        self.stop = threading.Event()
        self.records = {}
        self.retired = []

    @property
    def futures(self):
        return tuple(record.future for record in self.records.values() if record.future is not None)

    @property
    def pending(self):
        return bool(self.records or self.retired)

    def request(self, key, plans):
        if key in self.records:
            return False
        if self.stop.is_set():
            raise RuntimeError('Detector workspace preparation is closed')
        record = _Preparation(tuple(plans))
        self.records[key] = record
        try:
            record.future = self.executor.submit(self.build, record.plans, None, self.stop)
        except BaseException:
            del self.records[key]
            self.rollback(record.plans)
            raise
        record.future.add_done_callback(lambda _done: self.wake())
        return False

    def _retire(self, plans, refs):
        refs = tuple(refs)
        if any(reference() is not None for reference in refs):
            self.retired.append((plans, refs))
            for reference in refs:
                owner = reference()
                if owner is not None:
                    finalizer = weakref.finalize(owner, self.wake)
                    finalizer.atexit = False
                del owner
        else:
            self.rollback(plans)

    def settle_retired(self):
        for plans, refs in list(self.retired):
            if not any(reference() is not None for reference in refs):
                self.retired.remove((plans, refs))
                self.rollback(plans)

    def drain(self):
        self.settle_retired()
        for key, record in list(self.records.items()):
            future = record.future
            if future is None or not future.done():
                continue
            try:
                bundles = future.result()
            except CancelledError:
                if not future.cancelled():
                    raise
                del self.records[key]
                self.rollback(record.plans)
                continue
            except WorkspaceBirthFailure as error:
                del self.records[key]
                self._retire(record.plans, error.owner_refs)
                raise
            if self.adopt(record.plans, bundles):
                del self.records[key]
            else:
                # A new RAM promise invalidated the probe. Keep the same arrays and charge.
                plans = tuple(self.refresh(record.plans))
                replacement = self.executor.submit(self.build, plans, bundles, self.stop)
                record.plans, record.future = plans, replacement
                record.future.add_done_callback(lambda _done: self.wake())

    def shutdown(self, wait=True, *, cancel_futures=False):
        self.stop.set()
        records = tuple(self.records.items())
        for _key, record in records:
            if record.future is not None:
                record.future.cancel()
        cleanup = []
        failure = None
        try:
            for key, record in records:
                try:
                    bundles = record.future.result()
                except CancelledError:
                    if record.future.cancelled():
                        self.rollback(record.plans)
                        del self.records[key]
                    else:
                        failure = failure or record.future.exception()
                except WorkspaceBirthFailure as error:
                    self._retire(record.plans, error.owner_refs)
                    del self.records[key]
                except BaseException as error:
                    failure = failure or error
                else:
                    cleanup.append((key, record.plans, self.executor.submit(self.discard, bundles)))
            for key, plans, future in cleanup:
                try:
                    refs = future.result()
                except BaseException as error:
                    failure = failure or error
                else:
                    self._retire(plans, refs)
                    del self.records[key]
        finally:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.settle_retired()
        if failure is not None:
            raise failure

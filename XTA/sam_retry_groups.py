"""Bounded fresh-funded controllers for independent adaptive retry groups."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading
import time
import weakref


_RETRY_GROUP_POOL = None
_RETRY_GROUP_POOL_LOCK = threading.Lock()
_MAX_RETRY_GROUP_CONTROLLERS = 8
_CONTROLLER_SLOTS = threading.BoundedSemaphore(_MAX_RETRY_GROUP_CONTROLLERS)


class _ControllerPermit:
    def __init__(self, slots):
        self.slots, self.lock = slots, threading.Lock()

    def release(self):
        with self.lock:
            if self.slots is not None:
                self.slots.release()
                self.slots = None


def _executor():
    global _RETRY_GROUP_POOL
    with _RETRY_GROUP_POOL_LOCK:
        if _RETRY_GROUP_POOL is None:
            _RETRY_GROUP_POOL = ThreadPoolExecutor(max_workers=_MAX_RETRY_GROUP_CONTROLLERS,
                thread_name_prefix='sam-retry-group')
        return _RETRY_GROUP_POOL


def _event(name, **fields):
    from .runtime import runtime_trace_event
    runtime_trace_event(name, **fields)


class _SourceOwner:
    def __init__(self, run, context, runtime, scope, family, ordinals):
        self.run, self.context, self.runtime = run, context, runtime
        self.scope, self.family = scope, family
        self.ordinals = ordinals

    def execute(self, ordinal, profile, group_id, *, child):
        from .sam_tracker_runtime import _clear_producer_error_frames, sam_view_producer_family
        started = time.perf_counter()
        fields = dict(scope_id=self.scope, original_group_id=group_id, group_ordinal=self.ordinals[ordinal],
            mode='child' if child else 'serial', accounting_scope_id=profile.scope_id)
        _event('sam_retry_group_begin', **fields)
        result, failed, attempt_count = None, True, 0
        try:
            with sam_view_producer_family(self.family):
                result = self.run(ordinal, profile, child)
            failed = False
            return result
        except BaseException as error:
            attempt_count = int(getattr(error, '_sam_retry_attempt_count', 0))
            _clear_producer_error_frames(error)
            raise
        finally:
            _event('sam_retry_group_end', **fields, failed=failed,
                elapsed_seconds=time.perf_counter()-started,
                attempt_count=attempt_count if result is None else int(result.get('attempt_count',0)))


def _child(request, owner_ref, ordinal, group_id, parent_scope, original_ordinal=None,
           controller_permit=None):
    """Queued work owns scalar metadata and a gate; no borrowed source arrays."""
    try:
        with request() as profile:
            if profile is None:
                _event('sam_retry_group_declined', scope_id=parent_scope,
                    original_group_id=group_id, group_ordinal=ordinal if original_ordinal is None else original_ordinal,
                    reason=getattr(request,'decline_reason',None))
                return False, None
            owner = owner_ref()
            if owner is None:
                raise RuntimeError('SAM retry source owner expired before an admitted child')
            try:
                with owner.context._retry_child_resource_scope(profile, parent_scope):
                    return True, owner.execute(ordinal, profile, group_id, child=True)
            except BaseException as error:
                from .sam_tracker_runtime import _clear_producer_error_frames
                _clear_producer_error_frames(error)
                raise
    finally:
        if controller_permit is not None:
            controller_permit.release()


def execute_retry_groups(group_ids, run_group, *, parent_profile, context, runtime,
                         parent_scope, family=None, max_in_flight=8, group_ordinals=None):
    """Drain any completion; retry declined grants after funded work frees credit.

    Every group calls the same scientific runner. A child is allowed to touch
    the frozen source only after fresh rights and Context source ownership are
    established. If none of the bounded requests can acquire those rights,
    the original owner advances one group using its existing serial profile.
    """
    from .sam_resources import prepare_retry_child_resources
    ids = tuple(group_ids)
    ordinals = tuple(range(len(ids))) if group_ordinals is None else tuple(group_ordinals)
    if (len(ordinals)!=len(ids) or any(type(value) is not int or value<0 for value in ordinals)
            or len(set(ordinals))!=len(ordinals)):
        raise ValueError('SAM retry group ordinals must identify the finite original inventory')
    capacity = min(_MAX_RETRY_GROUP_CONTROLLERS,max(1,int(max_in_flight)),len(ids))
    owner = _SourceOwner(run_group,context,runtime,parent_scope,family,ordinals)
    owner_ref = weakref.ref(owner)
    pool = _executor()
    pending, owned, deferred, results = {}, [], set(), {}
    next_ordinal = 0
    refill_slots = capacity
    stats = dict(mode='parallel_unlimited', capacity=capacity, child_groups=0,
        serial_fallback_groups=0, resource_declines=0, controller_declines=0, peak_pending_groups=0)

    def submit(ordinal):
        # A queued controller would park the funded owner behind unrelated GPU work.
        if not _CONTROLLER_SLOTS.acquire(blocking=False):
            deferred.add(ordinal)
            stats['controller_declines'] += 1
            _event('sam_retry_group_declined', scope_id=parent_scope,
                original_group_id=ids[ordinal], group_ordinal=ordinals[ordinal], reason='controller_capacity')
            return False
        permit = _ControllerPermit(_CONTROLLER_SLOTS)
        try:
            request = prepare_retry_child_resources(parent_profile,
                f'{parent_scope}/retry-group/{ordinals[ordinal]}')
        except BaseException:
            permit.release()
            raise
        item = dict(request=request, hold=None, future=None, ordinal=ordinal, permit=permit)
        owned.append(item)  # Submission may raise after accepting the callback.
        item['hold'] = context._hold_retry_source(request)
        future = pool.submit(_child,request,owner_ref,ordinal,ids[ordinal],parent_scope,ordinals[ordinal],permit)
        item['future'] = future
        future.add_done_callback(lambda _future: permit.release())
        pending[future] = item
        stats['peak_pending_groups'] = max(stats['peak_pending_groups'],len(pending))
        return True

    def retire_holds():
        for item in owned:
            if item['request'].settled and item['hold'] is not None:
                item['hold'].close()

    def clear_future(future):
        if future is None or not future.done() or future.cancelled():
            return
        error = future.exception()
        if error is not None:
            from .sam_tracker_runtime import _clear_producer_error_frames
            _clear_producer_error_frames(error)

    def cleanup():
        for item in owned:
            if item['request'].cancel_if_unstarted():
                item['permit'].release()
            if item['future'] is not None:
                item['future'].cancel()
        # Keep the actual parent preparation/source live until every accepted
        # callback and its nested input/SDK owners settle. Runtime retirement
        # must run outside Context locks to resolve abandoned producer debt.
        last_retirement = 0.
        while True:
            for item in owned:
                clear_future(item['future'])
            retire_holds()
            unsettled = [item for item in owned if not item['request'].settled
                or item['future'] is not None and not item['future'].done()]
            if not unsettled:
                return
            now = time.monotonic()
            if now-last_retirement>=.25:
                try:
                    runtime.close()
                except BaseException:
                    pass  # The actual rights/source remain retained below.
                last_retirement = now
            futures = [item['future'] for item in unsettled if item['future'] is not None]
            if futures:
                wait(futures,timeout=.05,return_when=FIRST_COMPLETED)
            else:
                threading.Event().wait(.05)

    try:
        while len(results)<len(ids):
            if context._cancel.is_set() or getattr(runtime,'_cancel',context._cancel).is_set():
                raise RuntimeError('SAM independent retry groups cancelled')
            while refill_slots and len(pending)<capacity and (deferred or next_ordinal<len(ids)):
                if deferred:
                    ordinal = min(deferred)
                    deferred.remove(ordinal)
                else:
                    ordinal = next_ordinal
                    next_ordinal += 1
                if not submit(ordinal):
                    refill_slots = 0
                    break
                refill_slots -= 1
            if not pending:
                ordinal = min(deferred)
                deferred.remove(ordinal)
                results[ordinal] = owner.execute(ordinal,parent_profile,ids[ordinal],child=False)
                stats['serial_fallback_groups'] += 1
                refill_slots = capacity
                continue
            done,_ = wait(tuple(pending),timeout=.05,return_when=FIRST_COMPLETED)
            successes = []
            for future in sorted(done,key=lambda value:pending[value]['ordinal']):
                item = pending.pop(future)
                accepted,result = future.result()
                if not item['request'].settled:
                    raise RuntimeError('SAM retry child returned without retiring its actual resource owners')
                item['hold'].close()
                owned.remove(item)
                ordinal = item['ordinal']
                if accepted:
                    results[ordinal] = result
                    stats['child_groups'] += 1
                    successes.append(ordinal)
                else:
                    deferred.add(ordinal)
                    stats['resource_declines'] += 1
            # A decline is not a reason to exhaust the full group inventory
            # while funded siblings run. Refill on successful credit return.
            if successes:
                refill_slots = capacity-len(pending)
            if done and not successes and pending:
                continue
            if not pending and deferred and not refill_slots:
                ordinal = min(deferred)
                deferred.remove(ordinal)
                results[ordinal] = owner.execute(ordinal,parent_profile,ids[ordinal],child=False)
                stats['serial_fallback_groups'] += 1
                refill_slots = capacity
        retire_holds()
        owner.run = None
        run_group = None
        return tuple(results[index] for index in range(len(ids))),stats
    except BaseException as error:
        error._sam_retry_group_execution = dict(stats)
        try:
            context.cancel('SAM independent retry group failed')
        except BaseException as cancellation_error:
            if callable(getattr(error,'add_note',None)):
                error.add_note('SAM retry cancellation also failed: '+str(cancellation_error))
            try:
                runtime.cancel('SAM independent retry group failed')
            except BaseException:
                pass
        finally:
            cleanup()
            owner.run = None
            run_group = None
        from .sam_tracker_runtime import _clear_producer_error_frames
        _clear_producer_error_frames(error)
        raise

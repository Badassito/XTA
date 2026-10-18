"""Bounded fresh-funded controllers for independent adaptive retry groups."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import threading
import time
import weakref


_RETRY_GROUP_POOLS = weakref.WeakKeyDictionary()
_RETRY_GROUP_POOL_LOCK = threading.Lock()


class _ControllerPermit:
    def __init__(self, slots):
        self.slots, self.lock = slots, threading.Lock()

    def release(self):
        with self.lock:
            if self.slots is not None:
                self.slots.release()
                self.slots = None


class _RuntimeControllers:
    def __init__(self, capacity):
        self.capacity, self.active, self.lock = capacity, 0, threading.Lock()
        self.executor = ThreadPoolExecutor(max_workers=capacity, thread_name_prefix='sam-retry-group')

    def acquire(self, started_slots):
        with self.lock:
            if self.active >= min(self.capacity, started_slots):
                return None
            self.active += 1
        return _ControllerPermit(self)

    def release(self):
        with self.lock:
            if self.active <= 0:
                raise ValueError('SAM retry controller permit returned twice')
            self.active -= 1


def _controllers(runtime):
    with _RETRY_GROUP_POOL_LOCK:
        state = _RETRY_GROUP_POOLS.get(runtime)
        if state is None:
            capacity = len(getattr(runtime, '_configured_worker_slots', runtime.worker_slots))
            state = _RuntimeControllers(capacity)
            _RETRY_GROUP_POOLS[runtime] = state
            # Active source owners retain runtime until callbacks and rights settle.
            weakref.finalize(runtime, state.executor.shutdown, wait=False)
        return state


def _executor(runtime):
    return _controllers(runtime).executor


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
                         parent_scope, family=None, max_in_flight=None, group_ordinals=None):
    """Drain any completion; retry declined grants after funded work frees credit.

    Every group calls the same scientific runner. A child is allowed to touch
    the frozen source only after fresh rights and Context source ownership are
    established. If none of the bounded requests can acquire those rights,
    the original owner advances one group using its existing serial profile.
    A settled decline can also use that owner while accepted siblings run,
    within the same group bound and after the original tracker scope retires.
    """
    from .sam_resources import prepare_retry_child_resources
    ids = tuple(group_ids)
    ordinals = tuple(range(len(ids))) if group_ordinals is None else tuple(group_ordinals)
    if (len(ordinals)!=len(ids) or any(type(value) is not int or value<0 for value in ordinals)
            or len(set(ordinals))!=len(ordinals)):
        raise ValueError('SAM retry group ordinals must identify the finite original inventory')
    configured = len(getattr(runtime, '_configured_worker_slots', getattr(runtime, 'worker_slots', ())))
    limit = min(configured, configured if max_in_flight is None else max(1,int(max_in_flight)), len(ids))
    capacity = min(limit, len(getattr(runtime, 'worker_slots', ())))
    owner = _SourceOwner(run_group,context,runtime,parent_scope,family,ordinals)
    owner_ref = weakref.ref(owner)
    controllers = _controllers(runtime) if limit else None
    pool = _executor(runtime) if controllers is not None else None
    pending, owned, deferred, results = {}, [], set(), {}
    next_ordinal = 0
    refill_slots = capacity
    error_lock, first_error, cancellation_sent = threading.Lock(), [None], [False]
    stats = dict(mode='parallel_unlimited', capacity=capacity, child_groups=0,
        serial_fallback_groups=0, parent_inline_overlap_groups=0,
        resource_declines=0, controller_declines=0, peak_pending_groups=0)

    def remember_error(error):
        with error_lock:
            if first_error[0] is None:
                first_error[0] = error
                return error, True
            return first_error[0], False

    def check_child_error():
        with error_lock:
            error = first_error[0]
        if error is not None:
            raise error

    def cancel_owner(error):
        with error_lock:
            if cancellation_sent[0]:
                return
            cancellation_sent[0] = True
        # These cancellation hooks set flags and wake waiters; they never join
        # the child whose Future is invoking this callback. Keep the original
        # failure before waking an inline parent with a cancellation exception.
        live = owner_ref()
        if live is not None:
            try:
                live.context.cancel('SAM independent retry group failed')
            except BaseException as cancellation_error:
                if callable(getattr(error,'add_note',None)):
                    error.add_note('SAM retry cancellation also failed: '+str(cancellation_error))
                try:
                    live.runtime.cancel('SAM independent retry group failed')
                except BaseException:
                    pass

    def child_done(future, request_ref):
        try:
            error = future.exception()
        except BaseException as cancelled:
            error = cancelled
        request = request_ref()
        if error is None and request is not None and not request.settled:
            error = RuntimeError('SAM retry child returned without retiring its actual resource owners')
        if error is None:
            return
        _primary, first = remember_error(error)
        if first:
            cancel_owner(error)

    def parent_wave_available():
        condition = parent_profile._lease.pool.condition
        if not condition.acquire(blocking=False):
            return False
        try:
            parent_profile._validate_owner()
            return parent_profile._lease.tracker_scope_identity is None
        finally:
            condition.release()

    def submit(ordinal):
        # A queued controller would park the funded owner behind unrelated GPU work.
        permit = controllers.acquire(len(runtime.worker_slots))
        if permit is None:
            deferred.add(ordinal)
            stats['controller_declines'] += 1
            _event('sam_retry_group_declined', scope_id=parent_scope,
                original_group_id=ids[ordinal], group_ordinal=ordinals[ordinal], reason='controller_capacity')
            return False
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
        future.add_done_callback(lambda done, request_ref=weakref.ref(request):child_done(done,request_ref))
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
            check_child_error()
            if context._cancel.is_set() or getattr(runtime,'_cancel',context._cancel).is_set():
                raise RuntimeError('SAM independent retry groups cancelled')
            current_capacity = min(limit, len(getattr(runtime, 'worker_slots', ())))
            if current_capacity > capacity:
                refill_slots = max(refill_slots, current_capacity-len(pending))
            capacity = current_capacity
            stats['capacity'] = max(stats['capacity'], capacity)
            while refill_slots and len(pending)<capacity and (deferred or next_ordinal<len(ids)):
                check_child_error()
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
                if deferred:
                    ordinal = min(deferred)
                    deferred.remove(ordinal)
                else:
                    ordinal = next_ordinal
                    next_ordinal += 1
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
            # Only already-declined ordinals may use the funded original owner.
            # A resource decline reached this set after its request and source
            # hold settled above; a controller decline was never submitted.
            # Leave untouched inventory to the ordinary bounded child refill.
            if successes:
                refill_slots = capacity-len(pending)
            if pending and deferred and len(pending)<capacity and parent_wave_available():
                check_child_error()
                if context._cancel.is_set() or getattr(runtime,'_cancel',context._cancel).is_set():
                    raise RuntimeError('SAM independent retry groups cancelled')
                ordinal = min(deferred)
                deferred.remove(ordinal)
                results[ordinal] = owner.execute(ordinal,parent_profile,ids[ordinal],child=False)
                stats['serial_fallback_groups'] += 1
                stats['parent_inline_overlap_groups'] += 1
                continue
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
        primary, _first = remember_error(error)
        primary._sam_retry_group_execution = dict(stats)
        try:
            cancel_owner(primary)
        finally:
            cleanup()
            owner.run = None
            run_group = None
        from .sam_tracker_runtime import _clear_producer_error_frames
        _clear_producer_error_frames(error)
        if primary is not error:
            _clear_producer_error_frames(primary)
            raise primary
        raise

"""Python resource handles for the scheduler-owned native device arbiter.

No eligibility, occupancy, residency or fairness ledger lives in this module.
Python owns presentation strings and performs CUDA probes outside the native
owner lock; the same native transaction gates task dispatch and stage claims.
"""
from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time


@dataclass(frozen=True)
class _StageToken:
    owner: object
    epoch: int
    value: int


class NativeDeviceStageCoordinator:
    def __init__(self, stage_factory, overlap_factory, residency_factory, trace):
        self._owner = None
        self._metadata_lock = threading.RLock()
        self._identities = {}
        self._names = {}
        self._next_identity = 1
        self._wake_callback = None
        self._stage_factory = stage_factory
        self._overlap_factory = overlap_factory
        self._residency_factory = residency_factory
        self._trace = trace

    def bind(self, owner):
        if not callable(getattr(owner, 'stage', None)):
            raise RuntimeError('Native scheduler device-stage API is unavailable')
        with self._metadata_lock:
            if self._owner is owner:
                return
            if self._owner is not None:
                raise RuntimeError('A different native scheduler already owns the device stages')
            # Exercise the actual ABI now, before GPU/model work begins.
            owner.stage('snapshot', ())
            self._owner = owner
            self._identities.clear()
            self._names.clear()
            self._next_identity = 1
        self._configure_policy()

    def unbind(self, owner):
        with self._metadata_lock:
            if self._owner is not owner:
                raise RuntimeError('Device-stage unbind does not name the current native owner')
            snapshot = self.snapshot()
            if any(snapshot[name] for name in ('inference_inflight', 'stage_leases', 'overlap_leases',
                                               'resident_owners', 'provisional_stage_devices', 'auxiliary_inflight')):
                raise RuntimeError('Cannot unbind native scheduler while device owners remain live')
            self._owner = None
            self._wake_callback = None
            self._identities.clear()
            self._names.clear()
            self._next_identity = 1

    def unbind_aborted(self, owner, *, actors_settled):
        """Detach a failed run only after its process owners actually settled.

        A failed transport need not deliver the final inference/auxiliary ACK,
        so those old credits die with the aborted owner. Main-process CUDA and
        SAM-residency fences still require their ordinary actual release first.
        """
        with self._metadata_lock:
            if self._owner is not owner or actors_settled is not True:
                raise RuntimeError('Aborted native stage teardown requires the exact settled actor owner')
            if not owner.snapshot().get('failed'):
                raise RuntimeError('Aborted stage teardown requires a failed native scheduler')
            snapshot = self.snapshot()
            if any(snapshot[name] for name in ('stage_leases', 'overlap_leases', 'resident_owners',
                                               'provisional_stage_devices')):
                raise RuntimeError('Aborted native stage teardown still has unsettled CUDA owners')
            self._owner = None
            self._wake_callback = None
            self._identities.clear()
            self._names.clear()
            self._next_identity = 1

    def _required_owner(self):
        owner = self._owner
        if owner is None:
            raise RuntimeError('Native scheduler must be bound before device-stage use; no Python arbiter fallback exists')
        return owner

    def _call(self, operation, *fields):
        result = tuple(self._required_owner().stage(operation, tuple(int(v) for v in fields)))
        if operation not in {'snapshot', 'purpose', 'run_epoch', 'can_infer', 'can_aux', 'eligible',
                             'candidates', 'overlap_candidates', 'output_wait', 'floor_pending', 'inference_active'}:
            callback = self._wake_callback
            if callback is not None:
                # Native ready dispatch is already signalled by the transaction.
                # This notification wakes external Python producers, not policy.
                try:
                    callback()
                except Exception:
                    pass
        return result

    @staticmethod
    def _is_source_projection_retirement(purpose):
        return str(purpose).strip().lower().startswith((
            'spherical source projection ', 'radial source projection ', 'native source projection '))

    def _identity(self, category, text):
        text = str(text)
        key = (category, text)
        with self._metadata_lock:
            identity = self._identities.get(key)
            if identity is None:
                identity = self._next_identity
                self._next_identity += 1
                self._identities[key] = identity
                self._names[identity] = text
                if category == 'purpose':
                    kind = int(self._is_source_projection_retirement(text))
                    if text.strip().lower().startswith(('tta persistent sam ', 'sam ')):
                        kind |= 2
                    if text.endswith(' sibling ranks'):
                        kind |= 4
                    self._call('purpose', identity, kind)
            return identity

    def _purpose(self, purpose):
        return self._identity('purpose', purpose)

    def _configure_policy(self):
        def enabled(name):
            return os.environ.get(name, '1').strip().lower() not in {'0', 'false', 'no', 'off'}
        self._call('policy', enabled('YOLO_TTA_GPU_SPHERICAL_PRESSURE_RETIREMENT'),
                   enabled('YOLO_TTA_GPU_SPHERICAL_AGE_RETIREMENT'))

    def configure_workers(self, worker_devices):
        self._call('configure', *tuple(int(v) for v in worker_devices))
        self._configure_policy()

    def reset(self):
        self._call('reset')
        self._wake_callback = None

    def current_epoch(self):
        return int(self._call('run_epoch')[0])

    def set_wake_callback(self, callback):
        if callback is not None and not callable(callback):
            raise TypeError('Stage wake notification must be callable')
        self._wake_callback = callback

    def set_inference_priority_active(self, active):
        self._call('priority', bool(active))

    def set_inference_asset_retirement_pending(self, active):
        self._call('retiring', bool(active))

    def set_pending_inference_backlog(self, active):
        self._call('backlog', bool(active))

    def set_spherical_retirement_pressure(self, active):
        self._call('pressure', bool(active))

    def inference_active(self):
        return bool(self._call('inference_active')[0])

    def floor_retirement_pending(self, purpose):
        return bool(self._call('floor_pending', self._purpose(purpose))[0])

    def mark_inference_assets_retired(self, proof):
        # Typed proof validation is done by the public backprojection facade.
        self._call('retired', proof.device_index, proof.control_task_id, proof.coordinator_epoch)

    def can_dispatch_inference(self, device_index):
        return bool(self._call('can_infer', device_index)[0])

    def begin_inference(self, device_index):
        return bool(self._call('begin_infer', device_index)[0])

    def finish_inference(self, device_index):
        self._call('end_infer', device_index)

    def can_dispatch_auxiliary(self, device_index):
        return bool(self._call('can_aux', device_index)[0])

    def _request_spherical_retirement_turn(self, candidates, purpose):
        self._call('request', self._purpose(purpose), *candidates)

    def cancel_spherical_retirement_request(self, purpose, *, failed=False):
        # CPU-only scientific callers share projection cleanup but never register
        # a GPU purpose. Cancellation must not create a request/owner merely to
        # remove it. Known run-local identities still require the native owner.
        with self._metadata_lock:
            identity = self._identities.get(('purpose', str(purpose)))
            if identity is None:
                return
            self._call('cancel_request', identity, bool(failed))

    def _token_value(self, value):
        return (value.value if isinstance(value, _StageToken) and value.owner is self._owner
                and value.epoch == self.current_epoch() else 0)

    def _claim_stage_device(self, device, purpose, epoch, torch_mod=None, resident_token=None):
        owner = self._required_owner()
        resident = 0 if resident_token is None else self._token_value(resident_token)
        if resident_token is not None and not resident:
            return None
        purpose_id = self._purpose(purpose)
        reservation = self._call('prepare', device, purpose_id, epoch, resident)[0]
        if not reservation:
            return None
        committed = False
        try:
            if torch_mod is not None:
                try:
                    torch_mod.cuda.mem_get_info(torch_mod.device(f'cuda:{int(device)}'))
                except Exception:
                    if self._is_source_projection_retirement(purpose):
                        self.cancel_spherical_retirement_request(purpose, failed=True)
                    return None
            lease_id = self._call('commit', device, purpose_id, epoch, reservation, resident)[0]
            if not lease_id:
                return None
            token = _StageToken(owner, epoch, lease_id)
            try:
                result = self._stage_factory(self, device, purpose, token)
            except BaseException:
                self._call('release', device, purpose_id, lease_id)
                raise
            committed = True
            return result
        finally:
            if not committed and self._owner is owner:
                self._call('cancel_prepare', device, reservation)

    @staticmethod
    def _device_count(torch_mod):
        try:
            return int(torch_mod.cuda.device_count())
        except Exception:
            return 0

    def try_acquire_specific_stage(self, torch_mod, device_index, purpose):
        self._required_owner()
        device = int(device_index)
        if not 0 <= device < self._device_count(torch_mod):
            return None
        self._request_spherical_retirement_turn((device,), purpose)
        return self._claim_stage_device(device, purpose, self.current_epoch())

    def try_acquire_stage(self, torch_mod, purpose):
        self._required_owner()
        count = self._device_count(torch_mod)
        if count <= 0:
            return None
        self._request_spherical_retirement_turn(range(count), purpose)
        epoch = self.current_epoch()
        workers = self.snapshot()['worker_devices']
        candidates = [v for v in workers if 0 <= v < count] if workers else list(range(count))
        available = self._call('candidates', self._purpose(purpose), *candidates)
        if len(available) == 1:
            return self._claim_stage_device(available[0], purpose, epoch, torch_mod)
        estimates = []
        for device in available:
            try:
                free, _ = torch_mod.cuda.mem_get_info(torch_mod.device(f'cuda:{device}'))
                estimates.append((int(free), device))
            except Exception:
                continue
        for _, device in sorted(estimates, key=lambda item: (-item[0], item[1])):
            lease = self._claim_stage_device(device, purpose, epoch)
            if lease is not None:
                return lease
        if available and not estimates and self._is_source_projection_retirement(purpose):
            self.cancel_spherical_retirement_request(purpose, failed=True)
        return None

    def output_stage_wait_state(self, torch_mod, purpose):
        """Return native output capability and a race-safe change epoch.

        Status 0 has no selected physical GPU; 1 has an eligible device; 2 has
        selected GPUs whose native owners/gates can change. A failed CUDA probe
        while status remains 1 is unavailable execution, not contention.
        """
        self._required_owner()
        return self._call('output_wait', self._purpose(purpose), self._device_count(torch_mod))

    def wait_for_stage_change(self, epoch, timeout=None):
        return self._required_owner().wait_stage_change(int(epoch), timeout)

    def can_attempt_source_projection(self, purpose):
        workers = self.snapshot()['worker_devices']
        if not workers:
            return True
        self._request_spherical_retirement_turn(workers, purpose)
        return bool(self._call('candidates', self._purpose(purpose), *workers))

    def release_stage(self, device_index, purpose, *, token):
        value = self._token_value(token)
        if value:
            self._call('release', device_index, self._purpose(purpose), value)

    def promote_residency(self, device, purpose, stage_token):
        value = self._token_value(stage_token)
        if not value:
            raise RuntimeError('CUDA residency promotion requires the exact live native startup owner')
        native = self._call('promote', device, self._purpose(purpose), value)[0]
        self._trace('gpu_residency_acquired', device=f'cuda:{device}', purpose=purpose)
        return self._residency_factory(self, device, purpose, _StageToken(self._owner, self.current_epoch(), native))

    def try_acquire_resident_compute(self, torch_mod, device, purpose, resident_token):
        if not 0 <= device < self._device_count(torch_mod) or not self._token_value(resident_token):
            return None
        return self._claim_stage_device(device, purpose, self.current_epoch(), torch_mod, resident_token)

    def quarantine_residency(self, device, token, reason):
        value = self._token_value(token)
        if value:
            self._call('quarantine', device, value, self._identity('reason', reason))

    def release_residency(self, device, token):
        value = self._token_value(token)
        if not value:
            return False
        released = bool(self._call('release_resident', device, value)[0])
        if released:
            self._trace('gpu_residency_released', device=f'cuda:{device}')
        return released

    def try_acquire_overlap_stage(self, torch_mod, purpose, budget_bytes):
        purpose_id = self._purpose(purpose)
        budget = int(budget_bytes)
        epoch = self.current_epoch()
        candidates = self._call('overlap_candidates', purpose_id, budget)
        if not candidates:
            return None
        count = self._device_count(torch_mod)
        margin = int(max(0., float(os.environ.get('YOLO_TTA_GPU_PROJECTION_OVERLAP_MARGIN_GIB', '1'))) * 1024**3)
        estimates = []
        for device in candidates:
            if not 0 <= device < count:
                continue
            try:
                free, _ = torch_mod.cuda.mem_get_info(torch_mod.device(f'cuda:{device}'))
                estimates.append((int(free), device))
            except Exception:
                continue
        for free, device in sorted(estimates, key=lambda item: (-item[0], item[1])):
            lease_id = self._call('overlap', device, purpose_id, budget, epoch, free - margin)[0]
            if lease_id:
                token = _StageToken(self._owner, epoch, lease_id)
                try:
                    return self._overlap_factory(self, device, purpose, token, budget)
                except BaseException:
                    self._call('release_overlap', device, purpose_id, lease_id)
                    raise
        return None

    def release_overlap_stage(self, device_index, purpose, *, token):
        value = self._token_value(token)
        if value:
            self._call('release_overlap', device_index, self._purpose(purpose), value)

    def decline_overlap(self, purpose):
        self._call('decline_overlap', self._purpose(purpose))

    def enable_auxiliary(self, device, full_affinity=False):
        return bool(self._call('aux_enable', device, bool(full_affinity))[0])

    def revoke_auxiliary(self, device):
        return bool(self._call('aux_revoke', device)[0])

    def acquire_auxiliary(self, task_id, devices):
        device, lease, affinity = self._call('aux_acquire', task_id, *devices)
        return device, _StageToken(self._owner, self.current_epoch(), lease), bool(affinity)

    def complete_auxiliary(self, device, task_id, token):
        value = self._token_value(token)
        if not value:
            raise RuntimeError('Stale native auxiliary completion identity')
        self._call('aux_complete', device, task_id, value)

    def snapshot(self):
        values = self._call('snapshot')
        if len(values) < 16 or len(values) != 16 + 17 * values[15]:
            raise RuntimeError('Malformed native device-stage snapshot')
        h, rows = values[:16], [values[i:i + 17] for i in range(16, len(values), 17)]
        name = lambda identity: self._names.get(identity, 'native stage' if identity == 0 else str(identity))
        return dict(sample_monotonic_ns=time.monotonic_ns(), coordinator_epoch=h[0], change_epoch=h[1],
            worker_devices=[r[0] for r in rows if r[1]], retired_inference_devices=[r[0] for r in rows if r[13]],
            inference_inflight={r[0]: r[2] for r in rows if r[2]},
            stage_leases={r[0]: name(r[4]) for r in rows if r[3]},
            overlap_leases={r[0]: name(r[7]) for r in rows if r[6]},
            resident_owners={r[0]: dict(purpose=name(r[10]), lendable=not bool(r[11]),
                quarantined=bool(r[11]), reason=name(r[12]) if r[12] else '') for r in rows if r[9]},
            provisional_stage_devices=[r[0] for r in rows if r[5]],
            auxiliary_offered_devices=[r[0] for r in rows if r[14]],
            auxiliary_inflight={r[0]: r[16] for r in rows if r[15]},
            inference_priority_active=bool(h[2]), inference_asset_retirement_pending=bool(h[3]),
            pending_inference_backlog=bool(h[4]), spherical_retirement_pressure=bool(h[5]),
            spherical_retirement_reserved_device=None if h[6] < 0 else h[6],
            spherical_retirement_request_count=h[8], overlap_acquisitions=h[9], overlap_budget_declines=h[10],
            spherical_retirement_handoffs=h[11], spherical_retirement_aged_acquisitions=h[12],
            spherical_retirement_pressure_acquisitions=h[13], spherical_retirement_handoff_expirations=h[14])

"""Diagnostic-only extrapolation retry bounds; never an allocation certificate.

Only fixed scalar fields survive observation. Existing scope permissions are
read, never minted, validated, resized or retained by this module.
"""
from collections.abc import Mapping
import numpy as np

EVENT = 'sam_extrapolation_retry_attempt_memory_bounds'
FIELDS = (
    'scope_id', 'original_group_id', 'attempt_index', 'crop_mode', 'interval_identity',
    'crop_y0', 'crop_x0', 'crop_y1', 'crop_x1', 'job_count', 'maximum_frame_count',
    'maximum_seed_pixels', 'gray_payload_bytes', 'render_scratch_bytes',
    'maximum_session_cpu_bytes', 'maximum_raw_mask_bytes', 'consumer_transfer_bytes',
    'sdk_max_in_flight', 'attempt_wave_peak_bytes', 'assigned_wave_bytes',
    'evidence_fixed_bytes', 'evidence_geometry_bytes', 'reader_cache_bytes',
    'observation_allowance_bytes', 'metadata_allowance_bytes', 'image_allowance_bytes',
    'cache_descriptor_bytes', 'prepared_bank_peak_bytes', 'completion_wave_loan_peak_bytes',
    'packing_scratch_cap_peak_bytes', 'approved_retry_peak_bytes', 'unproved_owner_flags',
)
_SCOPE_FIELDS = (
    ('maximum_frame_count', 'maximum_frame_count'),
    ('maximum_seed_pixels', 'maximum_seed_pixels'),
    ('maximum_session_cpu_bytes', 'maximum_session_cpu_bytes'),
    ('maximum_raw_mask_bytes', 'maximum_raw_mask_bytes'),
    ('consumer_transfer_bytes', 'consumer_transfer_bytes'),
    ('sdk_max_in_flight', 'max_in_flight'),
    ('attempt_wave_peak_bytes', 'attempt_peak_bytes'),
    ('evidence_fixed_bytes', 'evidence_fixed_bytes'),
    ('cache_descriptor_bytes', 'maximum_cache_payload_bytes'),
    ('prepared_bank_peak_bytes', 'prepared_bank_bytes'),
    ('completion_wave_loan_peak_bytes', 'completion_wave_loan_bytes'),
)
_FLAG_NAMES = (
    'metadata_transfer_unproved', 'prepared_unknown', 'profile_unknown',
    'scope_owner_unknown', 'scope_limits_unknown', 'tiled_geometry_unknown',
    'geometry_unknown', 'packing_unknown', 'partial_observation', 'failed',
    'cancelled', 'no_scopes', 'diagnostic_read_failed', 'missing_scalar',
    'retirement_unproved',
)
_FLAGS = {name: 1 << index for index, name in enumerate(_FLAG_NAMES)}
_COUNT_CAP = (1 << 63)-1
_INTEGER_TYPES = frozenset((int, np.int8, np.int16, np.int32, np.int64,
                           np.uint8, np.uint16, np.uint32, np.uint64,
                           np.intp, np.uintp, np.longlong, np.ulonglong))
_FIELD_BITS = {name: 1 << index for index, name in enumerate(FIELDS)}


def _trace_sink():
    # Do not initialize observability, a sampler or a writer for this diagnostic.
    from . import runtime
    sink = runtime._RUNTIME_TELEMETRY
    return sink if sink is not None and sink.enabled and sink.trace_enabled else None


class RetryAttemptBounds:
    __slots__ = ('values', 'flags', 'scopes_seen', 'scopes_with_limits', 'emitted', 'invalid_fields')

    def __init__(self):
        self.values = dict.fromkeys(FIELDS)
        self.flags = _FLAGS['metadata_transfer_unproved']
        self.scopes_seen = self.scopes_with_limits = 0
        self.emitted = False
        self.invalid_fields = 0

    def flag(self, name):
        self.flags |= _FLAGS[name]

    def integer(self, value, *, positive=False):
        # Test exact scalar types before touching any user-defined conversion or
        # dtype property. In particular, bool and scalar subclasses are unknown.
        if type(value) not in _INTEGER_TYPES:
            self.flag('missing_scalar')
            return None
        result = int(value)
        if result < int(positive):
            self.flag('missing_scalar')
            return None
        return result

    def text(self, value):
        if type(value) is not str:
            self.flag('missing_scalar')
            return None
        return value[:512]

    def maximum(self, field, value):
        if value is None:
            self.invalidate(field)
        elif not self.invalid_fields & _FIELD_BITS[field]:
            prior = self.values[field]
            self.values[field] = value if prior is None else max(prior, value)

    def invalidate(self, field):
        self.invalid_fields |= _FIELD_BITS[field]
        self.values[field] = None

    def invalidate_scope(self):
        for target, _source in _SCOPE_FIELDS:
            self.invalidate(target)
        self.invalidate('evidence_geometry_bytes')
        self.invalidate('packing_scratch_cap_peak_bytes')


def begin_retry_attempt_bounds(metadata, decision, prepared, resource_profile):
    """Read frozen metadata only when an existing trace sink is enabled."""
    result = None
    try:
        if _trace_sink() is None:
            return None
        result = RetryAttemptBounds()
        values = result.values
        values['scope_id'] = result.text(metadata.get('scope_id', 'sam'))
        values['original_group_id'] = result.text(decision.original_id)
        values['interval_identity'] = result.text(decision.interval_identity)
        # Real decisions expose a defensive-copy record; read their immutable
        # scalar source directly without copying the surrounding receipt heap.
        from .sam_crop_retry import SamCropRetryDecision
        record = decision._record if isinstance(decision, SamCropRetryDecision) else decision.record
        values['attempt_index'] = result.integer(record.get('attempt_index'), positive=True)
        values['approved_retry_peak_bytes'] = result.integer(record.get('estimated_peak_bytes'))
        for index, field in enumerate(('crop_y0', 'crop_x0', 'crop_y1', 'crop_x1')):
            values[field] = result.integer(decision.crop_bbox_yx[index])
        from .workspace import _env_int
        values['render_scratch_bytes'] = max(1, _env_int('YOLO_TTA_SAM_RENDER_MAX_BYTES', 256*1024**2))
        values['reader_cache_bytes'] = 32*1024**2
        from .sam_resources import SamResourceProfile
        if not isinstance(resource_profile, SamResourceProfile):
            result.flag('profile_unknown')
        for target, source in (
                ('observation_allowance_bytes', 'assigned_observation_bytes'),
                ('metadata_allowance_bytes', 'assigned_contract_metadata_bytes'),
                ('image_allowance_bytes', 'assigned_image_bytes')):
            values[target] = result.integer(getattr(resource_profile, source, None))
        if prepared is None:
            result.flag('prepared_unknown')
            result.flag('partial_observation')
            return result
        values['crop_mode'] = result.text(prepared.crop_mode)
        if prepared.crop_mode == 'tiled':
            result.flag('tiled_geometry_unknown')
        elif prepared.crop_mode != 'whole':
            result.flag('geometry_unknown')
        jobs = prepared.tracker_jobs if prepared.crop_mode == 'tiled' else prepared.runs
        values['job_count'] = len(jobs)
        values['assigned_wave_bytes'] = result.integer(prepared.cpu_wave_admission.get('assigned_cpu_wave_bytes'))
        payload = 0
        for box in prepared.frame_crop_bounds.values():
            y0, x0, y1, x1 = (result.integer(box[index]) for index in range(4))
            if None in (y0, x0, y1, x1) or y1 <= y0 or x1 <= x0:
                result.flag('prepared_unknown')
                result.flag('partial_observation')
                break
            payload += (y1-y0)*(x1-x0)
        else:
            values['gray_payload_bytes'] = payload
    except BaseException:
        if result is not None:
            result.flag('diagnostic_read_failed')
            result.flag('partial_observation')
    return result


def observe_retry_scope(bounds, admission):
    """Copy every created scope's fixed scalars, without touching its lifetime."""
    if bounds is None:
        return
    try:
        if bounds.emitted:
            return
        bounds.scopes_seen = min(_COUNT_CAP, bounds.scopes_seen+1)
        from .sam_resources import SamTrackerScopeAdmission
        if not isinstance(admission, SamTrackerScopeAdmission):
            bounds.flag('scope_owner_unknown')
            bounds.flag('partial_observation')
            bounds.invalidate_scope()
            return
        limits = admission._limits
        if not isinstance(limits, Mapping) or limits.get('schema') != 'xta.sam_tracker_scope_admission/1':
            bounds.flag('scope_limits_unknown')
            bounds.flag('partial_observation')
            bounds.invalidate_scope()
            return
        bounds.scopes_with_limits = min(_COUNT_CAP, bounds.scopes_with_limits+1)
        for target, source in _SCOPE_FIELDS:
            bounds.maximum(target, bounds.integer(limits.get(source)))
        plane = bounds.integer(limits.get('maximum_evidence_plane_pixels'), positive=True)
        if bounds.values['crop_mode'] == 'whole' and plane is not None:
            bounds.maximum('evidence_geometry_bytes', 16*plane)
        else:
            bounds.flag('geometry_unknown')
            bounds.invalidate('evidence_geometry_bytes')
        # This is the permitted residual for this scope's protected maximum
        # plane, not measured use. Never combine maxima into a synthetic scope.
        unused = bounds.integer(limits.get('unused_attempt_wave_bytes'))
        fixed = bounds.integer(limits.get('evidence_fixed_bytes'))
        contract = bounds.integer(limits.get('evidence_contract_bytes'))
        loan = bounds.integer(limits.get('completion_wave_loan_bytes'))
        if None in (unused, fixed, contract, plane, loan):
            bounds.flag('packing_unknown')
            bounds.invalidate('packing_scratch_cap_peak_bytes')
        else:
            cap = max(0, unused-fixed-max(16*plane, contract)-loan)
            bounds.maximum('packing_scratch_cap_peak_bytes', cap)
    except BaseException:
        try:
            bounds.flag('diagnostic_read_failed')
            bounds.flag('partial_observation')
            bounds.invalidate_scope()
        except BaseException:
            pass


def finish_retry_attempt_bounds(bounds, *, status, metadata, decision, resource_profile):
    """Emit once after attempt unwind; an unprepared failure gets unknown fields.

    The replay owns one unconditional finally site. A fallback created from None
    is returned so other callers can retain its idempotence marker if necessary.
    """
    try:
        if bounds is None:
            bounds = begin_retry_attempt_bounds(metadata, decision, None, resource_profile)
        if bounds is None or bounds.emitted:
            return bounds
        bounds.emitted = True  # Never retry a failed optional diagnostic write.
        if type(status) is not str or status not in ('succeeded', 'failed', 'cancelled'):
            status = 'unknown'
            bounds.flag('partial_observation')
        if status != 'succeeded':
            bounds.flag('partial_observation')
            bounds.flag('retirement_unproved')
            if status in ('failed', 'cancelled'):
                bounds.flag(status)
        if not bounds.scopes_seen:
            bounds.flag('no_scopes')
            bounds.flag('partial_observation')
        flags = '|'.join(name for name in _FLAG_NAMES if bounds.flags & _FLAGS[name])
        bounds.values['unproved_owner_flags'] = (
            f'v=1;outcome={status};scopes_seen={bounds.scopes_seen};'
            f'scopes_with_limits={bounds.scopes_with_limits};flags={flags}')
        sink = _trace_sink()
        if sink is not None:
            sink.trace_event(EVENT, **bounds.values)
    except BaseException:
        pass
    return bounds

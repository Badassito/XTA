"""One bounded SAM crop retry, preserving the original seed and full interval.

This module decides geometry and accounts for additional tracking work. It never
renders images, constructs prompts, or grants CPU/GPU resource credits. Callers
must use live resource admission before allocating the admitted larger crop.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import copy
import hashlib
import json
import math
from pathlib import Path
import threading
from typing import Callable, Mapping

import numpy as np

SCHEMA = 'xta.sam_crop_retry/1'
CONTACT_SCHEMA = 'xta.sam_raw_crop_contacts/1'
CHILD_CONTACT_SCHEMA = 'xta.sam_raw_child_crop_contacts/1'
_SIDES = ('top', 'left', 'bottom', 'right')


def _integer(value, name, *, minimum=0):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return int(value)


def _geometry(crop_bbox_yx, canvas_shape_yx):
    if len(crop_bbox_yx) != 4 or len(canvas_shape_yx) != 2:
        raise ValueError('SAM crop requires a YX rectangle and canvas shape')
    box = tuple(_integer(value, 'crop coordinate') for value in crop_bbox_yx)
    canvas = tuple(_integer(value, 'canvas dimension', minimum=1) for value in canvas_shape_yx)
    if not (0 <= box[0] < box[2] <= canvas[0] and 0 <= box[1] < box[3] <= canvas[1]):
        raise ValueError('SAM crop must be a nonempty rectangle inside its declared canvas')
    return box, canvas


def raw_crop_boundary_contacts(mask, crop_bbox_yx, canvas_shape_yx) -> dict:
    """Count contacts on every RAW frame; declared canvas edges never trigger.

    Corner pixels are counted for each contacted side. No write, radius, score,
    topology, or unrelated-mask filtering is applied to these raw observations.
    """
    box, canvas = _geometry(crop_bbox_yx, canvas_shape_yx)
    raw = np.asarray(mask)
    if raw.ndim != 2 or raw.shape != (box[2]-box[0], box[3]-box[1]):
        raise ValueError('Raw SAM mask differs from its declared crop geometry')
    counts = dict(zip(_SIDES, (int(np.count_nonzero(raw[0, :])),
        int(np.count_nonzero(raw[:, 0])), int(np.count_nonzero(raw[-1, :])),
        int(np.count_nonzero(raw[:, -1])))))
    return crop_boundary_contacts_from_counts(counts, box, canvas)


def crop_boundary_contacts_from_counts(counts, crop_bbox_yx, canvas_shape_yx) -> dict:
    """Canonical contact record for independently authenticated raw edge counts."""
    box, canvas = _geometry(crop_bbox_yx, canvas_shape_yx)
    if set(counts) != set(_SIDES):
        raise ValueError('Raw crop boundary sides are incomplete')
    counts = {side: _integer(counts[side], 'raw boundary contact count') for side in _SIDES}
    limits = dict(zip(_SIDES, (box[3]-box[1], box[2]-box[0], box[3]-box[1], box[2]-box[0])))
    if any(counts[side] > limits[side] for side in _SIDES):
        raise ValueError('Raw boundary contact count exceeds its declared side length')
    internal = dict(zip(_SIDES, (box[0] > 0, box[1] > 0, box[2] < canvas[0], box[3] < canvas[1])))
    return dict(schema=CONTACT_SCHEMA, crop_bbox_yx=list(box), canvas_shape_yx=list(canvas),
        raw_frame_count=1, internal_contacts={s: counts[s] if internal[s] else 0 for s in _SIDES},
        canvas_edge_contacts={s: 0 if internal[s] else counts[s] for s in _SIDES})


def merge_crop_contacts(contacts) -> dict:
    """Merge all frames/runs of one original crop without retaining masks."""
    records = list(contacts)
    if not records:
        raise ValueError('At least one raw crop contact record is required')
    first = records[0]
    box, canvas = _geometry(first['crop_bbox_yx'], first['canvas_shape_yx'])
    merged = dict(schema=CONTACT_SCHEMA, crop_bbox_yx=list(box), canvas_shape_yx=list(canvas),
        raw_frame_count=0, internal_contacts={s: 0 for s in _SIDES}, canvas_edge_contacts={s: 0 for s in _SIDES})
    for record in records:
        if record.get('schema') != CONTACT_SCHEMA or _geometry(record['crop_bbox_yx'], record['canvas_shape_yx']) != (box, canvas):
            raise ValueError('Raw contact records must have the same declared crop and canvas')
        merged['raw_frame_count'] += _integer(record['raw_frame_count'], 'raw frame count')
        for kind in ('internal_contacts', 'canvas_edge_contacts'):
            if set(record[kind]) != set(_SIDES):
                raise ValueError('Raw crop contact sides are incomplete')
            for side in _SIDES:
                merged[kind][side] += _integer(record[kind][side], 'raw boundary contact count')
    return merged


def raw_child_crop_boundary_contacts(mask, child_crop_bbox_yx, group_crop_bbox_yx,
        canvas_shape_yx) -> dict:
    """Diagnose tile censoring separately; internal child edges NEVER retry.

    This feature enlarges outer group context only. A child may remain censored
    inside that context, and an unseeded ownership footprint remains unknown.
    Neither this diagnostic nor a successful outer retry certifies full coverage.
    """
    child, canvas = _geometry(child_crop_bbox_yx, canvas_shape_yx)
    group, _ = _geometry(group_crop_bbox_yx, canvas_shape_yx)
    if not (group[0] <= child[0] < child[2] <= group[2]
            and group[1] <= child[1] < child[3] <= group[3]):
        raise ValueError('SAM child crop must remain inside its declared group context')
    raw = raw_crop_boundary_contacts(mask, child, canvas)
    inside = dict(zip(_SIDES, (child[0] > group[0], child[1] > group[1],
        child[2] < group[2], child[3] < group[3])))
    internal = {side: raw['internal_contacts'][side] if inside[side] else 0 for side in _SIDES}
    return dict(schema=CHILD_CONTACT_SCHEMA, crop_bbox_yx=list(child), group_crop_bbox_yx=list(group),
        canvas_shape_yx=list(canvas), raw_frame_count=1, internal_child_contacts=internal,
        outer_group_contacts={side: 0 if inside[side] else raw['internal_contacts'][side] for side in _SIDES},
        canvas_edge_contacts=raw['canvas_edge_contacts'],
        extent_censored=bool(any(internal.values())),
        internal_child_enlargement='unsupported; outer_group_context_retry_only',
        coverage_proof=False)


def summarize_child_crop_contacts(contacts) -> dict:
    """Bounded scalar census across independent raw tile sessions and frames."""
    summary = dict(schema=CHILD_CONTACT_SCHEMA, raw_frame_count=0,
        internal_child_contacts={s: 0 for s in _SIDES}, outer_group_contacts={s: 0 for s in _SIDES},
        canvas_edge_contacts={s: 0 for s in _SIDES}, child_crops=[], extent_censored=False,
        internal_child_enlargement='unsupported; outer_group_context_retry_only', coverage_proof=False,
        count_basis='sum_per_side_independent_raw_child_observations; corners_and_repeated_sessions_counted')
    context = None
    boxes = set()
    for record in contacts:
        if record.get('schema') != CHILD_CONTACT_SCHEMA:
            raise ValueError('Unknown raw child crop contact schema')
        box, canvas = _geometry(record['crop_bbox_yx'], record['canvas_shape_yx'])
        group, _ = _geometry(record['group_crop_bbox_yx'], canvas)
        current = (group, canvas)
        if context is not None and context != current:
            raise ValueError('Raw child contact records must share their original group context')
        context = current
        boxes.add(box)
        summary['raw_frame_count'] += _integer(record['raw_frame_count'], 'raw child frame count')
        for kind in ('internal_child_contacts', 'outer_group_contacts', 'canvas_edge_contacts'):
            if set(record[kind]) != set(_SIDES):
                raise ValueError('Raw child crop contact sides are incomplete')
            for side in _SIDES:
                summary[kind][side] += _integer(record[kind][side], 'raw child contact count')
    if context is not None:
        summary.update(group_crop_bbox_yx=list(context[0]), canvas_shape_yx=list(context[1]))
    summary['child_crops'] = [list(box) for box in sorted(boxes)]
    summary['extent_censored'] = bool(any(summary['internal_child_contacts'].values()))
    return summary


@dataclass(frozen=True)
class SamCropRetryPolicy:
    enabled: bool = False
    expansion_min_pixels: int = 64
    expansion_fraction: float = .25
    max_area_ratio: float = 2.
    max_crop_pixels: int = 16_777_216
    max_extra_pixel_frames: int = 268_435_456
    extra_work_fraction: float = .25
    extra_work_floor_pixel_frames: int = 16_777_216
    max_extra_tracker_frames: int = 512
    max_retry_memory_bytes: int = 2*1024**3

    def __post_init__(self):
        if not isinstance(self.enabled, bool):
            raise ValueError('SAM adaptive crop enabled must be boolean')
        for key in ('expansion_min_pixels', 'max_crop_pixels', 'max_extra_pixel_frames',
                    'extra_work_floor_pixel_frames', 'max_retry_memory_bytes'):
            _integer(getattr(self, key), key, minimum=1)
        _integer(self.max_extra_tracker_frames, 'max_extra_tracker_frames', minimum=1)
        for key in ('expansion_fraction', 'max_area_ratio', 'extra_work_fraction'):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{key} must be finite and positive')
        if self.max_area_ratio <= 1:
            raise ValueError('max_area_ratio must permit an actual enlargement')

    def to_dict(self):
        settings = dict(schema=SCHEMA, **asdict(self), max_attempts_per_original=1,
            trigger='raw_internal_crop_edge_contact', seed_basis='same_frozen_original_seed_and_history',
            interval_basis='same_complete_original_interval', work_charge='full_retry_pixel_frames',
            implementation_sha256=IMPLEMENTATION_SHA256)
        settings['policy_sha256'] = hashlib.sha256(json.dumps(settings, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        return settings


@dataclass(frozen=True)
class SamCropRetryDecision:
    original_id: str
    retry: bool
    crop_bbox_yx: tuple[int, int, int, int]
    reason: str
    seed_identity: str
    interval_identity: str
    _record: dict

    @property
    def record(self):
        return copy.deepcopy(self._record)


class SamCropRetryController:
    """Atomic per-scope budget; failed attempts retain their full work charge."""
    def __init__(self, policy: SamCropRetryPolicy, *, baseline_pixel_frames: int,
            baseline_tracker_frames: int, largest_original_group_pixel_frames: int = 0,
            largest_original_group_tracker_frames: int = 0):
        if not isinstance(policy, SamCropRetryPolicy):
            raise TypeError('SAM crop retry requires an explicit frozen policy')
        self.policy = policy
        self.baseline_pixel_frames = _integer(baseline_pixel_frames, 'baseline pixel frames')
        self.baseline_tracker_frames = _integer(baseline_tracker_frames, 'baseline tracker frames')
        self.largest_original_group_pixel_frames = _integer(largest_original_group_pixel_frames, 'largest original group pixel frames')
        self.largest_original_group_tracker_frames = _integer(largest_original_group_tracker_frames, 'largest original group tracker frames')
        if self.largest_original_group_pixel_frames > self.baseline_pixel_frames or self.largest_original_group_tracker_frames > self.baseline_tracker_frames:
            raise ValueError('Largest original group work cannot exceed original scope work')
        self.extra_work_limit = min(policy.max_extra_pixel_frames,
            max(policy.extra_work_floor_pixel_frames, math.ceil(self.baseline_pixel_frames*policy.extra_work_fraction),
                2*self.largest_original_group_pixel_frames))
        self.extra_tracker_frame_limit = min(policy.max_extra_tracker_frames,
            max(self.largest_original_group_tracker_frames, math.ceil(self.baseline_tracker_frames*policy.extra_work_fraction)))
        self._charged = 0
        self._charged_tracker_frames = 0
        self._records = {}
        self._decisions = {}
        self._lock = threading.Lock()

    def reserve_retry(self, original_id: str, *, crop_bbox_yx, canvas_shape_yx,
            frame_count: int, seed_identity: str, interval_identity: str, contacts: Mapping,
            available_memory_bytes: int, memory_estimator: Callable[[tuple], int],
            work_estimator: Callable[[tuple], int] | None = None,
            tracker_frame_estimator: Callable[[tuple], int] | None = None) -> SamCropRetryDecision:
        """Reserve ONE full rerun before render/SDK allocation, or refuse honestly.

        ``available_memory_bytes`` must come from current operative admission,
        never a serialized evidence profile. ``memory_estimator`` must cover all
        simultaneous retry allocations (render, masks, contracts, and tracker).
        Callers must retain the original attempt and verify seed/interval identity
        again immediately before submitting the new prompt to SAM.
        """
        if any(not isinstance(value, str) or not value for value in (original_id, seed_identity, interval_identity)):
            raise ValueError('Crop retry requires original owner, frozen seed, and interval identities')
        box, canvas = _geometry(crop_bbox_yx, canvas_shape_yx)
        frames = _integer(frame_count, 'complete retry frame count', minimum=1)
        available = _integer(available_memory_bytes, 'current available memory bytes')
        validated = merge_crop_contacts([contacts])
        if tuple(validated['crop_bbox_yx']) != box or tuple(validated['canvas_shape_yx']) != canvas:
            raise ValueError('Raw crop contacts differ from the requested original geometry')
        if not callable(memory_estimator):
            raise ValueError('Crop retry requires a current complete memory estimate')
        if any(value is not None and not callable(value) for value in (work_estimator, tracker_frame_estimator)):
            raise ValueError('Retry work estimates must be callable or None')
        with self._lock:
            prior = self._decisions.get(original_id)
            if prior is not None:
                self.verify_retry_identity(prior, seed_identity=seed_identity, interval_identity=interval_identity)
                return SamCropRetryDecision(original_id, False, box, 'retry_already_considered',
                    seed_identity, interval_identity, {**prior.record, 'retry': False,
                        'reason': 'retry_already_considered', 'status': 'refused'})
            record = dict(schema=SCHEMA, original_id=original_id, original_crop_bbox_yx=list(box),
                canvas_shape_yx=list(canvas), seed_identity=seed_identity, interval_identity=interval_identity,
                frame_count=frames, contacts=validated, retry=False, status='refused', extent_censored=bool(any(validated['internal_contacts'].values())),
                extra_work_limit_pixel_frames=self.extra_work_limit, charged_before_pixel_frames=self._charged,
                extra_tracker_frame_limit=self.extra_tracker_frame_limit,
                charged_before_tracker_frames=self._charged_tracker_frames,
                policy_sha256=self.policy.to_dict()['policy_sha256'])
            proposed = box
            reason = 'disabled'
            if self.policy.enabled:
                sides = tuple(s for s in _SIDES if validated['internal_contacts'][s] > 0)
                reason = 'no_internal_crop_contact'
                if sides:
                    proposed, requested = self._enlargement(box, canvas, sides)
                    record['requested_crop_bbox_yx'] = list(requested)
                    area = (proposed[2]-proposed[0])*(proposed[3]-proposed[1])
                    work = (frames*area if work_estimator is None else
                            _integer(work_estimator(proposed), 'full retry pixel frames', minimum=1))
                    tracker_frames = (frames if tracker_frame_estimator is None else
                            _integer(tracker_frame_estimator(proposed), 'full retry tracker frames', minimum=1))
                    record.update(retry_crop_pixels=area, retry_pixel_frames=work,
                        retry_tracker_frames=tracker_frames,
                        work_estimate_basis='complete_crop_area_times_run_frames' if work_estimator is None else 'complete_independent_tracker_jobs')
                    reason = 'crop_area_limit'
                    if proposed != box:
                        reason = 'extra_work_budget_exhausted'
                        if work <= self.extra_work_limit-self._charged:
                            reason = 'extra_tracker_frame_budget_exhausted'
                            if tracker_frames <= self.extra_tracker_frame_limit-self._charged_tracker_frames:
                                required = _integer(memory_estimator(proposed), 'retry estimated peak bytes', minimum=1)
                                record.update(estimated_peak_bytes=required, available_memory_bytes=available,
                                    hard_memory_limit_bytes=self.policy.max_retry_memory_bytes)
                                reason = 'memory_limit'
                                if required <= min(available, self.policy.max_retry_memory_bytes):
                                    self._charged += work
                                    self._charged_tracker_frames += tracker_frames
                                    reason = 'retry_reserved'
                                    record.update(retry=True, status='reserved')
            record.update(crop_bbox_yx=list(proposed), reason=reason, charged_after_pixel_frames=self._charged,
                charged_after_tracker_frames=self._charged_tracker_frames)
            decision = SamCropRetryDecision(original_id, record['retry'], proposed, reason,
                seed_identity, interval_identity, record)
            self._decisions[original_id] = decision
            self._records[original_id] = copy.deepcopy(record)
            return decision

    def _enlargement(self, box, canvas, sides):
        h, w = box[2]-box[0], box[3]-box[1]
        dy = max(self.policy.expansion_min_pixels, math.ceil(h*self.policy.expansion_fraction))
        dx = max(self.policy.expansion_min_pixels, math.ceil(w*self.policy.expansion_fraction))
        scale = max(dy, dx)
        area_limit = min(self.policy.max_crop_pixels, math.floor(h*w*self.policy.max_area_ratio))
        def at(step):
            y, x = dy*step//scale, dx*step//scale
            return (max(0, box[0]-y) if 'top' in sides else box[0],
                    max(0, box[1]-x) if 'left' in sides else box[1],
                    min(canvas[0], box[2]+y) if 'bottom' in sides else box[2],
                    min(canvas[1], box[3]+x) if 'right' in sides else box[3])
        requested = at(scale)
        lo, hi = 0, scale
        while lo < hi:
            mid = (lo+hi+1)//2
            candidate = at(mid)
            if (candidate[2]-candidate[0])*(candidate[3]-candidate[1]) <= area_limit:
                lo = mid
            else:
                hi = mid-1
        return at(lo), requested

    @staticmethod
    def verify_retry_identity(decision, *, seed_identity: str, interval_identity: str):
        if decision.seed_identity != seed_identity or decision.interval_identity != interval_identity:
            raise ValueError('SAM crop retry changed its frozen original seed/history or interval')

    def complete_retry(self, decision: SamCropRetryDecision, *, status: str, detail=None):
        if status not in {'succeeded', 'failed', 'cancelled'}:
            raise ValueError('Retry completion must be succeeded, failed, or cancelled')
        with self._lock:
            if self._decisions.get(decision.original_id) is not decision or not decision.retry:
                raise ValueError('Retry completion does not own an admitted original attempt')
            record = self._records[decision.original_id]
            if record['status'] != 'reserved':
                raise ValueError('SAM crop retry was already completed')
            record.update(status=status, completion_detail=copy.deepcopy(detail or {}))
            return copy.deepcopy(record)

    def receipt(self):
        with self._lock:
            return dict(schema=SCHEMA, policy=self.policy.to_dict(), baseline_pixel_frames=self.baseline_pixel_frames,
                baseline_tracker_frames=self.baseline_tracker_frames,
                largest_original_group_pixel_frames=self.largest_original_group_pixel_frames,
                largest_original_group_tracker_frames=self.largest_original_group_tracker_frames,
                extra_work_limit_pixel_frames=self.extra_work_limit, charged_pixel_frames=self._charged,
                extra_tracker_frame_limit=self.extra_tracker_frame_limit, charged_tracker_frames=self._charged_tracker_frames,
                attempts=copy.deepcopy(self._records))


IMPLEMENTATION_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()

__all__ = ['SamCropRetryPolicy', 'SamCropRetryDecision', 'SamCropRetryController',
           'raw_crop_boundary_contacts', 'merge_crop_contacts', 'raw_child_crop_boundary_contacts',
           'summarize_child_crop_contacts', 'crop_boundary_contacts_from_counts', 'SCHEMA', 'IMPLEMENTATION_SHA256']

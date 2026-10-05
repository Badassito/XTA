"""Immutable post-interpolation terminal masks for one-direction SAM tails."""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import math
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import ndimage as ndi

from .sam_bridge_planning import (SamBridgeRunPlan, SamPlanningLimits, _observations,
                                  _continuations, _identity)

SCHEMA = 'xta.sam_extrapolation_plan/1'


class _BinaryBaseline:
    """Slice-only foreground view; composite integer IDs never split a body."""
    def __init__(self,source):
        self.source=source
        self.shape=source.shape
        self.dtype=np.dtype(bool)
    def __getitem__(self,frame):
        return np.asarray(self.source[frame])!=0
    def __array__(self,*args,**kwargs):
        raise RuntimeError('Post-interpolation binary topology is slice-only')


@dataclass(frozen=True)
class SamExtrapolationGroup:
    group_id: str
    observation_ids: tuple[str, ...]
    terminal_id: str
    terminal_frame: int
    direction: int
    context_bbox_yx: tuple[int, int, int, int]
    frame_indices: tuple[int, ...]
    output_frames: tuple[int, ...]
    terminal_radius: float
    status: str = 'planned'
    reasons: tuple[str, ...] = ()
    edges: tuple = ()
    frame_addresses: Mapping = field(default_factory=dict, compare=False)
    frame_addressing: Mapping = field(default_factory=dict, compare=False)
    original_group_id: str = ''


@dataclass(frozen=True)
class SamExtrapolationRunPlan(SamBridgeRunPlan):
    terminal_id: str = ''
    terminal_frame: int = 0
    output_frames: tuple[int, ...] = ()


@dataclass(frozen=True)
class SamExtrapolationPlan:
    observations: tuple
    groups: tuple[SamExtrapolationGroup, ...]
    runs: tuple[SamExtrapolationRunPlan, ...]
    native_shape_tyx: tuple[int, int, int]
    virtual_shape_tyx: tuple[int, int, int]
    status: str = 'planned'
    reasons: tuple[str, ...] = ()
    skipped_by_min_radius: int = 0
    frame_addressing: Mapping = field(default_factory=dict, compare=False)
    frame_addresses: Mapping = field(default_factory=dict, compare=False)
    crop_contract_version: str = SCHEMA
    requested_distance: int = 0
    effective_distance: int = 0
    observations_by_frame: Mapping = field(default_factory=dict,compare=False,repr=False)
    _observation_index_source: tuple | None = field(default=None,compare=False,repr=False)

    def __post_init__(self):
        # Group/cohort/retry replacements keep the exact same observation tuple
        # and reuse this immutable metadata index. A new inventory rebuilds it.
        if self._observation_index_source is not self.observations:
            indexed={}
            for observation in self.observations:
                indexed.setdefault(int(observation.frame_index),[]).append(observation)
            object.__setattr__(self,'observations_by_frame',MappingProxyType(
                {frame:tuple(values) for frame,values in indexed.items()}))
            object.__setattr__(self,'_observation_index_source',self.observations)

    @property
    def by_id(self):
        return MappingProxyType({obs.observation_id: obs for obs in self.observations})

    @property
    def needed_frames(self):
        return tuple(sorted({frame for run in self.runs for frame in run.expected_frames}))

    @property
    def frame_crop_bounds(self):
        groups = {g.group_id: g for g in self.groups}
        bounds = {}
        for run in self.runs:
            crop = groups[run.group_id].context_bbox_yx
            for frame in run.expected_frames:
                prior = bounds.get(frame, crop)
                bounds[frame] = (min(prior[0], crop[0]), min(prior[1], crop[1]),
                                 max(prior[2], crop[2]), max(prior[3], crop[3]))
        return MappingProxyType(bounds)


def terminal_radius(mask):
    """Exact bounded pixel radius; tight full-white rectangles are not infinite."""
    array = np.asarray(mask, bool)
    if array.ndim != 2 or not array.any():
        return 0.
    return float(ndi.distance_transform_edt(np.pad(array, 1)).max())


def _integer(value, name):
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(name+' must be a nonnegative integer')
    return int(value)


def plan_sam_extrapolation(observations, *, extrapolation_distance=0,
                           extrapolation_walk_back=1, extrapolation_min_radius=3.,
                           limits=None, scope_id='sam', observation_lineage=None,
                           canonical_labels=None, wrap_axis=False, eligible_terminals=None):
    """Freeze every tail before any tracking; walk-back uses one prior mask/run.

    Distance counts frames beyond the actual remaining terminal. The radius
    gate applies only to that terminal; neither walk-back seeds nor predictions
    are radius-filtered. Canonical detector IDs are not authoritative for a
    composite post-interpolation baseline.
    """
    distance = _integer(extrapolation_distance, 'extrapolation_distance')
    requested_distance = distance
    walk_back = _integer(extrapolation_walk_back, 'extrapolation_walk_back')
    radius = float(extrapolation_min_radius)
    if isinstance(extrapolation_min_radius, bool) or not math.isfinite(radius) or radius < 0:
        raise ValueError('extrapolation_min_radius must be finite and nonnegative')
    shape = tuple(map(int, observations.shape))
    if len(shape) != 3 or any(v < 1 for v in shape):
        raise ValueError('SAM extrapolation baseline requires a positive TYX canvas')
    if not distance:
        return SamExtrapolationPlan((), (), (), shape, shape, 'disabled')
    bounds = limits or SamPlanningLimits()
    if shape[1]*shape[2] > bounds.max_slice_pixels:
        return SamExtrapolationPlan((), (), (), shape, shape, 'unresolved', ('observation_slice_pixel_limit',),
                                    requested_distance=requested_distance,effective_distance=distance)
    volume = _BinaryBaseline(observations)
    addressing, addresses = {}, {}
    if wrap_axis:
        from .sam_cyclic import ExtrapolationCyclicObservationVolume
        distance = min(distance, max(0, shape[0]-1))
        if not distance:
            return SamExtrapolationPlan((), (), (), shape, shape, 'empty', ('cyclic_single_frame',),
                                        requested_distance=requested_distance,effective_distance=0)
        volume = ExtrapolationCyclicObservationVolume(volume, distance+min(walk_back,shape[0]-1))
        addressing, addresses = volume.frame_addressing, volume.frame_addresses
    lineage = dict(observation_lineage or {})
    lineage.update(source_stage='post_interpolation', observation_source='post_interpolation')
    observed, by_frame, failure = _observations(volume, scope=str(scope_id), labels=None,
                                               lineage=lineage, limits=bounds)
    if failure:
        return SamExtrapolationPlan(tuple(observed), (), (), shape, tuple(volume.shape),
                                    'unresolved', (failure,), frame_addressing=addressing,
                                    frame_addresses=addresses,requested_distance=requested_distance,
                                    effective_distance=distance)
    before, after = _continuations(observed, by_frame)
    # Alias copies are the same frozen observation, not separate families.
    parents = list(range(len(observed)))
    def root(index):
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index
    def join(a, b):
        a, b = root(a), root(b)
        parents[max(a,b)] = min(a,b)
    originals = {}
    for i, obs in enumerate(observed):
        key = obs.original_observation_id or obs.observation_id
        if key in originals:
            join(i, originals[key])
        else:
            originals[key] = i
        for neighbor in after[i]:
            join(i, neighbor)
    observed = [replace(obs, canonical_label=root(i)+1) for i, obs in enumerate(observed)]
    alias_by_root_frame = {(obs.original_observation_id or obs.observation_id, obs.frame_index): i
                          for i, obs in enumerate(observed)}
    groups, runs, skipped = [], [], 0
    counted_radius = set()
    for primary, original in enumerate(observed):
        if original.frame_index >= shape[0]:
            continue
        for direction in (-1, 1):
            index = primary
            if wrap_axis and ((direction < 0 and original.frame_index < distance)
                              or (direction > 0 and original.frame_index < min(walk_back,shape[0]-1))):
                index = alias_by_root_frame.get((original.original_observation_id, shape[0]+original.frame_index), primary)
            terminal = observed[index]
            outward = before if direction < 0 else after
            if outward[index]:
                continue
            identity = terminal.original_observation_id or terminal.observation_id
            if eligible_terminals is not None:
                allowed = (eligible_terminals(terminal, direction) if callable(eligible_terminals)
                           else identity in eligible_terminals or (identity,direction) in eligible_terminals)
                if not allowed:
                    continue
            measured = terminal_radius(terminal.mask_crop)
            if measured <= radius:
                if identity not in counted_radius:
                    skipped += 1
                    counted_radius.add(identity)
                continue
            remaining=(volume.shape[0]-1-terminal.frame_index) if direction>0 else terminal.frame_index
            output = tuple(terminal.frame_index+direction*n for n in range(1,min(distance,remaining)+1))
            if not output:
                continue
            seeds = [index]
            inward = after if direction < 0 else before
            current = index
            seen_seed_roots={terminal.original_observation_id or terminal.observation_id}
            for _ in range(walk_back):
                choices = inward[current]
                if not choices:
                    break
                current = min(choices, key=lambda j: (
                    sum((a-b)**2 for a,b in zip(observed[current].anchor_yx, observed[j].anchor_yx)),
                    observed[j].observation_id))
                seed_root=observed[current].original_observation_id or observed[current].observation_id
                if seed_root in seen_seed_roots:
                    break
                seen_seed_roots.add(seed_root)
                seeds.append(current)
            margin = max(128, bounds.context_margin_px)
            crop = (max(0,min(observed[j].bbox_yx[0] for j in seeds)-margin),
                    max(0,min(observed[j].bbox_yx[1] for j in seeds)-margin),
                    min(shape[1],max(observed[j].bbox_yx[2] for j in seeds)+margin),
                    min(shape[2],max(observed[j].bbox_yx[3] for j in seeds)+margin))
            frames = tuple(range(min((*output,*(observed[j].frame_index for j in seeds))),
                                 max((*output,*(observed[j].frame_index for j in seeds)))+1))
            pixels = (crop[2]-crop[0])*(crop[3]-crop[1])
            reasons = []
            if pixels > bounds.max_crop_pixels:
                reasons.append('context_crop_pixel_limit')
            if len(frames)*pixels > bounds.max_group_bytes:
                reasons.append('raw_session_memory_limit')
            if bounds.max_frames_per_group is not None and len(frames)>bounds.max_frames_per_group:
                reasons.append('session_frame_limit')
            gid = _identity('sam_extrap_group', scope_id, identity, direction, crop, frames)
            frame_addresses = MappingProxyType({f:addresses[f] for f in frames}) if addressing else {}
            group_addressing = MappingProxyType({**dict(addressing),'addresses':frame_addresses}) if addressing else {}
            group = SamExtrapolationGroup(gid, tuple(observed[j].observation_id for j in seeds),
                terminal.observation_id, terminal.frame_index, direction, crop, frames, output, measured,
                'unresolved' if reasons else 'planned', tuple(reasons),
                frame_addresses=frame_addresses, frame_addressing=group_addressing)
            groups.append(group)
            if reasons:
                continue
            for walk_index, seed in enumerate(seeds):
                expected = tuple(range(observed[seed].frame_index, output[-1]+direction, direction))
                rid = _identity('sam_extrap_run', gid, observed[seed].observation_id, expected)
                runs.append(SamExtrapolationRunPlan(rid,gid,(observed[seed].observation_id,),(),direction,
                    expected,(),walk_back_index=walk_index,terminal_id=terminal.observation_id,
                    terminal_frame=terminal.frame_index,output_frames=output))
    failures=tuple(sorted({reason for group in groups for reason in group.reasons}))
    status=('partial' if failures else 'planned') if runs else ('unresolved' if failures else 'empty')
    return SamExtrapolationPlan(tuple(observed),tuple(groups),tuple(runs),shape,tuple(volume.shape),
        status,reasons=failures+(('cyclic_horizon_period_limit',) if distance<requested_distance else ()),
        skipped_by_min_radius=skipped,frame_addressing=addressing,frame_addresses=addresses,
        requested_distance=requested_distance,effective_distance=distance)


def expanded_extrapolation_plan(plan,group_id,crop_bbox_yx):
    """One bounded retry from exactly the same frozen masks and intervals."""
    group=next(g for g in plan.groups if g.group_id==group_id)
    new=tuple(map(int,crop_bbox_yx));old=group.context_bbox_yx
    if (not 0<=new[0]<=old[0]<old[2]<=new[2]<=plan.native_shape_tyx[1]
            or not 0<=new[1]<=old[1]<old[3]<=new[3]<=plan.native_shape_tyx[2] or new==old):
        raise ValueError('SAM extrapolation retry must strictly enlarge its original fixed context')
    gid=_identity('sam_extrap_retry_group',group_id,new)
    expanded=replace(group,group_id=gid,context_bbox_yx=new,original_group_id=group.original_group_id or group_id)
    runs=tuple(replace(run,group_id=gid,run_id=_identity('sam_extrap_retry_run',run.run_id,new))
               for run in plan.runs if run.group_id==group_id)
    return replace(plan,groups=(expanded,),runs=runs,status='planned',reasons=())

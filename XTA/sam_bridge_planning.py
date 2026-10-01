"""Bounded, observation-only planning for native-view SAM interpolation.

SAM crops are planned in the consolidated view canvas. Detector tile rectangles
never enter this module. Every mask contract is fixed before inference; tracker
output cannot enlarge it. Canonical labels describe provenance, while stable
slice-local component IDs describe endpoints (including same-label daughters).
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
import math
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import ndimage as ndi


BBoxYX = tuple[int, int, int, int]  # y0, x0, y1, x1; upper bounds exclusive


def _readonly(mask: np.ndarray) -> np.ndarray:
    array = np.ascontiguousarray(mask, dtype=bool)
    # A bytes owner prevents callers from re-enabling WRITEABLE on the view.
    return np.frombuffer(array.tobytes(), dtype=bool).reshape(array.shape)


def _identity(kind: str, *values: object) -> str:
    payload = repr(values).encode("utf-8")
    return f"{kind}_{hashlib.sha256(payload).hexdigest()[:24]}"


@dataclass(frozen=True)
class SamPlanningLimits:
    max_observations: int = 200_000
    max_observation_bytes: int = 512 * 1024 * 1024
    max_slice_pixels: int = 16_777_216
    max_endpoints_per_group: int = 64
    max_observations_per_group: int = 512
    max_edges_per_group: int = 128
    max_proposed_edges: int = 100_000
    max_frames_per_group: int = 128
    max_crop_pixels: int = 4_194_304
    max_group_bytes: int = 256 * 1024 * 1024
    max_total_contract_bytes: int = 512 * 1024 * 1024
    context_margin_px: int = 24
    acceptance_margin_px: int = 8
    curvature_margin_px: int = 8

    def __post_init__(self) -> None:
        for key, value in vars(self).items():
            if int(value) < (0 if key.endswith("margin_px") else 1):
                raise ValueError(f"{key} has an invalid planning bound")


@dataclass(frozen=True)
class SamObservation:
    observation_id: str
    frame_index: int
    canonical_label: int
    component_index: int
    bbox_yx: BBoxYX
    mask_crop: np.ndarray = field(compare=False, repr=False)
    lineage: Mapping[str, Any] = field(default_factory=dict, compare=False)
    identity_limitation: str = "slice_connected_components_not_detector_instances"

    @property
    def area(self) -> int:
        return int(np.count_nonzero(self.mask_crop))

    @property
    def anchor_yx(self) -> tuple[float, float]:
        ys, xs = np.nonzero(self.mask_crop)
        return (float(ys.mean()) + self.bbox_yx[0],
                float(xs.mean()) + self.bbox_yx[1])

    def mask_in_crop(self, crop: BBoxYX) -> np.ndarray:
        y0, x0, y1, x1 = crop
        oy0, ox0, oy1, ox1 = self.bbox_yx
        if not (y0 <= oy0 <= oy1 <= y1 and x0 <= ox0 <= ox1 <= x1):
            raise ValueError(f"observation {self.observation_id} does not fit declared crop")
        mask = np.zeros((y1 - y0, x1 - x0), dtype=bool)
        mask[oy0 - y0:oy1 - y0, ox0 - x0:ox1 - x0] = self.mask_crop
        return _readonly(mask)


@dataclass(frozen=True)
class SamBridgeEdge:
    edge_id: str
    source_id: str
    target_id: str
    candidate_index: int = 1


@dataclass(frozen=True)
class SamBridgeGroup:
    group_id: str
    observation_ids: tuple[str, ...]
    edges: tuple[SamBridgeEdge, ...]
    context_bbox_yx: BBoxYX
    frame_indices: tuple[int, ...]
    acceptance_masks: np.ndarray = field(compare=False, repr=False)
    write_masks: np.ndarray = field(compare=False, repr=False)
    branch_evaluation_masks: Mapping[str, np.ndarray] = field(compare=False, repr=False)
    edge_write_masks: Mapping[str, np.ndarray] = field(compare=False, repr=False)
    known_foreground_masks: np.ndarray = field(compare=False, repr=False)
    unrelated_masks: np.ndarray = field(compare=False, repr=False)
    status: str = "planned"
    reasons: tuple[str, ...] = ()
    interpolation_min_radius: float = 0.0
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0)
    endpoint_ids: tuple[str, ...] = ()
    branch_permitted_masks: Mapping[str, np.ndarray] = field(default_factory=dict, compare=False, repr=False)
    edge_contract_masks: Mapping[str, np.ndarray] = field(default_factory=dict, compare=False, repr=False)

    @property
    def frame_start(self) -> int:
        return self.frame_indices[0]

    @property
    def frame_stop(self) -> int:
        """Exclusive view-native frame upper bound."""
        return self.frame_indices[-1] + 1

    def frame_offset(self, frame: int) -> int:
        index = int(frame) - self.frame_start
        if index < 0 or index >= len(self.frame_indices):
            raise KeyError(f"frame {frame} is outside group {self.group_id}")
        return index


@dataclass(frozen=True)
class SamBridgeRunPlan:
    run_id: str
    group_id: str
    seed_ids: tuple[str, ...]
    held_out_ids: tuple[str, ...]
    direction: int
    expected_frames: tuple[int, ...]
    edge_ids: tuple[str, ...]
    pass_index: int = 1
    walk_back_index: int = 0

    @property
    def seed_frame_index(self) -> int:
        return self.expected_frames[0]


@dataclass(frozen=True)
class SamBridgePlan:
    observations: tuple[SamObservation, ...]
    groups: tuple[SamBridgeGroup, ...]
    runs: tuple[SamBridgeRunPlan, ...]
    requested_passes: int
    completed_passes: int
    skipped_passes: int
    status: str = "planned"
    reasons: tuple[str, ...] = ()
    inventory_fingerprint: str = ""
    planning_fingerprint: str = ""

    @property
    def by_id(self) -> Mapping[str, SamObservation]:
        return MappingProxyType({item.observation_id: item for item in self.observations})

    @property
    def needed_frames(self) -> tuple[int, ...]:
        """Only frames actually visited by predeclared endpoint sessions."""
        return tuple(sorted({int(frame) for run in self.runs for frame in run.expected_frames}))

    @property
    def frame_crop_bounds(self) -> Mapping[int, BBoxYX]:
        """Fixed crop union per visited frame, without changing any run crop."""
        groups = {group.group_id: group for group in self.groups}
        bounds: dict[int, BBoxYX] = {}
        for run in self.runs:
            crop = groups[run.group_id].context_bbox_yx
            for frame in run.expected_frames:
                old = bounds.get(int(frame))
                bounds[int(frame)] = crop if old is None else (
                    min(old[0], crop[0]), min(old[1], crop[1]),
                    max(old[2], crop[2]), max(old[3], crop[3]))
        return MappingProxyType(bounds)


def _intersects(a: BBoxYX, b: BBoxYX, pad: int = 0) -> bool:
    return (a[0] - pad < b[2] and b[0] < a[2] + pad
            and a[1] - pad < b[3] and b[1] < a[3] + pad)


def _adjacent_masks(a: SamObservation, b: SamObservation) -> bool:
    if not _intersects(a.bbox_yx, b.bbox_yx, 1):
        return False
    ay0, ax0, ay1, ax1 = a.bbox_yx
    by0, bx0, by1, bx1 = b.bbox_yx
    crop = (min(ay0, by0) - 1, min(ax0, bx0) - 1,
            max(ay1, by1) + 1, max(ax1, bx1) + 1)
    return bool(np.any(ndi.binary_dilation(a.mask_in_crop(crop), structure=np.ones((3, 3)))
                       & b.mask_in_crop(crop)))


def _observations(volume: object, *, scope: str, labels: object | None,
                  lineage: Mapping[str, Any] | None, limits: SamPlanningLimits
                  ) -> tuple[list[SamObservation], dict[int, list[int]], str | None]:
    result: list[SamObservation] = []
    by_frame: dict[int, list[int]] = {}
    shape = tuple(int(v) for v in volume.shape)  # type: ignore[attr-defined]
    labelled = labels is not None or np.dtype(volume.dtype).kind in "iu"  # type: ignore[attr-defined]
    charge = 0
    for frame in range(shape[0]):
        plane = np.asarray(volume[frame])  # type: ignore[index]
        canonical = np.asarray(labels[frame]) if labels is not None else plane  # type: ignore[index]
        frame_items: list[int] = []
        component_index = 0
        foreground = plane != 0
        values = np.unique(canonical[foreground]) if labelled else (0,)
        for value in values:
            mask = foreground & (canonical == value) if labelled else foreground
            local, count = ndi.label(mask, structure=np.ones((3, 3), dtype=bool))
            for local_index, slices in enumerate(ndi.find_objects(local), start=1):
                if slices is None:
                    continue
                if len(result) >= limits.max_observations:
                    return result, by_frame, "observation_inventory_limit"
                component_index += 1
                sy, sx = slices
                bbox = (int(sy.start), int(sx.start), int(sy.stop), int(sx.stop))
                charge += (bbox[2] - bbox[0]) * (bbox[3] - bbox[1]) + 1024
                if charge > limits.max_observation_bytes:
                    return result, by_frame, "observation_inventory_memory_limit"
                crop = _readonly(local[slices] == local_index)
                digest = hashlib.sha256(np.packbits(crop).tobytes()).hexdigest()
                identifier = _identity("sam_obs", scope, frame, bbox, digest, int(value))
                roots = dict(lineage or {})
                roots.setdefault("observation_source", "detector")
                roots.setdefault("scope_id", scope)
                frame_items.append(len(result))
                result.append(SamObservation(identifier, frame, int(value), component_index,
                                             bbox, crop, MappingProxyType(roots)))
        by_frame[frame] = frame_items
    return result, by_frame, None


def _continuations(observations: Sequence[SamObservation], by_frame: Mapping[int, list[int]],
                   *, explicit_canonical_identity: bool = False
                   ) -> tuple[dict[int, list[int]], dict[int, list[int]]]:
    before = {index: [] for index in range(len(observations))}
    after = {index: [] for index in range(len(observations))}
    # Slice spatial buckets avoid a quadratic all-component comparison.
    for frame, indexes in by_frame.items():
        following = by_frame.get(frame + 1, [])
        buckets: dict[tuple[int, int], list[int]] = {}
        for target in following:
            y0, x0, y1, x1 = observations[target].bbox_yx
            for gy in range(y0 // 64, (y1 - 1) // 64 + 1):
                for gx in range(x0 // 64, (x1 - 1) // 64 + 1):
                    buckets.setdefault((gy, gx), []).append(target)
        for source in indexes:
            y0, x0, y1, x1 = observations[source].bbox_yx
            nearby: set[int] = set()
            for gy in range((y0 - 1) // 64, y1 // 64 + 1):
                for gx in range((x0 - 1) // 64, x1 // 64 + 1):
                    nearby.update(buckets.get((gy, gx), ()))
            for target in sorted(nearby):
                if (explicit_canonical_identity
                        and observations[source].canonical_label != observations[target].canonical_label):
                    continue
                if _adjacent_masks(observations[source], observations[target]):
                    after[source].append(target)
                    before[target].append(source)
    return before, after


def _canonicalize(observations: list[SamObservation], after: Mapping[int, list[int]]) -> None:
    parents = list(range(len(observations)))

    def root(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    for index, targets in after.items():
        for target in targets:
            left, right = root(index), root(target)
            parents[max(left, right)] = min(left, right)
    for index, observation in enumerate(observations):
        observations[index] = replace(observation, canonical_label=root(index) + 1)


def _cone_candidates(source: SamObservation, targets: Sequence[int],
                     observations: Sequence[SamObservation], distance: int,
                     angle: float, spacing: tuple[float, float, float],
                     canvas: tuple[int, int], limits: SamPlanningLimits
                     ) -> tuple[list[int], str | None]:
    if not targets or distance < 2:
        return [], None
    sz, sy, sx = spacing
    slope = math.tan(math.radians(angle))
    growth = max(0.0, slope * distance * sz)
    py, px = int(math.ceil(growth / sy)) + 2, int(math.ceil(growth / sx)) + 2
    y0, x0, y1, x1 = source.bbox_yx
    crop = (max(0, y0 - py), max(0, x0 - px),
            min(canvas[0], y1 + py), min(canvas[1], x1 + px))
    if (crop[2] - crop[0]) * (crop[3] - crop[1]) > limits.max_crop_pixels:
        return [], "projection_search_crop_limit"
    mask = source.mask_in_crop(crop)
    # One-cell padding treats pixels on the canvas edge as bounded foreground.
    padded = np.pad(mask, 1)
    sdf = (ndi.distance_transform_edt(padded, sampling=(sy, sx))
           - ndi.distance_transform_edt(~padded, sampling=(sy, sx)))[1:-1, 1:-1]
    found: list[tuple[int, float, str, int]] = []
    anchor = source.anchor_yx
    for index in targets:
        target = observations[index]
        steps = abs(target.frame_index - source.frame_index)
        if steps < 2 or steps > distance or not _intersects(crop, target.bbox_yx):
            continue
        ty0, tx0, ty1, tx1 = target.bbox_yx
        cy0, cx0 = max(ty0, crop[0]), max(tx0, crop[1])
        cy1, cx1 = min(ty1, crop[2]), min(tx1, crop[3])
        support = target.mask_crop[cy0 - ty0:cy1 - ty0, cx0 - tx0:cx1 - tx0]
        allowed = sdf[cy0 - crop[0]:cy1 - crop[0], cx0 - crop[1]:cx1 - crop[1]] >= -slope * steps * sz
        if not np.any(support & allowed):
            continue
        ys, xs = np.nonzero(support & allowed)
        nearest = float(np.min(((ys + cy0 - anchor[0]) * sy) ** 2
                               + ((xs + cx0 - anchor[1]) * sx) ** 2))
        found.append((steps, nearest, target.observation_id, index))
    found.sort()
    return [item[-1] for item in found], None


def _paint_shifted(destination: np.ndarray, observation: SamObservation,
                   crop: BBoxYX, shift_y: float, shift_x: float) -> None:
    ys, xs = np.nonzero(observation.mask_crop)
    iy = np.rint(ys + observation.bbox_yx[0] + shift_y - crop[0]).astype(np.int64)
    ix = np.rint(xs + observation.bbox_yx[1] + shift_x - crop[1]).astype(np.int64)
    inside = (iy >= 0) & (iy < destination.shape[0]) & (ix >= 0) & (ix < destination.shape[1])
    destination[iy[inside], ix[inside]] = True


def _corridor(a: SamObservation, b: SamObservation, frame: int,
              crop: BBoxYX, margin: int) -> np.ndarray:
    output = np.zeros((crop[2] - crop[0], crop[3] - crop[1]), dtype=bool)
    alpha = (frame - a.frame_index) / (b.frame_index - a.frame_index)
    ay, ax = a.anchor_yx
    by, bx = b.anchor_yx
    _paint_shifted(output, a, crop, alpha * (by - ay), alpha * (bx - ax))
    _paint_shifted(output, b, crop, (1 - alpha) * (ay - by), (1 - alpha) * (ax - bx))
    if margin:
        output = ndi.binary_dilation(output, iterations=margin)
    return output


def _empty_contracts() -> np.ndarray:
    return _readonly(np.empty((0, 0, 0), dtype=bool))


def plan_sam_bridges(
    observed_volume: object, *, interpolation_distance: int = 15,
    interpolation_candidates: int = 1, interpolation_walk_back: int = 1,
    interpolation_passes: int = 1, interpolation_min_radius: float = 3.0,
    interpolation_search_angle: float = 15.0, scope_id: str = "native",
    spacing_zyx: tuple[float, float, float] = (1.0, 1.0, 1.0),
    canonical_labels: object | None = None,
    observation_lineage: Mapping[str, Any] | None = None,
    limits: SamPlanningLimits | None = None,
    planning_pass_index: int = 1,
) -> SamBridgePlan:
    """Plan fixed family crops and independently detector-seeded tracker runs.

    Distance/candidate/angle/walk-back flags retain their source-search meanings.
    Passes bound distinct original-anchor rounds: this exhaustive planner completes
    one round and reports subsequent rounds exhausted. ``min_radius`` is retained
    for measurement of generated SAM bridge sections, never an SDF/seed veto.

    Binary input has connected-component provenance, not invented detector instance
    identity. Integer input or ``canonical_labels`` retains supplied labels. Every
    input is a detector-observation snapshot; callers must never pass mutable SAM
    unions as new observations.
    """
    bounds = limits or SamPlanningLimits()
    shape = tuple(int(value) for value in observed_volume.shape)  # type: ignore[attr-defined]
    if len(shape) != 3 or any(value <= 0 for value in shape):
        raise ValueError("SAM observations require a nonempty frame,y,x canvas")
    if canonical_labels is not None and tuple(canonical_labels.shape) != shape:  # type: ignore[attr-defined]
        raise ValueError("canonical labels and observation canvas shapes differ")
    if (interpolation_distance < 0 or interpolation_candidates < 1
            or interpolation_walk_back < 0 or interpolation_passes < 1
            or not math.isfinite(interpolation_min_radius) or interpolation_min_radius < 0
            or not math.isfinite(interpolation_search_angle)
            or not -90 < interpolation_search_angle < 90):
        raise ValueError("invalid SAM interpolation settings")
    if isinstance(planning_pass_index, bool) or not 1 <= int(planning_pass_index) <= int(interpolation_passes):
        raise ValueError("SAM planning pass must be within requested interpolation passes")
    spacing = tuple(float(value) for value in spacing_zyx)
    if len(spacing) != 3 or any(not math.isfinite(v) or v <= 0 for v in spacing):
        raise ValueError("spacing_zyx must contain three finite positive values")
    if interpolation_distance == 0:
        return SamBridgePlan((), (), (), interpolation_passes, 0, interpolation_passes,
                             "disabled", ("interpolation_distance_zero",))
    if int(planning_pass_index) > 1:
        # This planner exhausts every distinct observed-anchor hypothesis in its
        # first round. Later rounds do not scan images/masks or reseed bridges.
        return SamBridgePlan((), (), (), interpolation_passes, 1, interpolation_passes - 1,
                             "exhausted", ("original_anchor_hypotheses_exhausted",))
    if shape[1] * shape[2] > bounds.max_slice_pixels:
        return SamBridgePlan((), (), (), interpolation_passes, 0, interpolation_passes,
                             "unresolved", ("observation_slice_pixel_limit",))
    observations, by_frame, truncated = _observations(
        observed_volume, scope=scope_id, labels=canonical_labels,
        lineage=observation_lineage, limits=bounds)
    if truncated:
        return SamBridgePlan(tuple(observations), (), (), interpolation_passes, 0,
                             interpolation_passes, "unresolved", (truncated,))
    # Detector binary canvases also use conventional uint8 0/255 encoding.
    # A caller needing a literal canonical label 255 supplies canonical_labels.
    conventional_u8_binary = (np.dtype(observed_volume.dtype) == np.dtype(np.uint8)  # type: ignore[attr-defined]
                              and {item.canonical_label for item in observations} <= {1, 255}
                              and len({item.canonical_label for item in observations}) <= 1)
    explicit_canonical = (canonical_labels is not None
                          or (any(item.canonical_label > 1 for item in observations)
                              and not conventional_u8_binary))
    before, after = _continuations(observations, by_frame, explicit_canonical_identity=explicit_canonical)
    if not explicit_canonical:
        _canonicalize(observations, after)
    fingerprint = _identity("sam_input", scope_id, shape,
                            tuple(item.observation_id for item in observations),
                            repr(dict(observation_lineage or {})))
    planning_fingerprint = _identity("sam_planning", fingerprint, interpolation_distance,
                                     interpolation_candidates, interpolation_walk_back,
                                     interpolation_search_angle, spacing, tuple(vars(bounds).items()))
    # Required edges use the bounded nearest-candidate flag. Family inventory also
    # retains other compatible endpoints, so legitimate sibling support is measured
    # against a complete declared local family rather than an SDF candidate list.
    selected: dict[tuple[int, int], int] = {}
    direction_requests: dict[tuple[int, int], set[int]] = {}
    inventory: dict[int, set[int]] = {}
    endpoint_indexes = {index for index in range(len(observations))
                        if not before[index] or not after[index]}
    search_failures: dict[int, str] = {}
    proposed_count = 0
    for index in sorted(endpoint_indexes):
        source = observations[index]
        for direction, continuations, opposite in ((1, after, before), (-1, before, after)):
            if continuations[index]:
                continue
            possible: list[int] = []
            available = shape[0] - 1 - source.frame_index if direction > 0 else source.frame_index
            search_distance = min(interpolation_distance, available)
            for steps in range(2, search_distance + 1):
                frame = source.frame_index + direction * steps
                possible.extend(target for target in by_frame.get(frame, ()) if not opposite[target])
            found, error = _cone_candidates(source, possible, observations, search_distance,
                                            interpolation_search_angle, spacing, shape[1:], bounds)
            if error:
                search_failures[index] = error
                continue
            if found:
                proposed_count += len(found)
                if proposed_count > bounds.max_proposed_edges:
                    return SamBridgePlan(tuple(observations), (), (), interpolation_passes, 0,
                                         interpolation_passes, "unresolved", ("candidate_graph_edge_limit",),
                                         fingerprint, planning_fingerprint)
                inventory.setdefault(index, set()).update(found)
                direction_requests[(index, direction)] = set(found[:interpolation_candidates])
                for rank, target in enumerate(found[:interpolation_candidates], start=1):
                    key = (index, target) if direction > 0 else (target, index)
                    selected[key] = min(rank, selected.get(key, rank))
    # Connected candidate-family graph defines local hypotheses; it never itself
    # admits a bridge. Independent proposal selection remains policy-owned.
    graph: dict[int, set[int]] = {}
    for source, targets in inventory.items():
        for target in targets:
            graph.setdefault(source, set()).add(target)
            graph.setdefault(target, set()).add(source)
    if sum(len(neighbors) for neighbors in graph.values()) // 2 > bounds.max_proposed_edges:
        return SamBridgePlan(tuple(observations), (), (), interpolation_passes, 0,
                             interpolation_passes, "unresolved", ("candidate_graph_edge_limit",),
                             fingerprint, planning_fingerprint)
    for index in search_failures:
        graph.setdefault(index, set())
    # Union-find inventories connected families without unbounded frontier
    # expansion. Capped roots remain wholly unresolved: no leftover member may
    # become a deceptively complete separate group.
    parent = {index: index for index in graph}

    def graph_root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    for index, neighbors in graph.items():
        for neighbor in neighbors:
            left, right = graph_root(index), graph_root(neighbor)
            parent[max(left, right)] = min(left, right)
    families: dict[int, list[int]] = {}
    family_counts: dict[int, int] = {}
    for index in sorted(graph):
        root = graph_root(index)
        family_counts[root] = family_counts.get(root, 0) + 1
        members = families.setdefault(root, [])
        if len(members) <= bounds.max_endpoints_per_group:
            members.append(index)
    groups: list[SamBridgeGroup] = []
    runs: list[SamBridgeRunPlan] = []
    total_bytes = 0
    for first, members in sorted(families.items()):
        family = set(members)
        endpoints = set(family)
        pairs = [(a, b, rank) for (a, b), rank in selected.items() if a in family and b in family]
        reasons = [search_failures[item] for item in sorted(family) if item in search_failures]
        if family_counts[first] > bounds.max_endpoints_per_group:
            reasons.append("family_endpoint_limit")
        if len(pairs) > bounds.max_edges_per_group:
            reasons.append("family_edge_limit")
        lo = min(observations[item].frame_index for item in family)
        hi = max(observations[item].frame_index for item in family)
        # Walk-back observations are genuine adjacent detector continuations.
        # Extend the fixed interval before any jobs are constructed.
        walk_seeds: dict[tuple[int, int], list[int]] = {}
        for endpoint in sorted(endpoints):
            for direction, adjacency in ((1, before), (-1, after)):
                current = endpoint
                gathered: list[int] = []
                for _ in range(interpolation_walk_back):
                    neighbors = adjacency[current]
                    if not neighbors:
                        break
                    # Ambiguous continuation stays in family inventory; nearest
                    # deterministic branch supplies this additional seed.
                    current = min(neighbors, key=lambda item: (
                        sum((a - b) ** 2 for a, b in zip(observations[current].anchor_yx,
                                                         observations[item].anchor_yx)),
                        observations[item].observation_id))
                    gathered.append(current)
                    family.add(current)
                    lo = min(lo, observations[current].frame_index)
                    hi = max(hi, observations[current].frame_index)
                walk_seeds[(endpoint, direction)] = gathered
        # Include observed continuations throughout the full staggered interval.
        # Canonical identity is a provenance hint rather than a unique endpoint:
        # retain continuously observed same-family siblings near the repair even
        # when they were absent from the missing-connection candidate list.
        family_labels = {observations[item].canonical_label for item in family}
        family_envelope = (
            min(observations[item].bbox_yx[0] for item in family),
            min(observations[item].bbox_yx[1] for item in family),
            max(observations[item].bbox_yx[2] for item in family),
            max(observations[item].bbox_yx[3] for item in family),
        )
        sibling_pad = bounds.curvature_margin_px + bounds.acceptance_margin_px
        for frame in range(lo, hi + 1):
            for item in by_frame.get(frame, ()):
                record = observations[item]
                if (record.canonical_label in family_labels
                        and _intersects(record.bbox_yx, family_envelope, sibling_pad)):
                    family.add(item)
                    if len(family) > bounds.max_observations_per_group:
                        break
            if len(family) > bounds.max_observations_per_group:
                reasons.append("family_observation_limit")
                break
        pending = list(family)
        while pending:
            current = pending.pop()
            for nearby in before[current] + after[current]:
                if nearby not in family and lo <= observations[nearby].frame_index <= hi:
                    family.add(nearby)
                    pending.append(nearby)
            if len(family) > bounds.max_observations_per_group:
                reasons.append("family_observation_limit")
                break
        frames = tuple(range(lo, hi + 1))
        if len(frames) > bounds.max_frames_per_group:
            reasons.append("family_frame_limit")
        y0 = min(observations[item].bbox_yx[0] for item in family)
        x0 = min(observations[item].bbox_yx[1] for item in family)
        y1 = max(observations[item].bbox_yx[2] for item in family)
        x1 = max(observations[item].bbox_yx[3] for item in family)
        margin = bounds.context_margin_px + bounds.curvature_margin_px + bounds.acceptance_margin_px
        crop = (max(0, y0 - margin), max(0, x0 - margin),
                min(shape[1], y1 + margin), min(shape[2], x1 + margin))
        pixels = (crop[2] - crop[0]) * (crop[3] - crop[1])
        if pixels > bounds.max_crop_pixels:
            reasons.append("context_crop_pixel_limit")
        # Charge retained Boolean masks plus one float64 distance/geometry workspace.
        charge = pixels * (2 * len(frames) * (5 + 2 * len(pairs)) + 4 * len(endpoints) + 16)
        if charge > bounds.max_group_bytes:
            reasons.append("group_contract_memory_limit")
        if total_bytes + charge > bounds.max_total_contract_bytes:
            reasons.append("total_contract_memory_limit")
        observation_ids = tuple(observations[item].observation_id for item in sorted(family))
        edge_records = tuple(SamBridgeEdge(_identity("sam_edge", scope_id, observations[a].observation_id,
                                                    observations[b].observation_id),
                                           observations[a].observation_id, observations[b].observation_id, rank)
                             for a, b, rank in sorted(pairs))
        group_id = _identity("sam_group", scope_id, planning_fingerprint, observation_ids,
                             tuple(edge.edge_id for edge in edge_records), crop, frames, spacing)
        if reasons or not pairs:
            empty = _empty_contracts()
            groups.append(SamBridgeGroup(group_id, observation_ids, edge_records, crop, frames,
                                          empty, empty, MappingProxyType({}), MappingProxyType({}),
                                          empty, empty, "unresolved", tuple(sorted(set(reasons or ["no_missing_connection"]))),
                                          interpolation_min_radius, spacing,
                                          tuple(observations[item].observation_id for item in sorted(endpoints))))
            continue
        contract_shape = (len(frames), crop[2] - crop[0], crop[3] - crop[1])
        acceptance = np.zeros(contract_shape, dtype=bool)
        known = np.zeros(contract_shape, dtype=bool)
        original = np.zeros(contract_shape, dtype=bool)
        family_ids = set(family)
        for frame in frames:
            for item in by_frame.get(frame, ()):
                record = observations[item]
                if not _intersects(record.bbox_yx, crop):
                    continue
                # Other components may extend beyond the crop, so paste an intersection.
                oy0, ox0, oy1, ox1 = record.bbox_yx
                cy0, cx0 = max(oy0, crop[0]), max(ox0, crop[1])
                cy1, cx1 = min(oy1, crop[2]), min(ox1, crop[3])
                target = original[frame - lo, cy0 - crop[0]:cy1 - crop[0], cx0 - crop[1]:cx1 - crop[1]]
                target |= record.mask_crop[cy0 - oy0:cy1 - oy0, cx0 - ox0:cx1 - ox0]
                if item in family_ids:
                    known[frame - lo] |= record.mask_in_crop(crop)
        # Acceptance includes all inventoried candidate corridors and known sibling
        # continuations. Write contracts contain only requested missing branches.
        for source in sorted(endpoints):
            for target in graph.get(source, ()):
                if target not in endpoints or source > target:
                    continue
                a, b = sorted((observations[source], observations[target]), key=lambda item: item.frame_index)
                for frame in range(a.frame_index, b.frame_index + 1):
                    acceptance[frame - lo] |= _corridor(a, b, frame, crop,
                                                       bounds.curvature_margin_px + bounds.acceptance_margin_px)
        for offset in range(len(frames)):
            acceptance[offset] |= (ndi.binary_dilation(known[offset], iterations=bounds.acceptance_margin_px)
                                   if bounds.acceptance_margin_px else known[offset])
        writes = np.zeros(contract_shape, dtype=bool)
        per_edge: dict[str, np.ndarray] = {}
        edge_contracts: dict[str, np.ndarray] = {}
        evaluations: dict[str, np.ndarray] = {}
        permitted: dict[str, np.ndarray] = {}
        for (a_index, b_index, _), edge in zip(sorted(pairs), edge_records):
            a, b = observations[a_index], observations[b_index]
            branch = np.zeros(contract_shape, dtype=bool)
            for frame in range(a.frame_index, b.frame_index + 1):
                branch[frame - lo] = _corridor(a, b, frame, crop, bounds.curvature_margin_px)
            edge_contracts[edge.edge_id] = _readonly(branch)
            branch[a.frame_index - lo] = False
            branch[b.frame_index - lo] = False
            branch &= ~original
            writes |= branch
            per_edge[edge.edge_id] = _readonly(branch)
        for endpoint in sorted(endpoints):
            record = observations[endpoint]
            region = ndi.binary_dilation(record.mask_in_crop(crop),
                                          iterations=max(1, bounds.curvature_margin_px + bounds.acceptance_margin_px))
            # Assign nearby sibling support to its own evaluation region. This
            # spatial partition is predeclared, never derived from a predicted mask.
            competitors = [observations[item] for item in endpoints if item != endpoint
                           and observations[item].frame_index == record.frame_index]
            if competitors:
                own_distance = ndi.distance_transform_edt(~record.mask_in_crop(crop), sampling=spacing[1:])
                for sibling in competitors:
                    sibling_distance = ndi.distance_transform_edt(~sibling.mask_in_crop(crop), sampling=spacing[1:])
                    region &= own_distance <= sibling_distance
            evaluations[record.observation_id] = _readonly(region)
            other_branches = np.zeros(region.shape, dtype=bool)
            for source in sorted(endpoints):
                for target in graph.get(source, ()):
                    if target not in endpoints or source > target or endpoint in (source, target):
                        continue
                    a, b = sorted((observations[source], observations[target]), key=lambda item: item.frame_index)
                    if a.frame_index <= record.frame_index <= b.frame_index:
                        other_branches |= _corridor(a, b, record.frame_index, crop,
                                                    bounds.curvature_margin_px + bounds.acceptance_margin_px)
            for item in family:
                sibling = observations[item]
                if item != endpoint and sibling.frame_index == record.frame_index:
                    other_branches |= ndi.binary_dilation(sibling.mask_in_crop(crop),
                                                           iterations=max(1, bounds.acceptance_margin_px))
            other_branches &= ~record.mask_in_crop(crop)
            permitted[record.observation_id] = _readonly(other_branches)
        group = SamBridgeGroup(group_id, observation_ids, edge_records, crop, frames,
                               _readonly(acceptance), _readonly(writes), MappingProxyType(evaluations),
                               MappingProxyType(per_edge), _readonly(known), _readonly(original & ~known),
                               "planned", (), interpolation_min_radius, spacing,
                               tuple(observations[item].observation_id for item in sorted(endpoints)),
                               MappingProxyType(permitted), MappingProxyType(edge_contracts))
        groups.append(group)
        total_bytes += charge
        # One session per independently conditioned endpoint, and separately per
        # requested observed walk-back seed. No shared multi-object competition.
        for endpoint in sorted(endpoints):
            for direction in (1, -1):
                applicable = [(a, b, edge) for (a, b, _), edge in zip(sorted(pairs), edge_records)
                              if (a if direction > 0 else b) == endpoint
                              and (b if direction > 0 else a) in direction_requests.get((endpoint, direction), ())]
                if not applicable:
                    continue
                held_out = tuple(observations[b if direction > 0 else a].observation_id
                                 for a, b, _ in applicable)
                terminal = (max if direction > 0 else min)(
                    observations[b if direction > 0 else a].frame_index for a, b, _ in applicable)
                edge_ids = tuple(edge.edge_id for _, _, edge in applicable)
                for walk_index, seed in enumerate([endpoint] + walk_seeds.get((endpoint, direction), [])):
                    start = observations[seed].frame_index
                    coverage = tuple(range(start, terminal + direction, direction))
                    identifier = _identity("sam_run", group_id, observations[seed].observation_id,
                                           direction, held_out, edge_ids)
                    runs.append(SamBridgeRunPlan(identifier, group_id,
                                                (observations[seed].observation_id,), held_out,
                                                direction, coverage, edge_ids, 1, walk_index))
    status = "unresolved" if groups and all(group.status == "unresolved" for group in groups) else "planned"
    return SamBridgePlan(tuple(observations), tuple(groups), tuple(runs), interpolation_passes, 1,
                         interpolation_passes - 1, status,
                         ("original_anchor_hypotheses_exhausted",) if interpolation_passes > 1 else (),
                         fingerprint, planning_fingerprint)


__all__ = ["SamPlanningLimits", "SamObservation", "SamBridgeEdge", "SamBridgeGroup",
           "SamBridgeRunPlan", "SamBridgePlan", "plan_sam_bridges"]

"""Exact interpolation contract planes without retaining frame/edge stacks."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from ._deps import ndi
from .sam_bridge_planning import _corridor, _intersects, _readonly


def streaming_sam_contract_peak_bytes(crop_pixels, frame_count, edge_count, observation_count):
    """Producer scratch plus per-role metadata; inventory/packing are separate owners.

    96 bytes/pixel covers paint's ~49-byte coordinate peak, EDT float/int
    workspaces, immutable copies, frame caches and writer validation aliases.
    The permitted-support cache holds two integer planes (normally 8 bytes/pixel,
    at most 16); its construction never retains a corridor or sibling stack.
    The 2-KiB/role allowance covers generated names, evidence records and indexes.
    Retained candidate tuples charge 80 bytes per possible observation pair;
    construction pointer growth uses the metadata allowance. Source masks and
    the bounded packing queue remain separate owners.
    """
    pixels, frames, edges, observations = map(int, (crop_pixels, frame_count, edge_count, observation_count))
    if min(pixels, frames, observations) < 1 or edges < 0:
        raise ValueError('Streaming SAM contracts require positive crop/frame/observation counts')
    roles = (4 + 2 * edges) * frames + 3 * observations
    return 96 * pixels + 2048 * roles + 80 * observations * (observations - 1) // 2


def _include_permitted_support(first, second, plane, owner_a, owner_b=-2):
    """Intersect exclusions only where this Boolean contribution can be true.

    Derive bounds from the actual mask, preserving all existing corridor and
    dilation geometry. The axis reductions retain O(height + width) Boolean
    scratch; the owner rectangles are views of the existing two integer planes.
    """
    rows = plane.any(axis=1)
    if not rows.any():
        return
    y0, y1 = int(rows.argmax()), len(rows) - int(rows[::-1].argmax())
    columns = plane[y0:y1].any(axis=0)
    x0, x1 = int(columns.argmax()), len(columns) - int(columns[::-1].argmax())
    support = plane[y0:y1, x0:x1]
    first = first[y0:y1, x0:x1]
    second = second[y0:y1, x0:x1]
    if owner_a == -2:
        first[support] = second[support] = -2
        return
    unseen = support & (first == -1)
    first[unseen], second[unseen] = owner_a, owner_b
    first[support & (first != owner_a) & (first != owner_b)] = -2
    second[support & (second != owner_a) & (second != owner_b)] = -2


class StreamingSamGroupMasks(Mapping):
    """One last-frame cache; every returned plane has an immutable bytes owner."""

    def __init__(self, recipe):
        self.recipe = recipe
        self.group = recipe.descriptor
        if self.group.status != 'planned':
            raise ValueError('Streaming SAM contracts require a planned family recipe')
        self.crop = self.group.context_bbox_yx
        self.shape = (self.crop[2] - self.crop[0], self.crop[3] - self.crop[1])
        self.origin = tuple(self.group.crop_contract['legacy_raster_origin_yx'])
        self.observations = {recipe.observations[index].observation_id: recipe.observations[index]
                             for index in recipe.family}
        self.endpoint_indexes = {recipe.observations[index].observation_id: index
                                 for index in recipe.endpoints}
        self._endpoints_by_frame = {}
        for index in recipe.endpoints:
            self._endpoints_by_frame.setdefault(recipe.observations[index].frame_index, []).append(index)
        self._competition_frame = None
        self._competition_distance = None
        self._endpoint_slots = {index: slot for slot, index in enumerate(sorted(recipe.endpoints))}
        self._permitted_frame = None
        self._permitted_owners = None
        self.edges = {edge.edge_id: (recipe.observations[a], recipe.observations[b])
                      for (a, b, _), edge in zip(sorted(recipe.pairs), self.group.edges)}
        self._candidate_edge_inventory = tuple(self._candidate_edges())
        self._cached_frame = None
        self._cache = {}
        self._empty_plane = _readonly(np.zeros(self.shape, bool))

    def __len__(self):
        return (4 + 2 * len(self.edges)) * len(self.group.frame_indices) + 3 * len(self.group.observation_ids)

    def __iter__(self):
        for identifier in self.group.observation_ids:
            for kind in ('endpoint', 'evaluation', 'permitted'):
                yield f'{kind}:{identifier}'
        for kind in ('acceptance', 'write', 'unrelated', 'known_foreground'):
            for frame in self.group.frame_indices:
                yield f'{kind}:{frame}'
        for kind in ('edge_write', 'edge_contract'):
            for edge in self.group.edges:
                for frame in self.group.frame_indices:
                    yield f'{kind}:{edge.edge_id}:{frame}'

    def __getitem__(self, name):
        kind, separator, value = str(name).partition(':')
        if not separator:
            raise KeyError(name)
        if kind in ('endpoint', 'evaluation', 'permitted'):
            if value not in self.observations:
                raise KeyError(name)
            observation = self.observations[value]
            if kind == 'endpoint':
                return observation.mask_in_crop(self.crop)
            return self._endpoint_role(observation, kind)
        edge = None
        if kind in ('edge_write', 'edge_contract'):
            identifier, separator, value = value.rpartition(':')
            if not separator or identifier not in self.edges:
                raise KeyError(name)
            edge = self.edges[identifier]
        elif kind not in ('acceptance', 'write', 'unrelated', 'known_foreground'):
            raise KeyError(name)
        try:
            frame = int(value)
        except ValueError:
            raise KeyError(name) from None
        if str(frame) != value or not self.group.frame_start <= frame < self.group.frame_stop:
            raise KeyError(name)
        if edge is not None:
            return self._edge_plane(edge, frame, kind == 'edge_write')
        if kind == 'unrelated':
            return _readonly(self._frame_plane(frame, 'original') & ~self._frame_plane(frame, 'known'))
        return self._frame_plane(frame, 'known' if kind == 'known_foreground' else kind)

    def _candidate_edges(self):
        recipe = self.recipe
        for source in sorted(recipe.endpoints):
            for target in recipe.graph.get(source, ()):
                if target in recipe.endpoints and source < target:
                    a, b = sorted((recipe.observations[source], recipe.observations[target]),
                                  key=lambda item: item.frame_index)
                    yield source, target, a, b

    def _corridor(self, a, b, frame, margin):
        return _corridor(a, b, frame, self.crop, margin, raster_origin_yx=self.origin)

    def _corridor_union(self, edges, frame, margin):
        """Dilate shared-margin support once, retaining the single-edge ROI path.

        Every corridor clips its painted seeds to the same crop before dilation.
        Binary dilation distributes over that union, including crop boundaries;
        no per-edge raster stack or changed rounding lattice is needed.
        """
        edges = iter(edges)
        first = next(edges, None)
        if first is None:
            return np.zeros(self.shape, bool)
        second = next(edges, None)
        if second is None:
            return self._corridor(*first, frame, margin)
        plane = np.zeros(self.shape, bool)
        for a, b in (first, second):
            plane |= self._corridor(a, b, frame, 0)
        for a, b in edges:
            plane |= self._corridor(a, b, frame, 0)
        # SciPy iterations=0 means convergence, not an undilated result.
        return ndi.binary_dilation(plane, iterations=margin) if margin else plane

    def _frame_plane(self, frame, role):
        if frame != self._cached_frame:
            self._cache = {}
            self._cached_frame = frame
        if not self._cache:
            original = np.zeros(self.shape, bool)
            known = np.zeros(self.shape, bool)
            for index in self.recipe.by_frame.get(frame, ()):
                record = self.recipe.observations[index]
                if not _intersects(record.bbox_yx, self.crop):
                    continue
                oy0, ox0, oy1, ox1 = record.bbox_yx
                y0, x0 = max(oy0, self.crop[0]), max(ox0, self.crop[1])
                y1, x1 = min(oy1, self.crop[2]), min(ox1, self.crop[3])
                original[y0-self.crop[0]:y1-self.crop[0], x0-self.crop[1]:x1-self.crop[1]] |= (
                    record.mask_crop[y0-oy0:y1-oy0, x0-ox0:x1-ox0])
                if index in self.recipe.family:
                    known |= record.mask_in_crop(self.crop)
            self._cache.update(original=_readonly(original), known=_readonly(known))
        if role not in self._cache:
            bounds = self.recipe.bounds
            if role == 'write':
                plane = self._corridor_union(((a, b) for a, b in self.edges.values()
                    if a.frame_index < frame < b.frame_index), frame, bounds.curvature_margin_px)
                plane &= ~self._cache['original']
            elif role == 'acceptance':
                plane = self._corridor_union(((a, b) for _source, _target, a, b in self._candidate_edge_inventory
                    if a.frame_index <= frame <= b.frame_index), frame,
                    bounds.curvature_margin_px + bounds.acceptance_margin_px)
                known = self._cache['known']
                plane |= (ndi.binary_dilation(known, iterations=bounds.acceptance_margin_px)
                          if bounds.acceptance_margin_px else known)
            else:
                raise KeyError(role)
            self._cache[role] = _readonly(plane)
        return self._cache[role]

    def _edge_plane(self, edge, frame, writable):
        a, b = edge
        valid = a.frame_index < frame < b.frame_index if writable else a.frame_index <= frame <= b.frame_index
        if not valid:
            return self._empty_plane
        plane = self._corridor(a, b, frame, self.recipe.bounds.curvature_margin_px)
        if writable:
            plane &= ~self._frame_plane(frame, 'original')
        return _readonly(plane)

    def _endpoint_role(self, observation, role):
        frame = observation.frame_index
        endpoint = self.endpoint_indexes.get(observation.observation_id)
        if endpoint is None:
            if role == 'evaluation':
                return self._frame_plane(frame, 'acceptance')
            return _readonly(self._frame_plane(frame, 'known') & ~observation.mask_in_crop(self.crop))
        bounds = self.recipe.bounds
        if role == 'evaluation':
            region = ndi.binary_dilation(observation.mask_in_crop(self.crop),
                iterations=max(1, bounds.curvature_margin_px + bounds.acceptance_margin_px))
            if len(self._endpoints_by_frame[frame]) > 1:
                competition = self._endpoint_competition_distance(frame)
                own_distance = ndi.distance_transform_edt(~observation.mask_in_crop(self.crop),
                    sampling=self.group.spacing_zyx[1:])
                region &= own_distance <= competition
            return _readonly(region)
        owners = self._permitted_support_owners(frame)
        if owners is not None:
            first, second = owners
            slot = self._endpoint_slots[endpoint]
            permitted = (first != -1) & (first != slot) & (second != slot)
            permitted &= ~observation.mask_in_crop(self.crop)
            return _readonly(permitted)
        permitted = np.zeros(self.shape, bool)
        for source, target, a, b in self._candidate_edge_inventory:
            if endpoint not in (source, target) and a.frame_index <= frame <= b.frame_index:
                permitted |= self._corridor(a, b, frame,
                    bounds.curvature_margin_px + bounds.acceptance_margin_px)
        for index in self.recipe.family:
            sibling = self.recipe.observations[index]
            if index != endpoint and sibling.frame_index == frame:
                permitted |= ndi.binary_dilation(sibling.mask_in_crop(self.crop),
                    iterations=max(1, bounds.acceptance_margin_px))
        permitted &= ~observation.mask_in_crop(self.crop)
        return _readonly(permitted)

    def _permitted_support_owners(self, frame):
        """One frame's exact intersection of contributor exclusion identities.

        A candidate corridor is excluded only by its two endpoints. A sibling
        dilation is excluded only by that sibling's identity. At each pixel the
        intersection of these exclusion sets therefore contains at most two
        identities. Every other endpoint permits that pixel. This reduces the
        repeated all-siblings/all-candidates unions without retaining their masks.
        -1 denotes uncovered pixels; -2 denotes an empty intersection slot.
        """
        if frame == self._permitted_frame:
            return self._permitted_owners
        # Drop the prior owner before allocating its replacement. Failed builds
        # do not publish a partial cache and can be retried normally.
        self._permitted_frame = None
        self._permitted_owners = None
        endpoints = self._endpoints_by_frame[frame]
        if len(endpoints) == 1:
            self._permitted_frame = frame
            return None
        siblings = tuple(index for index in self.recipe.family
                         if self.recipe.observations[index].frame_index == frame)
        endpoint_set = set(endpoints)
        candidate_count = incident = 0
        for source, target, a, b in self._candidate_edge_inventory:
            if a.frame_index <= frame <= b.frame_index:
                candidate_count += 1
                incident += int(source in endpoint_set) + int(target in endpoint_set)
        repeated = len(endpoints) * (candidate_count + len(siblings) - 1) - incident
        if repeated <= candidate_count + len(siblings):
            self._permitted_frame = frame
            return None
        dtype = np.int32 if len(self._endpoint_slots) <= np.iinfo(np.int32).max else np.int64
        first = np.full(self.shape, -1, dtype=dtype)
        second = np.full(self.shape, -1, dtype=dtype)

        bounds = self.recipe.bounds
        for source, target, a, b in self._candidate_edge_inventory:
            if a.frame_index <= frame <= b.frame_index:
                _include_permitted_support(first, second, self._corridor(a, b, frame,
                    bounds.curvature_margin_px + bounds.acceptance_margin_px),
                    self._endpoint_slots[source], self._endpoint_slots[target])
        for index in siblings:
            sibling = self.recipe.observations[index]
            _include_permitted_support(first, second, ndi.binary_dilation(sibling.mask_in_crop(self.crop),
                iterations=max(1, bounds.acceptance_margin_px)), self._endpoint_slots.get(index, -2))
        first.setflags(write=False)
        second.setflags(write=False)
        self._permitted_owners = (first, second)
        self._permitted_frame = frame
        return self._permitted_owners

    def _endpoint_competition_distance(self, frame):
        if frame != self._competition_frame:
            self._competition_distance = None
            self._competition_frame = None
            union = np.zeros(self.shape, bool)
            clipped_empty = False
            for index in self._endpoints_by_frame[frame]:
                record = self.recipe.observations[index]
                oy0, ox0, oy1, ox1 = record.bbox_yx
                if not (self.crop[0] <= oy0 <= oy1 <= self.crop[2]
                        and self.crop[1] <= ox0 <= ox1 <= self.crop[3]):
                    raise ValueError(f'observation {record.observation_id} does not fit declared crop')
                y0, x0 = max(oy0, self.crop[0]), max(ox0, self.crop[1])
                y1, x1 = min(oy1, self.crop[2]), min(ox1, self.crop[3])
                if y0 >= y1 or x0 >= x1:
                    clipped_empty = True
                    continue
                mask = record.mask_crop[y0-oy0:y1-oy0, x0-ox0:x1-ox0]
                clipped_empty |= not mask.any()
                union[y0-self.crop[0]:y1-self.crop[0], x0-self.crop[1]:x1-self.crop[1]] |= mask
            distance = ndi.distance_transform_edt(~union, sampling=self.group.spacing_zyx[1:])
            if clipped_empty and union.any():
                # SciPy's empty-mask EDT has a finite outside-background contribution.
                np.minimum(distance, ndi.distance_transform_edt(np.ones(self.shape, bool),
                    sampling=self.group.spacing_zyx[1:]), out=distance)
            distance.setflags(write=False)
            self._competition_distance = distance
            self._competition_frame = frame
        return self._competition_distance


def stream_sam_group_contracts(recipe, min_radius):
    """Return exact existing writer metadata and lazily produced role planes.

    The caller keeps its scratch/inventory lease through writer.add_group(),
    whose bounded packing queue drains before returning. Retained caller planes
    are immutable, but arbitrary caller retention is outside that producer peak.
    """
    from .sam_interpolation import _plain
    group = recipe.descriptor
    masks = StreamingSamGroupMasks(recipe)
    endpoints = []
    for identifier in group.observation_ids:
        observation = masks.observations[identifier]
        endpoints.append(dict(observation_id=str(observation.observation_id),
            frame_index=int(observation.frame_index), canonical_label=int(observation.canonical_label),
            original_observation_id=str(observation.original_observation_id or observation.observation_id),
            native_frame_index=int(observation.native_frame_index if observation.native_frame_index is not None
                                   else observation.frame_index), mirror_u=bool(observation.mirror_u),
            lineage=_plain(observation.lineage)))
    metadata = dict(group_id=str(group.group_id), context_bbox_yx=tuple(map(int, group.context_bbox_yx)),
        selection_priority=list(group.crop_contract.get('selection_priority', (0, str(group.group_id)))),
        frame_indices=tuple(map(int, group.frame_indices)), endpoints=endpoints,
        edges=[_plain(edge) for edge in group.edges], status=str(group.status), reasons=_plain(group.reasons),
        complete=True, interpolation_min_radius=float(min_radius), connectivity=6,
        endpoint_identity_basis='slice_connected_components', endpoint_ids=list(group.endpoint_ids))
    if group.crop_contract:
        metadata['crop_contract'] = _plain(group.crop_contract)
    if group.frame_addressing:
        metadata.update(frame_addressing=_plain(group.frame_addressing),
            frame_addresses=_plain(group.frame_addresses), native_shape_tyx=list(group.native_shape_tyx))
    return metadata, masks

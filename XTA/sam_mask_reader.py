"""Bounded transaction-local caches for immutable SAM proposal masks.

The component-filter implementation remains unchanged. First reads retain its
validation, source guards and exact measurements; repeat reads reuse immutable
products. Payload and loaded-source identity are checked before and after each
transaction. No cache survives its owning outer transaction or crosses evidence attempts.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import struct
import threading

import numpy as np

from .sam_evidence import (SamEvidenceBundle, _decode_mask, _decode_raw_crop_boundary_contacts_with_foreground,
    native_output_shape_tyx, _freeze, _plain)
from . import sam_filtering as _filtering

_SOURCE_PATH = Path(__file__).resolve()
IMPLEMENTATION_SHA256 = hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest()
DEFAULT_CACHE_BYTES = 32 * 1024**2


def _validate_tiled_filter(value, tiled=False):
    if isinstance(value,_FilterSnapshot):
        return
    if isinstance(value,Mapping) and value.get("schema")==_filtering.SCHEMA:
        return
    declared=isinstance(value,Mapping) and (value.get("generation_mode")=="tiled" or
        (isinstance(value.get("resolved_policy"),Mapping) and value["resolved_policy"].get("version")==3))
    if (tiled or declared) and (not isinstance(value,Mapping) or "mask_filter" not in value):
        raise ValueError("Tiled SAM quality-v3 support requires its retained component-filter specification")


def _assert_source_unchanged():
    if hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest() != IMPLEMENTATION_SHA256:
        raise RuntimeError("SAM mask-reader implementation changed after loading")
    _filtering.assert_filter_implementation_unchanged()


class _FilterSnapshot(Mapping):
    """Validated deep immutable spec, confined to a reader transaction."""

    __slots__ = ("spec", "owner", "branch_selection")

    def __init__(self, value, owner):
        spec = _filtering._spec(value)
        object.__setattr__(self, "spec", None if spec is None else _freeze(_plain(spec)))
        object.__setattr__(self, "branch_selection", None)
        object.__setattr__(self, "owner", owner)

    def __setattr__(self, name, value):
        raise TypeError("SAM filter snapshots are immutable")

    def __getitem__(self, key):
        if self.spec is None:
            raise KeyError(key)
        return self.spec[key]

    def __iter__(self):
        return iter(self.spec or {})

    def __len__(self):
        return len(self.spec or {})


def _charge(value):
    if isinstance(value, np.ndarray):
        return int(value.nbytes) + 512
    if isinstance(value, bytes):
        return len(value) + 512
    if isinstance(value, tuple):
        return 128 + sum(_charge(item) for item in value)
    # Four times serialized metadata plus entry overhead conservatively includes
    # bounded component records, Python containers and their scalar objects.
    return len(json.dumps(_plain(value), separators=(",", ":"), allow_nan=False).encode("utf-8")) * 4 + 1024


class SamMaskReader:
    """One payload handle and byte-bounded raw/compact product LRUs.

    Arrays are bytes-backed and read-only. Eviction retires cache ownership;
    arrays already returned to callers remain valid. Filter measurements are
    copied on return, so callers cannot mutate another measurement or the cache.
    Mutable receipts are revalidated on every external access. Internal callers
    may bind a validated immutable ``filter_snapshot`` once per transaction.
    Up to half the total allowance preserves packed effective support and
    diagnostics across intrinsic lanes and ordered qualification. Other products
    can borrow unused compact capacity; their churn never evicts filter results.
    """

    def __init__(self, bundle, *, max_cache_bytes=DEFAULT_CACHE_BYTES):
        if not isinstance(bundle, SamEvidenceBundle):
            bundle = SamEvidenceBundle.open(bundle)
        if isinstance(max_cache_bytes, bool) or int(max_cache_bytes) != max_cache_bytes or int(max_cache_bytes) < 0:
            raise ValueError("SAM reader cache budget must be a nonnegative integer")
        self.bundle = bundle
        self.max_cache_bytes = int(max_cache_bytes)
        self._cache = OrderedDict()
        self._compact_cache = OrderedDict()
        self._compact_cache_limit = self.max_cache_bytes//2
        self._lock = threading.RLock()
        self._stream = None
        self._active = False
        self._closed = False
        self._identity = object()
        self._integrity_parent = None
        self._children = set()
        self._stats = dict(max_cache_bytes=self.max_cache_bytes, cache_bytes=0, peak_cache_bytes=0,
            cache_hits=0, cache_misses=0, cache_evictions=0, oversized_products=0,
            mask_decodes=0, filter_computations=0, effective_candidate_computations=0,
            packed_boundary_contact_scans=0,
            compact_filter_expansions=0, compact_filter_parent_hits=0,
            compact_filter_parent_exports=0,
            compact_cache_bytes=0, peak_compact_cache_bytes=0,
            filter_spec_validations=0, integrity_checks=0, transaction_complete=False,
            implementation_sha256=IMPLEMENTATION_SHA256)

    def __getattr__(self, name):
        return getattr(self.bundle, name)

    @property
    def active(self):
        return self._active

    @property
    def stats(self):
        from types import MappingProxyType
        return MappingProxyType(self._stats)

    def _require_active(self):
        if not self._active:
            raise RuntimeError("SAM mask reader requires an active transaction")

    def __enter__(self):
        if self._active or self._closed:
            raise RuntimeError("SAM mask reader transactions cannot be reused or nested")
        parent = self._integrity_parent
        if parent is None:
            _assert_source_unchanged()
            self.bundle.assert_unchanged()
            self._stats["integrity_checks"] += 1
        else:
            with parent._lock:
                parent._require_active()
                if self.bundle is not parent.bundle:
                    raise RuntimeError("SAM reader lane changed its outer evidence transaction")
                parent._children.add(self)
        try:
            self._stream = (self.bundle.directory / "masks.bin").open("rb")
        except BaseException:
            if parent is not None:
                with parent._lock:
                    parent._children.discard(self)
            raise
        self._active = True
        return self

    def close(self):
        with self._lock:
            if self._closed:
                return
            if self._children:
                raise RuntimeError("SAM outer mask transaction still has active reader lanes")
            try:
                if self._active:
                    if self._integrity_parent is None:
                        self.bundle.assert_unchanged()
                        _assert_source_unchanged()
                        self._stats["integrity_checks"] += 1
                    else:
                        self._integrity_parent._require_active()
                    self._stats["transaction_complete"] = True
            finally:
                if self._stream is not None:
                    self._stream.close()
                    self._stream = None
                self._cache.clear()
                self._compact_cache.clear()
                self._stats["cache_bytes"] = 0
                self._stats["compact_cache_bytes"] = 0
                self._active = False
                self._closed = True
                if self._integrity_parent is not None:
                    with self._integrity_parent._lock:
                        self._integrity_parent._children.discard(self)

    def fork(self, *, max_cache_bytes):
        """Borrow verified immutable metadata with a private cursor and cache.

        The caller must charge every lane's cache/workspace and join all lanes
        before outer exit. The outer transaction verifies the complete payload
        before admission and after the joined reads; cached lane hits cannot
        waive that final corruption check.
        """
        self._require_active()
        child = SamMaskReader(self.bundle, max_cache_bytes=max_cache_bytes)
        child._integrity_parent = self
        return child

    def borrowed_filter_snapshot(self, value):
        """Rebind only an immutable snapshot issued by this lane's parent."""
        self._require_active()
        parent = self._integrity_parent
        if (parent is None or not isinstance(value, _FilterSnapshot)
                or value.owner is not parent._identity):
            raise ValueError("SAM filter snapshot must belong to the active outer transaction")
        parent._require_active()
        snapshot = object.__new__(_FilterSnapshot)
        object.__setattr__(snapshot, "owner", self._identity)
        object.__setattr__(snapshot, "spec", value.spec)
        object.__setattr__(snapshot, "branch_selection", value.branch_selection)
        return snapshot

    def __exit__(self, *exc):
        self.close()

    def _product(self, key, create):
        with self._lock:
            self._require_active()
            cache = self._compact_cache if key[0] == 'compact_effective_raw' else self._cache
            if key in cache:
                self._stats["cache_hits"] += 1
                value, charge = cache.pop(key)
                cache[key] = value, charge
                return value
            self._stats["cache_misses"] += 1
            value = create()
            self._remember_product(key, value)
            return value

    def _cached_product(self, key):
        """Borrow an immutable cached value without doing work under this lock."""
        with self._lock:
            self._require_active()
            cache = self._compact_cache if key[0] == 'compact_effective_raw' else self._cache
            if key not in cache:
                return None
            self._stats["cache_hits"] += 1
            value, charge = cache.pop(key)
            cache[key] = value, charge
            return value

    def _remember_product(self, key, value):
        """Share immutable products within this reader's existing byte allowance."""
        with self._lock:
            self._require_active()
            compact = key[0] == 'compact_effective_raw'
            cache = self._compact_cache if compact else self._cache
            if key in cache:
                return True
            charge = _charge(value)
            limit = self._compact_cache_limit if compact else self.max_cache_bytes-self._stats['compact_cache_bytes']
            if charge > limit:
                self._stats["oversized_products"] += 1
                return False
            owned = self._stats['compact_cache_bytes'] if compact else self._stats['cache_bytes']-self._stats['compact_cache_bytes']
            while cache and owned+charge > limit:
                _, (_, retired_charge) = cache.popitem(last=False)
                owned -= retired_charge
                self._stats["cache_bytes"] -= retired_charge
                if compact:
                    self._stats['compact_cache_bytes'] -= retired_charge
                self._stats["cache_evictions"] += 1
            # Ordinary products may borrow currently unused compact capacity.
            # Reclaim that loan before a compact insertion, without allowing
            # ordinary churn to retire expensive filtered products.
            while compact and self._cache and self._stats['cache_bytes']+charge > self.max_cache_bytes:
                _, (_, retired_charge) = self._cache.popitem(last=False)
                self._stats['cache_bytes'] -= retired_charge
                self._stats['cache_evictions'] += 1
            cache[key] = value, charge
            self._stats["cache_bytes"] += charge
            if compact:
                self._stats['compact_cache_bytes'] += charge
                self._stats['peak_compact_cache_bytes'] = max(
                    self._stats['peak_compact_cache_bytes'], self._stats['compact_cache_bytes'])
            self._stats["peak_cache_bytes"] = max(self._stats["peak_cache_bytes"], self._stats["cache_bytes"])
            return True

    def mask(self, key):
        key = str(key)
        def decode():
            self._stats["mask_decodes"] += 1
            return _decode_mask(self._stream, self.bundle.records[key], self.bundle.max_mask_bytes)
        return self._product(("mask", self.evidence_fingerprint, key), decode)

    def raw_mask(self, run_id, frame):
        return self.mask(self.runs[str(run_id)]["raw_mask_keys"][str(int(frame))])

    def raw_crop_boundary_contacts(self, run_id, frame, *, crop_bbox_yx=None, canvas_shape_yx=None):
        """Verify full packed evidence, then read only four raw crop edges."""
        return self.raw_crop_boundary_contacts_with_foreground(run_id, frame,
            crop_bbox_yx=crop_bbox_yx, canvas_shape_yx=canvas_shape_yx)[0]

    def raw_crop_boundary_contacts_with_foreground(self, run_id, frame, *,
            crop_bbox_yx=None, canvas_shape_yx=None):
        """Reuse one validated raw census for contacts and raw-only empty stopping."""
        run_id, frame = str(run_id), int(frame)
        run = self.runs[run_id]
        if run.get('generation_mode') == 'tiled' or run.get('tile_evidence'):
            raise ValueError('Tiled raw contacts require the dense overlapping halo union')
        group = self.groups[str(run['group_id'])]
        shape = native_output_shape_tyx(self)
        box = tuple(group['context_bbox_yx'])
        if crop_bbox_yx is not None and tuple(crop_bbox_yx) != box:
            raise ValueError('Raw SAM contact crop differs from the caller\'s original planned geometry')
        if canvas_shape_yx is not None and tuple(canvas_shape_yx) != shape[1:]:
            raise ValueError('Raw SAM contact canvas differs from the caller\'s original planned geometry')
        key = run['raw_mask_keys'][str(frame)]
        def census():
            self._stats['packed_boundary_contact_scans'] += 1
            contacts, foreground = _decode_raw_crop_boundary_contacts_with_foreground(self._stream, self.records[key],
                self.bundle.max_mask_bytes, box, shape[1:])
            return json.dumps(dict(contacts=contacts, foreground=foreground),
                separators=(',', ':'), allow_nan=False).encode('utf-8')
        encoded = self._product(('raw_crop_boundary_contacts_with_foreground', self.evidence_fingerprint,
            key, box, tuple(shape[1:])), census)
        decoded = json.loads(encoded)
        return decoded['contacts'], decoded['foreground']

    def candidate_mask(self, run_id, frame):
        return self.mask(self.runs[str(run_id)]["candidate_mask_keys"][str(int(frame))])

    def group_mask(self, group_id, name):
        return self.mask(self.groups[str(group_id)]["mask_keys"][str(name)])

    def availability_mask(self, run_id, frame):
        run=self.runs[str(run_id)]
        keys=run.get("availability_mask_keys",{})
        if str(int(frame)) in keys:
            return self.mask(keys[str(int(frame))])
        return self.bundle.availability_mask(run_id,frame)

    def tile_raw_mask(self, run_id, tile_id, frame):
        tile=next((item for item in self.runs[str(run_id)].get("tile_evidence",()) if item["tile_id"]==str(tile_id)),None)
        if tile is None:
            raise ValueError("Unknown independent SAM tile identity")
        return self.mask(tile["raw_mask_keys"][str(int(frame))])

    def halo_union_mask(self, run_id, frame):
        run=self.runs[str(run_id)]
        tiles=run.get("tile_evidence",())
        if not tiles:
            return self.bundle.halo_union_mask(run_id,frame)
        key=("raw_halo_union",self.evidence_fingerprint,str(run_id),int(frame))
        def assemble():
            group=self.groups[run["group_id"]]
            y0,x0,y1,x1=group["context_bbox_yx"]
            union=np.zeros((y1-y0,x1-x0),bool)
            for tile in tiles:
                if str(int(frame)) not in tile["raw_mask_keys"]:
                    continue
                a0,b0,a1,b1=tile["crop_bbox_yx"]
                union[a0-y0:a1-y0,b0-x0:b1-x0] |= self.tile_raw_mask(run_id,tile["tile_id"],frame)
            return np.frombuffer(union.tobytes(),dtype=np.bool_).reshape(union.shape)
        return self._product(key,assemble)

    def measure_effective_halo_union(self, run_id, frame, value=None):
        spec=self._filter_spec(value)
        identity="legacy_unfiltered" if spec is None else str(spec["sha256"])
        key=("effective_halo_union",self.evidence_fingerprint,identity,str(run_id),int(frame))
        def calculate():
            raw=self.halo_union_mask(run_id,frame)
            group_id=self.runs[str(run_id)]["group_id"]
            threshold=0. if spec is None else float(spec["thresholds_by_group"][group_id])
            enabled=False if spec is None else bool(spec["enabled"])
            mask,measurement=_filtering.filter_sam_components(raw,threshold,enabled=enabled)
            return mask,json.dumps(measurement,separators=(",",":"),allow_nan=False).encode("utf-8")
        mask,encoded=self._product(key,calculate)
        return mask,json.loads(encoded)

    def filter_snapshot(self, value):
        with self._lock:
            self._require_active()
            if isinstance(value, Mapping) and isinstance(value.get('mask_filter'), _FilterSnapshot) and 'branch_selection' not in value:
                snapshot = value['mask_filter']
                if snapshot.owner is not self._identity:
                    raise ValueError('SAM filter snapshots belong to one reader transaction')
                from .sam_branch_selection import branch_selection_required
                if branch_selection_required(value) and snapshot.branch_selection is None:
                    raise ValueError('SAM branch-aware support requires its retained branch selection recipe')
                if snapshot.branch_selection is not None and set(value.get('selected_run_ids', ())) - set(snapshot.branch_selection['selected_edge_ids_by_run']):
                    raise ValueError('SAM branch selection differs from its selected contributors')
                return snapshot
            _validate_tiled_filter(value,self.scope.get("sam_crop_mode")=="tiled")
            self._stats["filter_spec_validations"] += 1
            snapshot = _FilterSnapshot(value, self._identity)
            from .sam_branch_selection import validate_branch_selection
            branch = validate_branch_selection(value, self, mask_filter_sha256=None if snapshot.spec is None else snapshot.spec['sha256'])
            object.__setattr__(snapshot, 'branch_selection', branch)
            return snapshot

    def _branch_filter_overlay(self, prefix, recipe, *, max_index_bytes):
        """Validate a new chunk and share this transaction's immutable prefix.

        The index allowance is caller-admitted topology slack, not permission
        from receipt metadata. Insufficient slack uses the former full merge
        and validation path. That path has the same three metadata owners as
        before: retained prefix, mutable merge, frozen snapshot. Neither result
        retains the old prefix's indexes; replacing the caller's accepted
        snapshot retires them, while immutable packed records may be shared.
        """
        with self._lock:
            self._require_active()
            if not isinstance(prefix, _FilterSnapshot) or prefix.owner is not self._identity:
                raise ValueError('SAM filter snapshots belong to one reader transaction')
            if isinstance(max_index_bytes, bool) or not isinstance(max_index_bytes, int) or max_index_bytes < 0:
                raise ValueError('SAM branch index allowance must be a nonnegative integer')
            from .sam_branch_selection import (validate_branch_selection,
                _validated_branch_overlay, merge_connected_edge_selections)
            incoming = validate_branch_selection(dict(branch_selection=recipe), self,
                mask_filter_sha256=None if prefix.spec is None else prefix.spec['sha256'])
            if incoming is None:
                raise ValueError('SAM branch overlay requires a complete branch recipe')
            branch = _validated_branch_overlay(prefix.branch_selection, incoming,
                max_index_bytes=max_index_bytes)
            if branch is None:
                recipes = [value for value in (prefix.branch_selection, incoming) if value is not None]
                merged = merge_connected_edge_selections(recipes)
                # Retire the validated incoming copy before the full freeze;
                # merged now owns its independent plain records.
                del recipes, incoming
                return self.filter_snapshot(dict(mask_filter=prefix, branch_selection=merged))
            snapshot = object.__new__(_FilterSnapshot)
            object.__setattr__(snapshot, 'owner', self._identity)
            object.__setattr__(snapshot, 'spec', prefix.spec)
            object.__setattr__(snapshot, 'branch_selection', branch)
            return snapshot

    def _branch_index_bytes(self, snapshot):
        self._require_active()
        if not isinstance(snapshot, _FilterSnapshot) or snapshot.owner is not self._identity:
            raise ValueError('SAM filter snapshots belong to one reader transaction')
        from .sam_branch_selection import _BranchPrefix
        branch = snapshot.branch_selection
        return branch.index_bytes if isinstance(branch, _BranchPrefix) else 0

    def _branch_selection(self, value, spec):
        if isinstance(value, Mapping) and isinstance(value.get('mask_filter'), _FilterSnapshot):
            snapshot = value['mask_filter']
            if snapshot.owner is not self._identity:
                raise ValueError('SAM filter snapshots belong to one reader transaction')
            if 'branch_selection' not in value:
                from .sam_branch_selection import branch_selection_required
                if branch_selection_required(value) and snapshot.branch_selection is None:
                    raise ValueError('SAM branch-aware support requires its retained branch selection recipe')
                return snapshot.branch_selection
        elif isinstance(value, _FilterSnapshot):
            if value.owner is not self._identity:
                raise ValueError('SAM filter snapshots belong to one reader transaction')
            return value.branch_selection
        from .sam_branch_selection import validate_branch_selection
        return validate_branch_selection(value, self, mask_filter_sha256=None if spec is None else spec['sha256'])

    def _filter_spec(self, value):
        self._require_active()
        if isinstance(value, Mapping) and isinstance(value.get("mask_filter"), _FilterSnapshot):
            value = value["mask_filter"]
        if isinstance(value, _FilterSnapshot):
            if value.owner is not self._identity:
                raise ValueError("SAM filter snapshots belong to one reader transaction")
            return value.spec
        _validate_tiled_filter(value,self.scope.get("sam_crop_mode")=="tiled")
        self._stats["filter_spec_validations"] += 1
        return _filtering._spec(value)

    def _effective_product(self, run_id, frame, spec):
        identity = "legacy_unfiltered" if spec is None else str(spec["sha256"])
        key = ("compact_effective_raw", self.evidence_fingerprint, identity, str(run_id), int(frame))
        def calculate():
            parent = self._integrity_parent
            if parent is not None:
                cached = parent._cached_product(key)
                if cached is not None:
                    self._stats["compact_filter_parent_hits"] += 1
                    return cached
            self._stats["filter_computations"] += 1
            mask, measurements = _filtering.measure_effective_raw_mask(self, run_id, frame, spec)
            encoded = json.dumps(measurements, separators=(",", ":"), allow_nan=False).encode("utf-8")
            compact = (struct.pack('<QQ', *mask.shape)
                + np.packbits(mask.reshape(-1), bitorder="little").tobytes(), encoded)
            if parent is not None:
                # The parent cache was charged before intrinsic lane admission.
                # Lanes compute privately; only immutable insertion holds its
                # lock. Finished lanes can retire without discarding this work.
                if parent._remember_product(key, compact):
                    self._stats["compact_filter_parent_exports"] += 1
            return compact
        packed, encoded = self._product(key, calculate)
        record = self.records[self.runs[str(run_id)]["raw_mask_keys"][str(int(frame))]]
        shape = tuple(record["shape"])
        if len(packed) < 16 or struct.unpack_from('<QQ', packed) != shape:
            raise ValueError('SAM compact filter product differs from its authenticated raw shape')
        pixels = int(np.prod(shape))
        if len(packed) != 16+(pixels+7)//8:
            raise ValueError('SAM compact filter product has malformed packed bounds')
        unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8, offset=16), count=pixels, bitorder="little")
        self._stats["compact_filter_expansions"] += 1
        mask = np.frombuffer(unpacked.tobytes(), dtype=np.bool_).reshape(shape)
        return mask, encoded

    def measure_effective_raw_mask(self, run_id, frame, value=None):
        spec = self._filter_spec(value)
        mask, measurements = self._effective_product(run_id, frame, spec)
        return mask, json.loads(measurements)

    def effective_raw_mask(self, run_id, frame, value=None):
        return self._effective_product(run_id, frame, self._filter_spec(value))[0]

    def effective_candidate_mask(self, run_id, frame, value=None):
        spec = self._filter_spec(value)
        branch = self._branch_selection(value, spec)
        identity = "legacy_unfiltered" if spec is None else str(spec["sha256"])
        from .sam_branch_selection import candidate_branch_identity
        branch_identity = None if branch is None else candidate_branch_identity(branch, run_id, frame)
        key = ("effective_candidate", self.evidence_fingerprint, identity, branch_identity, str(run_id), int(frame))
        def calculate():
            self._stats["effective_candidate_computations"] += 1
            raw = self._effective_product(run_id, frame, spec)[0]
            candidate = (raw if branch is not None and branch.get('write_domain') == 'fixed_context'
                         else self.candidate_mask(run_id, frame) & raw)
            if branch is not None:
                from .sam_branch_selection import apply_edge_selection_to_candidate, decode_owner_support_plane, packet_identity
                def path_reader(edge_id, owner, stored_frame):
                    packed = branch['edges'][edge_id]['owner_support'][owner][str(stored_frame)]
                    return self._product(('branch_plane', self.evidence_fingerprint, edge_id, owner, stored_frame, packet_identity(packed)),
                        lambda: decode_owner_support_plane(packed, max_plane_bytes=self.bundle.max_mask_bytes))
                candidate = apply_edge_selection_to_candidate(self, run_id, frame, candidate, value,
                    recipe=branch, path_reader=path_reader)
            return np.frombuffer(candidate.tobytes(), dtype=np.bool_).reshape(candidate.shape)
        return self._product(key, calculate)


def measure_effective_raw_mask(bundle, run_id, frame, receipt_or_filter=None):
    if isinstance(bundle, SamMaskReader):
        return bundle.measure_effective_raw_mask(run_id, frame, receipt_or_filter)
    _validate_tiled_filter(receipt_or_filter,bundle.runs[str(run_id)].get("generation_mode")=="tiled")
    return _filtering.measure_effective_raw_mask(bundle, run_id, frame, receipt_or_filter)


def effective_raw_mask(bundle, run_id, frame, receipt_or_filter=None):
    if isinstance(bundle, SamMaskReader):
        return bundle.effective_raw_mask(run_id, frame, receipt_or_filter)
    _validate_tiled_filter(receipt_or_filter,bundle.runs[str(run_id)].get("generation_mode")=="tiled")
    return _filtering.effective_raw_mask(bundle, run_id, frame, receipt_or_filter)


def effective_candidate_mask(bundle, run_id, frame, receipt_or_filter=None):
    if isinstance(bundle, SamMaskReader):
        return bundle.effective_candidate_mask(run_id, frame, receipt_or_filter)
    _validate_tiled_filter(receipt_or_filter,bundle.runs[str(run_id)].get("generation_mode")=="tiled")
    from .sam_branch_selection import validate_branch_selection, apply_edge_selection_to_candidate
    spec = _filtering._spec(receipt_or_filter)
    branch = validate_branch_selection(receipt_or_filter, bundle, mask_filter_sha256=None if spec is None else spec['sha256'])
    candidate = (_filtering.effective_raw_mask(bundle, run_id, frame, receipt_or_filter)
        if branch is not None and branch.get('write_domain') == 'fixed_context' else
        bundle.candidate_mask(run_id, frame) & _filtering.effective_raw_mask(bundle, run_id, frame, receipt_or_filter))
    return apply_edge_selection_to_candidate(bundle, run_id, frame, candidate, receipt_or_filter, recipe=branch) if branch is not None else candidate

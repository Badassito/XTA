"""Bounded transaction-local caches for immutable SAM proposal masks.

The component-filter implementation remains unchanged. First reads retain its
validation, source guards and exact measurements; repeat reads reuse immutable
products. Payload and loaded-source identity are checked before and after each
transaction. No cache survives its reader or crosses evidence attempts.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import hashlib
import json
from pathlib import Path
import threading

import numpy as np

from .sam_evidence import SamEvidenceBundle, _decode_mask, _freeze, _plain
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

    __slots__ = ("spec", "owner")

    def __init__(self, value, owner):
        spec = _filtering._spec(value)
        object.__setattr__(self, "spec", None if spec is None else _freeze(_plain(spec)))
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
    """One payload handle and a byte-bounded LRU of raw/effective products.

    Arrays are bytes-backed and read-only. Eviction retires cache ownership;
    arrays already returned to callers remain valid. Filter measurements are
    copied on return, so callers cannot mutate another measurement or the cache.
    Mutable receipts are revalidated on every external access. Internal callers
    may bind a validated immutable ``filter_snapshot`` once per transaction.
    """

    def __init__(self, bundle, *, max_cache_bytes=DEFAULT_CACHE_BYTES):
        if not isinstance(bundle, SamEvidenceBundle):
            bundle = SamEvidenceBundle.open(bundle)
        if isinstance(max_cache_bytes, bool) or int(max_cache_bytes) != max_cache_bytes or int(max_cache_bytes) < 0:
            raise ValueError("SAM reader cache budget must be a nonnegative integer")
        self.bundle = bundle
        self.max_cache_bytes = int(max_cache_bytes)
        self._cache = OrderedDict()
        self._lock = threading.RLock()
        self._stream = None
        self._active = False
        self._closed = False
        self._identity = object()
        self._stats = dict(max_cache_bytes=self.max_cache_bytes, cache_bytes=0, peak_cache_bytes=0,
            cache_hits=0, cache_misses=0, cache_evictions=0, oversized_products=0,
            mask_decodes=0, filter_computations=0, effective_candidate_computations=0,
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
        _assert_source_unchanged()
        self.bundle.assert_unchanged()
        self._stats["integrity_checks"] += 1
        self._stream = (self.bundle.directory / "masks.bin").open("rb")
        self._active = True
        return self

    def close(self):
        with self._lock:
            if self._closed:
                return
            try:
                if self._active:
                    self.bundle.assert_unchanged()
                    _assert_source_unchanged()
                    self._stats["integrity_checks"] += 1
                    self._stats["transaction_complete"] = True
            finally:
                if self._stream is not None:
                    self._stream.close()
                self._cache.clear()
                self._stats["cache_bytes"] = 0
                self._active = False
                self._closed = True

    def __exit__(self, *exc):
        self.close()

    def _product(self, key, create):
        with self._lock:
            self._require_active()
            if key in self._cache:
                self._stats["cache_hits"] += 1
                value, charge = self._cache.pop(key)
                self._cache[key] = value, charge
                return value
            self._stats["cache_misses"] += 1
            value = create()
            charge = _charge(value)
            if charge > self.max_cache_bytes:
                self._stats["oversized_products"] += 1
                return value
            while self._cache and self._stats["cache_bytes"] + charge > self.max_cache_bytes:
                _, (_, retired_charge) = self._cache.popitem(last=False)
                self._stats["cache_bytes"] -= retired_charge
                self._stats["cache_evictions"] += 1
            self._cache[key] = value, charge
            self._stats["cache_bytes"] += charge
            self._stats["peak_cache_bytes"] = max(self._stats["peak_cache_bytes"], self._stats["cache_bytes"])
            return value

    def mask(self, key):
        key = str(key)
        def decode():
            self._stats["mask_decodes"] += 1
            return _decode_mask(self._stream, self.bundle.records[key], self.bundle.max_mask_bytes)
        return self._product(("mask", self.evidence_fingerprint, key), decode)

    def raw_mask(self, run_id, frame):
        return self.mask(self.runs[str(run_id)]["raw_mask_keys"][str(int(frame))])

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
            _validate_tiled_filter(value,self.scope.get("sam_crop_mode")=="tiled")
            self._stats["filter_spec_validations"] += 1
            return _FilterSnapshot(value, self._identity)

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
        key = ("effective_raw", self.evidence_fingerprint, identity, str(run_id), int(frame))
        def calculate():
            self._stats["filter_computations"] += 1
            mask, measurements = _filtering.measure_effective_raw_mask(self, run_id, frame, spec)
            encoded = json.dumps(measurements, separators=(",", ":"), allow_nan=False).encode("utf-8")
            return mask, encoded
        return self._product(key, calculate)

    def measure_effective_raw_mask(self, run_id, frame, value=None):
        spec = self._filter_spec(value)
        mask, measurements = self._effective_product(run_id, frame, spec)
        return mask, json.loads(measurements)

    def effective_raw_mask(self, run_id, frame, value=None):
        return self._effective_product(run_id, frame, self._filter_spec(value))[0]

    def effective_candidate_mask(self, run_id, frame, value=None):
        spec = self._filter_spec(value)
        identity = "legacy_unfiltered" if spec is None else str(spec["sha256"])
        key = ("effective_candidate", self.evidence_fingerprint, identity, str(run_id), int(frame))
        def calculate():
            self._stats["effective_candidate_computations"] += 1
            raw = self._effective_product(run_id, frame, spec)[0]
            candidate = self.candidate_mask(run_id, frame) & raw
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
    return _filtering.effective_candidate_mask(bundle, run_id, frame, receipt_or_filter)

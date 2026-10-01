"""Bounded worker-owned reuse of immutable, prompt-independent SAM features.

Only the pinned feature adapter opts in. Tracker state and normalized images
are never cached. Tensor storage is counted once, including positional tensors
and evicted payloads still retained by active sessions.
"""
from __future__ import annotations

from collections import OrderedDict
from collections.abc import Mapping
import operator
import threading
import weakref

from .lta_sam import PINNED_SAM_PACKAGE_TREE_SHA256


def _clone_tree(value):
    """Rebuild mutable containers while retaining immutable tensor storage."""
    if isinstance(value, Mapping):
        return {key: _clone_tree(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_clone_tree(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_clone_tree(item) for item in value)
    if hasattr(value, 'tensors') and hasattr(value, 'mask'):
        if value.mask is not None:
            raise ValueError('Reusable tracker features must have no tensor mask')
        return type(value)(value.tensors, None)
    return value


def _storages(features):
    """Return unique backing stores, separating positional-storage identity."""
    import torch
    result, positions = {}, set()
    pending = [(features, False)]
    while pending:
        value, position = pending.pop()
        if isinstance(value, torch.Tensor):
            if value.requires_grad or value.grad_fn is not None:
                raise ValueError('Reusable tracker features must not retain an autograd graph')
            storage = value.untyped_storage()
            key = (str(value.device), int(storage._cdata))
            result[key] = storage
            if position:
                positions.add(key)
        elif isinstance(value, Mapping):
            for name, item in value.items():
                pending.append((item, position or name == 'vision_pos_enc'))
        elif isinstance(value, (tuple, list)):
            for item in value:
                pending.append((item, position))
        elif hasattr(value, 'tensors') and hasattr(value, 'mask'):
            if value.mask is not None:
                raise ValueError('Reusable tracker features must have no tensor mask')
            pending.append((value.tensors, position))
    if not result or any(storage.nbytes() <= 0 for storage in result.values()):
        raise ValueError('Reusable tracker features require nonempty tensor storage')
    return result, positions


class LruTrackerFeatureCache:
    """An isolated predictor's exact frame-feature LRU with a live byte cap.

    ``memory_probe`` may report effective free device bytes (driver free plus
    reusable allocator reservation). A low probe causes eviction or admission
    refusal, never a model/precision change. The byte cap accounts for payload
    retained by active sessions after eviction, rather than only dictionary
    membership. Process/allocator overhead is reported separately by runtime.
    """

    def __init__(self, max_bytes, *, memory_probe=None, headroom_bytes=0,
                 model_identity='', max_entries=1024, position_source_identity=''):
        if isinstance(max_bytes, bool) or isinstance(headroom_bytes, bool):
            raise TypeError('Feature cache budgets must be integer byte counts')
        self.max_bytes = int(operator.index(max_bytes))
        self.headroom_bytes = int(operator.index(headroom_bytes))
        if isinstance(max_entries, bool):
            raise TypeError('Feature cache entry limit must be an integer')
        self.max_entries = int(operator.index(max_entries))
        if self.max_bytes < 0 or self.headroom_bytes < 0 or self.max_entries < 1:
            raise ValueError('Feature cache budgets must be nonnegative and entry limit positive')
        if memory_probe is not None and not callable(memory_probe):
            raise TypeError('Feature cache memory_probe must be callable or None')
        self.memory_probe = memory_probe
        self.model_identity = str(model_identity)
        self.position_source_identity = str(position_source_identity)
        self._model = None
        self._closed = False
        self._entries = OrderedDict()
        self._live = {}
        self._positions = set()
        self._canonical_positions = OrderedDict()
        self._validated_positions = set()
        self._lock = threading.RLock()
        self._counts = dict(hits=0, misses=0, admissions=0, evictions=0,
                            rejected_oversize=0, rejected_active_bytes=0,
                            rejected_headroom=0, model_invalidations=0,
                            canonical_position_hits=0, canonical_position_misses=0,
                            position_validations=0, position_validation_bytes=0,
                            position_spec_evictions=0)
        self._peak_live_bytes = 0

    def _bind(self, model):
        if self._closed:
            raise RuntimeError('Tracker feature cache is closed')
        if model is None:
            raise ValueError('Tracker feature cache requires an explicit model owner')
        if self._model is None:
            self._model = model
        elif self._model is not model:
            self.clear()
            self._model = model
            self._counts['model_invalidations'] += 1

    def _live_bytes(self):
        return sum(size for reference, size in self._live.values() if reference() is not None)

    def _resident_keys(self):
        return {key for _features, keys in self._entries.values() for key in keys}

    def _remove_dead(self, key, reference):
        with self._lock:
            if key in self._live and self._live[key][0] is reference:
                self._live.pop(key)
                self._positions.discard(key)

    def _register(self, stores, positions):
        owner_ref = weakref.ref(self)
        for key, storage in stores.items():
            if key not in self._live or self._live[key][0]() is None:
                def release(reference, key=key, owner_ref=owner_ref):
                    owner = owner_ref()
                    if owner is not None:
                        owner._remove_dead(key, reference)
                reference = weakref.ref(storage, release)
                self._live[key] = (reference, int(storage.nbytes()))
        self._positions.update(positions)
        self._peak_live_bytes = max(self._peak_live_bytes, self._live_bytes())

    def _evict(self):
        if self._entries:
            self._entries.popitem(last=False)
            self._counts['evictions'] += 1

    def canonicalize_positions(self, features, *, model, salt=()):
        """Reuse pinned frame-invariant positions after one exact witness.

        The first alternate tensor for each branch/level/geometry/precision is
        compared bit for bit. The pinned positional-encoding source is frame
        independent; later frames reuse that verified invariant without GPU
        hashing or repeated equality synchronizations. Weak references retain
        no extra positional allocation beyond cache/session tensor ownership.
        """
        if self.position_source_identity != PINNED_SAM_PACKAGE_TREE_SHA256:
            return _clone_tree(features)
        import torch
        hash(salt)
        result = _clone_tree(features)
        owner_ref = weakref.ref(self)
        with self._lock:
            self._bind(model)
            for branch_name, branch in result.items():
                if not isinstance(branch, Mapping) or 'vision_pos_enc' not in branch:
                    continue
                positions = list(branch['vision_pos_enc'])
                for level, position in enumerate(positions):
                    if not isinstance(position, torch.Tensor) or position.requires_grad or position.grad_fn is not None:
                        raise ValueError('Canonical tracker positions require graph-free tensors')
                    key = (salt, branch_name, level, tuple(position.shape),
                           tuple(position.stride()), position.dtype, str(position.device))
                    saved = self._canonical_positions.get(key)
                    canonical = None if saved is None else saved()
                    if canonical is None:
                        while len(self._canonical_positions) >= self.max_entries * 6:
                            old_key, _old_reference = self._canonical_positions.popitem(last=False)
                            self._validated_positions.discard(old_key)
                            self._counts['position_spec_evictions'] += 1
                        def release(reference, key=key, owner_ref=owner_ref):
                            owner = owner_ref()
                            if owner is not None:
                                with owner._lock:
                                    if owner._canonical_positions.get(key) is reference:
                                        owner._canonical_positions.pop(key)
                                        owner._validated_positions.discard(key)
                        self._canonical_positions[key] = weakref.ref(position, release)
                        self._counts['canonical_position_misses'] += 1
                        continue
                    if key not in self._validated_positions:
                        same_storage = (canonical.untyped_storage()._cdata == position.untyped_storage()._cdata
                                        and canonical.storage_offset() == position.storage_offset())
                        if not same_storage:
                            # Byte views also distinguish signed zeros; copies
                            # for strided encodings occur only for this witness.
                            if not torch.equal(canonical.contiguous().view(torch.uint8),
                                               position.contiguous().view(torch.uint8)):
                                raise RuntimeError('Pinned tracker positional encodings are not frame invariant')
                            self._counts['position_validation_bytes'] += position.numel() * position.element_size()
                        self._validated_positions.add(key)
                        self._counts['position_validations'] += 1
                    positions[level] = canonical
                    self._canonical_positions.move_to_end(key)
                    self._counts['canonical_position_hits'] += 1
                branch['vision_pos_enc'] = positions
        return result

    def get(self, key, *, model):
        hash(key)
        with self._lock:
            self._bind(model)
            entry = self._entries.get(key)
            if entry is None:
                self._counts['misses'] += 1
                return None
            self._entries.move_to_end(key)
            self._counts['hits'] += 1
            return _clone_tree(entry[0])

    def put(self, key, features, *, model):
        hash(key)
        stores, positions = _storages(features)
        payload_bytes = sum(int(storage.nbytes()) for storage in stores.values())
        with self._lock:
            self._bind(model)
            if payload_bytes > self.max_bytes:
                self._counts['rejected_oversize'] += 1
                return False
            if key in self._entries:
                self._entries.pop(key)
                self._counts['evictions'] += 1
            # Local storage wrappers are weak-reference anchors while deciding
            # admission; they hold no extra tensor allocation or tracker state.
            def footprint():
                new_bytes = sum(int(storage.nbytes()) for identity, storage in stores.items()
                                if identity not in self._live or self._live[identity][0]() is None)
                return self._live_bytes() + new_bytes
            while self._entries and (footprint() > self.max_bytes or len(self._entries) >= self.max_entries):
                self._evict()
            if footprint() > self.max_bytes:
                self._counts['rejected_active_bytes'] += 1
                return False
            if self.memory_probe is not None:
                while self._entries and int(self.memory_probe()) < self.headroom_bytes:
                    self._evict()
                if int(self.memory_probe()) < self.headroom_bytes:
                    self._counts['rejected_headroom'] += 1
                    return False
            self._entries[key] = (_clone_tree(features), frozenset(stores))
            self._register(stores, positions)
            self._counts['admissions'] += 1
            return True

    def clear(self):
        """Retire resident ownership; active session references remain counted."""
        with self._lock:
            self._counts['evictions'] += len(self._entries)
            self._entries.clear()
            self._canonical_positions.clear()
            self._validated_positions.clear()

    def close(self):
        """Release model ownership when the isolated worker shuts down."""
        with self._lock:
            self.clear()
            self._model = None
            self._closed = True

    def snapshot(self):
        with self._lock:
            resident = self._resident_keys()
            live_keys = {key for key, (reference, _size) in self._live.items() if reference() is not None}
            byte_count = lambda keys: sum(self._live[key][1] for key in keys)
            return dict(self._counts, max_bytes=self.max_bytes,
                        headroom_bytes=self.headroom_bytes, max_entries=self.max_entries,
                        entries=len(self._entries), resident_bytes=byte_count(resident & live_keys),
                        live_bytes=byte_count(live_keys),
                        evicted_live_bytes=byte_count(live_keys - resident),
                        positional_bytes=byte_count(live_keys & self._positions),
                        positional_specs=len(self._canonical_positions),
                        peak_live_bytes=self._peak_live_bytes, model_identity=self.model_identity,
                        position_source_identity=self.position_source_identity,
                        closed=self._closed)


__all__ = ['LruTrackerFeatureCache']

"""Stage mask-only reconciliation of a Slicer cutoff review on its own grid."""

from dataclasses import dataclass
import math
from types import SimpleNamespace

import numpy as np

from .reconciliation import EvidenceLayer, evidence_role, reconcile
from .reconciliation_policy import load_reconciliation_policy, resolve_reconciliation, validate_policy


class ReconciliationCancelled(InterruptedError):
    """A preview was cancelled before it changed the review."""


@dataclass(frozen=True)
class CutoffResult:
    cutoff: int
    packed_mask: np.ndarray
    added: int
    removed: int


@dataclass(frozen=True, eq=False)
class StagedReconciliation:
    store: object
    revision: int
    shape: tuple[int, int, int]
    bounds: tuple[tuple[int, int, int], tuple[int, int, int]]
    global_id: str
    policy_name: str
    policy_sha256: str
    cutoffs: tuple[CutoffResult, ...]


def load_preview_policy(path, memory_mib):
    """Load a user-selected trusted local policy with normal TTA discovery."""
    if path is None:
        return {"name": "Union", "mode": "union", "grouping": "views"}, ""

    settings = resolve_reconciliation(SimpleNamespace(
        reconciliation=str(path), reconciliation_memory_mib=memory_mib,
    ))
    policy = load_reconciliation_policy(settings, check_capability=_check_review_capability)
    _review_policy(policy)
    return policy, settings.sha256


def _check_review_capability(policy):
    if not isinstance(policy, dict):
        return  # The normal policy validator reports the malformed value.
    if policy.get("mode") == "confidence":
        raise ValueError("This review does not provide same-grid observed confidence for reconciliation.")
    if policy.get("select_proposals") is not None or policy.get("sam_bridge_policy") is not None:
        raise ValueError("Review reconciliation cannot reselect SAM proposals.")


def _review_policy(policy):
    _check_review_capability(policy)
    policy = validate_policy(policy)
    if policy["mode"] == "union" and policy["decide"] is None:
        return policy, False
    if policy["grouping"] != "views":
        raise ValueError("This review does not provide same-grid section geometry for reconciliation.")
    return policy, True


@dataclass(frozen=True, eq=False)
class _Layer:
    data: np.ndarray
    origin: tuple[int, int, int]
    storage: str
    words: int

    @classmethod
    def copy_of(cls, layer):
        data = layer.data.copy(order="C")
        data.flags.writeable = False
        return cls(data, tuple(map(int, layer.origin)), layer.storage, layer.words)

    def slab(self, index, z0, z1, shape):
        out = np.zeros((z1 - z0, *shape[1:]), dtype=bool)
        origin = (z0, 0, 0)
        lo = tuple(max(a, b) for a, b in zip(origin, self.origin, strict=True))
        hi = tuple(min(a, b + length) for a, b, length in
                   zip((z1, *shape[1:]), self.origin, self.data.shape[:3], strict=True))
        if any(b <= a for a, b in zip(lo, hi, strict=True)):
            return out
        local = tuple(slice(a - b, c - b) for a, b, c in zip(lo, self.origin, hi, strict=True))
        destination = tuple(slice(a - b, c - b) for a, b, c in zip(lo, origin, hi, strict=True))
        data = self.data[local]
        if self.storage == "level":
            out[destination] = data > index
        else:
            word = data[..., index // 64] if self.words == 2 else data
            out[destination] = (word & np.asarray(1 << (index % 64), dtype=word.dtype)) != 0
        return out


def _metadata(plan, layer, index, uses_view_groups):
    info = plan.layers.get(layer.lid, {})
    rows = info.get("rows", {})
    row = rows.get(index)
    if row is None:
        row = next(iter(rows.values()), {})
    kind = row.get("mask_kind")
    if uses_view_groups:
        view = row.get("physical_view_name") or row.get("view_name")
        if not view:
            raise ValueError(f"Review layer {layer.lid} has no view identity for weighted reconciliation.")
        evidence_role(row)
        if index not in rows and any(
            ((other.get("physical_view_name") or other.get("view_name")), other.get("mask_kind")) != (view, kind)
            for other in rows.values()
        ):
            raise ValueError(f"Review layer {layer.lid} lacks consistent evidence metadata for this cutoff.")
    elif kind not in ("yolo", "bridge", "extrapolation", "union"):
        kind = "yolo" if layer.role == "direct" else "union"
    return {"mask_kind": kind, "view_name": row.get("view_name"),
            "physical_view_name": row.get("physical_view_name")}


def prepare_reconciliation(store, plan, policy, *, memory_mib=256, policy_sha256="", progress=None):
    """Read one frozen review revision, compute all cutoffs, and stage packed output."""
    policy, uses_view_groups = _review_policy(policy)
    shape = tuple(plan.shape)
    if tuple(store.shape) != shape or len(plan.cutoffs) != store.n_cutoffs:
        raise ValueError("Reconciliation needs a matching store and cutoff grid.")
    if store._txn is not None or store.diverged is not None:
        raise RuntimeError("Reconciliation needs a settled edit revision.")
    global_layer = store.global_layer()
    if global_layer is None:
        raise ValueError("Reconciliation needs a review Global layer.")
    generation = store.edit_revision
    additive = tuple(layer for layer in store.layers.values() if layer.additive)
    if not additive:
        raise ValueError("Reconciliation needs at least one editable additive layer.")
    if type(memory_mib) is not int or memory_mib < 1:
        raise ValueError("Reconciliation memory budget must be a positive number of MiB.")
    packed_shape = (*shape[:2], (shape[2] + 7) // 8)
    packed_bytes = math.prod(packed_shape) * store.n_cutoffs
    snapshot_bytes = global_layer.data.nbytes + sum(layer.data.nbytes for layer in additive)
    # Stage packed output as each slab arrives. Reserve comparison and
    # packing scratch in addition to the core's own slab workspace.
    comparison_bytes = 3 * min(shape[0], 16) * math.prod(shape[1:])
    retained_bytes = snapshot_bytes + packed_bytes + comparison_bytes
    voting_mib = memory_mib - math.ceil(retained_bytes / 1024**2)
    if voting_mib < 1:
        raise ValueError(f"Review snapshots and masks exceed the {memory_mib} MiB reconciliation budget.")
    snapshots = tuple(_Layer.copy_of(layer) for layer in additive)
    original_global = _Layer.copy_of(global_layer)
    results = []
    starts = np.array(shape, dtype=np.int64)
    stops = np.zeros(3, dtype=np.int64)
    for index, cutoff in enumerate(plan.cutoffs):
        def notify(stage, current, total):
            if progress is not None:
                progress(cutoff, stage, current, total)

        notify("setup", 0, shape[0])
        evidence = []
        for layer, snapshot in zip(additive, snapshots, strict=True):
            metadata = _metadata(plan, layer, index, uses_view_groups)
            evidence.append(EvidenceLayer(layer.lid, shape, metadata,
                            lambda z0, z1, source=snapshot, k=index: source.slab(k, z0, z1, shape)))
        packed = np.zeros(packed_shape, dtype=np.uint8)
        added = removed = 0

        def write(z0, z1, value):
            nonlocal added, removed
            prior = original_global.slab(index, z0, z1, shape)
            mask = value.view(bool)
            packed[z0:z1] = np.packbits(mask, axis=-1, bitorder="little")
            added += int(np.count_nonzero(mask & ~prior))
            removed += int(np.count_nonzero(prior & ~mask))
            if mask.any():
                for axis, reduce_axes in enumerate(((1, 2), (0, 2), (0, 1))):
                    nonempty = np.flatnonzero(mask.any(axis=reduce_axes))
                    offset = z0 if axis == 0 else 0
                    starts[axis] = min(starts[axis], offset + nonempty[0])
                    stops[axis] = max(stops[axis], offset + nonempty[-1] + 1)

        reconcile(evidence, shape_tyx=shape, policy=policy, write_slab=write,
                  memory_mib=voting_mib, progress=notify)
        packed.flags.writeable = False
        results.append(CutoffResult(cutoff, packed, added, removed))
    if np.any(starts >= stops):
        starts[:] = 0
        stops[:] = 1
    # Replacement can promote the new cropped family to a bitset. This
    # includes room to generate the existing Global's undo payload.
    crop_voxels = math.prod(stops - starts)
    new_bytes = crop_voxels * (4 + store.bits_dtype.itemsize * store.bits_words)
    if any(r.added or r.removed for r in results):
        commit_bytes = packed_bytes + new_bytes + 2 * original_global.data.nbytes
        if commit_bytes > memory_mib * 1024**2:
            raise ValueError(f"Review Global would exceed the {memory_mib} MiB reconciliation budget.")
    bounds = tuple(map(int, starts)), tuple(map(int, stops))
    return StagedReconciliation(store, generation, shape, bounds, global_layer.lid,
                                str(policy["name"]), policy_sha256, tuple(results))


def apply_reconciliation(store, staged, family_type):
    """Adopt a still-current stage as exactly one reversible Global operation."""
    if store is not staged.store or store.edit_revision != staged.revision:
        raise RuntimeError("The review changed during reconciliation. Run Reconcile Global again.")
    if store._txn is not None or store.diverged is not None:
        raise RuntimeError("Reconciliation needs a settled edit revision.")

    def apply():
        layer = store.global_layer()
        if layer is None or layer.lid != staged.global_id:
            raise RuntimeError("The review Global changed during reconciliation.")
        builder = store.begin_entry("reconcile_global")
        if any(result.added or result.removed for result in staged.cutoffs):
            (z0, y0, x0), (z1, y1, x1) = staged.bounds
            family = family_type((z1 - z0, y1 - y0, x1 - x0), len(staged.cutoffs))
            for result in staged.cutoffs:
                packed = result.packed_mask[z0:z1, y0:y1, x0 // 8:(x1 + 7) // 8]
                mask = np.unpackbits(packed, axis=-1, bitorder="little").view(bool)
                mask = mask[..., x0 % 8:x0 % 8 + x1 - x0]
                family.add(mask)
            replacement = family.layer(layer.lid, "global", staged.bounds[0], visible=layer.visible)
            builder.replace_layer(layer.lid, replacement.data, replacement.origin)
        builder.set_stale(False)
        return builder.commit(report={
            "kind": "reconcile_global",
            "policy": staged.policy_name,
            "policy_sha256": staged.policy_sha256,
            "changes": {str(result.cutoff): [result.added, result.removed] for result in staged.cutoffs},
        })

    return store._atomic(apply)

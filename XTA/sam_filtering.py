"""Shared, receipt-defined SAM component filtering before spatial clipping.

Raw evidence stays unchanged. A component is one full crop-space, eight-connected
2D component; its maximum Euclidean distance-transform radius determines whether
the entire component survives. Selected additions, quality measurements, replay,
and final-survival checks must all use these accessors with the same receipt.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
from scipy import ndimage

from ._deps import _numba

SCHEMA = "xta.sam_component_filter/1"
_SOURCE_PATH = Path(__file__).resolve()
IMPLEMENTATION_SHA256 = hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest()
_MAX_COMPONENT_RECORDS = 128
_QUALIFIED_LEGACY_IMPLEMENTATION = "bac0f301626e20500c3b8beab261b1f9ea64289de2af2add26b883507495a780"
_QUALIFIED_PREVIOUS_IMPLEMENTATION = "96278bbc549acb81c54f1e6f8124faa1d091ffbca378e98cd73d09040f44368c"
_QUALIFIED_NUMERICAL_SOURCE = "c98e939e8187951f7f77a78c3401800bf2d85405c4acd9cde8c65f9ae8f33fd3"


def assert_filter_implementation_unchanged():
    if hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest() != IMPLEMENTATION_SHA256:
        raise RuntimeError("SAM component filter implementation changed after loading")


def _fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def _readonly(mask):
    return np.frombuffer(np.ascontiguousarray(mask, dtype=np.bool_).tobytes(), dtype=np.bool_).reshape(mask.shape)


def build_mask_filter(bundle, *, enabled=True, min_radius=None):
    """Capture the fixed group thresholds in a portable, auditable filter spec."""
    assert_filter_implementation_unchanged()
    if not isinstance(enabled, bool):
        raise ValueError("SAM component filter enabled must be boolean")
    if min_radius is not None and (isinstance(min_radius, bool) or not np.isfinite(float(min_radius)) or float(min_radius) < 0):
        raise ValueError("SAM component-filter override must be finite and nonnegative")
    thresholds = {str(key): float(group.get("interpolation_min_radius", 0.) if min_radius is None else min_radius)
                  for key, group in bundle.groups.items()}
    if any(not np.isfinite(value) or value < 0 for value in thresholds.values()):
        raise ValueError("SAM component-filter radii must be finite and nonnegative")
    spec = dict(schema=SCHEMA, enabled=enabled, connectivity=8,
        radius_units="view_native_pixels", comparison="maximum_inscribed_radius<=threshold",
        measurement_domain="full_raw_crop_before_acceptance_and_write",
        threshold_source="group.interpolation_min_radius" if min_radius is None else "policy.component_min_radius",
        thresholds_by_group=thresholds, implementation_sha256=IMPLEMENTATION_SHA256)
    spec["sha256"] = _fingerprint(spec)
    return spec


def _compatible_filter_implementation(identifier):
    """Accept the qualified v25 receipts only while their exact numerics remain."""
    if identifier == IMPLEMENTATION_SHA256:
        return True
    if identifier not in (_QUALIFIED_LEGACY_IMPLEMENTATION, _QUALIFIED_PREVIOUS_IMPLEMENTATION):
        return False
    # The bounded radius implementation preserves the original padded EDT and
    # complete-component decision. Bind historical receipts to every numerical
    # helper, including the compiled reduction's Python source.
    try:
        parts = [function.__name__ + "\n" +
                 inspect.getsource(getattr(function, "py_func", function)).replace("\r\n", "\n").rstrip("\n") + "\n"
                 for function in (_readonly, _component_maxima, _component_inscribed_radii, filter_sam_components)]
    except (OSError, TypeError):
        return False
    return hashlib.sha256("".join(parts).encode("utf-8")).hexdigest() == _QUALIFIED_NUMERICAL_SOURCE


def _spec(value):
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("SAM component filter requires a receipt or filter mapping")
    if value.get("schema") == SCHEMA:
        result = value
    elif "mask_filter" in value:
        result = value["mask_filter"]
    else:
        # Old receipts represented unfiltered candidate unions. Never change
        # their meaning by guessing today's policy from their settings/name.
        if str(value.get("schema", "")).startswith("xta.sam_component_filter/"):
            raise ValueError("Unsupported SAM component filter schema")
        resolved = value.get("resolved_policy", {})
        version = resolved.get("version") if isinstance(resolved, Mapping) else None
        name = str(value.get("policy_name", ""))
        named_version = next((v for v in (2, 3, 4, 5) if name.endswith(f"_v{v}")), None)
        modern = (version in (2, 3, 4, 5, "2", "3", "4", "5") or named_version is not None
                  or "component_filter_implementation_sha256" in value
                  or isinstance(resolved, Mapping) and "enforce_interpolation_min_radius" in resolved)
        if modern:
            raise ValueError(f"SAM quality-v{version or named_version or 'current'} selection receipt is missing its component filter spec")
        return None
    if not isinstance(result, Mapping) or result.get("schema") != SCHEMA:
        raise ValueError("Unsupported SAM component filter schema")
    plain = {key: dict(item) if isinstance(item, Mapping) else item for key, item in result.items()}
    if result.get("sha256") != _fingerprint({key: item for key, item in plain.items() if key != "sha256"}):
        raise ValueError("SAM component filter spec changed or its fingerprint is invalid")
    if (result.get("connectivity") != 8 or result.get("measurement_domain") != "full_raw_crop_before_acceptance_and_write"
            or result.get("comparison") != "maximum_inscribed_radius<=threshold"
            or result.get("radius_units") != "view_native_pixels"):
        raise ValueError("Unsupported SAM component filter geometry semantics")
    if not isinstance(result.get("enabled"), bool):
        raise ValueError("SAM component filter enabled must be boolean")
    thresholds = result.get("thresholds_by_group")
    if (not isinstance(thresholds, Mapping) or any(isinstance(v, bool) or not np.isfinite(float(v)) or float(v) < 0 for v in thresholds.values())):
        raise ValueError("SAM component filter thresholds must be finite and nonnegative")
    if result.get("enabled") and not _compatible_filter_implementation(result.get("implementation_sha256")):
        raise ValueError("SAM component filter implementation differs from its selection receipt")
    return result


@_numba.njit(nogil=True, cache=False)
def _component_maxima(labels, distances, count):
    """Linear numeric reduction without sorting the complete label image."""
    maxima = np.zeros(count, np.float64)
    for y in range(labels.shape[0]):
        for x in range(labels.shape[1]):
            label = int(labels[y, x])
            if 0 < label <= count and distances[y, x] > maxima[label - 1]:
                maxima[label - 1] = distances[y, x]
    return maxima


def _component_inscribed_radii(raw, labels, count):
    """Exact padded EDT maxima with bounded work for sparse components.

    Every component's bounding rectangle is surrounded by background. Other
    disconnected components cannot supply a nearer zero than the background
    separating them, so its own cropped EDT has the same foreground distances.
    Overlapping/nested rectangles fall back to one full-plane EDT and a linear
    reduction, keeping the total transformed area bounded by the full plane.
    """
    count = int(count)
    if count <= 0:
        return np.empty(0, np.float64)
    regions = ndimage.find_objects(labels, max_label=count)
    cells = sum((region[0].stop-region[0].start+2) * (region[1].stop-region[1].start+2)
                for region in regions if region is not None)
    full_cells = (raw.shape[0]+2) * (raw.shape[1]+2)
    if cells > full_cells:
        distance = ndimage.distance_transform_edt(np.pad(raw, 1))[1:-1, 1:-1]
        return _component_maxima(labels, distance, count)
    radii = np.zeros(count, np.float64)
    for index, region in enumerate(regions):
        if region is None:
            continue
        height, width = region[0].stop-region[0].start, region[1].stop-region[1].start
        if min(height, width) <= 2:
            radii[index] = 1.
            continue
        component = labels[region] == index + 1
        if component.all():
            radii[index] = min((component.shape[0]+1)//2, (component.shape[1]+1)//2)
        else:
            distance = ndimage.distance_transform_edt(np.pad(component, 1))
            radii[index] = distance.max()
    return radii


def filter_sam_components(raw_mask, min_radius, *, enabled=True):
    """Return immutable effective support and bounded component diagnostics.

    Zero is an exact geometric no-op. Filtering uses the full binary mask,
    including components outside all spatial contracts. It never erodes a
    surviving component or discards a thin spur attached to a substantial body.
    """
    assert_filter_implementation_unchanged()
    raw = np.asarray(raw_mask)
    threshold = float(min_radius)
    if raw.ndim != 2 or not np.isin(raw, (0, 1)).all():
        raise ValueError("SAM component filtering requires a binary two-dimensional raw mask")
    if not np.isfinite(threshold) or threshold < 0 or not isinstance(enabled, bool):
        raise ValueError("SAM component filter threshold must be finite/nonnegative and enabled boolean")
    raw = np.asarray(raw, dtype=np.bool_)
    raw_count = int(np.count_nonzero(raw))
    diagnostic = dict(enabled=enabled, threshold=threshold, connectivity=8,
        raw_foreground=raw_count, effective_foreground=raw_count, removed_foreground=0,
        raw_component_count=None, retained_component_count=None, removed_component_count=0,
        components=[], omitted_component_records=0,
        status="disabled" if not enabled else "disabled_zero_threshold" if threshold == 0 else "measured")
    if not enabled or threshold == 0:
        return _readonly(raw), diagnostic
    labels, count = ndimage.label(raw, structure=np.ones((3, 3), dtype=bool))
    diagnostic["raw_component_count"] = int(count)
    if not count:
        diagnostic["retained_component_count"] = 0
        return _readonly(raw), diagnostic
    radii = _component_inscribed_radii(raw, labels, count)
    sizes = np.bincount(labels.reshape(-1), minlength=count+1)[1:]
    keep = np.concatenate(([False], radii > threshold))
    effective = keep[labels]
    removed_count = int(np.count_nonzero(radii <= threshold))
    diagnostic.update(effective_foreground=int(np.count_nonzero(effective)),
        removed_foreground=raw_count-int(np.count_nonzero(effective)),
        retained_component_count=int(count)-removed_count, removed_component_count=removed_count,
        components=[dict(component_id=index+1, foreground=int(sizes[index]), maximum_inscribed_radius=float(radii[index]),
                         removed=bool(radii[index] <= threshold)) for index in range(min(int(count), _MAX_COMPONENT_RECORDS))],
        omitted_component_records=max(0, int(count)-_MAX_COMPONENT_RECORDS))
    return _readonly(effective), diagnostic


def measure_effective_raw_mask(bundle, run_id, frame, receipt_or_filter=None):
    """Read raw ownership and apply the exact recorded component filter."""
    spec = _spec(receipt_or_filter)
    run = bundle.runs[str(run_id)]
    raw = bundle.raw_mask(run_id, frame)
    if spec is None:
        return filter_sam_components(raw, 0., enabled=False)
    if str(run["group_id"]) not in spec.get("thresholds_by_group", {}):
        raise ValueError("SAM component filter lacks the selected run's group threshold")
    threshold = float(spec["thresholds_by_group"][str(run["group_id"])])
    return filter_sam_components(raw, threshold, enabled=spec["enabled"])


def effective_raw_mask(bundle, run_id, frame, receipt_or_filter=None):
    return measure_effective_raw_mask(bundle, run_id, frame, receipt_or_filter)[0]


def effective_candidate_mask(bundle, run_id, frame, receipt_or_filter=None):
    """Apply full-raw component filtering, then preserve original write ownership."""
    raw = effective_raw_mask(bundle, run_id, frame, receipt_or_filter)
    return _readonly(bundle.candidate_mask(run_id, frame) & raw)

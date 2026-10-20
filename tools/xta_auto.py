"""Evidence-guided native thin-neck cuts; optional companion to xta_native_filter.

Only ambiguous native components enter this module. All returned masks are subsets
of their input masks. Radii are in the XTA native INDEX lattice, not millimetres.
"""
from __future__ import annotations

import math
import numpy as np
from scipy import ndimage as ndi

FACE = ndi.generate_binary_structure(3, 1)


def split_thin_neck(mask, keep_evidence, remove_evidence, *, max_radius=2,
                    max_keep_loss=0.01, min_remove_fraction=0.90):
    """Return (kept_mask_or_None, audit). None means leave this component alone.

    Separating thick cores supply watershed markers, NOT the upsampled preview
    boundary. Reconstruct against the original native foreground. No opening or
    erosion is used as the final surface. Overlapping footprint votes are unknown.
    """
    from skimage.segmentation import watershed

    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 3 or keep_evidence.shape != mask.shape or remove_evidence.shape != mask.shape:
        raise ValueError('Auto component/evidence arrays must have matching 3D shapes.')
    if max_radius < 1 or not 0 <= max_keep_loss < 1 or not 0 < min_remove_fraction <= 1:
        raise ValueError('Invalid auto cut parameters.')
    # An explicit exterior avoids scipy EDT's all-foreground / edge convention.
    native = np.pad(mask, 1)
    keep = np.pad(mask & keep_evidence & ~remove_evidence, 1)
    remove = np.pad(mask & remove_evidence & ~keep_evidence, 1)
    keep_n, remove_n = int(keep.sum()), int(remove.sum())
    audit = {'decision': 'keep_unresolved', 'exclusive_keep_voxels': keep_n,
             'exclusive_remove_voxels': remove_n, 'attempts': []}
    if not keep_n or not remove_n:
        audit['reason'] = 'no_exclusive_keep_or_remove_evidence'
        return None, audit
    distance = ndi.distance_transform_edt(native)
    for radius in range(1, int(max_radius) + 1):
        cores, n = ndi.label(distance > radius, structure=FACE)
        attempt = {'radius_native_voxels': radius, 'core_count': int(n)}
        audit['attempts'].append(attempt)
        flags = np.zeros(n + 1, np.uint8)
        flags[np.unique(cores[keep])] |= 1
        flags[np.unique(cores[remove])] |= 2
        flags[0] = 0
        # No vote-ratio rule that can erase an explicitly retained thick core.
        if np.any(flags == 3):
            attempt['reason'] = 'thick_core_still_has_mixed_evidence'
            continue
        if not np.any(flags == 1) or not np.any(flags == 2):
            attempt['reason'] = 'no_separate_keep_and_remove_cores'
            continue
        types = np.ones(n + 1, np.int32)  # unobserved thick cores are protected
        types[flags == 2] = 2
        types[0] = 0
        markers = types[cores]
        basins = watershed(-distance, markers, connectivity=FACE, mask=native,
                           compactness=0, watershed_line=True)
        proposed = (basins == 1) & native
        # Detach floating pieces not connected to a retained (or unknown) core.
        connected, count = ndi.label(proposed, structure=FACE)
        good = np.zeros(count + 1, bool)
        good[np.unique(connected[markers == 1])] = True
        good[0] = False
        proposed = good[connected]
        keep_loss = int(np.count_nonzero(keep & ~proposed)) / keep_n
        remove_fraction = int(np.count_nonzero(remove & ~proposed)) / remove_n
        separator = native & (basins == 0)
        max_separator_distance = float(distance[separator].max(initial=0))
        attempt.update(keep_evidence_loss_fraction=keep_loss,
                       remove_evidence_removed_fraction=remove_fraction,
                       separator_max_distance_native_voxels=max_separator_distance)
        # All core interiors must survive on their requested sides. A cut confined
        # to a <=radius boundary shell alone is not sufficient evidence of a split.
        if keep_loss > max_keep_loss + 1e-12:
            attempt['reason'] = 'keep_evidence_loss_guard'
            continue
        if remove_fraction + 1e-12 < min_remove_fraction:
            attempt['reason'] = 'remove_evidence_coverage_guard'
            continue
        if max_separator_distance > radius + 1e-12:
            raise RuntimeError('Watershed crossed a protected thick core.')
        if not np.any(proposed) or np.any(proposed & ~native):
            raise RuntimeError('Auto split violated foreground preservation.')
        result = proposed[1:-1, 1:-1, 1:-1].copy()
        audit.update(decision='auto_split', reason='separated_native_cores',
                     radius_native_voxels=radius,
                     removed_native_voxels=int(np.count_nonzero(mask & ~result)),
                     retained_native_voxels=int(result.sum()),
                     keep_evidence_loss_fraction=keep_loss,
                     remove_evidence_removed_fraction=remove_fraction,
                     separator_max_distance_native_voxels=max_separator_distance,
                     protected_unobserved_cores=int(np.count_nonzero(flags[1:] == 0)))
        attempt['reason'] = 'accepted'
        return result, audit
    audit['reason'] = 'no_acceptable_thin_neck_split'
    return None, audit


def mixed_bounds(labels, roots, mixed, depth):
    """One bounded-slab scan; no full-volume root remapping allocation."""
    if hasattr(labels, 'component_bounds'):
        return labels.component_bounds(roots, mixed)
    ids = np.flatnonzero(mixed)
    lut = np.zeros(len(roots), np.int32)
    lut[ids] = np.arange(1, len(ids) + 1)
    provisional = lut[roots]
    bounds = [None] * len(ids)
    for z in range(0, labels.shape[0], depth):
        slab = provisional[labels[z:z+depth]]
        for j, box in enumerate(ndi.find_objects(slab, max_label=len(ids))):
            if box is None:
                continue
            current = [[box[0].start + z, box[0].stop + z],
                       [box[1].start, box[1].stop], [box[2].start, box[2].stop]]
            if bounds[j] is None:
                bounds[j] = current
            else:
                for a in range(3):
                    bounds[j][a][0] = min(bounds[j][a][0], current[a][0])
                    bounds[j][a][1] = max(bounds[j][a][1], current[a][1])
    return [(int(i), tuple(slice(a, b) for a, b in box))
            for i, box in zip(ids, bounds) if box is not None]


def gather_region(support, low, box):
    axes = [tuple(a[s] for a in pair) for pair, s in zip(support.indices, box)]
    out = np.zeros(tuple(s.stop - s.start for s in box), bool)
    for zz in axes[0]:
        for yy in axes[1]:
            for xx in axes[2]:
                out |= low[np.ix_(zz, yy, xx)] != 0
    return out


def project_region(support, mask, box, destination):
    """Positive-footprint projection for a small audit overlay, not final geometry."""
    axes = [tuple(a[s] for a in pair) for pair, s in zip(support.indices, box)]
    for z, frame in enumerate(mask):
        y, x = np.nonzero(frame)
        if not len(y):
            continue
        for zz in set(int(a[z]) for a in axes[0]):
            for ys in axes[1]:
                for xs in axes[2]:
                    np.logical_or.at(destination[zz], (ys[y], xs[x]), True)


def resolve_conflicts(labels, roots, mixed, support, keep, removed, *, depth, args):
    """Zero only accepted deletions in disk-backed labels; other components untouched.

    Oversize boxes are not tiled independently: that would change topology and EDT.
    They remain unmodified and are recorded, allowing an unattended conservative run.
    """
    records = []
    resolved = np.zeros(len(roots), bool)
    preview = np.zeros(support.low, bool)
    boxes = mixed_bounds(labels, roots, mixed, depth)
    for ordinal, (component, box) in enumerate(boxes):
        roi_shape = tuple(s.stop-s.start for s in box)
        padded_voxels = math.prod(n + 2 for n in roi_shape)
        print(f'  auto {ordinal+1}/{len(boxes)}: component {component}, ROI {roi_shape}', flush=True)
        record = {'component_id': component, 'bbox_tyx': [[s.start, s.stop] for s in box],
                  'padded_roi_voxels': padded_voxels}
        if padded_voxels > args.auto_max_roi_voxels:
            record.update(decision='keep_unresolved', reason='roi_voxel_limit')
        else:
            view = labels.region(box) if hasattr(labels, 'region') else labels[box]
            mask = roots[view] == component
            k = gather_region(support, keep, box)
            r = gather_region(support, removed, box)
            result, audit = split_thin_neck(
                mask, k, r, max_radius=args.auto_max_radius,
                max_keep_loss=args.auto_max_keep_loss,
                min_remove_fraction=args.auto_min_remove_fraction)
            record.update(audit)
            if result is not None:
                deleted = mask & ~result
                project_region(support, deleted, box, preview)
                if hasattr(labels, 'clear_region'):
                    labels.clear_region(box, deleted)
                else:
                    view[deleted] = 0
                resolved[component] = True
                del deleted, result
            del view, mask, k, r
        records.append(record)
    return resolved, records, preview

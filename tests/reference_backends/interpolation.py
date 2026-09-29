"""Historical Python interpolation candidate planner, kept only as a test oracle."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np

from XTA.interpolation import SliceComponentTableCache, SliceEndpointSeed, SliceProjectionCandidate


def _find_slice_projection_candidates_python(
    labels_real: object,
    seed: SliceEndpointSeed,
    max_slice_distance: int,
    search_angle_deg: float,
    max_candidates: int,
    wrap_axis: bool = False,
    component_cache: Optional[SliceComponentTableCache] = None,
    slice_luts: Optional['SliceLocalLabelLUTs'] = None,
) -> List[SliceProjectionCandidate]:
    # Local import keeps the package dependency graph acyclic.
    from XTA.topology import SparseSliceLabelStore

    if int(max_slice_distance) <= 0 or int(max_candidates) <= 0:
        return []

    s0, y0, x0 = seed.point
    num_slices = int(labels_real.shape[0])
    if num_slices <= 0:
        return []

    local_cache = (
        component_cache
        if component_cache is not None
        else SliceComponentTableCache(labels_real, slice_luts=slice_luts)
    )
    source_record, source_anchor = local_cache.find_record_for_point(int(s0), int(seed.label), (int(y0), int(x0)))
    if source_record is None or source_anchor is None or int(source_record.area) <= 0:
        return []

    max_steps = min(int(max_slice_distance), max(0, int(num_slices) - 1)) if bool(wrap_axis) else int(max_slice_distance)
    if int(max_steps) <= 0:
        return []

    cropped_sdf = local_cache.get_projection_sdf(
        source_record,
        max_slice_distance=int(max_steps),
        search_angle_deg=float(search_angle_deg),
    )
    sdf = np.asarray(cropped_sdf.sdf, dtype=np.float32)
    crop_y0 = int(cropped_sdf.origin_y)
    crop_x0 = int(cropped_sdf.origin_x)
    crop_y1 = int(crop_y0 + int(sdf.shape[0]))
    crop_x1 = int(crop_x0 + int(sdf.shape[1]))

    slope = math.tan(math.radians(float(search_angle_deg)))
    full_w = int(labels_real.shape[2])
    # target_label -> (step, unrolled d2, candidate); d2 is kept separately because the
    # candidate's target_point is in the target slice's own (actual) coordinates, which
    # differ from unrolled projection coordinates for wrap-crossing steps.
    found: Dict[int, Tuple[int, int, SliceProjectionCandidate]] = {}

    for step in range(1, int(max_steps) + 1):
        s_raw = int(s0 + int(seed.direction_sign) * step)
        mirrored = False
        if bool(wrap_axis):
            s = int(s_raw % int(num_slices))
            # a step across the azimuthal 0°/180° wrap lands in a frame whose
            # u axis is REVERSED relative to the projection cone. max_steps <= num_slices-1
            # caps the walk at a single crossing.
            mirrored = bool(s_raw < 0 or s_raw >= int(num_slices))
        elif s_raw < 0 or s_raw >= num_slices:
            break
        else:
            s = s_raw

        threshold = -float(slope) * float(step)
        projection = sdf >= threshold
        if not np.any(projection):
            if float(search_angle_deg) < 0.0:
                break
            continue

        if isinstance(labels_real, SparseSliceLabelStore):
            # materialize only the SDF window. A wrap-crossing projection requests
            # the corresponding actual-u window and reverses that small crop back into
            # unrolled projection coordinates.
            if mirrored:
                actual_x0 = int(full_w - crop_x1)
                actual_x1 = int(full_w - crop_x0)
                labels_crop = labels_real.read_window(
                    int(s), crop_y0, crop_y1, actual_x0, actual_x1,
                )[:, ::-1]
            else:
                labels_crop = labels_real.read_window(
                    int(s), crop_y0, crop_y1, crop_x0, crop_x1,
                )
        else:
            slice_view = np.asarray(labels_real[int(s)])
            if mirrored:
                slice_view = slice_view[:, ::-1]
            labels_crop = slice_view[crop_y0:crop_y1, crop_x0:crop_x1]
        if slice_luts is not None:
            # local-id raster — canonicalize just the SDF crop (small gather)
            # instead of relying on a full-volume compact relabel pass.
            labels_crop = slice_luts.lut_for(int(s))[labels_crop]
        overlap = projection & (labels_crop > 0) & (labels_crop != int(seed.label))
        if not np.any(overlap):
            continue

        ys_local, xs_local = np.nonzero(overlap)
        lbls = labels_crop[ys_local, xs_local].astype(np.int64, copy=False)
        for target_label in np.unique(lbls):
            target_label_i = int(target_label)
            if target_label_i <= 0 or target_label_i == int(seed.label) or target_label_i in found:
                continue
            use = lbls == target_label_i
            ys_t = ys_local[use]
            xs_t = xs_local[use]
            if ys_t.size == 0:
                continue
            ys_global = ys_t.astype(np.int64, copy=False) + int(crop_y0)
            xs_global = xs_t.astype(np.int64, copy=False) + int(crop_x0)
            d2 = (ys_global - int(source_anchor[0])) ** 2 + (xs_global - int(source_anchor[1])) ** 2
            idx = int(np.argmin(d2))
            x_actual = int(full_w - 1 - int(xs_global[idx])) if mirrored else int(xs_global[idx])
            found[target_label_i] = (int(step), int(d2[idx]), SliceProjectionCandidate(
                source_label=int(seed.label),
                target_label=target_label_i,
                source_point=(int(s0), int(y0), int(x0)),
                target_point=(int(s), int(ys_global[idx]), x_actual),
                slice_distance=int(step),
            ))

        if len(found) >= int(max_candidates):
            break

    ordered = sorted(
        found.items(),
        key=lambda item: (int(item[1][0]), int(item[1][1]), int(item[0])),
    )
    return [candidate for _label, (_step, _d2, candidate) in ordered[: int(max_candidates)]]


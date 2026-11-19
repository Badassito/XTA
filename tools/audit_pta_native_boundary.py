#!/usr/bin/env python3
"""Compare one original native F2 plane to prepared F2, before/after manual edits.

The original predates manual cleanup. Overlap-region metrics reduce unrelated
object removals but do not constitute exact per-object source correspondence.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

import cv2
import numpy as np
from scipy import ndimage as ndi

if __package__:
    from .nrrd_reference_grid import inspect_reference_raster, read_reference_planes
else:
    from nrrd_reference_grid import inspect_reference_raster, read_reference_planes


def plane(path, t):
    """Return reference plane ``t`` as bool, the reference (t,Y,X) shape and the plane's SHA-256.

    A cropped (segment bounding box) layer is expanded onto its reference grid, so the
    plane, shape and hash equal those of the same layer's full raster.
    """
    raster = inspect_reference_raster(path)
    if raster.encoding not in ('gzip', 'gz') or raster.stored_shape is None or len(raster.stored_shape) != 3:
        raise ValueError('Expected gzip uint8 NRRD')
    planes, shape = read_reference_planes(raster, t, t + 1)
    data = memoryview(np.ascontiguousarray(planes[0])).cast('B')
    return planes[0] > 0, shape, hashlib.sha256(data).hexdigest()


def metrics(a, b):
    inter, union = np.count_nonzero(a & b), np.count_nonzero(a | b)
    return {'original_native_pixels': int(a.sum()), 'prepared_pixels': int(b.sum()),
            'added_pixels': int(np.count_nonzero(b & ~a)), 'removed_pixels': int(np.count_nonzero(a & ~b)),
            'iou': float(inter / union) if union else 1,
            'prepared_area_change_percent': float(100 * (b.sum() / a.sum() - 1)) if a.any() else None}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--native', type=Path, required=True)
    p.add_argument('--prepared', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--t', type=int, default=1134)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=True)
    retained = a.output_dir / 'f2_original_native_global.nrrd'
    if retained.absolute() != a.native.absolute() and not retained.exists():
        shutil.copyfile(a.native, retained)
    print('Original compressed native retained; reading single native plane.', flush=True)
    native, ns, nh = plane(retained, a.t)
    prepared, ps, ph = plane(a.prepared, a.t)
    if ns != ps:
        raise ValueError('Shape mismatch')
    print('Planes loaded; measuring overlap regions.', flush=True)
    labels, count = ndi.label(native, structure=np.ones((3, 3), bool))
    touched = np.unique(labels[prepared]); touched = touched[touched != 0]
    matched_original = np.isin(labels, touched)
    # Include nearby new prepared edges while separating distant deleted objects.
    near = ndi.binary_dilation(matched_original, iterations=8)
    restricted_prepared = prepared & near
    overlay = np.zeros((*native.shape, 3), np.uint8)
    overlay[native] = (0, 200, 0)
    overlay[prepared] = (200, 0, 200)
    overlay[native & prepared] = (220, 220, 220)
    cv2.imwrite(str(a.output_dir / 'f2_original_native_vs_prepared_t1134.png'), overlay)
    result = {'source_paths': {'native_original': str(a.native), 'native_retained': str(retained),
                             'prepared': str(a.prepared)},
              'source_shape_tyx': list(ns), 'native_t': a.t,
              'plane_payload_sha256': {'original': nh, 'prepared': ph},
              'whole_plane': metrics(native, prepared),
              'overlapping_original_components_and_prepared_within_8px': metrics(matched_original, restricted_prepared),
              'original_native_component_count': count,
              'original_components_touched_by_prepared_count': len(touched),
              'limitation': 'Original native Global predates manual preview deletions. Overlap-based component association can merge adjacent objects and excludes prepared pixels beyond an 8px dilation. Counts are not exact per-object source correspondence or anatomical truth.'}
    (a.output_dir / 'f2_original_native_boundary_t1134.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2), flush=True)


if __name__ == '__main__':
    main()

#!/usr/bin/env python3
"""Report shared coverage native frame counts without loading a model.

Counts are before in-plane angles, external policy copies and optional tiles.
The example uses one upright Spherical cube, one Radial axis and one auto
Azimuthal axis. Legacy reference counts are metadata, not selectable plans.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def plan(shape, imgsz):
    from XTA.config import AzimuthalViewRequest, RadialViewRequest, SphericalViewRequest
    from XTA.unification.runtime import compile_physical_views

    compiled = compile_physical_views(t_dim=shape[0], height=shape[1], width=shape[2],
        cartesian_views=(), azimuthal_requests=(AzimuthalViewRequest('transverse'),),
        tilted_groups=(), azimuthal_native_raster=imgsz,
        radial_requests=(RadialViewRequest('transverse'),), radial_patch_size=imgsz,
        spherical_requests=(SphericalViewRequest('transverse'),), spherical_patch_size=imgsz)
    families = {}
    for family in ('spherical', 'radial', 'azimuthal'):
        selected = [view for view in compiled.views if view.family == family]
        first = selected[0]
        actual = sum(view.num_slices for view in selected)
        reference = (first.sampling_reference_frames or actual) if family == 'spherical' else sum(
            view.sampling_reference_frames or view.num_slices for view in selected)
        assert 0 < actual <= reference
        families[family] = {
            'native_frames': actual,
            'legacy_reference_frames': reference,
            'reduction_percent_from_legacy_reference': 100 * (1 - actual / reference),
            'trajectories': len(selected),
            'certificates': sorted(set(view.sampling_certificate for view in selected if view.sampling_certificate)),
            'fallback_reasons': sorted(set(view.sampling_reason for view in selected if view.sampling_reason)),
        }
        if family == 'spherical':
            radii = sorted({radius for view in selected for radius in view.spherical_radii})
            bands = []
            for view in selected:
                band = dict(minimum_radius=view.spherical_min_radius, maximum_radius=view.spherical_max_radius,
                            radius_count=view.num_slices, radius_step=view.spherical_step,
                            face_intervals=view.spherical_face_intervals,
                            error_bound_squared=view.sampling_error_bound_sq)
                if band not in bands:
                    bands.append(band)
            families[family].update(radius_count=len(radii), radius_step=max((b-a for a, b in zip(radii, radii[1:])), default=0.),
                                    face_intervals=max(view.spherical_face_intervals for view in selected),
                                    error_bound_squared=max(view.sampling_error_bound_sq for view in selected), radius_bands=bands)
        elif family == 'radial':
            families[family].update(global_radius_count=first.radial_global_count or first.num_slices,
                                    radius_step=first.radial_step,
                                    error_bound_squared=first.sampling_error_bound_sq)
        else:
            families[family]['azimuth_spacing_degrees'] = compiled.azimuthal_azimuth_angles[0]
    return {'shape_t_y_x': list(shape), 'imgsz': imgsz,
            'coverage_meaning': 'positive native intensity interpolation support in the declared domain',
            'counts_before_angles_policies_tiles': True, 'plans': {'coverage': families}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--shape', type=int, nargs=3, metavar=('T', 'Y', 'X'), required=True)
    parser.add_argument('--imgsz', type=int, default=3072)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.shape) < 2 or args.imgsz < 1:
        parser.error('shape axes must be >=2 and imgsz must be positive')
    record = plan(args.shape, args.imgsz)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    main()

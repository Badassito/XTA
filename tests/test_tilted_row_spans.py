"""Row clipping must preserve frozen native-pull addresses and span ownership."""
import math

import numpy as np
import pytest

from XTA import geometry, projection_coverage_cpu as fast
from tests.test_native_pull_equivalence import reference


@pytest.mark.parametrize('base', ['transverse', 'sagittal', 'coronal'])
@pytest.mark.parametrize('direction,angle', [('vertical', 30.), ('horizontal', -30.)])
@pytest.mark.parametrize('shape', [(7, 11, 15), (19, 9, 23)])
@pytest.mark.parametrize('numeric', [False, True])
@pytest.mark.parametrize('family', ['tilted', 'tilted-azimuthal', 'azimuthal'])
@pytest.mark.parametrize('cached', [False, True])
def test_partial_rows_and_empty_bbox_match_frozen_oracle(base, direction, angle, shape, numeric, family, cached):
    work = (13, 17, 19)
    view = next(v for v in geometry._build_tilted_view_infos(*work,
        tilt_views=(base,), tilt_angles=(30.,), tilt_directions=(direction,))
        if v.tilt_angle_deg == angle)
    if family != 'tilted':
        view = geometry._build_azimuthal_view_info(*work, base_view=base,
            azimuth_angle=45., azimuthal_native_raster=7, request_token='row-spans',
            tilted_source=view if family == 'tilted-azimuthal' else None)
    # Distinct physical frames, and a strided source to prevent contiguous-only
    # assumptions from hiding behind the row-oriented destination traversal.
    owner = np.random.default_rng(160381).integers(0, 256,
        (view.num_slices, 14, 14), dtype=np.uint8)
    source = owner[:, ::2, ::2]
    source.flags.writeable = False
    before = source.copy()
    plan = fast.prepare_native_pull_plan(view, source.shape, shape,
        max_plan_bytes=1024**2, cache_plane=cached)
    total = math.prod(shape)
    for bbox in (None, (1, 2, 3, shape[0]-1, shape[1]-1, shape[2]-2),
                 (0, 0, 4, *shape[:2], 4), (2, 3, 1, 2, 5, 8)):
        expected = reference(source, view, shape, 0, total, numeric=numeric, bbox=bbox)
        owner_out = np.full(total*2+4, 247, np.uint8)
        actual = owner_out[2:-2:2]
        # Multiple lengths cross partial rows and complete planes. Output is
        # deliberately strided; sentinel bytes outside each span must survive.
        first = 0
        for length in (1, shape[2]-2, shape[2]+3, shape[1]*shape[2]+7, total):
            stop = min(total, first+length)
            fast.pull_native_flat_into(source, plan, actual[first:stop],
                first_flat=first, scalar_max=numeric, destination_bbox_tyx=bbox)
            first = stop
            if first == total:
                break
        np.testing.assert_array_equal(actual, expected)
        assert np.all(owner_out[1::2] == 247)
        assert np.all(owner_out[:2] == 247) and np.all(owner_out[-2:] == 247)
    np.testing.assert_array_equal(source, before)

"""Native shell crops retain global geometry and at most one gray level of drift."""
from collections import defaultdict

import numpy as np
import pytest

from XTA import cylindrical_geometry, geometry, spherical_geometry
from XTA.config import resolve_tilted_view_groups


SHAPE = (37, 43, 47)
SOURCE = np.random.default_rng(155392).integers(0, 256, SHAPE, np.uint8)
MASK = (SOURCE % 7 < 3).astype(np.uint8)*255


def _views():
    bases = ('transverse', 'sagittal', 'coronal')
    views = geometry.get_view_infos(*SHAPE, cartesian_views=(),
        radial_views=(*bases, *(f'tilted_{base}' for base in bases)),
        radial_min_radius=.7, radial_patch_size=40,
        spherical_views=('transverse', 'tilted_transverse'),
        spherical_min_radius=.7, spherical_patch_size=40,
        tilt_groups=resolve_tilted_view_groups(['transverse,sagittal,coronal:31:both']))
    recipes = defaultdict(list)
    for view in views:
        if view.family == 'radial':
            key = (view.family, view.radial_base_view, view.tilt_direction, view.tilt_angle_deg)
        elif view.family == 'spherical':
            key = (view.family, view.spherical_face, view.spherical_rotation_xyz)
        else:
            continue
        recipes[key].append(view)
    return tuple(view for family in recipes.values()
                 for view in (family[:1]+family[-1:] if len(family) > 1 else family))


VIEWS = _views()


def _renderer(view):
    return cylindrical_geometry if view.family == 'radial' else spherical_geometry


@pytest.mark.parametrize('view', VIEWS, ids=lambda view: view.name)
def test_native_shell_crop_matches_full_geometry_and_gray8(view):
    renderer = _renderer(view)
    height, width = view.src_h, view.src_w
    boxes = ((1, 3, 34, width-1), (31, 1, height, width-3),
             (height-3, width-4, height, width), (0, width-1, height, width))
    for frame in sorted({0, view.num_slices//2, view.num_slices-1}):
        intensity = renderer.render_shell_frame(SOURCE, view, frame)
        categorical = renderer.render_shell_frame(MASK, view, frame, categorical=True)
        np.testing.assert_array_equal(renderer.render_shell_frame(SOURCE, view, frame,
            bbox_yx=(0, 0, height, width)), intensity)
        for y0, x0, y1, x1 in boxes:
            actual = renderer.render_shell_frame(SOURCE, view, frame, bbox_yx=(y0, x0, y1, x1))
            expected = intensity[y0:y1, x0:x1]
            assert actual.dtype == np.uint8 and actual.flags.c_contiguous
            assert actual.shape == expected.shape
            delta = np.abs(actual.astype(np.int16)-expected.astype(np.int16))
            assert int(delta.max(initial=0)) <= 1
            np.testing.assert_array_equal(renderer.render_shell_frame(MASK, view, frame,
                categorical=True, bbox_yx=(y0, x0, y1, x1)), categorical[y0:y1, x0:x1])


@pytest.mark.parametrize('family', ('radial', 'spherical'))
@pytest.mark.parametrize('reverse', (False, True))
def test_native_shell_crop_lends_strided_and_reversed_sources(family, reverse):
    view = next(view for view in VIEWS if view.family == family)
    backing = np.empty((2*SHAPE[0], SHAPE[1], 2*SHAPE[2]), np.uint8)
    backing[::2, :, ::2] = SOURCE
    source = backing[::2, :, ::2]
    if reverse:
        source = source[::-1, ::-1, ::-1]
    assert not source.flags.c_contiguous
    renderer = _renderer(view)
    expected = renderer.render_shell_frame(source, view, 0)[5:35, 7:29]
    actual = renderer.render_shell_frame(source, view, 0, bbox_yx=np.array((5, 7, 35, 29), np.int32))
    assert int(np.abs(actual.astype(np.int16)-expected.astype(np.int16)).max(initial=0)) <= 1


@pytest.mark.parametrize('family', ('radial', 'spherical'))
@pytest.mark.parametrize('box', ((-1, 0, 4, 4), (0, -1, 4, 4), (0, 0, 41, 4),
    (0, 0, 4, 41), (4, 0, 4, 4), (0, 4, 4, 4), (5, 0, 4, 4),
    (0, 0, 4), (0, 0, 4, 4, 4), (False, 0, 4, 4), (np.bool_(False), 0, 4, 4),
    (0., 0, 4, 4), ('0', 0, 4, 4), 4))
def test_native_shell_crop_rejects_unbounded_or_noninteger_boxes_before_sampling(monkeypatch, family, box):
    view = next(view for view in VIEWS if view.family == family)
    renderer = _renderer(view)
    monkeypatch.setattr(renderer, 'shell_coordinates',
        lambda *args, **kwargs: pytest.fail('Invalid crop allocated native coordinates'))
    with pytest.raises(ValueError, match='crop bounds'):
        renderer.render_shell_frame(SOURCE, view, 0, bbox_yx=box)

"""Only captured canonical pixels may supply the corresponding SAM canvas."""
from dataclasses import replace
import json

import numpy as np
import pytest

from XTA import geometry
from XTA.sam_view_geometry import sam_gray_capture_spec, sam_gray_cache_geometry_record
from tests.test_sam_view_orientations import SOURCE_SHAPE, VIEWS


def spec_for(view, tmp_path, *, side=4):
    job = geometry.build_aug_job_for_variant(view, side, tmp_path)
    return sam_gray_capture_spec(view, job, geometry.DEFAULT_CHANNEL_FORMAT,
        source_identity='source:size:mtime', source_shape_tyx=SOURCE_SHAPE,
        source_processing_shape_tyx=SOURCE_SHAPE, source_resize_semantics='native')


@pytest.mark.parametrize('view', [v for v in VIEWS if v.family != 'orthogonal'], ids=lambda v: v.name)
def test_real_curved_and_tilted_recipes_have_canonical_capture(view, tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_DELAY_NATIVE_EXPANSION', '1')
    base, rotated = geometry.expand_views_into_tta_variants((view,), (0., 31.))
    spec = spec_for(base, tmp_path)
    assert spec is not None
    assert spec['native_centers'] == list(range(view.num_slices))
    assert spec['canvas_shape'] == [view.num_slices, 4, 4]
    np.testing.assert_array_equal(spec['output_to_processing_affine'], [[1, 0, 0], [0, 1, 0]])
    assert spec['physical_view_record'] == sam_gray_cache_geometry_record(rotated,
        spec['canvas_shape'], SOURCE_SHAPE, source_processing_shape_tyx=SOURCE_SHAPE)
    assert spec_for(rotated, tmp_path) is None
    assert json.loads(json.dumps(spec, allow_nan=False)) == spec


def test_cache_geometry_distinguishes_patch_origin_and_source_grid(tmp_path):
    view = next(v for v in VIEWS if v.family == 'spherical')
    base = geometry.expand_views_into_tta_variants((view,), (0.,))[0]
    spec = spec_for(base, tmp_path)
    moved = replace(base, spherical_u_origin=base.spherical_u_origin + 1)
    assert spec_for(moved, tmp_path)['physical_view_record'] != spec['physical_view_record']
    resized = sam_gray_cache_geometry_record(base, spec['canvas_shape'], (14, 18, 22),
        source_processing_shape_tyx=SOURCE_SHAPE)
    assert resized != spec['physical_view_record']


def test_cheap_cartesian_and_incompatible_canvas_keep_source_renderer(tmp_path, monkeypatch):
    cartesian = next(v for v in VIEWS if v.family == 'orthogonal')
    base = geometry.expand_views_into_tta_variants((cartesian,), (0.,))[0]
    assert spec_for(base, tmp_path) is None
    shell = next(v for v in VIEWS if v.family == 'spherical')
    base = geometry.expand_views_into_tta_variants((shell,), (0.,))[0]
    monkeypatch.setenv('YOLO_TTA_DELAY_NATIVE_EXPANSION', '0')
    assert spec_for(base, tmp_path, side=4) is None


@pytest.mark.parametrize('disabled', ('0', 'false', 'off'))
def test_capture_backout_keeps_real_canonical_views_on_source_renderer(tmp_path, monkeypatch, disabled):
    view = next(v for v in VIEWS if v.family == 'spherical')
    base = geometry.expand_views_into_tta_variants((view,), (0.,))[0]
    monkeypatch.delenv('YOLO_TTA_SAM_GRAY_CACHE', raising=False)
    assert spec_for(base, tmp_path) is not None
    monkeypatch.setenv('YOLO_TTA_SAM_GRAY_CACHE', disabled)
    assert spec_for(base, tmp_path) is None
    monkeypatch.setenv('YOLO_TTA_SAM_GRAY_CACHE', '1')
    assert spec_for(base, tmp_path) is not None


def test_nonidentity_capture_affine_cannot_misalign_seeds(tmp_path, monkeypatch):
    base = geometry.expand_views_into_tta_variants(
        (next(v for v in VIEWS if v.family == 'spherical'),), (0.,))[0]
    monkeypatch.setattr(geometry, 'output_to_view_processing_affine',
        lambda *_: np.array([[1., 0., .125], [0., 1., 0.]]))
    assert spec_for(base, tmp_path) is None

"""Policy identities preserve the physical angle throughout actual tile planning."""
from __future__ import annotations

import numpy as np
import pytest

from XTA.geometry import (
    TileConfig, _build_cartesian_view, build_aug_job_for_variant,
    build_dense_tile_jobs_for_aug, build_dense_tile_raster_plan,
    expand_views_into_policy_variants, expand_views_into_tta_variants,
)


@pytest.mark.parametrize('angle', [0., 30., -15.5])
def test_policy_tile_plan_uses_explicit_angle_and_preserves_output_identity(angle, tmp_path):
    physical = _build_cartesian_view('transverse', 4, 64, 64)
    variants = expand_views_into_policy_variants(
        expand_views_into_tta_variants([physical], [angle]), 3)
    reference_tiles = None
    plan_ids = set()
    for view in variants:
        job = build_aug_job_for_variant(view, 128, tmp_path)
        tiles = build_dense_tile_jobs_for_aug(view, job, TileConfig(32, 32, 'cfg1'), 128, tmp_path)
        assert tiles
        if reference_tiles is None:
            reference_tiles = tiles
        assert len(tiles) == len(reference_tiles)
        for reference, tile in zip(reference_tiles, tiles):
            plan = build_dense_tile_raster_plan(view, tile)
            assert plan.in_plane_variant.angle_deg == pytest.approx(angle % 360.)
            assert plan.metadata['runtime_view_id'] == view.name
            assert plan.metadata['runtime_job_id'] == tile.tile_id
            assert plan.metadata['tile_config_id'] == 'cfg1'
            np.testing.assert_array_equal(tile.M_src_to_out, reference.M_src_to_out)
            np.testing.assert_array_equal(tile.M_out_to_src, reference.M_out_to_src)
            plan_ids.add(plan.digest)
    assert len(plan_ids) == len(reference_tiles) * len(variants)

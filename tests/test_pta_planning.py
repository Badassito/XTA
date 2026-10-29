"""Planning ownership and output contracts across shared and native geometry."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from XTA import pta_planning as planning
from XTA import pta_rendering as rendering
from XTA.pta_config import parse_pta_args
from XTA.pta_cpr import build_cpr_geometry
from XTA.pta_workers import WarningLog
from XTA.unification.runtime import compile_physical_views


def _view(**changes):
    return replace(rendering.ViewInfo(
        name="transverse", display_name="transverse", family="transverse",
        num_slices=6, src_h=5, src_w=7, pad_mode="clamp",
        full_t=6, full_h=5, full_w=7,
    ), **changes)



def test_native_plan_names_tiles_metadata_and_publication_flags(tmp_path):
    variant = rendering.ChannelVariant("C3S1", "custom", 3, 1, True, (1, 0, -1))
    plan = planning.build_plans_for_views(
        views=[_view()], out_dir=tmp_path, stem="sample",
        tile_configs=[rendering.TileConfig(4, 3, "small")], save_overlay=True,
        imgsz=0, label_enabled=True, image_format="nvjpeg", channel_variants=[variant],
        center_eligible_frame_indices=(4, 1, 4),
        transverse_source_encoded_indices=(1, 2, 3, 4, 5, 6),
        publish_images=False, publish_labels=False,
    )[0]
    assert plan.tag == "transverse_C3S1_reverse"
    assert plan.view.__class__ is rendering.ViewInfo
    assert plan.source_encoded_indices == (1, 2, 3, 4, 5, 6)
    assert plan.eligible_frame_indices == (1, 4)
    assert tuple(planning.render_plan_frame_indices(plan)) == (1, 4)
    assert plan.img_pattern.endswith("sample_transverse_C3S1_reverse_%04d.jpg")
    assert plan.lbl_pattern.endswith("sample_transverse_C3S1_reverse_%04d.txt")
    assert plan.overlay_path.name == "sample_transverse_C3S1_reverse_Overlay.mkv"
    assert [(tile.x, tile.y) for tile in plan.tile_layout] == [(0, 0), (3, 0), (0, 1), (3, 1)]
    assert all((tile.out_w, tile.out_h) == (4, 4) for tile in plan.tile_layout)
    assert all(not tile.publish_images and not tile.publish_labels for tile in plan.tile_layout)
    assert plan.canonical_plan is None
    assert plan.shared_aug_job is None
    assert not (tmp_path / "images").exists()
    assert not (tmp_path / "labels").exists()
    assert (tmp_path / "overlays").is_dir()
    assert {key: plan.stats[key] for key in (
        "frames", "view_frames", "skipped_center_frames", "image_format",
        "channel_offsets", "full_output_size", "annotated_center_frames",
        "skipped_unannotated_centers", "skipped_discontinuous_centers", "eligible_center_frames",
    )} == dict(
        frames=2, view_frames=6, skipped_center_frames=4, image_format="jpg",
        channel_offsets=[1, 0, -1], full_output_size=[7, 5], annotated_center_frames=2,
        skipped_unannotated_centers=4, skipped_discontinuous_centers=0, eligible_center_frames=2,
    )


@pytest.mark.parametrize("imgsz", [0, 8])
def test_canonical_view_affine_tiles_and_sampling_identity(tmp_path, imgsz):
    config = parse_pta_args([
        "--input", "dataset", "--enable_cartesian", "transverse,sagittal",
        "--enable_azimuthal", "transverse:60", "--enable_tilted", "coronal:15:horizontal",
    ])
    views, compiled = planning.compile_v18_pta_views(
        t_dim=6, h=5, w=7, config=config, azimuthal_native_raster=8,
    )
    expected = compile_physical_views(
        t_dim=6, height=5, width=7, cartesian_views=config.cartesian_views,
        azimuthal_requests=config.azimuthal_requests, tilted_groups=config.tilted_groups,
        azimuthal_native_raster=8, augmentation_angles=(0.0,),
    )
    assert [view.name for view in views] == [view.name for view in expected.views]
    plans = planning.build_plans_for_views(
        views=views, out_dir=tmp_path, stem="shared", tile_configs=[rendering.TileConfig(4, 3, "s")],
        save_overlay=False, imgsz=imgsz, label_enabled=True,
    )
    for plan, shared in zip(plans, compiled.views):
        assert plan.view.shared_view is shared
        assert plan.shared_aug_job.aff is plan.aff.shared_affine
        assert plan.canonical_plan.digest == plan.stats["canonical_plan_digest"]
        assert plan.canonical_plan.output_shape == (plan.aff.out_h, plan.aff.out_w)
        assert [tile.canonical_plan.digest for tile in plan.tile_layout] == plan.stats["canonical_tile_plan_digests"]
        assert all(tile.shared_job is not None for tile in plan.tile_layout)
        assert plan.aff.native_output is (imgsz == 0)
        if imgsz == 0:
            assert (plan.aff.out_w, plan.aff.out_h) == (shared.src_w, shared.src_h)
            np.testing.assert_array_equal(plan.aff.M_src_to_out, np.eye(3, dtype=np.float32)[:2])
        else:
            assert (plan.aff.out_w, plan.aff.out_h) == (8, 8)
            np.testing.assert_array_equal(plan.aff.M_src_to_out, plan.aff.shared_affine.M_src_to_out)


def test_partial_centers_discontinuities_and_frame_major_order(tmp_path, capsys):
    variants = [rendering.DEFAULT_CHANNEL_VARIANT, *[
        rendering.ChannelVariant("C3S1", "custom", 3, 1, reverse, offsets)
        for reverse, offsets in ((False, (-1, 0, 1)), (True, (1, 0, -1)))
    ]]
    warnings = WarningLog()
    plans = planning.build_plans_for_views(
        views=[_view()], out_dir=tmp_path, stem="partial", tile_configs=[],
        save_overlay=False, imgsz=0, label_enabled=True, channel_variants=variants,
        center_eligible_frame_indices=(1, 4), transverse_source_encoded_indices=(1, 2, 4, 6, 7, 8),
        warnings=warnings,
    )
    assert [plan.eligible_frame_indices for plan in plans] == [(1, 4), (4,), (4,)]
    assert [plan.source_encoded_indices for plan in plans] == [(), (1, 2, 4, 6, 7, 8), (1, 2, 4, 6, 7, 8)]
    assert list(planning.iter_render_source_frame_jobs_round_robin(plans)) == [(0, 1), (0, 4), (1, 4), (2, 4)]
    assert list(planning.iter_render_source_frame_jobs_round_robin([])) == []
    assert warnings.counts["channel_stack_discontinuity_centers_skipped"] == 1
    assert "center=2, missing=[3]" in capsys.readouterr().err
    assert [plan.stats["skipped_discontinuous_centers"] for plan in plans] == [0, 1, 1]


@pytest.mark.parametrize("centers,encoded,message", [
    ((6,), (), "annotated center index"),
    (None, (1, 2), "encoded-index mapping"),
    (None, (1, 2, 3, 3, 4, 5), "strictly increasing"),
])
def test_invalid_center_and_encoded_lattices_retain_errors(tmp_path, centers, encoded, message):
    with pytest.raises(ValueError, match=message):
        planning.build_plans_for_views(
            views=[_view()], out_dir=tmp_path, stem="invalid", tile_configs=[],
            save_overlay=False, imgsz=0, label_enabled=True,
            center_eligible_frame_indices=centers, transverse_source_encoded_indices=encoded,
        )


def test_cpr_geometry_passes_through_without_canonical_builtin_plan(tmp_path):
    geometry = build_cpr_geometry(np.array([[1, 2, 3], [4, 2, 3]]), (6, 5, 7), raster_size=9)
    view = _view(name="cpr_000", display_name="cpr_000", family="cpr",
                 num_slices=geometry.num_slices, src_h=geometry.width,
                 src_w=geometry.width, cpr_geometry=geometry)
    plan = planning.build_plans_for_views(
        views=[view], out_dir=tmp_path, stem="curved", tile_configs=[rendering.TileConfig(4, 4, "s")],
        save_overlay=False, imgsz=0, label_enabled=True, center_eligible_frame_indices=(0,),
    )[0]
    assert plan.view is view
    assert plan.view.cpr_geometry is geometry
    assert plan.canonical_plan is None
    assert all(tile.canonical_plan is None and tile.shared_job is None for tile in plan.tile_layout)
    assert tuple(planning.render_plan_frame_indices(plan)) == tuple(range(geometry.num_slices))

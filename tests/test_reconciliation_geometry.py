from __future__ import annotations

import math

import numpy as np
import pytest

from XTA.reconciliation_geometry import section_codes, section_descriptor


def context(shape=(5, 5, 5), *, source=None, angles=(0, 45, 90, 135)):
    views = {}
    axes = {"transverse": ("x", "y", "t"), "sagittal": ("x", "t", "y"), "coronal": ("y", "t", "x")}
    for base, (horizontal, vertical, stack) in axes.items():
        views[base] = {"name": base, "family": "orthogonal", "horizontal_axis": horizontal,
                       "vertical_axis": vertical, "stack_axis": stack, "physical_view_name": base}
        name = f"azimuthal_{base}"
        views[name] = {"name": name, "family": "azimuthal", "azimuthal_base_view": base,
                       "physical_view_name": name, "azimuths_deg": angles}
    return {"views_by_name": views, "processing_shape_tyx": shape, "source_shape_tyx": source or shape}


def desc(name, ctx):
    return section_descriptor({"physical_view_name": name}, geometry_context=ctx)


def test_cartesian_names_follow_actual_stack_axes():
    ctx = context()
    assert desc("transverse", ctx)["normal_xyz"] == (0, 0, 1)
    assert desc("sagittal", ctx)["normal_xyz"] == (0, 1, 0)
    assert desc("coronal", ctx)["normal_xyz"] == (1, 0, 0)
    assert len({section_codes(desc(name, ctx), 0, 1, (5, 5, 5), ctx) for name in ("transverse", "sagittal", "coronal")}) == 3


def test_transverse_azimuth_axes_share_cartesian_plane_codes():
    ctx = context()
    codes = section_codes(desc("azimuthal_transverse", ctx), 0, 5, (5, 5, 5), ctx)
    sagittal = section_codes(desc("sagittal", ctx), 0, 5, (5, 5, 5), ctx)
    coronal = section_codes(desc("coronal", ctx), 0, 5, (5, 5, 5), ctx)
    assert np.all(codes[:, 2, :] == sagittal)  # theta 0: radial x, stack t
    assert np.all(codes[:, [0, 1, 3, 4], 2] == coronal)  # theta 90: radial y
    assert codes.dtype == np.uint32


def test_other_azimuth_bases_match_actual_plane_axes():
    ctx = context()
    sagittal = section_codes(desc("azimuthal_sagittal", ctx), 0, 5, (5, 5, 5), ctx)
    coronal = section_codes(desc("azimuthal_coronal", ctx), 0, 5, (5, 5, 5), ctx)
    transverse_code = section_codes(desc("transverse", ctx), 0, 5, (5, 5, 5), ctx)
    coronal_code = section_codes(desc("coronal", ctx), 0, 5, (5, 5, 5), ctx)
    sagittal_code = section_codes(desc("sagittal", ctx), 0, 5, (5, 5, 5), ctx)
    assert np.all(sagittal[2, :, :] == transverse_code)
    assert np.all(sagittal[[0, 1, 3, 4], :, 2] == coronal_code)
    assert np.all(coronal[2, :, :] == transverse_code)
    assert np.all(coronal[[0, 1, 3, 4], 2, :] == sagittal_code)


@pytest.mark.parametrize("size", [4, 5, 6, 7])
def test_odd_even_voxel_centers_have_correct_azimuthal_symmetry(size):
    ctx = context((3, size, size))
    codes = section_codes(desc("azimuthal_transverse", ctx), 0, 3, (3, size, size), ctx)
    np.testing.assert_array_equal(codes, codes[:, ::-1, ::-1])
    # Equal x/y offsets select the actual 45-degree sampled section.
    expected = section_codes({"kind": "plane", "normal_xyz": (-1, 1, 0)}, 0, 3, (3, size, size))
    for i in range(size):
        if size % 2 and i == size // 2:
            continue  # The center belongs to every diameter; deterministic theta 0.
        assert np.all(codes[:, i, i] == expected)


def test_dense_azimuth_near_90_wraps_to_coronal_orientation_bin():
    ctx = context(angles=(0, 89.99, 179.99))
    codes = section_codes(desc("azimuthal_transverse", ctx), 0, 5, (5, 5, 5), ctx)
    coronal = section_codes(desc("coronal", ctx), 0, 5, (5, 5, 5), ctx)
    assert np.all(codes[:, [0, 1, 3, 4], 2] == coronal)


def test_nearest_recorded_azimuth_is_used_instead_of_continuous_angle():
    ctx = context(angles=(0, 90))
    codes = section_codes(desc("azimuthal_transverse", ctx), 0, 5, (5, 5, 5), ctx)
    allowed = {int(section_codes(desc(n, ctx), 0, 5, (5, 5, 5), ctx)) for n in ("sagittal", "coronal")}
    assert set(np.unique(codes)) == allowed


def test_tilted_azimuth_uses_same_plane_orientation_as_upright():
    ctx = context()
    original = ctx["views_by_name"]["azimuthal_transverse"]
    for direction in ("horizontal", "vertical"):
        name = f"azimuthal_tilted_transverse_{direction}_p30"
        ctx["views_by_name"][name] = {**original, "name": name, "physical_view_name": name,
                                       "tilt_direction": direction, "tilt_angle_deg": 30, "azimuthal_tilted_source": True}
        tilted, upright = desc(name, ctx), desc("azimuthal_transverse", ctx)
        assert tilted["group_key"] == upright["group_key"]
        np.testing.assert_array_equal(section_codes(tilted, 0, 5, (5, 5, 5), ctx),
                                      section_codes(upright, 0, 5, (5, 5, 5), ctx))


def test_tilt_normal_accounts_for_source_temporal_scale_and_sign():
    ctx = context((10, 5, 5), source=(5, 5, 5))
    normal = desc("tilted_transverse_vertical_p30", ctx)["normal_xyz"]
    expected = np.array([0, -math.tan(math.radians(30)), 2])
    expected /= np.linalg.norm(expected)
    np.testing.assert_allclose(normal, expected)
    negative = desc("tilted_transverse_vertical_m30", ctx)["normal_xyz"]
    assert normal != negative
    assert section_codes({"kind": "plane", "normal_xyz": normal}, 0, 1, (5, 5, 5)) == section_codes(
        {"kind": "plane", "normal_xyz": tuple(-v for v in normal)}, 0, 1, (5, 5, 5))


def test_azimuth_polar_coordinates_use_processing_grid_and_source_normal_scale():
    ctx = context((10, 5, 5), source=(5, 5, 5), angles=(0, 45, 90, 135))
    azimuth = desc("azimuthal_sagittal", ctx)
    codes = section_codes(azimuth, 0, 5, (5, 5, 5), ctx)
    # Output (t=3,x=4) maps to processing offsets (t=2,x=2), theta=45.
    expected_normal = (-1, 0, 2)
    expected = section_codes({"kind": "plane", "normal_xyz": expected_normal}, 0, 5, (5, 5, 5))
    assert np.all(codes[3, :, 4] == expected)


def test_tiles_and_tta_duplicates_share_fixed_plane_group():
    ctx = context()
    entries = [{"physical_view_name": "coronal", "view_name": "coronal__tta_a0", "source": "fullframe"},
               {"physical_view_name": "coronal", "view_name": "coronal__tta_a90", "source": "tile", "tile_config_id": "other"}]
    groups = [section_descriptor(m, geometry_context=ctx)["group_key"] for m in entries]
    assert groups[0] == groups[1]


def test_all_spherical_patches_rotations_share_one_surface_code():
    names = ["spherical_upright_px_patch_u0_v0", "spherical_vertical_p30_nz_patch_u1_v1", "spherical_horizontal_m30_py_patch_u0_v1"]
    descriptors = [section_descriptor({"physical_view_name": name, "view_family": "spherical"}) for name in names]
    assert {d["group_key"] for d in descriptors} == {"sphere"}
    assert len({int(section_codes(d, 0, 1, (3, 3, 3))) for d in descriptors}) == 1


def test_radial_patches_and_tilts_share_only_their_base_cylinder():
    names = ["radial_transverse_patch_u0_h0", "radial_tilted_transverse_vertical_p30_patch_u4_h1", "radial_sagittal_patch_u0_h0"]
    descriptors = [section_descriptor({"physical_view_name": name, "view_family": "radial"}) for name in names]
    assert descriptors[0]["group_key"] == descriptors[1]["group_key"] != descriptors[2]["group_key"]
    assert section_codes(descriptors[0], 0, 1, (3, 3, 3)) == section_codes(descriptors[1], 0, 1, (3, 3, 3))


def test_absent_azimuth_sampling_fails_instead_of_inventing_section_geometry():
    descriptor = section_descriptor({"physical_view_name": "azimuthal_transverse", "view_family": "azimuthal"})
    with pytest.raises(ValueError, match="processing_shape"):
        section_codes(descriptor, 0, 1, (5, 5, 5))
    with pytest.raises(ValueError, match="azimuths"):
        section_codes(descriptor, 0, 1, (5, 5, 5), {"processing_shape_tyx": (5, 5, 5)})


@pytest.mark.parametrize("tolerance", [0, 0.001, 91, float("nan")])
def test_invalid_angular_tolerance_fails(tolerance):
    with pytest.raises(ValueError, match="angular_tolerance"):
        section_codes({"kind": "plane", "normal_xyz": (0, 0, 1)}, 0, 1, (5, 5, 5), angular_tolerance_deg=tolerance)

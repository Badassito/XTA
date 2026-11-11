"""Forced NRRD reconciliation preserves categorical labels and slice identity."""

from unittest import mock

import numpy as np
import pytest

from XTA import pta_inputs


def write_mask(tmp_path, source, *, encoding="raw", canonical=True, directions=True):
    nrrd = pytest.importorskip("nrrd")
    path = tmp_path / "mask.nrrd"
    header = {"encoding": encoding}
    if directions:
        header["space directions"] = (
            np.eye(3) if canonical else np.asarray([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
        )
    data = source.transpose(2, 1, 0) if canonical else source
    nrrd.write(str(path), data, header=header)
    return path


def load_mask(path, target_shape, **kwargs):
    return pta_inputs.load_nrrd_mask_for_volume(
        path, target_shape, warnings=pta_inputs.WarningLog(), **kwargs
    )


@pytest.mark.parametrize("encoding", ["raw", "gzip"])
@pytest.mark.parametrize("force", [False, True])
def test_matching_mask_preserves_existing_binary_import(tmp_path, encoding, force):
    source = np.zeros((7, 11, 17), np.uint8)
    source[1:6, 3:9, 4:13] = 7
    path = write_mask(tmp_path, source, encoding=encoding)
    actual = load_mask(path, source.shape, force=force)
    np.testing.assert_array_equal(actual, source > 0)
    assert actual.dtype == np.uint8
    assert actual.flags.c_contiguous


def test_strict_loader_rejects_even_one_voxel_difference(tmp_path):
    path = write_mask(tmp_path, np.ones((7, 11, 17), np.uint8))
    with pytest.raises(ValueError, match="does not match"):
        load_mask(path, (8, 11, 17))


@pytest.mark.parametrize("encoding", ["raw", "gzip"])
@pytest.mark.parametrize("target_shape", [(10, 14, 20), (4, 8, 14)])
def test_force_accepts_three_voxels_on_every_axis_without_full_reader(
    tmp_path, encoding, target_shape
):
    nrrd = pytest.importorskip("nrrd")
    source = np.zeros((7, 11, 17), np.uint8)
    source[0, 0, 0] = 2
    source[-1, -1, -1] = 9
    path = write_mask(tmp_path, source, encoding=encoding)
    with mock.patch.object(nrrd, "read", side_effect=AssertionError("full NRRD reader")):
        actual = load_mask(path, target_shape, force=True)
    assert actual.shape == target_shape
    assert actual[0, 0, 0] == actual[-1, -1, -1] == 1
    assert set(np.unique(actual)) == {0, 1}


@pytest.mark.parametrize("target_shape", [(11, 11, 17), (7, 15, 17), (7, 11, 21)])
def test_force_rejects_four_voxel_difference_without_plane_mapping(tmp_path, target_shape):
    path = write_mask(tmp_path, np.ones((7, 11, 17), np.uint8))
    with pytest.raises(ValueError):
        load_mask(path, target_shape, force=True)


@pytest.mark.parametrize("axis", [0, 1, 2])
def test_force_uses_endpoint_aligned_nearest_neighbor_for_each_axis(tmp_path, axis):
    source_shape = [6, 9, 12]
    source_shape[axis] = 4
    source = np.zeros(source_shape, np.uint8)
    for index in (1, 3):
        selection = [slice(None)] * 3
        selection[axis] = index
        source[tuple(selection)] = 5
    target_shape = list(source_shape)
    target_shape[axis] = 7
    path = write_mask(tmp_path, source)
    actual = load_mask(path, tuple(target_shape), force=True)
    # Endpoints are retained. Halfway positions follow PTA's round-to-even convention.
    expected_slices = [0, 0, 1, 0, 0, 0, 1]
    for index, value in enumerate(expected_slices):
        assert np.all(np.take(actual, index, axis=axis) == value)


def test_noncontiguous_source_resizes_xy_and_preserves_plane_order(tmp_path):
    source = np.zeros((6, 9, 14), np.uint8)
    source[[0, 2, 5], 2:7, 3:11] = 3
    path = write_mask(tmp_path, source)
    actual = load_mask(path, (6, 12, 17), force=True, source_contiguous=False)
    assert actual.shape == (6, 12, 17)
    np.testing.assert_array_equal(np.any(actual, axis=(1, 2)), [True, False, True, False, False, True])


def test_noncontiguous_source_rejects_depth_resize_without_mapping(tmp_path):
    path = write_mask(tmp_path, np.ones((6, 9, 14), np.uint8))
    with pytest.raises(ValueError):
        load_mask(path, (7, 9, 14), force=True, source_contiguous=False)


def test_explicit_plane_mapping_keeps_z_identity_despite_large_depth_difference(tmp_path):
    source = np.zeros((8, 9, 14), np.uint8)
    source[0, :, :] = 2
    source[6, :, :] = 5
    path = write_mask(tmp_path, source)
    actual = load_mask(
        path, (3, 12, 17), force=True, source_plane_indices=(6, 2, 0)
    )
    assert actual.shape == (3, 12, 17)
    assert np.all(actual[0] == 1)
    assert np.all(actual[1] == 0)
    assert np.all(actual[2] == 1)


def test_explicit_plane_mapping_still_enforces_xy_tolerance(tmp_path):
    path = write_mask(tmp_path, np.ones((8, 9, 14), np.uint8))
    with pytest.raises(ValueError):
        load_mask(path, (3, 13, 14), force=True, source_plane_indices=(6, 2, 0))


def test_explicit_unannotated_plane_is_mask_empty(tmp_path):
    path = write_mask(tmp_path, np.ones((7, 9, 14), np.uint8))
    actual, matched_positions = pta_inputs.load_indexed_nrrd_mask_for_volume(
        path,
        (3, 9, 14),
        warnings=pta_inputs.WarningLog(),
        force=True,
        source_plane_indices=(0, -1, 6),
    )
    assert np.all(actual[0] == 1)
    assert np.all(actual[1] == 0)
    assert np.all(actual[2] == 1)
    assert matched_positions == (0, 2)


@pytest.mark.parametrize("mapping", [(0, 1), (-1, 7, 8)])
def test_invalid_explicit_plane_mapping_rejects(tmp_path, mapping):
    path = write_mask(tmp_path, np.ones((7, 9, 14), np.uint8))
    with pytest.raises(ValueError):
        load_mask(path, (3, 9, 14), force=True, source_plane_indices=mapping)


def test_outside_indexed_planes_are_unannotated_at_either_end(tmp_path):
    path = write_mask(tmp_path, np.ones((7, 9, 14), np.uint8))
    actual, matched_positions = pta_inputs.load_indexed_nrrd_mask_for_volume(
        path,
        (4, 9, 14),
        warnings=pta_inputs.WarningLog(),
        force=True,
        source_plane_indices=(-2, 0, 6, 7),
    )
    assert matched_positions == (1, 2)
    assert np.all(actual[0] == 0)
    assert np.all(actual[1:3] == 1)
    assert np.all(actual[3] == 0)


def test_force_supports_header_defined_noncanonical_axis_order(tmp_path):
    source = np.zeros((7, 11, 17), np.int16)
    source[0, :, :] = -3
    source[-1, :, :] = 5
    path = write_mask(tmp_path, source, canonical=False)
    actual = load_mask(path, (10, 14, 20), force=True)
    assert actual.shape == (10, 14, 20)
    assert np.all(actual[0] == 0)
    assert np.all(actual[-1] == 1)


def test_force_resizes_detached_nrrd(tmp_path):
    nrrd = pytest.importorskip("nrrd")
    source = np.zeros((7, 11, 17), np.uint8)
    source[-1, -1, -1] = 1
    path = tmp_path / "mask.nhdr"
    nrrd.write(str(path), source.transpose(2, 1, 0), header={"space directions": np.eye(3)})
    actual = load_mask(path, (10, 14, 20), force=True)
    assert actual.shape == (10, 14, 20)
    assert actual[-1, -1, -1] == 1


def test_force_rejects_ambiguous_axes_without_metadata(tmp_path):
    path = write_mask(tmp_path, np.ones((7, 8, 9), np.uint8), directions=False)
    with pytest.raises(ValueError, match="ambiguous"):
        load_mask(path, (8, 9, 10), force=True)


def test_force_can_infer_unique_axis_order_without_metadata(tmp_path):
    source = np.zeros((7, 13, 19), np.uint8)
    source[-1, -1, -1] = 7
    path = write_mask(tmp_path, source, directions=False)
    actual = load_mask(path, (10, 16, 22), force=True)
    assert actual.shape == (10, 16, 22)
    assert actual[-1, -1, -1] == 1


def test_force_keeps_header_axis_directions_authoritative(tmp_path):
    # Raw array shape matches the image, but the header maps its first axis to X.
    path = write_mask(tmp_path, np.ones((13, 8, 4), np.uint8))
    with pytest.raises(ValueError):
        load_mask(path, (4, 8, 13), force=True)


@pytest.mark.parametrize("corruption", ["short", "long"])
def test_resized_raw_mask_validates_entire_source_payload(tmp_path, corruption):
    source = np.ones((7, 9, 13), np.uint8)
    path = write_mask(tmp_path, source)
    payload = path.read_bytes()
    path.write_bytes(payload[:-1] if corruption == "short" else payload + b"x")
    with pytest.raises((ValueError, EOFError, OSError)):
        load_mask(path, (4, 9, 13), force=True)

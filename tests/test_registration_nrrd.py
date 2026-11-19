"""Scalar NRRD registration keeps category values, geometry and observation coverage."""

import numpy as np
import pytest

from XTA.registration.nrrd import transfer_nrrd_labelmap
from XTA.registration.packet import SCHEMA, SCHEMA_VERSION, RegistrationPacket

nrrd = pytest.importorskip("nrrd")


def packet(shape, moving_shape=None, matrix=None, spacing=(1.0, 1.0, 1.0), masks=None):
    value = np.eye(4) if matrix is None else np.asarray(matrix, float)
    scans = {
        role: dict(
            filename=role + ".mkv",
            sha256="a" * 64,
            shape_zyx=list(size),
            spacing_xyz=list(spacing),
            spacing_units="unknown",
        )
        for role, size in [("fixed", shape), ("moving", moving_shape or shape)]
    }
    return RegistrationPacket(
        dict(
            schema=SCHEMA,
            schema_version=SCHEMA_VERSION,
            pair="synthetic",
            status="validated",
            **scans,
            mapping=dict(
                fixed_to_moving_xyz=value.tolist(),
                moving_to_fixed_xyz=np.linalg.inv(value).tolist(),
            ),
        ),
        masks or {},
    )


def write(path, data, encoding="raw", header=None):
    nrrd.write(str(path), data, header={"encoding": encoding, **(header or {})}, index_order="C")


def load(path):
    return nrrd.read(str(path), index_order="C")


@pytest.mark.parametrize("encoding", ["raw", "gzip"])
def test_oblique_axis_change_preserves_uint64_ids_and_masks(tmp_path, encoding):
    source = np.zeros((4, 5, 6), np.uint64)
    source[1:3, 1:4, 1:5] = 2**63 + 17
    source[2, 3, 4] = 2**63 + 123
    transform = np.array([[0, -1, 0, 4], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], float)
    fixed_mask = np.ones((5, 6), bool)
    fixed_mask[2, 2] = False
    moving_mask = np.ones((6, 5), bool)
    moving_mask[4, 3] = False
    pair = packet(
        source.shape, (4, 6, 5), transform, masks={"fixed": fixed_mask, "moving": moving_mask}
    )
    original = tmp_path / "source.nrrd"
    write(original, source, encoding)
    output = tmp_path / "target.nrrd"
    result = transfer_nrrd_labelmap(
        pair,
        original,
        output,
        source_grid_aligned=True,
        work_dir=tmp_path / "scratch",
        encoding=encoding,
        block_rows=2,
    )
    actual, header = load(output)
    coverage, _ = load(result["coverage_output"])
    expected = source.transpose(0, 2, 1)[:, :, ::-1].copy()
    expected_coverage = np.broadcast_to((fixed_mask.T[:, ::-1] & moving_mask), expected.shape)
    expected[~expected_coverage] = 0
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(coverage, expected_coverage)
    assert actual.dtype == np.uint64
    assert int(actual.max()) == 2**63 + 123
    np.testing.assert_array_equal(header["space directions"], np.eye(3))
    assert "space" not in header
    assert result["unavailable_voxels"] == int((~expected_coverage).sum())
    assert not list((tmp_path / "scratch").iterdir())


def test_cropped_reordered_labels_use_source_physical_geometry(tmp_path):
    shape = (5, 6, 7)
    spacing = (2.0, 3.0, 4.0)
    transform = np.eye(4)
    transform[:3, 3] = [1, -1, 0]
    pair = packet(shape, matrix=transform, spacing=spacing)
    source_header = dict(
        dimension=3,
        sizes=np.array(shape[::-1]),
        space="left-posterior-superior",
        **{
            "space directions": np.diag(spacing),
            "space origin": [10.0, 20.0, 30.0],
            "space units": ["mm"] * 3,
        },
    )
    # Label X follows source Z, label Y source Y, label Z source X.
    label_header = {
        "space": "LPS",
        "space directions": np.array([[0, 0, 4], [0, 3, 0], [2, 0, 0]]),
        "space origin": np.array([14, 23, 34]),
        "space units": ["mm"] * 3,
    }
    source_labels = np.arange(1, 19, dtype=np.uint16).reshape(2, 3, 3)
    original = tmp_path / "cropped.nrrd"
    write(original, source_labels, "gzip", label_header)
    target_header = {
        **source_header,
        "space origin": np.array([101.0, -50.0, 8.0]),
        "space directions": np.array([[0, 2, 0], [-3, 0, 0], [0, 0, 4.0]]),
    }
    output = tmp_path / "mapped.nrrd"
    result = transfer_nrrd_labelmap(
        pair,
        original,
        output,
        source_reference=source_header,
        target_reference=target_header,
        block_rows=1,
    )
    actual, header = load(output)
    coverage, _ = load(result["coverage_output"])
    expected = np.zeros(shape, np.uint16)
    expected[1:4, 0:3, 3:5] = source_labels.transpose(2, 1, 0)
    known = np.zeros(shape, np.uint8)
    known[1:4, 0:3, 3:5] = 1
    np.testing.assert_array_equal(actual, expected)
    np.testing.assert_array_equal(coverage, known)
    np.testing.assert_array_equal(header["space origin"], target_header["space origin"])
    np.testing.assert_array_equal(header["space directions"], target_header["space directions"])
    assert header["space units"] == ["mm"] * 3


def test_ras_label_coordinates_convert_to_lps_reference(tmp_path):
    shape = (3, 4, 5)
    pair = packet(shape)
    labels = np.arange(np.prod(shape), dtype=np.int16).reshape(shape)
    lps = {
        "dimension": 3,
        "sizes": np.array(shape[::-1]),
        "space": "LPS",
        "space directions": np.eye(3),
        "space origin": [10, 20, 30],
    }
    ras = {
        "space": "RAS",
        "space directions": np.diag([-1.0, -1.0, 1.0]),
        "space origin": [-10, -20, 30],
    }
    original = tmp_path / "ras.nrrd"
    write(original, labels, header=ras)
    result = transfer_nrrd_labelmap(pair, original, tmp_path / "mapped.nrrd", source_reference=lps)
    np.testing.assert_array_equal(load(result["output"])[0], labels)


def test_reverse_direction_and_negative_label_values(tmp_path):
    shape = (3, 4, 6)
    transform = np.eye(4)
    transform[0, 3] = 1
    pair = packet(shape, matrix=transform)
    source = np.zeros(shape, np.int16)
    source[1, 2, 3] = -8
    original = tmp_path / "moving.nrrd"
    write(original, source)
    result = transfer_nrrd_labelmap(
        pair,
        original,
        tmp_path / "fixed.nrrd",
        source_grid_aligned=True,
        direction="moving-to-fixed",
        background_value=-1,
    )
    actual, _ = load(result["output"])
    assert actual[1, 2, 2] == -8
    assert np.all(actual[:, :, -1] == -1)
    assert result["source_role"] == "moving"


@pytest.mark.parametrize(
    "metadata",
    [{"Segment0_ID": "example"}, {"Segmentation_MasterRepresentation": "Binary labelmap"}],
)
def test_slicer_semantics_rejected_before_output_publication(tmp_path, metadata):
    data = np.ones((3, 4, 5), np.uint8)
    original = tmp_path / "seg.nrrd"
    write(original, data, header=metadata)
    output = tmp_path / "out.nrrd"
    with pytest.raises(ValueError, match="segment-aware"):
        transfer_nrrd_labelmap(packet(data.shape), original, output, source_grid_aligned=True)
    assert not output.exists()
    assert not output.with_name("out_coverage.nrrd").exists()
    assert not list(tmp_path.glob("*.partial"))


def test_ambiguous_alignment_and_wrong_reference_spacing_rejected(tmp_path):
    shape = (3, 4, 5)
    original = tmp_path / "source.nrrd"
    write(original, np.zeros(shape, np.uint8))
    with pytest.raises(ValueError, match="source_reference"):
        transfer_nrrd_labelmap(packet(shape), original, tmp_path / "out.nrrd")
    bad_reference = {
        "dimension": 3,
        "sizes": np.array(shape[::-1]),
        "space directions": np.eye(3) * 2,
        "space origin": [0, 0, 0],
    }
    with pytest.raises(ValueError, match="voxel sizes"):
        transfer_nrrd_labelmap(
            packet(shape), original, tmp_path / "out.nrrd", source_reference=bad_reference
        )


def test_truncated_payload_and_publication_failure_leave_no_partial_result(tmp_path, monkeypatch):
    shape = (3, 4, 5)
    original = tmp_path / "source.nrrd"
    write(original, np.ones(shape, np.uint8), "gzip")
    output = tmp_path / "out.nrrd"
    from XTA.registration import nrrd as module

    publish = module._publish_complete_file
    count = 0

    def failing_link(source, destination):
        nonlocal count
        count += 1
        if count == 2:
            raise OSError("simulated coverage publication failure")
        return publish(source, destination)

    monkeypatch.setattr(module, "_publish_complete_file", failing_link)
    with pytest.raises(OSError, match="simulated"):
        transfer_nrrd_labelmap(packet(shape), original, output, source_grid_aligned=True)
    assert not output.exists()
    assert not output.with_name("out_coverage.nrrd").exists()
    assert not list(tmp_path.glob("*.partial"))
    original.write_bytes(original.read_bytes()[:-4])
    with pytest.raises((EOFError, ValueError)):
        transfer_nrrd_labelmap(packet(shape), original, output, source_grid_aligned=True)
    assert not output.exists()


def test_existing_outputs_are_preserved(tmp_path):
    data = np.ones((2, 3, 4), np.uint8)
    original = tmp_path / "source.nrrd"
    write(original, data)
    output = tmp_path / "out.nrrd"
    output.write_bytes(b"existing output")
    with pytest.raises(FileExistsError):
        transfer_nrrd_labelmap(packet(data.shape), original, output, source_grid_aligned=True)
    assert output.read_bytes() == b"existing output"


@pytest.mark.parametrize("fail_at", [2, 3])
def test_publication_rollback_preserves_another_writers_replacement(tmp_path, monkeypatch, fail_at):
    from XTA.registration import nrrd as module

    original = tmp_path / "source.nrrd"
    source = np.ones((2, 3, 4), np.uint8)
    write(original, source)
    output = tmp_path / "target.nrrd"
    coverage = tmp_path / "target_coverage.nrrd"
    replaced = output if fail_at == 2 else coverage
    publish = module._publish_complete_file
    count = 0

    def publish_then_replace(stage, destination):
        nonlocal count
        count += 1
        if count == fail_at:
            replacement = tmp_path / "external.nrrd"
            replacement.write_bytes(b"replacement published by another writer")
            module.os.replace(replacement, replaced)
            raise FileExistsError("simulated competing publication")
        publish(stage, destination)

    monkeypatch.setattr(module, "_publish_complete_file", publish_then_replace)
    with pytest.raises(FileExistsError, match="competing"):
        transfer_nrrd_labelmap(packet(source.shape), original, output, source_grid_aligned=True)
    assert replaced.read_bytes() == b"replacement published by another writer"
    assert not (coverage if fail_at == 2 else output).exists()
    assert not output.with_suffix(".nrrd.registration.json").exists()
    assert not list(tmp_path.glob("*.partial"))

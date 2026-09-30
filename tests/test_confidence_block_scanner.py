"""Confidence crop scanning preserves the full-plane block and publication contract."""

from __future__ import annotations

import json
from unittest import mock

import numpy as np
import pytest

from XTA import confidence_storage as storage


class CropReader:
    known_z_bounds = (0, 3)

    def __init__(self) -> None:
        self.shape = (3, 25, 29)
        self.planes = np.zeros(self.shape, dtype=np.uint8)
        self.crops: dict[int, list[tuple[int, int, int, int, np.ndarray]]] = {}
        rng = np.random.default_rng(2405)
        for z, bounds in (
            (0, (1, 15, 3, 28)),   # ragged crop crosses two row bands and four columns
            (1, (2, 8, 1, 11)),    # two disjoint crops on one slice
            (1, (17, 24, 10, 27)),
        ):
            y0, y1, x0, x1 = bounds
            values = rng.integers(0, 6, (y1 - y0, x1 - x0), dtype=np.uint8)
            values[values < 4] = 0
            values[0, 0] = np.uint8(255)
            values[-1, -1] = np.uint8(1)
            self.planes[z, y0:y1, x0:x1] = values
            self.crops.setdefault(z, []).append((y0, y1, x0, x1, values))

    def iter_crops(self, z: int):
        yield from self.crops.get(z, ())

    def __call__(self, z: int) -> np.ndarray:
        return self.planes[z]


def _blocks_match(actual, expected) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert left[:4] == right[:4]
        assert left[4].flags.c_contiguous
        np.testing.assert_array_equal(left[4], right[4])


@pytest.mark.parametrize("band_limit", [8 * 1024 * 1024, 1])
def test_crop_scan_matches_full_plane_with_ragged_and_split_crops(band_limit: int) -> None:
    reader = CropReader()
    with mock.patch.object(storage, "_CROP_OCCUPANCY_BAND_BYTES", band_limit):
        for z in range(reader.shape[0]):
            actual = list(storage.crop_blocks(reader.iter_crops(z), reader.shape[1:], 8))
            expected = list(storage.plane_blocks(reader(z), 8))
            _blocks_match(actual, expected)


def test_crop_and_full_plane_writers_publish_identical_artifacts(tmp_path) -> None:
    reader = CropReader()

    class FullPlaneReader:
        known_z_bounds = reader.known_z_bounds

        def __call__(self, z: int) -> np.ndarray:
            return reader(z)

    options = dict(layer_key="scan-parity", model_name="model", provenance={"source": "fixture"},
                   coordinate_space="native_view_processing", source_shape=(3, 25, 29),
                   block_size=8, max_numeric_bytes=1_000_000)
    crop_metadata = storage.write_blocks(tmp_path / "crop", reader.shape, reader, **options)
    full_metadata = storage.write_blocks(tmp_path / "full", reader.shape, FullPlaneReader(), **options)
    assert crop_metadata == full_metadata
    for name in ("index.bin", "scores.u8.zlib", "metadata.json"):
        assert (tmp_path / "crop" / name).read_bytes() == (tmp_path / "full" / name).read_bytes()
    assert json.loads((tmp_path / "crop" / "metadata.json").read_text()) == crop_metadata
    assert not list((tmp_path / "crop").glob("*.partial"))


def test_repeated_or_out_of_order_global_cells_still_fail() -> None:
    value = np.asarray([[1]], dtype=np.uint8)
    cases = (
        [(0, 1, 0, 1, value), (0, 1, 0, 1, value)],  # duplicate
        [(0, 1, 2, 3, value), (0, 1, 0, 1, value)],  # reversed cells
        [(0, 1, 0, 1, value), (1, 2, 1, 2, value)],  # disjoint pixels in one cell
    )
    for crops in cases:
        with pytest.raises(ValueError, match="disjoint, ordered block cells"):
            list(storage.crop_blocks(crops, (4, 4), 2))


def test_staging_limit_failure_does_not_publish_partial_artifacts(tmp_path) -> None:
    destination = tmp_path / "limited"
    with pytest.raises(storage.ConfidenceStageLimit):
        storage.write_blocks(destination, CropReader().shape, CropReader(),
                             layer_key="scan-limit", model_name="model", provenance={},
                             coordinate_space="native_view_processing", source_shape=(3, 25, 29),
                             block_size=8, max_numeric_bytes=1)
    assert list(destination.iterdir()) == []


def test_metadata_failure_keeps_completion_marker_absent(tmp_path) -> None:
    destination = tmp_path / "metadata-failure"
    reader = CropReader()
    with mock.patch("XTA.confidence_evidence._write_json_atomic", side_effect=OSError("metadata write failed")):
        with pytest.raises(OSError, match="metadata write failed"):
            storage.write_blocks(destination, reader.shape, reader,
                                 layer_key="scan-failure", model_name="model", provenance={},
                                 coordinate_space="native_view_processing", source_shape=(3, 25, 29),
                                 block_size=8)
    assert not (destination / "metadata.json").exists()
    assert (destination / "index.bin").exists()
    assert (destination / "scores.u8.zlib").exists()
    assert not list(destination.glob("*.partial"))

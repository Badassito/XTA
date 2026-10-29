"""Canonical input readers preserve sparse decode and annotation contracts."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta_inputs
from XTA.pta_workers import WarningLog


def test_sparse_png_reader_keeps_lattice_and_reports_selected_planes(tmp_path):
    images = []
    for index in range(5):
        path = tmp_path / f'sample_{index:04d}.png'
        assert pta_inputs.cv2.imwrite(str(path), np.full((3, 4), index + 10, np.uint8))
        images.append(path)
    warnings = WarningLog()
    actual, block = pta_inputs.load_image_sequence_to_memory(
        images, warnings=warnings, workers=2, required_positions=(2, 4),
    )
    assert block is None
    assert actual.shape == (5, 3, 4)
    np.testing.assert_array_equal(actual[:, 0, 0], [10, 0, 12, 0, 14])
    assert warnings.counts['partial_sequence_required_frames'] == 1


def test_sequence_decode_failure_releases_new_allocator_block(tmp_path):
    first, second = tmp_path / 'first.png', tmp_path / 'second.png'
    assert pta_inputs.cv2.imwrite(str(first), np.ones((3, 4), np.uint8))
    assert pta_inputs.cv2.imwrite(str(second), np.ones((2, 4), np.uint8))
    block = SimpleNamespace(release=mock.Mock())
    with pytest.raises(ValueError, match='dimension mismatch'):
        pta_inputs.load_image_sequence_to_memory(
            [first, second], warnings=WarningLog(), workers=1,
            allocator=lambda shape: (np.empty(shape, np.uint8), block),
        )
    block.release.assert_called_once()


def test_yolo_parser_preserves_valid_polygons_and_rejects_degenerate_lines(tmp_path):
    path = tmp_path / 'sample.txt'
    path.write_text(
        '0 -1 0 2 0 .5 2\n'
        '0 .1 .1 .1 .1 .2 .2\n'
        '0 nan 0 1 0 0 1\n'
        '0 text 0 1 0 0 1\n'
        '0 .1 .2\n', encoding='utf-8',
    )
    warnings = WarningLog()
    assert pta_inputs.read_yolo_polygons_normalized(path, warnings=warnings) == (
        ((0., 0.), (1., 0.), (.5, 1.)),
    )
    assert warnings.counts['degenerate_input_polygon'] == 2
    assert warnings.counts['invalid_label_coordinate'] == 1
    assert warnings.counts['invalid_label_line'] == 1

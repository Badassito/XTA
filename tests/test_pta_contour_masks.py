"""Polygon export accepts nonbinary masks without changing source pixels."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.pta_publication import mask_has_yolo_polygon, mask_to_yolo_lines


def export(mask):
    warnings = []
    lines = mask_to_yolo_lines(
        mask, warnings=SimpleNamespace(add=lambda *args: warnings.append(args)), context='case',
    )
    return lines, warnings


def masks():
    empty = np.zeros((12, 16), np.uint8)
    point = empty.copy()
    point[4, 5] = 1
    line = empty.copy()
    line[2, 2:12] = 1
    hole = empty.copy()
    hole[1:11, 2:14] = 1
    hole[4:8, 6:10] = 0
    nested = hole.copy()
    nested[5:7, 7:9] = 1
    fragmented = empty.copy()
    fragmented[::2, ::2] = 1
    return [empty, point, line, hole, nested, fragmented, np.ones_like(empty)]


@pytest.mark.parametrize('mask', masks())
@pytest.mark.parametrize('layout', ['contiguous', 'transposed', 'reverse', 'readonly'])
@pytest.mark.parametrize('dtype', [np.bool_, np.uint8, np.uint16, np.int16, np.float32])
def test_mask_cast_and_layout_keep_exact_labels_warnings_and_source(mask, layout, dtype):
    source = mask.astype(dtype)
    if dtype != np.bool_:
        source *= 255
        source[0, 0] = 256 if dtype != np.uint8 else 2
    if layout == 'transposed':
        source = source.T
    elif layout == 'reverse':
        source = source[::-1, ::-1]
    elif layout == 'readonly':
        source.flags.writeable = False
    before = source.copy()
    # The uint8 conversion precedes thresholding in the established contract.
    normalized = (np.asarray(source, dtype=np.uint8) > 0).astype(np.uint8)
    assert export(source) == export(normalized)
    assert mask_has_yolo_polygon(source) == mask_has_yolo_polygon(normalized)
    np.testing.assert_array_equal(source, before)


def test_polygon_hole_export_keeps_coordinates_order_and_warning():
    assert export(masks()[3]) == (
        ['0 0.125000 0.083333 0.125000 0.833333 0.812500 0.833333 0.812500 0.083333'],
        [('holes_discarded_during_polygon_export', 'case')],
    )


def test_known_empty_bypasses_mask_access():
    assert mask_to_yolo_lines(
        object(), warnings=SimpleNamespace(add=None), context='case', known_empty=True,
    ) == []


@pytest.mark.parametrize('shape', [(0,), (0, 3), (2, 0), (1, 0, 2)])
def test_zero_sized_masks_remain_empty(shape):
    assert export(np.empty(shape, np.uint8)) == ([], [])
    assert not mask_has_yolo_polygon(np.empty(shape, np.uint8))

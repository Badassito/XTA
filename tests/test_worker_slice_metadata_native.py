"""Worker receipt geometry and its existing MSB-first row occupancy encoding."""
import numpy as np
import pytest

from XTA.workers import _binary_slice_metadata_from_array


@pytest.mark.parametrize("width", [0, 1, 7, 8, 9, 16, 17, 33])
@pytest.mark.parametrize("layout", ["pitched", "reversed", "broadcast", "bool"])
def test_slice_metadata_preserves_half_open_bounds_and_msb_first_rows(width, layout):
    raw = bytearray(1 + 4 * 19 * (width + 11))
    source = np.ndarray((4, 19, width + 11), np.uint8, buffer=raw, offset=1)[:, :, :width]
    if width:
        source[1, [0, 7, 8, 18], width - 1] = 255
        source[2, 1:18:3, :max(1, width - 1)] = 128
        source[3, 15, 0] = 1
    if layout == "reversed":
        source = source[::-1, ::-1, ::-1]
    elif layout == "broadcast":
        source = np.broadcast_to(source[1, :, :1], source.shape)
    elif layout == "bool":
        # Aliasing one-byte nonzero values as bool also exercises byte normalization.
        source = source.view(bool)
    source.setflags(write=False)

    truth = source != 0
    rows = truth.any(axis=2)
    expected = np.zeros((source.shape[0], 4), np.int64)
    for z, plane in enumerate(truth):
        ys, xs = np.nonzero(plane)
        if len(ys):
            expected[z] = [ys.min(), ys.max() + 1, xs.min(), xs.max() + 1]
    receipt = _binary_slice_metadata_from_array(source)
    np.testing.assert_array_equal(receipt["slice_any"], rows.any(axis=1), strict=True)
    np.testing.assert_array_equal(receipt["slice_bboxes"], expected, strict=True)
    np.testing.assert_array_equal(receipt["slice_row_any"], np.packbits(rows, axis=1), strict=True)
    np.testing.assert_array_equal(receipt["slice_row_count"], np.array([19], np.int64), strict=True)


def test_other_numeric_masks_keep_truth_semantics():
    source = np.asarray([[[0., np.nan], [-2., 0.]], [[0., -0.], [0., 0.]]])
    metadata = _binary_slice_metadata_from_array(source)
    np.testing.assert_array_equal(metadata["slice_any"], [True, False])
    np.testing.assert_array_equal(metadata["slice_bboxes"], [[0, 2, 0, 2], [0, 0, 0, 0]])
    np.testing.assert_array_equal(metadata["slice_row_any"], [[192], [0]])

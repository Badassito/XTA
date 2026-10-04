"""Inspection deltas stay separate and subtract at the displayed resolution."""
import json

import numpy as np
import pytest

from tools.audit_sam_bridge_overlap import NewVoxelDiagnostic, open_payload
from XTA.outputs import _read_layer_slice_in_output_shape


def read_diagnostic(path):
    stream, payload, shape = open_payload(path)
    try:
        return np.frombuffer(payload.read(), np.uint8).reshape(shape)
    finally:
        payload.close()
        stream.close()


def test_new_voxel_diagnostic_is_exact_and_non_recomposable(tmp_path):
    shape = (3, 4, 5)
    detector = np.zeros(shape, bool)
    forward, backward = np.zeros(shape, bool), np.zeros(shape, bool)
    detector[1, 2, 2] = True
    forward[1, 2, 2:4] = True
    backward[2, 1, 1] = True
    expected = (forward | backward) & ~detector
    path = tmp_path/'sam_new_voxels.seg.nrrd'
    diagnostic = NewVoxelDiagnostic(path, shape)
    for frame in range(shape[0]):
        diagnostic.consume(frame, expected[frame])
    metadata = diagnostic.finalize(files=('detector','forward','backward'), policy_ids=('full-policy-hash',))
    np.testing.assert_array_equal(read_diagnostic(path), expected.astype(np.uint8))
    assert metadata['diagnostic_only'] and metadata['recomposition_op'] == 'none'
    assert metadata['voxel_count'] == 2
    assert json.loads(path.with_suffix('.json').read_text(encoding='utf-8')) == metadata
    with pytest.raises(FileExistsError):
        NewVoxelDiagnostic(path, shape)


def test_low_quality_diagnostic_subtracts_after_pooling_each_input(tmp_path):
    original = np.zeros((5, 5, 5), np.uint8)
    bridge = np.zeros_like(original)
    original[1, 1, 1], bridge[2, 1, 1] = 1, 1
    shape = (1, 1, 1)
    # Native contributions are disjoint, but both map into the one displayed bin.
    detector_low = _read_layer_slice_in_output_shape(original, shape, 0)
    bridge_low = _read_layer_slice_in_output_shape(bridge, shape, 0)
    path = tmp_path/'low_new_voxels.seg.nrrd'
    diagnostic = NewVoxelDiagnostic(path, shape)
    diagnostic.consume(0, (bridge_low != 0) & ~(detector_low != 0))
    metadata = diagnostic.finalize(files=('detector_low','bridge_low'), policy_ids=('policy',))
    assert metadata['voxel_count'] == 0
    assert not read_diagnostic(path).any()
    # Pooling a prior full-resolution subtraction would keep that overlapping bin.
    assert _read_layer_slice_in_output_shape(bridge & ~original, shape, 0).any()

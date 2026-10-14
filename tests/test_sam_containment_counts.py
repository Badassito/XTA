"""Fused whole-crop containment preserves finite-plane SciPy diagnostics."""
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest
from scipy import ndimage

from XTA import sam_policy
from tests.test_sam_guarded_rescue import _bundle
from tools.qualify_sam_policy_throughput import compare_selections


def _reference(raw, effective, acceptance):
    boundary = acceptance & ~ndimage.binary_erosion(
        acceptance, structure=np.ones((3, 3), bool), border_value=0)
    return (int(np.count_nonzero(raw)), int(np.count_nonzero(effective)),
            int(np.count_nonzero(raw & ~acceptance)), int(np.count_nonzero(effective & ~acceptance)),
            int(np.count_nonzero(raw & boundary)), int(np.count_nonzero(effective & boundary)))


def test_exhaustive_tiny_planes_match_finite_border_reference():
    planes = [np.array([(value >> bit) & 1 for bit in range(4)], bool).reshape(2, 2)
              for value in range(16)]
    for acceptance in planes:
        for raw in planes:
            for effective in planes:
                assert sam_policy._containment_counts(raw, effective, acceptance) == _reference(raw, effective, acceptance)


@pytest.mark.parametrize("shape", [(0, 0), (1, 1), (1, 17), (17, 1), (31, 33), (64, 65), (512, 513)])
def test_empty_dense_sparse_readonly_and_strided_planes(shape):
    rng = np.random.default_rng(5406)
    for density in (0., .001, .1, .5, 1.):
        raw = rng.random(shape) < density
        effective = raw & (rng.random(shape) < .8)
        for acceptance in (np.zeros(shape, bool), np.ones(shape, bool), rng.random(shape) < .8):
            for arrays in ((raw, effective, acceptance), tuple(a.T for a in (raw, effective, acceptance))):
                for array in arrays:
                    array.flags.writeable = False
                assert sam_policy._containment_counts(*arrays) == _reference(*arrays)


@pytest.mark.parametrize("index", range(3))
@pytest.mark.parametrize("kind", ["shape", "dtype"])
def test_invalid_plane_is_rejected_before_native_indexing(index, kind):
    arrays = [np.ones((3, 4), bool) for _ in range(3)]
    arrays[index] = np.ones((3, 3), bool) if kind == "shape" else arrays[index].astype(np.uint8)
    with pytest.raises(ValueError, match="matching boolean planes"):
        sam_policy._containment_counts(*arrays)


def test_parallel_readonly_counts_are_independent():
    rng = np.random.default_rng(975)
    cases = [tuple(rng.random((128, 129)) < density for density in (.1, .08, .8)) for _ in range(32)]
    for case in cases:
        for array in case:
            array.flags.writeable = False
    expected = [_reference(*case) for case in cases]
    with ThreadPoolExecutor(max_workers=8) as pool:
        assert list(pool.map(lambda case: sam_policy._containment_counts(*case), cases)) == expected


@pytest.mark.parametrize("options", [{}, {"flaw": "detached"}, {"flaw": "far"},
                                     {"flaw": "flood"}, {"censor": True}, {"bad_endpoint": True}])
def test_full_selection_receipts_match_scipy_reference(tmp_path, monkeypatch, options):
    bundle, _, _, _ = _bundle(tmp_path, **options)
    actual = sam_policy.select_sam_proposals(bundle)
    monkeypatch.setattr(sam_policy, "_containment_counts", _reference)
    expected = sam_policy.select_sam_proposals(bundle)
    assert compare_selections(expected, actual)["exact_quality"]


def test_corrupt_evidence_still_fails_authentication(tmp_path):
    bundle, _, _, _ = _bundle(tmp_path)
    path = bundle.directory / "masks.bin"
    data = path.read_bytes()
    path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
    with pytest.raises(ValueError, match="payload changed"):
        sam_policy.select_sam_proposals(bundle)

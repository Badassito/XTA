"""SAM component filtering never re-reads its own source file.

The module identity is hashed once at import. Historical receipts stay bound to
the numerical helpers through ``_QUALIFIED_NUMERICAL_SOURCE``, whose digest is
also read once, at import. Effective masks equal the v25.0.2 filter.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from XTA import sam_filtering
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_cyclic_evidence_replay import bundle_at

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "5320fbef0655293a85719593f36fffc8a213cea6"  # v25.0.2


@pytest.fixture
def old_filtering(tmp_path, monkeypatch):
    try:
        source = subprocess.run(["git", "show", f"{REFERENCE_COMMIT}:XTA/sam_filtering.py"], cwd=ROOT,
                                capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"reference commit {REFERENCE_COMMIT[:7]} is unavailable")
    assert hashlib.sha256(source).hexdigest() == sam_filtering._QUALIFIED_V25_SOURCE_GUARD_IMPLEMENTATION
    assert source != Path(sam_filtering.__file__).read_bytes()
    package = tmp_path / "oldpkg"
    package.mkdir()
    path = package / "sam_filtering.py"
    path.write_bytes(source)
    # Relative imports (``._deps``, ``.sam_mask_reader``) resolve inside XTA.
    spec = importlib.util.spec_from_file_location("XTA._old_sam_filtering_for_gil_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _frames(bundle):
    return [(run_id, int(frame)) for run_id, run in sorted(bundle.runs.items())
            for frame in sorted(run["raw_mask_keys"], key=int)]


def _forbid_filter_source_reads(monkeypatch):
    source = Path(sam_filtering.__file__).resolve()
    reads = []
    original = Path.read_bytes

    def read_bytes(self):
        if Path(self).resolve() == source:
            reads.append(self)
            raise AssertionError("SAM filter source was re-read")
        return original(self)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return reads


# --- old vs new identity ---------------------------------------------------------------------

@pytest.mark.parametrize("radius", [None, 0.0, 0.5, 1.0, 2.5])
def test_effective_masks_match_v25_filter_on_bundle_and_reader(tmp_path, old_filtering, radius):
    bundle = bundle_at(tmp_path)
    new_spec = sam_filtering.build_mask_filter(bundle, min_radius=radius)
    old_spec = old_filtering.build_mask_filter(bundle, min_radius=radius)
    assert {k: v for k, v in new_spec.items() if k not in ("implementation_sha256", "sha256")} == \
        {k: v for k, v in old_spec.items() if k not in ("implementation_sha256", "sha256")}
    frames = _frames(bundle)
    assert frames
    for disabled in (False, True):
        new_value = None if disabled else new_spec
        old_value = None if disabled else old_spec
        old = [old_filtering.measure_effective_raw_mask(bundle, r, f, old_value) for r, f in frames]
        plain = [sam_filtering.measure_effective_raw_mask(bundle, r, f, new_value) for r, f in frames]
        # Historical (v25.0.2) specs remain accepted by the edited module.
        historical = [sam_filtering.measure_effective_raw_mask(bundle, r, f, old_value) for r, f in frames]
        with bundle.reader() as reader:
            guarded = [sam_filtering.measure_effective_raw_mask(reader, r, f, new_value) for r, f in frames]
            via_reader = [reader.measure_effective_raw_mask(r, f, new_value) for r, f in frames]
        for (om, od), *others in zip(old, plain, historical, guarded, via_reader):
            for nm, nd in others:
                assert nm.dtype == om.dtype and nm.shape == om.shape and nm.strides == om.strides
                assert nm.flags.writeable == om.flags.writeable
                assert nm.tobytes() == om.tobytes() and nd == od


def test_filter_numerics_stay_pinned_and_previous_release_is_qualified():
    assert sam_filtering.IMPLEMENTATION_SHA256 != sam_filtering._QUALIFIED_V25_SOURCE_GUARD_IMPLEMENTATION
    assert sam_filtering.IMPLEMENTATION_SHA256 != sam_filtering._QUALIFIED_V26_SOURCE_GUARD_IMPLEMENTATION
    assert sam_filtering._compatible_filter_implementation(sam_filtering._QUALIFIED_V25_SOURCE_GUARD_IMPLEMENTATION)
    assert sam_filtering._compatible_filter_implementation(sam_filtering._QUALIFIED_V26_SOURCE_GUARD_IMPLEMENTATION)
    assert (sam_filtering._numerical_source_sha256(sam_filtering._numerical_helpers())
            == sam_filtering._QUALIFIED_NUMERICAL_SOURCE)
    assert sam_filtering._compatible_filter_implementation(sam_filtering.IMPLEMENTATION_SHA256)
    assert not sam_filtering._compatible_filter_implementation("0" * 64)


def test_unqualified_receipt_implementation_is_still_rejected(tmp_path):
    bundle = bundle_at(tmp_path)
    spec = dict(sam_filtering.build_mask_filter(bundle, min_radius=1.0))
    spec["implementation_sha256"] = "0" * 64
    spec.pop("sha256")
    spec["sha256"] = sam_filtering._fingerprint(spec)
    with pytest.raises(ValueError, match="implementation differs"):
        sam_filtering._spec(spec)


# --- per-mask work does no source I/O ----------------------------------------------------------

def test_no_filter_path_reads_the_filter_source(tmp_path, monkeypatch):
    bundle = bundle_at(tmp_path)
    receipt = select_sam_proposals(bundle, policy={"sam_bridge_policy": "permissive"})
    spec = sam_filtering.build_mask_filter(bundle, min_radius=1.0)
    historical = dict(spec, implementation_sha256=sam_filtering._QUALIFIED_V26_SOURCE_GUARD_IMPLEMENTATION)
    historical.pop("sha256")
    historical["sha256"] = sam_filtering._fingerprint(historical)
    frames = _frames(bundle)
    reads = _forbid_filter_source_reads(monkeypatch)
    for run_id, frame in frames:
        sam_filtering.measure_effective_raw_mask(bundle, run_id, frame, spec)
        sam_filtering.measure_effective_raw_mask(bundle, run_id, frame, historical)
    sam_filtering.build_mask_filter(bundle, min_radius=1.0)
    sam_filtering.filter_sam_components(np.ones((3, 3), bool), 1.0)
    with bundle.reader() as reader:
        snapshot = reader.filter_snapshot(receipt)
        for run_id, frame in frames:
            reader.measure_effective_raw_mask(run_id, frame, snapshot)
            reader.effective_candidate_mask(run_id, frame, snapshot)
        assert reader.stats["filter_computations"] == len(frames)
        lane = reader.fork(max_cache_bytes=1 << 20)
        with lane:
            borrowed = lane.borrowed_filter_snapshot(snapshot)
            for run_id, frame in frames:
                lane.measure_effective_raw_mask(run_id, frame, borrowed)
    assert reads == []

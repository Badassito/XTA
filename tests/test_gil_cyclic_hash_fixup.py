"""GIL-convoy fix follow-up: the per-mask SAM filter source hash inside reader transactions.

``filter_sam_components`` re-reads and SHA-256-hashes ``sam_filtering.py`` on every
decoded mask. Its text is pinned by ``_QUALIFIED_NUMERICAL_SOURCE``, so the call line
stays; instead the guard returns early only while ``measure_effective_raw_mask``
filters for an active SamMaskReader transaction, whose root re-hashes the source on
entry and on close. Everything else keeps the per-call check.
"""
from __future__ import annotations

import hashlib
import importlib.util
import subprocess
import sys
import threading
from pathlib import Path

import numpy as np
import pytest

from XTA import sam_filtering
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_cyclic_evidence_replay import bundle_at

ROOT = Path(__file__).resolve().parents[1]
REFERENCE_COMMIT = "5320fbef0655293a85719593f36fffc8a213cea6"  # v25.0.2
CHANGED = "SAM component filter implementation changed after loading"


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


def _missing(monkeypatch, tmp_path):
    monkeypatch.setattr(sam_filtering, "_SOURCE_PATH", tmp_path / "missing" / "sam_filtering.py")


def _changed(monkeypatch, tmp_path):
    changed = tmp_path / "changed_sam_filtering.py"
    changed.write_bytes(Path(sam_filtering.__file__).read_bytes() + b"\n# changed\n")
    monkeypatch.setattr(sam_filtering, "_SOURCE_PATH", changed)


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
    assert sam_filtering._compatible_filter_implementation(sam_filtering._QUALIFIED_V25_SOURCE_GUARD_IMPLEMENTATION)
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


# --- per-mask work inside a transaction does no source I/O ----------------------------------

def test_reader_transaction_masks_never_touch_the_filter_source(tmp_path, monkeypatch):
    bundle = bundle_at(tmp_path)
    receipt = select_sam_proposals(bundle, policy={"sam_bridge_policy": "permissive"})
    reader = bundle.reader().__enter__()
    try:
        with monkeypatch.context() as patch:
            _missing(patch, tmp_path)
            snapshot = reader.filter_snapshot(receipt)
            for run_id, frame in _frames(bundle):
                reader.measure_effective_raw_mask(run_id, frame, snapshot)
                reader.effective_candidate_mask(run_id, frame, snapshot)
            assert reader.stats["filter_computations"] == len(_frames(bundle))
            # Outside the per-mask bracket the guard still hashes the file.
            with pytest.raises(FileNotFoundError):
                sam_filtering.assert_filter_implementation_unchanged()
            with pytest.raises(FileNotFoundError):
                sam_filtering.filter_sam_components(np.ones((3, 3), bool), 1.0)
    finally:
        reader.close()


def test_fork_lanes_in_worker_threads_are_guarded_by_their_root(tmp_path, monkeypatch):
    bundle = bundle_at(tmp_path)
    receipt = select_sam_proposals(bundle, policy={"sam_bridge_policy": "permissive"})
    errors = []
    with bundle.reader() as reader:
        snapshot = reader.filter_snapshot(receipt)
        with monkeypatch.context() as patch:
            _missing(patch, tmp_path)
            lane = reader.fork(max_cache_bytes=1 << 20)

            def work():
                try:
                    with lane:
                        borrowed = lane.borrowed_filter_snapshot(snapshot)
                        for run_id, frame in _frames(bundle):
                            lane.measure_effective_raw_mask(run_id, frame, borrowed)
                except BaseException as error:
                    errors.append(error)
            thread = threading.Thread(target=work)
            thread.start()
            thread.join()
    assert errors == []


def test_change_inside_a_transaction_is_rejected_on_close(tmp_path, monkeypatch):
    bundle = bundle_at(tmp_path)
    receipt = select_sam_proposals(bundle, policy={"sam_bridge_policy": "permissive"})
    with pytest.raises(RuntimeError, match=CHANGED):
        with bundle.reader() as reader:
            snapshot = reader.filter_snapshot(receipt)
            _changed(monkeypatch, tmp_path)
            for run_id, frame in _frames(bundle):
                reader.measure_effective_raw_mask(run_id, frame, snapshot)
    # The next transaction is refused at entry.
    with pytest.raises(RuntimeError, match=CHANGED):
        bundle.reader().__enter__()


def test_plain_bundle_and_closed_reader_keep_the_per_call_check(tmp_path, monkeypatch):
    bundle = bundle_at(tmp_path)
    spec = sam_filtering.build_mask_filter(bundle, min_radius=1.0)
    run_id, frame = _frames(bundle)[0]
    reader = bundle.reader()
    with reader:
        pass
    _changed(monkeypatch, tmp_path)
    with pytest.raises(RuntimeError, match=CHANGED):
        sam_filtering.measure_effective_raw_mask(bundle, run_id, frame, spec)
    assert not sam_filtering._source_checked_transaction(reader)
    assert not sam_filtering._source_checked_transaction(bundle)
    with pytest.raises(RuntimeError, match=CHANGED):
        sam_filtering.filter_sam_components(np.ones((3, 3), bool), 1.0)


def test_guard_flag_is_scoped_to_the_call_and_thread(tmp_path, monkeypatch):
    seen = []

    def probe(raw, threshold, *, enabled=True):
        seen.append(getattr(sam_filtering._TRANSACTION_GUARDED, "active", False))
        # Another thread is never exempt while this one is bracketed.
        other = []
        thread = threading.Thread(target=lambda: other.append(
            getattr(sam_filtering._TRANSACTION_GUARDED, "active", False)))
        thread.start()
        thread.join()
        seen.append(other[0])
        raise ValueError("probe")
    monkeypatch.setattr(sam_filtering, "filter_sam_components", probe)
    with pytest.raises(ValueError, match="probe"):
        sam_filtering._filter_in_checked_transaction(np.ones((2, 2), bool), 1.0, True)
    assert seen == [True, False]
    assert getattr(sam_filtering._TRANSACTION_GUARDED, "active", False) is False

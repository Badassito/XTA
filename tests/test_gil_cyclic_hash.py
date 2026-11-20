"""GIL-convoy fix: the cyclic helpers never re-read their own source.

The old helpers re-read and SHA-256-hashed ``sam_cyclic.py`` on every address,
transform and validation call (about 1.3 M calls per Azimuthal view). These tests
prove the helpers, evidence, selection and publication paths perform no source
file I/O and that their results are identical to the v25.0.2 (5320fbe) implementation.
The module identity is hashed once, at import.
"""
from __future__ import annotations

import importlib.util
import itertools
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from XTA import sam_cyclic, sam_evidence, sam_interpolation, sam_policy
from XTA.sam_evidence import (SamEvidenceBundle, evidence_frame_geometry, iter_selected_native_crops,
    native_output_shape_tyx, selected_native_plane)
from XTA.sam_policy import select_sam_proposals
from tests.test_sam_cyclic_evidence_replay import bundle_at

ROOT = Path(__file__).resolve().parents[1]
_HELPERS = ("address_for_unfolded_index", "transform_crop_between_frame_addresses",
            "validate_cyclic_frame_addressing", "build_cyclic_frame_addressing", "mirror_bbox_yx",
            "CyclicFrameAddresses", "CyclicObservationVolume", "ExtrapolationCyclicObservationVolume")


# v25.0.2, the last release whose helpers re-hash their source on every call. Pinned
# (not HEAD) so the comparison stays meaningful after this change is committed.
REFERENCE_COMMIT = "5320fbef0655293a85719593f36fffc8a213cea6"
# SHA-256 of XTA/sam_cyclic.py at REFERENCE_COMMIT.
REFERENCE_CYCLIC_SHA256 = "5caa2371c6da22543388380fe2dc487dc05305fdfdf8c802d069e3065756b89c"


def _reference_source(relative):
    try:
        return subprocess.run(["git", "show", f"{REFERENCE_COMMIT}:{relative}"], cwd=ROOT,
                              capture_output=True, check=True).stdout
    except (OSError, subprocess.CalledProcessError):
        pytest.skip(f"reference commit {REFERENCE_COMMIT[:7]} is unavailable")


@pytest.fixture
def old_cyclic(tmp_path, monkeypatch):
    """The v25.0.2 helper module, loaded from its own file (so its per-call hash passes)."""
    import hashlib
    source = _reference_source("XTA/sam_cyclic.py")
    assert hashlib.sha256(source).hexdigest() == REFERENCE_CYCLIC_SHA256
    assert b"def assert_cyclic_implementation_unchanged" in source
    if source == Path(sam_cyclic.__file__).read_bytes():
        pytest.fail("reference sam_cyclic.py equals the working tree; the comparison would be tautological")
    path = tmp_path / "old_sam_cyclic.py"
    path.write_bytes(source)
    spec = importlib.util.spec_from_file_location("_old_sam_cyclic_for_gil_test", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    return module


def _outcome(function, *args, **kwargs):
    try:
        value = function(*args, **kwargs)
    except Exception as error:  # Compare failure semantics exactly.
        return ("error", type(error).__name__, str(error))
    return ("value", value)


def _plain_mapping(value):
    if hasattr(value, "items"):
        return {key: (type(item).__name__, _plain_mapping(item)) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value).__name__, [_plain_mapping(item) for item in value]
    return value


def _swap_in_old(monkeypatch, old):
    for name in _HELPERS:
        monkeypatch.setattr(sam_cyclic, name, getattr(old, name))


def _record_source_reads(monkeypatch):
    """Record (and refuse) any read of ``sam_cyclic.py`` after import."""
    source = Path(sam_cyclic.__file__).resolve()
    reads = []
    original = Path.read_bytes

    def read_bytes(self):
        if Path(self).resolve() == source:
            reads.append(self)
            raise AssertionError("SAM cyclic source was re-read")
        return original(self)
    monkeypatch.setattr(Path, "read_bytes", read_bytes)
    return reads


# --- old vs new identity of every per-call helper ------------------------------------------

def test_address_helper_matches_head_for_valid_and_invalid_inputs(old_cyclic):
    cases = [((index, count), dict(period_degrees=period))
             for index in range(-40, 41) for count in (1, 2, 3, 7, 8) for period in (180.0, 360.0)]
    cases += [((5, 8), {}), ((np.int64(9), np.int32(8)), {}), ((True, 8), {}), ((1.0, 8), {}),
              ((1, 0), {}), ((1, -3), {}), ((1, 8), dict(period_degrees=90.0)),
              ((1, 8), dict(period_degrees=True)), (("1", 8), {})]
    for args, kwargs in cases:
        new = _outcome(sam_cyclic.address_for_unfolded_index, *args, **kwargs)
        old = _outcome(old_cyclic.address_for_unfolded_index, *args, **kwargs)
        assert new[0] == old[0]
        if new[0] == "value":
            assert _plain_mapping(new[1]) == _plain_mapping(old[1])
            assert type(new[1]) is type(old[1])
        else:
            assert new == old


def test_transform_helper_matches_head_bitwise(old_cyclic):
    rng = np.random.default_rng(20261102)
    width = 27
    for _ in range(300):
        y0, x0 = int(rng.integers(0, 10)), int(rng.integers(0, 20))
        y1, x1 = y0 + int(rng.integers(1, 6)), x0 + int(rng.integers(1, width - x0 + 1))
        dtype = rng.choice([np.bool_, np.uint8])
        mask = rng.integers(0, 2, (y1 - y0, x1 - x0)).astype(dtype)
        if rng.integers(0, 2):
            mask.setflags(write=False)
        native = int(rng.integers(0, 5))
        source = sam_cyclic.address_for_unfolded_index(native + 5 * int(rng.integers(-2, 3)), 5)
        target = sam_cyclic.address_for_unfolded_index(native + 5 * int(rng.integers(-2, 3)), 5)
        new_mask, new_bbox = sam_cyclic.transform_crop_between_frame_addresses(
            mask, (y0, x0, y1, x1), source, target, width)
        old_mask, old_bbox = old_cyclic.transform_crop_between_frame_addresses(
            mask, (y0, x0, y1, x1), source, target, width)
        assert new_bbox == old_bbox and type(new_bbox) is type(old_bbox)
        assert new_mask.dtype == old_mask.dtype and new_mask.strides == old_mask.strides
        assert new_mask.flags.writeable == old_mask.flags.writeable
        assert new_mask.tobytes() == old_mask.tobytes()
        assert np.shares_memory(new_mask, mask) == np.shares_memory(old_mask, mask)
    address = sam_cyclic.address_for_unfolded_index
    bad = [(np.ones((2, 2), bool), (0, 0, 2, 3), address(0, 4), address(4, 4), 9),
           (np.ones((2, 2), bool), (0, 0, 2, 2), address(0, 4), address(5, 4), 9),
           (np.ones((2, 2), bool), (0, 8, 2, 10), address(0, 4), address(4, 4), 9),
           (np.ones((2, 2), bool), (0, 0, 2, 2), address(0, 4), address(4, 4), 0)]
    for args in bad:
        assert (_outcome(sam_cyclic.transform_crop_between_frame_addresses, *args)
                == _outcome(old_cyclic.transform_crop_between_frame_addresses, *args))


def test_validation_and_frame_maps_match_head(old_cyclic):
    for period, aliases, native in itertools.product((180.0, 360.0), (0, 1, 3), (4, 5)):
        recipe = dict(sam_cyclic.build_cyclic_frame_addressing((native, 6, 9), aliases, period_degrees=period))
        assert _plain_mapping(recipe) == _plain_mapping(
            old_cyclic.build_cyclic_frame_addressing((native, 6, 9), aliases, period_degrees=period))
        frames = tuple(range(native + aliases))
        addresses = {frame: dict(sam_cyclic.address_for_unfolded_index(frame, native, period_degrees=period))
                     for frame in frames}
        variants = [recipe, {**recipe, "addresses": addresses},
                    {**recipe, "addresses": {str(k): v for k, v in addresses.items()}}]
        corrupt = {k: dict(v) for k, v in addresses.items()}
        corrupt[frames[-1]]["mirror_u"] = not corrupt[frames[-1]]["mirror_u"]
        variants += [{**recipe, "addresses": corrupt}, {**recipe, "alias_frames": native},
                     {**recipe, "schema": "other"}, {}, {**recipe, "period_degrees": 90.0}]
        for metadata in variants:
            for expected in (None, frames, frames[:-1]):
                new = _outcome(sam_cyclic.validate_cyclic_frame_addressing, metadata, expected_frames=expected)
                old = _outcome(old_cyclic.validate_cyclic_frame_addressing, metadata, expected_frames=expected)
                assert new[0] == old[0]
                if new[0] == "error":
                    assert new == old
                elif isinstance(new[1], sam_cyclic.CyclicFrameAddresses):
                    assert [_plain_mapping(new[1][i]) for i in new[1]] == [_plain_mapping(old[1][i]) for i in old[1]]
                    assert (new[1].native_count, new[1].evidence_count, new[1].period_degrees,
                            new[1].addressing_schema) == (old[1].native_count, old[1].evidence_count,
                            old[1].period_degrees, old[1].addressing_schema)
                else:
                    assert _plain_mapping(new[1]) == _plain_mapping(old[1])
        volume = np.arange(native * 6 * 9, dtype=np.uint8).reshape(native, 6, 9)
        if aliases < native:
            new_volume = sam_cyclic.CyclicObservationVolume(volume, aliases, period_degrees=period)
            old_volume = old_cyclic.CyclicObservationVolume(volume, aliases, period_degrees=period)
            for frame in range(new_volume.shape[0]):
                assert new_volume[frame].tobytes() == old_volume[frame].tobytes()
        new_tail = sam_cyclic.ExtrapolationCyclicObservationVolume(volume, aliases, period_degrees=period)
        old_tail = old_cyclic.ExtrapolationCyclicObservationVolume(volume, aliases, period_degrees=period)
        assert _plain_mapping(new_tail.frame_addressing) == _plain_mapping(old_tail.frame_addressing)
        for frame in range(new_tail.shape[0]):
            assert new_tail[frame].tobytes() == old_tail[frame].tobytes()


def test_selection_and_native_folds_match_head_helpers(tmp_path, monkeypatch, old_cyclic):
    bundle = bundle_at(tmp_path)
    policy = {"sam_bridge_policy": "permissive"}

    def observe():
        receipt = select_sam_proposals(bundle, policy=policy)
        crops = [(g, s, n, b, m.tobytes()) for g, s, n, b, m in iter_selected_native_crops(bundle, receipt)]
        planes = [selected_native_plane(bundle, receipt, frame).tobytes() for frame in range(4)]
        sam_planes = [sam_interpolation.selected_sam_plane(bundle, receipt, frame, (12, 16)).tobytes()
                      for frame in range(4)]
        with bundle.reader() as reader:
            snapshot = reader.filter_snapshot(receipt)
            group = bundle.groups["seam"]
            contact = sam_policy._pair_contact(reader, group, ["forward"], group, ["forward"], 26, snapshot)
        receipt = {key: value for key, value in receipt.items() if key not in ("selection_resources",)}
        return json.dumps(receipt, sort_keys=True, default=str), crops, planes, sam_planes, contact

    new = observe()
    with monkeypatch.context() as patch:
        _swap_in_old(patch, old_cyclic)
        old = observe()
    assert new == old


def _contact_variant(gid, group, masks, run, raw, *, bbox=None, flip=False):
    """A copy of the seam family with distinct endpoint roots (so contact is really evaluated)."""
    from copy import deepcopy
    group = deepcopy(group)
    renamed = {"A": f"{gid}-A", "B-alias": f"{gid}-B-alias"}
    group["group_id"] = gid
    for endpoint in group["endpoints"]:
        endpoint["observation_id"] = renamed[endpoint["observation_id"]]
        endpoint["original_observation_id"] = f"{gid}-{endpoint['original_observation_id']}"
    group["edges"] = [dict(edge_id=f"{gid}-edge", source_id=renamed["A"], target_id=renamed["B-alias"])]
    if bbox is not None:
        group["context_bbox_yx"] = bbox
    copy = (lambda array: array[:, ::-1].copy()) if flip else (lambda array: array.copy())
    renamed_masks = {}
    for key, value in masks.items():
        kind, _, name = key.partition(":")
        renamed_masks[f"{kind}:{renamed.get(name, name)}"] = copy(value)
    masks = renamed_masks
    run = {**deepcopy(run), "run_id": f"{gid}-forward", "group_id": gid,
           "seed_ids": [renamed["A"]], "held_out_ids": [renamed["B-alias"]]}
    return group, masks, run, {frame: copy(value) for frame, value in raw.items()}


@pytest.mark.parametrize("period", [180.0, 360.0])
def test_pair_contact_true_and_false_outcomes_match_head_helpers(tmp_path, monkeypatch, old_cyclic, period):
    """Distinct families that touch (directly, across the seam mirror) and that do not.

    The self-pair in the test above short-circuits on shared endpoint roots, so it
    never reaches the cyclic transform; these pairs do, with both outcomes.
    """
    from tests.test_sam_cyclic_evidence_replay import fixture
    scope, group, masks, run, raw = fixture(period)
    families = [(group, masks, run, raw), _contact_variant("twin", group, masks, run, raw),
                _contact_variant("mirror", group, masks, run, raw,
                                 bbox=sam_cyclic.mirror_bbox_yx(group["context_bbox_yx"], 16), flip=True),
                _contact_variant("far", group, masks, run, raw, bbox=(2, 7, 10, 15))]
    with sam_evidence.SamEvidenceWriter(tmp_path / "evidence", scope) as writer:
        for family in families:
            writer.add_group(family[0], family[1])
            writer.add_run(family[2], family[3])
        bundle = writer.commit()
    receipt = select_sam_proposals(bundle, policy={"sam_bridge_policy": "permissive"})
    runs = {gid: ["forward" if gid == "seam" else f"{gid}-forward"] for gid in bundle.groups}
    transforms = []

    def observe():
        with bundle.reader() as reader:
            snapshot = reader.filter_snapshot(receipt)
            return [sam_policy._pair_contact(reader, bundle.groups[a], runs[a], bundle.groups[b], runs[b],
                                             connectivity, snapshot)
                    for a, b in itertools.product(sorted(bundle.groups), repeat=2) for connectivity in (6, 18, 26)]

    def counting(original):
        def wrapper(*args, **kwargs):
            transforms.append(1)
            return original(*args, **kwargs)
        return wrapper

    with monkeypatch.context() as patch:
        patch.setattr(sam_cyclic, "transform_crop_between_frame_addresses",
                      counting(sam_cyclic.transform_crop_between_frame_addresses))
        new = observe()
    new_transforms, transforms[:] = len(transforms), []
    with monkeypatch.context() as patch:
        _swap_in_old(patch, old_cyclic)
        patch.setattr(sam_cyclic, "transform_crop_between_frame_addresses",
                      counting(old_cyclic.transform_crop_between_frame_addresses))
        old = observe()
    assert new == old
    assert True in new and False in new
    assert new_transforms == len(transforms) > 0
    # Seam/twin touch directly; the mirrored copy touches through the seam fold.
    contact = dict(zip([(a, b, c) for a, b in itertools.product(sorted(bundle.groups), repeat=2)
                        for c in (6, 18, 26)], new))
    assert contact[("seam", "twin", 26)] and contact[("seam", "mirror", 26)]
    assert not contact[("seam", "far", 26)] and not contact[("seam", "seam", 26)]


# --- hot paths perform no source I/O ----------------------------------------------------------

def test_per_pair_and_per_frame_paths_never_touch_the_source_file(tmp_path, monkeypatch):
    bundle = bundle_at(tmp_path)
    reads = _record_source_reads(monkeypatch)
    receipt = select_sam_proposals(bundle, policy={"sam_bridge_policy": "permissive"})
    address = sam_cyclic.address_for_unfolded_index(9, 4)
    sam_cyclic.transform_crop_between_frame_addresses(np.ones((2, 3), bool), (0, 0, 2, 3), address,
                                                      sam_cyclic.address_for_unfolded_index(1, 4), 16)
    group = bundle.groups["seam"]
    sam_cyclic.validate_cyclic_frame_addressing(group["frame_addressing"], expected_frames=group["frame_indices"])
    assert dict(sam_cyclic.CyclicFrameAddresses(4, 6, 180.0)[5])["mirror_u"] is True
    assert native_output_shape_tyx(bundle) == (4, 12, 16)
    evidence_frame_geometry(bundle)
    sam_interpolation._publication_frame_addresses(bundle, group)
    with bundle.reader() as reader:
        snapshot = reader.filter_snapshot(receipt)
        for _ in range(3):
            assert sam_policy._cyclic_group_geometry(reader, group) is not None
            sam_policy._pair_contact(reader, group, ["forward"], group, ["forward"], 26, snapshot)
        for frame in range(4):
            sam_interpolation.selected_sam_plane(reader, receipt, frame, (12, 16),
                                                 _records=(bundle.groups, bundle.runs))
    list(iter_selected_native_crops(bundle, receipt))
    selected_native_plane(bundle, receipt, 0)
    sam_cyclic.build_cyclic_frame_addressing((4, 12, 16), 2)
    sam_cyclic.ExtrapolationCyclicObservationVolume(np.zeros((4, 12, 16), np.uint8), 2)
    SamEvidenceBundle.open(bundle.directory)
    bundle_at(tmp_path / "written")
    merged, _, _, _ = sam_interpolation._publish_directions(
        bundle, receipt, np.zeros((4, 12, 16), np.uint8), tmp_path / "outputs", {}, 1, tmp_path / "merged")
    if isinstance(merged, np.memmap):
        merged._mmap.close()
    assert reads == []

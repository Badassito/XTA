"""Recovery geometry remains anchored to original source pixels and evidence."""
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tools import recover_sam_refused_family as recovery


def test_contract_rebase_preserves_world_pixels_when_crop_moves_and_grows():
    original = np.zeros((2, 6, 8), bool)
    original[0, 1:4, 2:5] = True
    original[1, 3:6, 5:8] = True
    old, new = (10, 20, 16, 28), (8, 17, 18, 30)
    result = recovery.remap_contract(original, old, new)
    assert result.shape == (2, 10, 13)
    assert np.array_equal(result[:, 2:8, 3:11], original)
    assert result.sum() == original.sum()
    assert not result[:, :2].any()


def test_contract_shrink_declares_intersection_without_coordinate_shift():
    original = np.zeros((6, 8), bool)
    original[1:5, 1:7] = True
    result = recovery.remap_contract(original, (10, 20, 16, 28), (12, 23, 18, 30))
    assert np.array_equal(result[:4, :5], original[2:6, 3:8])
    assert not result[4:].any()
    assert not result[:, 5:].any()


@pytest.mark.parametrize("mask", [np.ones((6, 8), np.uint8), np.ones((5, 8), bool)])
def test_bad_geometry_is_rejected(mask):
    with pytest.raises(ValueError, match="shape/dtype"):
        recovery.remap_contract(mask, (10, 20, 16, 28), (8, 17, 18, 30))


def test_sealed_plan_mutation_is_rejected(tmp_path):
    from tools.sam_crop_strategy_geometry import _hash
    value = {"families": [], "source_frame_start": 590}
    value["plan_sha256"] = _hash(value)
    path = tmp_path/"strategy_plan.json"
    path.write_text(json.dumps(value))
    assert recovery._plan(SimpleNamespace(output=tmp_path))["source_frame_start"] == 590
    value["source_frame_start"] = 591
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Sealed"):
        recovery._plan(SimpleNamespace(output=tmp_path))


def test_reference_read_checks_exact_original_and_sdf_bytes(tmp_path):
    original = tmp_path/"original.npy"
    sdf = tmp_path/"sdf.npy"
    np.save(original, np.zeros((3, 12, 14), np.uint8))
    np.save(sdf, np.zeros((3, 12, 14), bool))
    reference = dict(original_observations_file=str(original), original_observations_sha256=recovery._sha(original),
                     selected_additions_file=str(sdf), selected_additions_sha256=recovery._sha(sdf),
                     source_frame_start=100, source_shape_tyx=[3,12,14], evaluation_cache_local=1)
    directory = tmp_path/"sdf_references"/"fixture"
    directory.mkdir(parents=True)
    (directory/"reference.json").write_text(json.dumps(reference))
    args = SimpleNamespace(experiment=tmp_path, dataset="fixture")
    assert recovery._source(args)[1:] == (100, (3,12,14), 1)
    original.write_bytes(original.read_bytes()+b"changed")
    with pytest.raises(ValueError, match="original_observations"):
        recovery._source(args)


def test_bundle_inventory_filters_modes_and_families(tmp_path):
    for mode, family in (("whole","A"), ("whole","B"), ("tiled","A")):
        directory = tmp_path/"bundles"/mode/family
        directory.mkdir(parents=True)
        (directory/"manifest.json").write_text("{}")
    assert list(recovery.selected_bundles(tmp_path, ["A"], ["whole","tiled"])) == [("A","whole"), ("A","tiled")]


@pytest.mark.parametrize("identity", ["../other", "C:/elsewhere", "A/B", ".hidden", "A..B"])
def test_artifact_identities_cannot_escape_the_recovery_directory(identity):
    with pytest.raises(ValueError, match="safe file"):
        recovery.safe_output_key(identity)


def test_plain_artifact_identity_remains_stable():
    assert recovery.safe_output_key("joint_lower_B1") == "joint_lower_B1"


def test_prepare_uses_only_retained_refused_original_seeds():
    groups = {
        "small": dict(group_id="small", complete=False, endpoints=[dict(observation_id="S")], mask_keys={"endpoint_local:S":"s"}),
        "large": dict(group_id="large", complete=False, endpoints=[dict(observation_id="L")], mask_keys={"endpoint_local:L":"l"}),
        "planned": dict(group_id="planned", complete=True, endpoints=[], mask_keys={}),
    }
    class Reader:
        def __enter__(self):
            return self
        def __exit__(self, *_):
            pass
        def group_mask(self, group, _):
            return np.ones((2, 2) if group == "small" else (4, 4), bool)
    bundle = SimpleNamespace(groups=groups, reader=lambda: Reader())
    assert recovery.select_refused_group(bundle)["group_id"] == "large"
    assert recovery.select_refused_group(bundle, "small")["group_id"] == "small"
    with pytest.raises(ValueError, match="No retained"):
        recovery.select_refused_group(bundle, "planned")

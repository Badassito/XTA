"""Single-seed continuation must not depend on a future endpoint or revive."""
import numpy as np
import pytest

from tools.evaluate_sam_extrapolation import (
    _largest_island, plan_terminal_free, replay_run, select_prefix,
)
from tests.test_sam_evidence_policy import build_bundle, fixture_group, fixture_run


def _seed():
    mask = np.zeros((24, 24), bool)
    mask[10:14, 10:14] = True
    return mask


def test_growth_then_shrink_is_preserved_without_endpoint_envelope():
    seed = _seed()
    grown = np.zeros_like(seed)
    grown[7:17, 7:17] = True
    frames = {1: grown, 2: seed}
    selected, receipt = select_prefix(frames, {1: .9, 2: .9}, [0, 1, 2], seed)
    assert list(selected) == [1, 2]
    assert receipt["measurements"][0]["area_vs_seed"] == 6.25
    assert receipt["measurements"][1]["area_vs_previous"] == .16
    assert not receipt["future_endpoint_used_for_acceptance"]


@pytest.mark.parametrize("failure", ["empty", "score", "missing", "unavailable"])
def test_failed_prefix_never_resumes_at_recovered_frame(failure):
    seed = _seed()
    frames = {1: seed, 2: seed.copy(), 3: seed}
    scores = {1: .9, 2: .9, 3: .9}
    statuses = {"1": "observed", "2": "observed", "3": "observed"}
    if failure == "empty":
        frames[2][:] = False
    elif failure == "score":
        scores[2] = .1
    elif failure == "missing":
        del frames[2]
    else:
        statuses["2"] = "unavailable"
    selected, receipt = select_prefix(frames, scores, [0, 1, 2, 3], seed, observation_status=statuses)
    assert list(selected) == [1]
    assert receipt["stop"]["frame"] == 2


def test_largest_island_tie_is_deterministic_and_removes_noise():
    seed = _seed()
    noisy = seed.copy()
    noisy[2, 2] = True
    assert np.array_equal(_largest_island(noisy), seed)
    tied = np.zeros((10, 10), bool)
    tied[1:3, 1:3] = True
    tied[6:8, 6:8] = True
    assert _largest_island(tied)[1, 1]
    assert not _largest_island(tied)[6, 6]


def test_cautious_stops_before_boundary_touch_and_disconnected_motion():
    seed = _seed()
    moved = np.zeros_like(seed)
    moved[10:14, 19:23] = True
    selected, receipt = select_prefix({1: moved}, {1: .9}, [0, 1], seed, motion=2)
    assert not selected and receipt["stop"]["reason"] == "disconnected_motion"
    moved[10:14, 23] = True
    _, receipt = select_prefix({1: moved}, {1: .9}, [0, 1], seed, stop_crop_contact=True)
    assert receipt["stop"]["reason"] == "crop_contact"


def test_terminal_free_plan_is_invariant_to_all_nonseed_predictions():
    volume = np.zeros((15, 64, 80), np.uint8)
    volume[4, 20:32, 20:32] = 1
    volume[10, 24:36, 26:38] = 1
    first = plan_terminal_free(volume, (4, 10), horizon=4, padding=8)
    changed = volume.copy()
    changed[:4] = 1
    changed[5:10] = 1
    changed[11:] = 1
    second = plan_terminal_free(changed, (4, 10), horizon=4, padding=8)
    assert len(first) == len(second) == 2
    for a, b in zip(first, second):
        assert np.array_equal(a.pop("seed_mask"), b.pop("seed_mask"))
        assert a == b
    assert first[0]["frame_start"] == 0 and first[0]["frame_stop"] == 5
    assert first[1]["frame_start"] == 10 and first[1]["frame_stop"] == 15


def test_replay_does_not_use_future_acceptance_or_endpoint_shape(tmp_path):
    group, masks, raw = fixture_group()
    raw[4][:] = False  # Opposite endpoint failure cannot kill a good prefix.
    run = fixture_run("single", group, tracker_scores={str(i): .9 for i in range(5)})
    bundle = build_bundle(tmp_path, [(run, raw)], group=group, masks=masks)
    with bundle.reader() as reader:
        selected, receipt = replay_run(reader, "single", horizon=3)
    assert list(selected) == [1, 2, 3]
    assert receipt["stop"]["reason"] == "horizon"


def test_propagated_seed_frame_can_differ_when_exact_injection_is_verified(tmp_path):
    group, masks, raw = fixture_group()
    raw[0][3, 5] = True
    run = fixture_run("single", group, tracker_scores={str(i): .9 for i in range(5)},
                      runtime_receipt={"adapter_receipt": {
                          "seed_roundtrip_passed": True, "seed_roundtrip_exact": True}})
    bundle = build_bundle(tmp_path, [(run, raw)], group=group, masks=masks)
    with bundle.reader() as reader:
        selected, _ = replay_run(reader, "single", horizon=2)
    assert list(selected) == [1, 2]


@pytest.mark.parametrize("value", [0, 65, True, 1.5])
def test_bad_horizon_is_rejected(value):
    with pytest.raises(ValueError, match="horizon"):
        select_prefix({1: _seed()}, {1: .9}, [0, 1], _seed(), horizon=value)


def test_reverse_addresses_keep_seed_out_of_output_and_clamp_horizon():
    frames = {4: _seed(), 3: _seed(), 2: _seed(), 1: _seed()}
    selected, receipt = select_prefix(frames, {i: .9 for i in frames}, [4, 3, 2, 1], _seed(), horizon=2)
    assert list(selected) == [3, 2]
    assert receipt["stop"] == {"reason": "horizon", "frame": 1}


def test_out_of_order_addresses_and_invalid_scores_are_rejected():
    with pytest.raises(ValueError, match="contiguous"):
        select_prefix({1: _seed()}, {1: .9}, [0, 1, 0], _seed())
    with pytest.raises(ValueError, match="finite probability"):
        select_prefix({1: _seed()}, {1: float("nan")}, [0, 1], _seed())


def test_live_integer_removed_status_stops_even_unfiltered_raw_variant():
    seed = _seed()
    selected, receipt = select_prefix({1: seed, 2: seed, 3: seed}, {1: .9, 2: None, 3: .9},
                            [0, 1, 2, 3], seed, min_score=0., largest_island=False,
                            observation_status={1: "observed", 2: "removed", 3: "observed"})
    assert list(selected) == [1]
    assert receipt["stop"] == {"reason": "unavailable_observation", "frame": 2}

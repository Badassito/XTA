import numpy as np
import pytest

from XTA.lta_interval_execution import DirectionalHandoff
from XTA.lta_tracklets import blend_directional_masks
from XTA.lta_union_artifacts import LtaUnionWriter
from XTA.lta_windows import plan_domain_windows, plan_neighbor_anchor_domains


def test_positive_and_explicit_empty_neighbors_stop_each_tracker():
    domains = plan_neighbor_anchor_domains([9, 39, 47, 83, 84, 89], 126, background_frames=[45, 81, 125])
    assert [(anchor, domain.frame_start, domain.frame_stop) for anchor, domain in domains.items()] == [
        (9, 9, 39), (39, 10, 45), (47, 46, 81), (83, 82, 84), (84, 84, 89), (89, 85, 125),
    ]
    for domain in domains.values():
        windows = plan_domain_windows(domain)
        assert min(window.frame_start for window in windows) == domain.frame_start
        assert max(window.frame_stop for window in windows) == domain.frame_stop
        assert all(window.frame_count <= 30 for window in windows)
    single = plan_neighbor_anchor_domains([64], 126)
    assert single[64].frame_start == 0 and single[64].frame_stop == 126
    outer = plan_neighbor_anchor_domains([9, 39], 126)
    assert [(domain.frame_start, domain.frame_stop) for domain in outer.values()] == [(9, 39), (10, 40)]
    with pytest.raises(ValueError, match="both foreground and background"):
        plan_neighbor_anchor_domains([64], 126, background_frames=[64])


def test_centered_distances_translate_one_branch_without_adding_a_duplicate():
    forward = np.zeros((42, 65), dtype=np.uint8)
    backward = np.zeros_like(forward)
    forward[10:21, 10:21] = 1
    backward[10:21, 20:31] = 1
    # A distant, unmatched branch remains verbatim.
    backward[28:31, 44:47] = 1
    expected = np.zeros_like(forward)
    expected[10:21, 15:26] = 1
    expected[28:31, 44:47] = 1
    assert np.array_equal(blend_directional_masks(forward, backward, 0.5), expected)
    assert np.array_equal(blend_directional_masks(forward, backward * 0, 0.5), forward)


def test_hollow_component_center_is_exact_across_bounded_bands(monkeypatch):
    from XTA.lta_tracklets import _centered_component

    labels = np.ones((550, 600), dtype=np.uint8)
    labels[125:426, 150:451] = 0
    y, x = np.nonzero(labels)
    cx, cy = x.mean(), y.mean()
    nearest = np.argmin((y - cy) ** 2 + (x - cx) ** 2)
    original_nonzero = np.nonzero

    def bounded_nonzero(band):
        assert band.size <= 65536
        return original_nonzero(band)

    monkeypatch.setattr(np, "nonzero", bounded_nonzero)
    stats = [(0,) * 5, (0, 0, 600, 550, len(y))]
    _, _, center = _centered_component(labels, stats, np.array([(0, 0), (cx, cy)]), 1)
    assert center == (y[nearest], x[nearest])


def _write_union(tmp_path, name, start, mask):
    with LtaUnionWriter(tmp_path / name, shape=mask.shape, frame_start=start) as writer:
        writer.append_chunk(start, mask)
        return writer.finish()


@pytest.mark.parametrize("reverse_arrival", [False, True])
def test_interval_reduction_combines_byte_unaligned_tiles_and_missing_direction(tmp_path, reverse_arrival):
    domains = plan_neighbor_anchor_domains([1, 5, 9], 11, background_frames=[7])
    union = np.zeros((11, 11, 19), dtype=np.uint8)
    mask = np.zeros((3, 6, 7), dtype=np.uint8)
    mask[0:2, 1:4, 1:4] = 1
    forward = _write_union(tmp_path, "forward.bits", 2, mask)
    mask[:] = 0
    mask[0, 1:4, 2:5] = 1
    mask[2, 3:6, 0:3] = 1
    backward = _write_union(tmp_path, "backward.bits", 2, mask)
    artifacts = [(forward, 1, [(2, 4)]), (backward, 5, [(2, 3), (4, 5)])]
    observed = []
    with DirectionalHandoff(tmp_path, union, domains) as handoff:
        assert set(handoff.frames) == {2, 3, 4}
        for descriptor, anchor, visited in artifacts[::(-1 if reverse_arrival else 1)]:
            handoff.reduce(
                descriptor, tile_xyxy=(3, 2, 10, 8), frame_start=2, frame_stop=5, anchor_frame=anchor,
                windows=({"adapter": {"model_visited_frame_ranges": visited}},),
            )
        handoff.compose(observe=lambda *counts: observed.append(counts))
    # Frame 2 has both observations, frame 3 only forward, and frame 4 only backward.
    assert observed == [(2, 6, 12)]
    assert np.count_nonzero(union[2]) == 9
    assert np.array_equal(np.argwhere(union[3]), np.argwhere(np.pad(
        np.ones((3, 3), np.uint8), ((3, 5), (4, 12)),
    )))
    assert np.count_nonzero(union[4, 5:8, 3:6]) == 9
    assert np.count_nonzero(union) == 27
    assert not (tmp_path / "opposing_interval_masks.bits").exists()


def test_observed_empty_side_counts_but_an_unvisited_side_does_not(tmp_path):
    domains = plan_neighbor_anchor_domains([1, 4], 6)
    union = np.zeros((6, 5, 11), dtype=np.uint8)
    forward = _write_union(tmp_path, "forward.bits", 2, np.ones((2, 5, 11), dtype=np.uint8))
    backward = _write_union(tmp_path, "backward.bits", 2, np.zeros((2, 5, 11), dtype=np.uint8))
    observed = []
    with DirectionalHandoff(tmp_path, union, domains) as handoff:
        with pytest.raises(RuntimeError, match="crossed"):
            handoff.reduce(
                forward, tile_xyxy=(0, 0, 11, 5), frame_start=2, frame_stop=5, anchor_frame=1, windows=(),
            )
        assert not np.any(union)
        handoff.reduce(forward, tile_xyxy=(0, 0, 11, 5), frame_start=2, frame_stop=4, anchor_frame=1, windows=())
        handoff.reduce(
            backward, tile_xyxy=(0, 0, 11, 5), frame_start=2, frame_stop=4, anchor_frame=4,
            windows=({"adapter": {"model_visited_frame_ranges": [(2, 3)]}},),
        )
        handoff.compose(observe=lambda *counts: observed.append(counts))
    assert observed == [(2, 55, 55)]
    assert np.count_nonzero(union) == 110


@pytest.mark.parametrize(
    "left_bounds,right_bounds,expected_pixels,expected_disagreement",
    [
        ((0, 9), (9, 24), 48, []),
        ((0, 11), (7, 24), 48, [(2, 16, 16)]),
        ((0, 20), (2, 24), 24, [(2, 48, 48)]),
    ],
)
def test_handoff_respects_joint_spatial_observation(
    tmp_path, left_bounds, right_bounds, expected_pixels, expected_disagreement,
):
    union = np.zeros((5, 12, 24), dtype=np.uint8)
    domains = plan_neighbor_anchor_domains([1, 3], len(union))
    observed = []
    with DirectionalHandoff(tmp_path, union, domains) as handoff:
        for anchor, (first, stop), x in ((1, left_bounds, 5), (3, right_bounds, 10)):
            masks = np.zeros((1, 12, stop - first), dtype=np.uint8)
            masks[0, 2:10, x - first:x - first + 3] = 1
            descriptor = _write_union(tmp_path, f"anchor-{anchor}.bits", 2, masks)
            handoff.reduce(
                descriptor, tile_xyxy=(first, 0, stop, 12), frame_start=2, frame_stop=3,
                anchor_frame=anchor, windows=({"adapter": {"model_visited_frame_ranges": [(2, 3)]}},),
            )
        handoff.compose(observe=lambda *counts: observed.append(counts))
    assert np.count_nonzero(union[2]) == expected_pixels
    assert observed == expected_disagreement
    if expected_pixels == 48:
        assert np.all(union[2, 2:10, 5:8])
        assert np.all(union[2, 2:10, 10:13])
        assert not np.any(union[2, :, 8:10])


def test_relay_keeps_injected_prompt_foreground_without_claiming_background(tmp_path):
    union = np.zeros((5, 8, 14), dtype=np.uint8)
    domains = plan_neighbor_anchor_domains([1, 3], len(union))
    observed = []
    with DirectionalHandoff(tmp_path, union, domains) as handoff:
        for anchor, x, prompt in ((1, 4, 1), (3, 7, 2)):
            mask = np.zeros((1, 8, 14), dtype=np.uint8)
            mask[0, 2:6, x:x + 2] = 1
            descriptor = _write_union(tmp_path, f"seed-{anchor}.bits", 2, mask)
            handoff.reduce(
                descriptor, tile_xyxy=(0, 0, 14, 8), frame_start=2, frame_stop=3,
                anchor_frame=anchor,
                windows=({"window": {"prompt_frame": prompt},
                          "adapter": {"model_visited_frame_ranges": [(2, 3)]}},),
            )
        handoff.compose(observe=lambda *counts: observed.append(counts))
    assert np.count_nonzero(union[2]) == 16
    assert not observed

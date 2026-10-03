from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest

from tools.qualify_sam_tiled_schedule import TRIALS, compare_trials, legacy_parent_batches


def test_legacy_override_changes_only_order_inside_current_parent_cohorts():
    prepared = SimpleNamespace(crop_mode="tiled", tracker_jobs=tuple(
        SimpleNamespace(original_run_index=index // 3) for index in range(12)))
    current = ((0, 3, 1, 4, 2, 5), (6, 9, 7, 10, 8, 11))
    result = legacy_parent_batches(prepared, current, 1)
    assert result == ((0, 1, 2, 3, 4, 5), (6, 7, 8, 9, 10, 11))
    assert [set(batch) for batch in current] == [set(batch) for batch in result]
    assert current == ((0, 3, 1, 4, 2, 5), (6, 9, 7, 10, 8, 11))
    with pytest.raises(ValueError, match="exactly one worker"):
        legacy_parent_batches(prepared, current, 2)


def _matrix():
    content = dict(masks={"raw": {"decoded_sha256": "raw-content"},
                          "candidate": {"decoded_sha256": "candidate-content"}},
                   children={"parent/tile": {"tracker_scores": {"0": 0.75}}},
                   parents={"parent": {"seed_ids": ["original"]}})
    result = dict(scopes={"parent": dict(content=content, counters={"cache_hits": 0})},
                  selected_parent_additions={"shape": [11, 100, 120], "decoded_sha256": "selected"})
    return [dict(slot=slot, schedule=schedule, result=copy.deepcopy(result))
            for slot, schedule, _repeat in TRIALS]


def test_abba_comparison_ignores_performance_counters_but_rejects_changed_raw_or_scores():
    trials = _matrix()
    trials[1]["result"]["scopes"]["parent"]["counters"]["cache_hits"] = 33
    assert compare_trials(trials)["exact_all_child_raw_owned_candidate_availability_masks"]
    trials[2]["result"]["scopes"]["parent"]["content"]["children"]["parent/tile"]["tracker_scores"]["0"] = 0.76
    with pytest.raises(AssertionError, match="scores"):
        compare_trials(trials)
    trials = _matrix()
    trials[3]["result"]["scopes"]["parent"]["content"]["masks"]["candidate"]["decoded_sha256"] = "changed"
    with pytest.raises(AssertionError, match="masks"):
        compare_trials(trials)


def test_abba_requires_complete_slots_and_selected_output_identity():
    with pytest.raises(ValueError, match="complete A1 B1 B2 A2"):
        compare_trials(_matrix()[:3])
    trials = _matrix()
    trials[-1]["result"]["selected_parent_additions"]["decoded_sha256"] = "changed-output"
    with pytest.raises(AssertionError, match="selected parent"):
        compare_trials(trials)

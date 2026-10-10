"""Offline sample layout and ownership guards; no decode, large arrays or GPU."""
import math
import numpy as np
import pytest

from tools.prepare_gray_cache_experiment import (consecutive_window, histogram_stats,
                                                owned_task_path, sample_specs)


@pytest.mark.parametrize('fraction', (.35, .70))
def test_windows_keep_true_adjacent_frames_and_stay_inside_view(fraction):
    frames = consecutive_window(41, fraction)
    assert len(frames) == 16 and frames == tuple(range(frames[0], frames[0]+16))
    assert 0 <= frames[0] < frames[-1] < 41


def test_no_short_window_or_fake_adjacency():
    with pytest.raises(ValueError):
        consecutive_window(15, .35)
    with pytest.raises(ValueError):
        consecutive_window(16, 0.)
    assert consecutive_window(16, .70) == tuple(range(16))


def test_real_view_recipe_selects_two_samples_per_priority_family():
    specs = sample_specs((63, 65, 67), size=32)
    assert len(specs) == 8
    assert sorted(view.family for _, view, *_ in specs) == sorted(
        ['spherical', 'spherical', 'radial', 'radial', 'azimuthal', 'azimuthal', 'tilted', 'tilted'])
    assert {view.tilt_direction for _, view, *_ in specs if view.family == 'tilted'} == {'vertical', 'horizontal'}
    assert any(view.spherical_tilted_source for _, view, *_ in specs if view.family == 'spherical')
    assert all(view.sampling_policy == 'coverage' for _, view, *_ in specs
               if view.family in {'spherical', 'radial'})
    from XTA.backprojection import azimuthal_full_coverage_angle_deg
    from XTA.geometry import azimuthal_target_diameter
    for _, view, *_ in specs:
        if view.family == 'azimuthal':
            assert math.isclose(view.azimuths_deg[1]-view.azimuths_deg[0],
                azimuthal_full_coverage_angle_deg(azimuthal_target_diameter(
                    view.azimuthal_request_token, 63, 65, 67)))


def test_histogram_metrics_keep_padding_bias_visible():
    image = np.array([0, 0, 20, 100, 255], np.uint8)
    result = histogram_stats(np.bincount(image, minlength=256))
    assert result['zero_fraction'] == .4 and result['full_fraction'] == .2
    assert result['pixels'] == image.size
    assert result['mean'] == float(image.mean())
    assert math.isclose(result['std'], float(image.std()))


def test_actual_native_f3_curved_recipes_use_coverage_without_source_allocation():
    specs = sample_specs((1931, 3064, 3022))
    for _, view, _, frames in specs:
        assert len(frames) == 16 and frames == tuple(range(frames[0], frames[0]+16))
        if view.family in {'spherical', 'radial'}:
            assert view.sampling_policy == 'coverage' and view.sampling_certificate


def test_native_retirement_refuses_unowned_or_outside_files(tmp_path):
    own = tmp_path/'native-source-owned.gray8.dat'
    own.write_bytes(b'0')
    assert owned_task_path(own, tmp_path) == own.resolve()
    foreign = tmp_path/'sample.npy'
    foreign.write_bytes(b'0')
    with pytest.raises(RuntimeError):
        owned_task_path(foreign, tmp_path)
    child = tmp_path/'child'
    child.mkdir()
    outside = child/'native-source-other.gray8.dat'
    outside.write_bytes(b'0')
    with pytest.raises(RuntimeError):
        owned_task_path(outside, tmp_path)

"""The SAM crop environment is launch-scoped and does not alter detector grids."""
from dataclasses import asdict
import os
from unittest import mock

import pytest

from XTA import cli
from XTA.config import (activate_sam_crop_mode, build_argparser, resolve_backend_devices,
    resolve_backend_models, resolve_interpolation_settings, resolve_sam_crop_mode)


def resolve(arguments=()):
    args = build_argparser().parse_args(['--input', 'source.mkv', '--device', '0',
        '--model', 'gpu:detector.engine', 'sam:bundle', '--interpolation_backend', 'sam', *arguments])
    return resolve_interpolation_settings(args, resolve_backend_models(args.model),
        resolve_backend_devices(args.device))


def test_crop_mode_defaults_to_whole_and_has_no_cli_selector(monkeypatch):
    monkeypatch.delenv('YOLO_TTA_SAM_CROP_MODE', raising=False)
    assert resolve().sam_crop_mode == 'whole'
    assert '--sam_crop_mode' not in build_argparser().format_help()
    with pytest.raises(SystemExit):
        resolve(['--sam_crop_mode', 'tiled'])


@pytest.mark.parametrize('mode', ('whole', 'tiled', ' TiLeD '))
def test_environment_only_mode_is_normalized_and_retained(monkeypatch, mode):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', mode)
    settings = resolve()
    assert settings.sam_crop_mode == mode.strip().lower()
    assert asdict(settings)['sam_crop_mode'] == settings.sam_crop_mode


@pytest.mark.parametrize('mode', ('both', '', '1260', 'native', 'outer'))
def test_invalid_active_mode_fails_before_runtime(monkeypatch, mode):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', mode)
    with mock.patch('XTA.tta_mode.run') as runtime:
        with pytest.raises(SystemExit):
            cli._run_tta(['--input', 'source.mkv', '--device', '0', '--model',
                'gpu:detector.engine', 'sam:bundle', '--interpolation_backend', 'sam'])
        runtime.assert_not_called()


@pytest.mark.parametrize('arguments', (['--interpolation_backend', 'sdf'],
    ['--interpolation_distance', '0']))
def test_inactive_sam_does_not_read_or_validate_unused_crop_environment(monkeypatch, arguments):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'unused-invalid-value')
    with mock.patch('XTA.config.resolve_sam_crop_mode', side_effect=AssertionError('unused env read')):
        settings = resolve(arguments)
    assert settings.sam_crop_mode is None
    assert settings.sam_model is None
    assert settings.sam_devices == ()


def test_snapshot_survives_environment_change_and_restores_next_run(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'tiled')
    selected = resolve().sam_crop_mode
    with activate_sam_crop_mode(selected):
        monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
        assert resolve().sam_crop_mode == 'tiled'
    assert resolve().sam_crop_mode == 'whole'


def test_cli_preflight_and_runtime_share_one_environment_snapshot(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'tiled')
    observed = []
    def run():
        os.environ['YOLO_TTA_SAM_CROP_MODE'] = 'whole'
        observed.append(resolve().sam_crop_mode)
    with mock.patch('XTA.tta_mode.run', side_effect=run):
        cli._run_tta(['--input', 'source.mkv', '--device', '0', '--model',
            'gpu:detector.engine', 'sam:bundle', '--interpolation_backend', 'sam'])
    assert observed == ['tiled']
    assert resolve_sam_crop_mode() == 'whole'


def test_nested_explicit_snapshot_validates_inner_mode_and_restores_outer(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    with activate_sam_crop_mode('whole'):
        assert resolve_sam_crop_mode() == 'whole'
        with activate_sam_crop_mode('tiled'):
            assert resolve_sam_crop_mode() == 'tiled'
        assert resolve_sam_crop_mode() == 'whole'
        with pytest.raises(ValueError):
            with activate_sam_crop_mode('invalid'):
                pass
        assert resolve_sam_crop_mode() == 'whole'


def test_direct_generator_consumes_launch_snapshot_and_explicit_mode(monkeypatch):
    import numpy as np
    from XTA.sam_interpolation import prepare_sam_interpolation_pass
    from XTA.sam_crop_tiling import resolve_sam_crop_mode as resolve_generation_mode
    observed = np.zeros((3, 9, 13), np.uint8)
    observed[0, 4, 6] = observed[2, 4, 6] = 1
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
    with activate_sam_crop_mode('tiled'):
        assert resolve_generation_mode() == 'tiled'
        assert resolve_generation_mode('whole') == 'whole'
        prepared = prepare_sam_interpolation_pass(observed, scope='native-mode-fixture', min_radius=0)
        assert prepared.crop_mode == 'tiled'
        explicit = prepare_sam_interpolation_pass(observed, scope='native-mode-fixture',
            min_radius=0, crop_mode='whole')
        assert explicit.crop_mode == 'whole'
        assert prepared.settings_sha256 != explicit.settings_sha256


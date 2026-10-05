from dataclasses import asdict
from unittest import mock

import pytest

from XTA import cli
from XTA.config import (activate_sam_adaptive_crop, build_argparser,
    resolve_backend_devices, resolve_backend_models, resolve_interpolation_settings,
    resolve_sam_adaptive_crop)

ENV = 'YOLO_TTA_SAM_ADAPTIVE_CROP'


def _settings(*extra):
    args = build_argparser().parse_args(['--input', 'source.mkv', '--device', '0',
        '--model', 'gpu:detector.engine', 'sam:bundle', *extra])
    return resolve_interpolation_settings(args, resolve_backend_models(args.model), resolve_backend_devices(args.device))


def test_defaults_are_off_and_recordable_without_new_public_cli(monkeypatch):
    monkeypatch.delenv(ENV, raising=False)
    settings = _settings('--interpolation_backend', 'sam')
    assert settings.sam_adaptive_crop is False
    assert asdict(settings)['sam_adaptive_crop'] is False
    assert '--sam_adaptive_crop' not in build_argparser().format_help()


@pytest.mark.parametrize('value,expected', [('0', False), ('1', True), (' 1 ', True)])
@pytest.mark.parametrize('options', [('--interpolation_backend', 'sam'),
    ('--interpolation_distance', '0', '--extrapolation_distance', '4')])
def test_active_interpolation_and_extrapolation_share_switch(monkeypatch, value, expected, options):
    monkeypatch.setenv(ENV, value)
    assert _settings(*options).sam_adaptive_crop is expected


@pytest.mark.parametrize('value', ['', '2', 'true', 'off', 'auto'])
def test_bad_active_env_fails_before_runtime(monkeypatch, value):
    monkeypatch.setenv(ENV, value)
    with mock.patch('XTA.tta_mode.run') as runtime:
        with pytest.raises(SystemExit):
            cli._run_tta(['--input', 'source.mkv', '--device', '0', '--model',
                'gpu:detector.engine', 'sam:bundle', '--extrapolation_distance', '4'])
        runtime.assert_not_called()


def test_inactive_sam_ignores_bad_unused_switch(monkeypatch):
    monkeypatch.setenv(ENV, 'unused-invalid')
    with mock.patch('XTA.config.resolve_sam_adaptive_crop', side_effect=AssertionError('unused env')):
        assert _settings().sam_adaptive_crop is None


@pytest.mark.parametrize('original', [None, '0', '1'])
def test_launch_pins_even_the_default_off_through_env_mutation(monkeypatch, original):
    if original is None:
        monkeypatch.delenv(ENV, raising=False)
    else:
        monkeypatch.setenv(ENV, original)
    observed = []
    def runtime():
        monkeypatch.setenv(ENV, 'invalid-after-launch')
        observed.append(resolve_sam_adaptive_crop())
    with mock.patch('XTA.tta_mode.run', side_effect=runtime):
        cli._run_tta(['--input', 'source.mkv', '--device', '0', '--model',
            'gpu:detector.engine', 'sam:bundle', '--interpolation_backend', 'sam'])
    assert observed == [original == '1']


def test_nested_snapshots_restore_original_validated_value(monkeypatch):
    monkeypatch.setenv(ENV, '1')
    with activate_sam_adaptive_crop(False):
        assert resolve_sam_adaptive_crop() is False
        with activate_sam_adaptive_crop(True):
            assert resolve_sam_adaptive_crop() is True
        assert resolve_sam_adaptive_crop() is False
    assert resolve_sam_adaptive_crop() is True

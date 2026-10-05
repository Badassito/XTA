"""Extrapolation activates SAM independently of interpolation selection."""
from dataclasses import asdict
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import cli
from XTA.config import (build_argparser, resolve_backend_devices, resolve_backend_models,
    resolve_interpolation_settings, resolve_sam_crop_mode, resolve_sam_tight_crop_guard)


def _resolve(*extra, models=('gpu:detector.engine', 'sam:bundle'), devices=('0',)):
    arguments = ['--input', 'source.mkv', '--model', *models, '--device', *devices, *extra]
    args = build_argparser().parse_args(arguments)
    settings = resolve_interpolation_settings(args, resolve_backend_models(args.model), resolve_backend_devices(args.device))
    return args, settings


def test_extrapolation_defaults_disable_without_changing_interpolation():
    args, settings = _resolve(models=('gpu:detector.engine',))
    assert (args.extrapolation_distance, args.extrapolation_walk_back, args.extrapolation_min_radius) == (0, 1, 3.)
    assert settings.enabled and settings.backend == 'sdf'
    assert not settings.extrapolation_enabled and not settings.sam_enabled
    assert settings.sam_model is None and settings.sam_devices == ()
    assert asdict(settings)['extrapolation_distance'] == 0


@pytest.mark.parametrize('backend', ['sdf', 'sam'])
@pytest.mark.parametrize('interpolation_distance', [0, 5])
@pytest.mark.parametrize('extrapolation_distance', [0, 7])
def test_independent_activation_matrix(backend, interpolation_distance, extrapolation_distance):
    _, settings = _resolve('--interpolation_backend', backend, '--interpolation_distance', str(interpolation_distance),
        '--extrapolation_distance', str(extrapolation_distance), '--extrapolation_walk_back', '0',
        '--extrapolation_min_radius', '1.25')
    needed = extrapolation_distance > 0 or (backend == 'sam' and interpolation_distance > 0)
    assert settings.backend == backend
    assert settings.enabled is (interpolation_distance > 0)
    assert settings.extrapolation_enabled is (extrapolation_distance > 0)
    assert settings.sam_enabled is needed
    assert settings.sam_model == ('bundle' if needed else None)
    assert settings.sam_devices == (('cuda:0',) if needed else ())
    assert settings.extrapolation_walk_back == 0 and settings.extrapolation_min_radius == 1.25


@pytest.mark.parametrize('option,value', [('--extrapolation_distance', '-1'), ('--extrapolation_distance', '1.5'),
    ('--extrapolation_walk_back', '-1'), ('--extrapolation_walk_back', '1.5'),
    ('--extrapolation_min_radius', '-.1'), ('--extrapolation_min_radius', 'nan'),
    ('--extrapolation_min_radius', 'inf'), ('--extrapolation_min_radius', '-inf')])
def test_invalid_extrapolation_flags_fail_at_static_parser(option, value):
    with pytest.raises(SystemExit):
        _resolve(option+'='+value)


@pytest.mark.parametrize('backend,distance', [('sdf', 5), ('sdf', 0), ('sam', 0)])
def test_extrapolation_requires_sam_bundle_even_when_interpolation_does_not(backend, distance):
    with pytest.raises(ValueError, match='extrapolation_distance.*sam:PATH'):
        _resolve('--interpolation_backend', backend, '--interpolation_distance', str(distance),
            '--extrapolation_distance', '2', models=('gpu:detector.engine',))


def test_cpu_detector_needs_explicit_sam_pool_for_extrapolation():
    with pytest.raises(ValueError, match='CPU-only detector requires --sam_device'):
        _resolve('--extrapolation_distance', '2', models=('cpu:detector.xml', 'sam:bundle'), devices=('cpu',))
    _, settings = _resolve('--extrapolation_distance', '2', '--sam_device', '2,0',
        models=('cpu:detector.xml', 'sam:bundle'), devices=('cpu',))
    assert settings.sam_devices == ('cuda:2', 'cuda:0')
    assert settings.backend == 'sdf' and settings.sam_enabled


def test_extrapolation_only_ignores_unused_interpolation_guard(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_TIGHT_CROP_GUARD', 'unused-invalid')
    with mock.patch('XTA.config.resolve_sam_tight_crop_guard', side_effect=AssertionError('unused guard read')):
        _, settings = _resolve('--interpolation_distance', '0', '--extrapolation_distance', '4')
    assert settings.sam_enabled and settings.sam_tight_crop_guard is None


def test_disabled_sam_work_ignores_unused_crop_and_guard_environments(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'unused-invalid')
    monkeypatch.setenv('YOLO_TTA_SAM_TIGHT_CROP_GUARD', 'unused-invalid')
    _, settings = _resolve('--interpolation_distance', '0', '--extrapolation_distance', '0',
        models=('cpu:detector.xml',), devices=('cpu',))
    assert not settings.sam_enabled
    assert settings.sam_crop_mode is None and settings.sam_tight_crop_guard is None


def test_real_cli_rejects_missing_extrapolation_bundle_before_runtime():
    with mock.patch('XTA.tta_mode.run') as runtime:
        with pytest.raises(SystemExit):
            cli._run_tta(['--input', 'source.mkv', '--device', '0', '--model', 'gpu:detector.engine',
                '--interpolation_distance', '0', '--extrapolation_distance', '2'])
        runtime.assert_not_called()


def test_real_cli_pins_shared_crop_and_skips_unused_guard_for_extrapolation(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'tiled')
    monkeypatch.setenv('YOLO_TTA_SAM_TIGHT_CROP_GUARD', 'unused-invalid')
    observed = []
    def runtime():
        monkeypatch.setenv('YOLO_TTA_SAM_CROP_MODE', 'whole')
        observed.extend([resolve_sam_crop_mode(), resolve_sam_tight_crop_guard()])
    with mock.patch('XTA.tta_mode.run', side_effect=runtime):
        cli._run_tta(['--input', 'source.mkv', '--device', '0', '--model', 'gpu:detector.engine', 'sam:bundle',
            '--interpolation_distance', '0', '--extrapolation_distance', '2'])
    assert observed == ['tiled', None]


@pytest.mark.parametrize('updates', [dict(extrapolation_distance=True), dict(extrapolation_distance=1.5),
    dict(extrapolation_walk_back=-1), dict(extrapolation_min_radius=float('nan')), dict(extrapolation_min_radius=True)])
def test_direct_resolution_does_not_truncate_invalid_extrapolation_settings(updates):
    args, _ = _resolve()
    arguments = SimpleNamespace(**{**vars(args), **updates})
    with pytest.raises(ValueError, match='extrapolation_'):
        resolve_interpolation_settings(arguments, resolve_backend_models(args.model), resolve_backend_devices(args.device))


def test_option_help_distinguishes_terminal_gate_and_has_no_extrapolation_backend():
    parser = build_argparser()
    help_text = parser.format_help()
    assert '--extrapolation_backend' not in help_text
    actions = {action.dest: action for action in parser._actions}
    assert 'terminal seeds' in actions['extrapolation_min_radius'].help
    assert 'not radius-filtered' in actions['extrapolation_min_radius'].help
    assert 'raw SAM emptiness' in actions['extrapolation_distance'].help

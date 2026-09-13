"""Portable CLI contracts for the CUDA probe; these do not qualify GPU execution."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys

import pytest

from XTA.config import build_argparser, resolve_channel_format
from tools import tta_augmentation_smoke as smoke

ROOT = Path(__file__).resolve().parents[1]


def test_probe_uses_main_cli_option_names():
    production_flags = build_argparser()._option_string_actions
    for flag in ('--device', '--channel_format', '--imgsz', '--batch'):
        assert flag in production_flags
    args = smoke.parse_args([
        '--device', '2', '--channel_format', 'C5S2', '--imgsz', '128', '--batch', '6',
    ])
    assert args.device == 2
    assert args.imgsz == 128
    assert args.batch == 6
    assert args.channel_format == resolve_channel_format('C5S2')


def test_probe_defaults_preserve_r1_dimensions():
    args = smoke.parse_args([])
    assert (args.device, args.imgsz, args.batch) == (0, 256, 4)
    assert args.channel_format.token == 'RGB'
    assert args.channel_format.channel_count == 3
    assert args.profiles == ['light', 'baseline', 'heavy', 'superheavy']
    assert not args.compile
    assert args.report == Path('tta_augmentation_cuda_report.json')


@pytest.mark.parametrize('token', [
    'grey', 'gray', 'GREY', 'RGB', 'rgb', 'C1S1', 'C1S3', 'C3S1', 'C5S2', 'c7s4',
])
def test_channel_tokens_match_production(token):
    main_args = build_argparser().parse_args([
        '--input', 'fixture.mkv', '--model', 'gpu:fixture.engine',
        '--device', '0', '--channel_format', token,
    ])
    args = smoke.parse_args(['--channel_format', token])
    assert args.channel_format == resolve_channel_format(main_args.channel_format)


@pytest.mark.parametrize('token', ['3', 'C2S1', 'C0S1', 'C5S0', 'C5S-1', 'C5', 'CMYK'])
def test_invalid_channel_formats_fail_at_parse_time(token, capsys):
    with pytest.raises(SystemExit) as exc:
        smoke.parse_args(['--channel_format', token])
    assert exc.value.code == 2
    assert '--channel_format' in capsys.readouterr().err


@pytest.mark.parametrize('value', ['-1', 'cuda:0', 'cpu', '0,1', '0:cpu'])
def test_device_requires_one_nonnegative_logical_index(value, capsys):
    with pytest.raises(SystemExit) as exc:
        smoke.parse_args(['--device', value])
    assert exc.value.code == 2
    assert '--device' in capsys.readouterr().err


@pytest.mark.parametrize('argv', [
    ['--size', '256'], ['--channels', '3'], ['--channel', 'RGB'],
    ['--imgsz', '15'], ['--imgsz', '0'], ['--batch', '1'],
])
def test_obsolete_or_invalid_arguments_are_rejected(argv):
    with pytest.raises(SystemExit) as exc:
        smoke.parse_args(argv)
    assert exc.value.code == 2


def test_minimum_probe_dimensions():
    args = smoke.parse_args(['--imgsz', '16', '--batch', '2', '--channel_format', 'grey'])
    assert (args.imgsz, args.batch, args.channel_format.channel_count) == (16, 2, 1)


def test_help_and_parsing_do_not_import_inference_dependencies():
    code = """
import sys
initial_modules = set(sys.modules)
from tools.tta_augmentation_smoke import parse_args
try:
    parse_args(['--help'])
except SystemExit as exc:
    assert exc.code == 0
parse_args(['--device', '0', '--channel_format', 'C5S2', '--imgsz', '128'])
assert not {'torch', 'numpy', 'cv2', 'ultralytics'}.intersection(set(sys.modules) - initial_modules)
"""
    result = subprocess.run(
        [sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True,
        check=True, timeout=30,
    )
    assert '--channel_format' in result.stdout
    assert '--imgsz' in result.stdout
    assert '--channels' not in result.stdout
    assert '--size' not in result.stdout


@pytest.mark.parametrize('token', ['grey', 'RGB', 'C5S2'])
def test_no_cuda_report_records_resolved_cli_settings(token, tmp_path, monkeypatch):
    import torch

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: False)
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE', '0')
    report_path = tmp_path / 'reports' / f'{token}.json'
    code = smoke.main([
        '--device', '0', '--channel_format', token, '--imgsz', '128', '--batch', '2',
        '--report', str(report_path),
    ])
    report = json.loads(report_path.read_text())
    layout = resolve_channel_format(token)
    assert code == 2  # Unavailable, not a passing GPU test.
    assert report['status'] == 'not_run'
    assert report['profiles'] == []
    assert report['cuda_available'] is False
    assert report['device'] == 0
    assert report['imgsz'] == 128
    assert report['batch'] == 2
    assert report['channel_format'] == layout.token
    assert report['channel_count'] == layout.channel_count
    assert report['channel_stride'] == layout.stride
    assert 'no volume-slice sampling' in report['input_fixture']
    assert 'size' not in report and 'channels' not in report


def test_device_index_maps_to_torch_cuda_device(tmp_path, monkeypatch):
    import torch

    selected = []

    def stop_before_gpu_execution(device):
        selected.append(device)
        raise RuntimeError('test sentinel: GPU execution intentionally stopped')

    monkeypatch.setattr(torch.cuda, 'is_available', lambda: True)
    monkeypatch.setattr(torch.cuda, 'set_device', stop_before_gpu_execution)
    monkeypatch.setenv('PTA_GPU_TORCH_COMPILE', '0')
    report_path = tmp_path / 'device.json'
    assert smoke.main(['--device', '2', '--report', str(report_path)]) == 1
    assert selected == [torch.device('cuda:2')]
    report = json.loads(report_path.read_text())
    assert report['status'] == 'failed'
    assert 'test sentinel' in report['error']
    assert report['device'] == 2


def test_forward_geometry_is_checked_before_stochastic_photometry():
    import torch

    class Policy:
        def _apply_intensity_noise(self, images, seeds, parameters):
            # A discrete jump models a Poisson draw changing after tiny rate drift.
            return torch.where(images > .5, images + .1, images)

    policy = Policy()
    expected = torch.full((4, 3, 16, 16), .5)
    actual = expected.clone()
    actual[3, 1, 8, 8] += 1e-5
    expected_result, expected_spatial = smoke._capture_prephotometry(
        policy, lambda: policy._apply_intensity_noise(expected, [0, 1, 100, 101], []))
    actual_result, actual_spatial = smoke._capture_prephotometry(
        policy, lambda: policy._apply_intensity_noise(actual, [0, 1, 100, 101], []))
    assert (actual_result - expected_result).abs().max().item() > .1
    metrics = smoke._check_forward_geometry(expected_spatial, actual_spatial, 'fixture')
    assert metrics['forward_max_normalized_difference'] < .0001
    assert '_apply_intensity_noise' not in vars(policy)
    broken = expected_spatial.clone()
    broken[0, 0, 8, 8] += .02
    with pytest.raises(AssertionError, match='forward geometry mismatch'):
        smoke._check_forward_geometry(expected_spatial, broken, 'fixture')


def test_photometry_capture_restores_policy_after_failure():
    from types import SimpleNamespace
    import torch

    def photometry(images, *_):
        raise RuntimeError('photometry failure')

    policy = SimpleNamespace(_apply_intensity_noise=photometry)
    with pytest.raises(RuntimeError, match='photometry failure'):
        smoke._capture_prephotometry(policy, lambda: policy._apply_intensity_noise(torch.zeros(1)))
    assert policy._apply_intensity_noise is photometry


def test_photometry_is_qualified_using_identical_spatial_input():
    from types import SimpleNamespace
    import torch

    policy = SimpleNamespace(_apply_intensity_noise=lambda images, seeds, parameters: images + .1)
    spatial = torch.full((2, 3, 16, 16), .5)
    actual = ((spatial + .1) * 255).round() / 255
    smoke._check_photometry(policy, spatial, actual, [0, 1], [], 'fixture')
    with pytest.raises(AssertionError, match='photometry differs'):
        smoke._check_photometry(policy, spatial, spatial, [0, 1], [], 'fixture')

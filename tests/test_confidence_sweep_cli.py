"""Exact sweep grammar, scalar compatibility, and explicit cleanup policy."""
from __future__ import annotations

import argparse
import contextlib
import io

import pytest

from XTA.config import (
    ConfidenceSweepSpec,
    build_argparser,
    confidence_sweep_branch_cutoff,
    parse_confidence_sweep_options,
)


REQUIRED = ['--input', 'input.mkv', '--model', 'gpu:model.engine']
BRANCH_ENV = 'XTA_YOLO_SWEEP_CUTOFF_HUNDREDTHS'


def parse(*arguments):
    parser = build_argparser()
    tokens = [*REQUIRED, *arguments]
    parsed = parser.parse_args(tokens)
    return parse_confidence_sweep_options(parser, parsed, tokens)


@pytest.mark.parametrize('value', ['0', '1', '.12345678901234567', '1e-20', '-0.0'])
def test_scalar_confidence_keeps_full_float_precision(value):
    args = parse('--conf', value)
    assert type(args.conf) is float
    assert args.conf == float(value)
    assert args.conf_sweep is None
    assert args.min_conf == .30
    assert not hasattr(args, 'conf_sweep_min_conf_implicit')


def test_default_scalar_contract_stays_unchanged():
    args = parse()
    assert args.conf == .15
    assert args.conf_sweep is None
    assert args.min_conf == .30


@pytest.mark.parametrize('value,levels', [
    ('.5:.1:1', (50, 60, 70, 80, 90, 100)),
    ('0:.01:1', tuple(range(101))),
    ('.99:.01:1', (99, 100)),
    ('.7:.01:.7', (70,)),
    ('0:1:1', (0, 100)),
    ('5e-1:1e-1:1e0', (50, 60, 70, 80, 90, 100)),
    ('.5000:.1000:1.0000', (50, 60, 70, 80, 90, 100)),
])
def test_ranges_use_inclusive_integer_hundredths(value, levels):
    args = parse('--conf=' + value)
    assert args.conf_sweep.levels_hundredths == levels
    assert args.conf_sweep.values == tuple(level / 100 for level in levels)
    assert args.conf_sweep.floor == args.conf == levels[0] / 100
    assert args.conf_sweep.highest == levels[-1] / 100
    assert args.min_conf == 0.0
    assert args.conf_sweep_min_conf_implicit


def test_grouped_confidence_combines_ranges_and_cutoffs_in_ascending_order():
    args = parse('--conf', '.90:.02:1', '.80', '.50:.10:.90', '.94')
    levels = (50, 60, 70, 80, 90, 92, 94, 96, 98, 100)
    assert args.conf_sweep.levels_hundredths == levels
    assert args.conf_sweep.values == tuple(level / 100 for level in levels)
    assert args.conf == args.conf_sweep.floor == .50
    assert args.conf_sweep.highest == 1.0
    assert args.min_conf == 0.0
    assert args.conf_sweep_min_conf_implicit


@pytest.mark.parametrize('invalid_group', ['.90:.03:1', '.123'])
def test_every_group_must_use_the_exact_hundredth_grid(invalid_group):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as error:
        parse('--conf', '.5:.1:1', invalid_group)
    assert error.value.code == 2
    assert '--conf' in stderr.getvalue()
    assert invalid_group in stderr.getvalue()


@pytest.mark.parametrize('value', [
    '.5:.1', '.5:.1:1:1', ':.1:1', '.5::1', '.5:.1:',
    '.5:0:1', '.5:-.1:1', '.8:.1:.5', '-.1:.1:1', '.5:.1:1.01',
    '.501:.1:1', '.5:.001:1', '.5:.03:.6', '.5:1.01:.5',
    'nan:.1:1', '.5:inf:1', '.5:.1:-inf', 'bad:.1:1',
    '.50000000000000000000000000000000000000000001:.1:1',
    '1e-1000000000:.01:1',
])
def test_invalid_ranges_fail_as_argparse_errors(value):
    stderr = io.StringIO()
    with contextlib.redirect_stderr(stderr), pytest.raises(SystemExit) as error:
        parse('--conf=' + value)
    assert error.value.code == 2
    assert '--conf' in stderr.getvalue()


@pytest.mark.parametrize('value', ['nan', 'inf', '-inf', '-.01', '1.01', 'bad'])
def test_invalid_scalar_confidence_remains_an_argparse_error(value):
    with contextlib.redirect_stderr(io.StringIO()), pytest.raises(SystemExit) as error:
        parse('--conf=' + value)
    assert error.value.code == 2


@pytest.mark.parametrize('value', ['0', '.2', '.99', '1'])
@pytest.mark.parametrize('before', [True, False])
def test_explicit_min_conf_is_fixed_even_below_sweep_cutoffs(value, before):
    conf = ['--conf', '.5:.1:1']
    minimum = ['--min_conf=' + value]
    args = parse(*(minimum + conf if before else conf + minimum))
    assert args.min_conf == float(value)
    assert not args.conf_sweep_min_conf_implicit


def test_duplicate_confidence_flags_reset_stale_sweep_and_keep_last_value():
    scalar = parse('--conf', '.5:.1:1', '.9:.02:1', '--conf=.12345678901234567')
    assert scalar.conf == float('.12345678901234567')
    assert scalar.conf_sweep is None
    assert scalar.min_conf == .30
    sweep = parse('--conf=.12345678901234567', '--conf=.5:.1:1', '--conf=.6:.2:1')
    assert sweep.conf == .6
    assert sweep.conf_sweep.levels_hundredths == (60, 80, 100)
    assert sweep.min_conf == 0


def test_duplicate_min_confidence_flags_keep_last_explicit_value():
    args = parse('--min_conf=.9', '--conf=.5:.1:1', '--min_conf', '.2')
    assert args.min_conf == .2
    assert not args.conf_sweep_min_conf_implicit


def test_scalar_helper_does_not_change_existing_validation_policy():
    args = parse('--conf=.9', '--min_conf=.2')
    assert args.min_conf == .2
    assert args.conf_sweep is None
    # Scalar pipeline rejection remains authoritative outside this resolver.
    namespace = argparse.Namespace(conf=.9, min_conf=.3)
    assert parse_confidence_sweep_options(None, namespace, []) is namespace
    assert vars(namespace) == {'conf': .9, 'min_conf': .3}


@pytest.mark.parametrize('values', [(0, 1, 100), (50, 10, 100), (70, 1, 70)])
def test_spec_constructor_accepts_supported_exact_grids(values):
    assert ConfidenceSweepSpec(*values).levels_hundredths[-1] == values[-1]


@pytest.mark.parametrize('values', [(True, 1, 100), (0., 1, 100), (0, 0, 100),
                                    (-1, 1, 100), (0, 1, 101), (90, 1, 50), (50, 3, 60)])
def test_spec_constructor_rejects_invalid_programmatic_grids(values):
    with pytest.raises(ValueError):
        ConfidenceSweepSpec(*values)


def test_internal_branch_cutoff_is_absent_by_default(monkeypatch):
    monkeypatch.delenv(BRANCH_ENV, raising=False)
    assert confidence_sweep_branch_cutoff() is None


@pytest.mark.parametrize('value,expected', [('0', 0), ('1', 1), ('70', 70), ('100', 100),
                                           ('00070', 70), ('0000', 0)])
def test_internal_branch_cutoff_accepts_ascii_integer_hundredths(monkeypatch, value, expected):
    monkeypatch.setenv(BRANCH_ENV, value)
    assert confidence_sweep_branch_cutoff() == expected


@pytest.mark.parametrize('value', ['', ' ', ' 70', '70 ', '+70', '-1', '101', '70.0',
                                  '7e1', 'nan', 'inf', '７０', '٧٠', '1_0', '1000', '9' * 5000])
def test_internal_branch_cutoff_rejects_other_representations(monkeypatch, value):
    monkeypatch.setenv(BRANCH_ENV, value)
    with pytest.raises(ValueError, match=BRANCH_ENV):
        confidence_sweep_branch_cutoff()


def test_help_documents_implicit_sweep_cleanup_default():
    help_text = ' '.join(build_argparser().format_help().split())
    assert 'START:STEP:END' in help_text
    assert 'Sweep defaults --min_conf to 0 unless explicitly supplied' in help_text

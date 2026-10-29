"""The resolved PTA path keeps canonical channels; legacy helpers stay callable."""

from unittest import mock

import pytest

from XTA import pta
from XTA.config import resolve_channel_format
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options


def test_runtime_preserves_the_resolved_channel_object():
    config = parse_pta_args(["--input", "dataset", "--channel_format", "c5s2"])
    with mock.patch("XTA.config.resolve_channel_format", side_effect=AssertionError("reparse")):
        options = build_runtime_options(config)
    assert options.channel_format == ["C5S2"]
    assert options._pta_channel_formats == (config.channel_format,)
    assert options._pta_channel_formats[0] is config.channel_format


@pytest.mark.parametrize(
    ("token", "expected"),
    [
        ("gray", [("gray", False, (0,))]),
        ("RGB", [("RGB", False, (0, 0, 0))]),
        ("C1S3", [("C1S3", False, (0,))]),
        ("C5S2", [("C5S2", False, (-4, -2, 0, 2, 4)),
                   ("C5S2_reverse", True, (4, 2, 0, -2, -4))]),
    ],
)
def test_canonical_and_legacy_layouts_expand_to_identical_variants(token, expected):
    canonical = resolve_channel_format(token)
    legacy = pta.resolve_channel_formats(token)
    # Resolution is complete before expansion. This catches accidental token
    # round-trips through the shared channel parser for either input shape.
    with mock.patch("XTA.unification.channels.resolve_channel_format",
                    side_effect=AssertionError("reparse")):
        actual = pta.expand_channel_variants([canonical])
        assert pta.expand_channel_variants(legacy) == actual
    assert [(value.tag_token, value.reverse, value.offsets) for value in actual] == expected


def test_main_uses_the_canonical_layout_without_legacy_resolution():
    config = parse_pta_args(["--input", "dataset", "--channel_format", "C3S2"])
    options = build_runtime_options(config)
    stopped = RuntimeError("stop after channel selection")
    with (mock.patch.object(pta, "resolve_channel_formats", side_effect=AssertionError("reparse")),
          mock.patch.object(pta, "expand_channel_variants", side_effect=stopped) as expand):
        with pytest.raises(RuntimeError, match="stop after channel selection"):
            pta.main(args=options, argv=[])
    assert expand.call_args.args[0] == [config.channel_format]
    assert expand.call_args.args[0][0] is config.channel_format


def test_main_keeps_legacy_namespace_channel_resolution():
    options = build_runtime_options(parse_pta_args(["--input", "dataset"]))
    del options._pta_channel_formats
    options.channel_format = ["grey", "RGB", "C3S1", "C3S2"]
    stopped = RuntimeError("stop after channel selection")
    with (mock.patch.object(pta, "resolve_channel_formats", wraps=pta.resolve_channel_formats) as resolve,
          mock.patch.object(pta, "expand_channel_variants", side_effect=stopped) as expand):
        with pytest.raises(RuntimeError, match="stop after channel selection"):
            pta.main(args=options, argv=[])
    resolve.assert_called_once_with(options.channel_format)
    assert [value.token for value in expand.call_args.args[0]] == ["gray", "RGB", "C3S1", "C3S2"]


def test_legacy_resolver_preserves_deduplication_order_and_custom_count_rule():
    formats = pta.resolve_channel_formats(["grey, RGB", "c5s2 C5S1", "GRAY RGB C5S2"])
    assert [value.token for value in formats] == ["gray", "RGB", "C5S2", "C5S1"]
    assert all(isinstance(value, pta.ChannelFormat) for value in formats)
    with pytest.raises(ValueError, match="All custom --channel_format values must use the same C"):
        pta.resolve_channel_formats(["C3S1", "C5S1"])

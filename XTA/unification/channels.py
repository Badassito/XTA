"""Mode-owned channel expansion over the existing canonical TTA grammar."""

from __future__ import annotations

from typing import TYPE_CHECKING, Tuple

from .contracts import ChannelLayout, ChannelVariant, PipelineMode

if TYPE_CHECKING:
    from XTA.config import ChannelFormat


def channel_layout_from_format(fmt: "ChannelFormat") -> ChannelLayout:
    """Copy one already-resolved ``config.ChannelFormat`` into the validated contract layout."""

    return ChannelLayout(
        token=fmt.token,
        kind=fmt.kind,
        channel_count=fmt.channel_count,
        stride=fmt.stride,
        offsets=tuple(fmt.offsets),
    )


def expand_channel_variants(
    mode: PipelineMode | str,
    layout: ChannelLayout,
) -> Tuple[ChannelVariant, ...]:
    """Apply the unified mode contract to one canonical channel layout.

    TTA and LTA supply only the canonical ascending order. PTA adds the
    reversed channel order only when reversal changes the actual sequence.
    """

    resolved_mode = PipelineMode.coerce(mode)
    if not isinstance(layout, ChannelLayout):
        raise TypeError("layout must be a ChannelLayout")
    ascending = ChannelVariant(layout=layout, direction="ascending")
    if resolved_mode is not PipelineMode.PTA:
        return (ascending,)
    reversed_offsets = tuple(reversed(layout.offsets))
    if reversed_offsets == layout.offsets:
        return (ascending,)
    return (
        ascending,
        ChannelVariant(layout=layout, direction="reversed"),
    )


__all__ = (
    "channel_layout_from_format",
    "expand_channel_variants",
)

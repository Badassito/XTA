"""Global LTA lineage identity and tile-local mask rebasing.

``LtaLineageId`` scopes one lineage to its volume, physical/runtime view and
tile configuration.  ``rebase_tile_mask`` translates a source-tile mask into a
neighbor tile's local coordinates through their shared overlap, or across a
zero-width/height touching port when stride equals tile size.  Production
relay observation and episode coalescing live in ``lta_worker_adapter`` and
``lta_relay_episodes``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .lta_tiles import TilePlan


@dataclass(frozen=True, order=True)
class LtaLineageId:
    """Globally scoped identity for one LTA lineage."""

    volume_id: str
    physical_view_id: str
    runtime_view_id: str
    lineage_id: str
    tile_config_id: str = "fullframe"

    def __post_init__(self) -> None:
        for name in (
            "volume_id",
            "physical_view_id",
            "runtime_view_id",
            "lineage_id",
            "tile_config_id",
        ):
            value = str(getattr(self, name)).strip()
            if not value:
                raise ValueError(f"{name} must not be empty")
            object.__setattr__(self, name, value)

    @property
    def token(self) -> str:
        return "::".join(
            (
                self.volume_id,
                self.physical_view_id,
                self.runtime_view_id,
                self.tile_config_id,
                self.lineage_id,
            )
        )


def rebase_tile_mask(mask: Any, source: TilePlan, destination: TilePlan) -> Any:
    """Translate overlap or a touching border into a destination-local mask.

    Overlapping tiles preserve global pixel coordinates.  When stride equals
    tile size, the source border is shifted by one pixel onto the destination
    border so a tracker can continue across a zero-width/height handoff port.
    """

    import numpy as np

    source_mask = np.asarray(mask, dtype=bool)
    if source_mask.shape != (source.height, source.width):
        raise ValueError(
            f"source mask shape {source_mask.shape} does not match tile "
            f"{(source.height, source.width)}"
        )
    sx0, sy0, sx1, sy1 = source.xyxy
    dx0, dy0, dx1, dy1 = destination.xyxy
    gx0, gy0 = max(sx0, dx0), max(sy0, dy0)
    gx1, gy1 = min(sx1, dx1), min(sy1, dy1)
    output = np.zeros((destination.height, destination.width), dtype=np.bool_)
    if gx1 < gx0 or gy1 < gy0:
        return output

    if gx1 > gx0:
        source_x = slice(gx0 - sx0, gx1 - sx0)
        destination_x = slice(gx0 - dx0, gx1 - dx0)
    elif sx1 == dx0:
        source_x = slice(source.width - 1, source.width)
        destination_x = slice(0, 1)
    elif dx1 == sx0:
        source_x = slice(0, 1)
        destination_x = slice(destination.width - 1, destination.width)
    else:
        return output

    if gy1 > gy0:
        source_y = slice(gy0 - sy0, gy1 - sy0)
        destination_y = slice(gy0 - dy0, gy1 - dy0)
    elif sy1 == dy0:
        source_y = slice(source.height - 1, source.height)
        destination_y = slice(0, 1)
    elif dy1 == sy0:
        source_y = slice(0, 1)
        destination_y = slice(destination.height - 1, destination.height)
    else:
        return output

    output[destination_y, destination_x] = source_mask[source_y, source_x]
    return output


__all__ = (
    "LtaLineageId",
    "rebase_tile_mask",
)

"""Dependency-light public surface for the unified TTA/PTA/LTA contract layer."""

from .channels import channel_layout_from_format, expand_channel_variants
from .contracts import (
    BackendSamplingImplementation,
    ChannelLayout,
    ChannelVariant,
    DataRole,
    ForwardSamplingPolicy,
    FrameAddress,
    InPlaneVariant,
    PipelineMode,
    RasterPlan,
    RenderRequestBatch,
    RenderItem,
    TileLayout,
)
from .views import ResolvedViewVariant, expand_view_variants, resolve_in_plane_variants
from .sampling import (
    build_forward_raster_plan,
    forward_sampling_execution_record,
    forward_sampling_policy,
    raster_plan_from_spawn_spec,
    raster_plan_spawn_spec,
    require_forward_sampling,
)
from .tiles import TileConfig, resolve_tile_groups


__all__ = (
    "BackendSamplingImplementation",
    "ChannelLayout",
    "ChannelVariant",
    "DataRole",
    "ForwardSamplingPolicy",
    "FrameAddress",
    "InPlaneVariant",
    "PipelineMode",
    "RasterPlan",
    "RenderRequestBatch",
    "RenderItem",
    "ResolvedViewVariant",
    "TileLayout",
    "channel_layout_from_format",
    "expand_channel_variants",
    "expand_view_variants",
    "build_forward_raster_plan",
    "forward_sampling_execution_record",
    "forward_sampling_policy",
    "raster_plan_from_spawn_spec",
    "raster_plan_spawn_spec",
    "require_forward_sampling",
    "TileConfig",
    "resolve_tile_groups",
    "resolve_in_plane_variants",
)

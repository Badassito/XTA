"""Diagnostic compatibility entry points for the retired CUDA scatter backend.

Use native destination pull through :mod:`XTA.backprojection`. This shim keeps
its former constructor's explicit refusal without retaining unusable kernels.
"""
from __future__ import annotations

from .cylindrical_cuda_projection import (
    RadialCudaProjectionUnsafeFailure, _BLOCK_BYTES, _UPLOAD_BYTES, _RESERVE_BYTES,
)
from .tilted_azimuthal_projection import TILTED_AZIMUTHAL_CUDA_UNAVAILABLE_REASON


class TiltedAzimuthalCudaProjectionUnavailable(RuntimeError):
    """The retired scatter backend cannot satisfy native destination coverage."""


class TiltedAzimuthalCudaProjectionUnsafeFailure(RadialCudaProjectionUnsafeFailure):
    """Preserve the former backend's non-replayable failure type for callers."""


class TiltedAzimuthalCudaProjector:
    """Reject retired scatter before inspecting inputs or initializing CUDA."""

    def __init__(self, source, plan, device_index=0, *, initial_packed=None, first_frame=0,
                 block_bytes=_BLOCK_BYTES, upload_bytes=_UPLOAD_BYTES, reserve_bytes=_RESERVE_BYTES):
        raise TiltedAzimuthalCudaProjectionUnavailable(TILTED_AZIMUTHAL_CUDA_UNAVAILABLE_REASON)

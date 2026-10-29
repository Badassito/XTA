"""Diagnostic compatibility entry points for retired tilted-Azimuthal scatter.

Rounded forward scatter tables do not cover native destination cells. Active
projection uses native destination pull in :mod:`XTA.backprojection`.
"""
from __future__ import annotations


class TiltedAzimuthalPlanUnavailable(RuntimeError):
    """The retired scatter plan cannot satisfy native destination coverage."""


TILTED_AZIMUTHAL_CUDA_UNAVAILABLE_REASON = (
    'Legacy Tilted Azimuthal CUDA scatter does not cover native destination cells; '
    'use canonical native destination pull'
)


def tilted_azimuthal_cuda_capability():
    """Refuse retired scatter before geometry allocation or GPU admission."""
    return False, TILTED_AZIMUTHAL_CUDA_UNAVAILABLE_REASON


def build_tilted_azimuthal_plan(source, view, output_shape, known_row_occupancy=None,
                                known_slice_bboxes=None, *, max_plan_bytes=512 * 1024**2):
    """Reject the former scatter-plan interface without inspecting its inputs."""
    raise TiltedAzimuthalPlanUnavailable(TILTED_AZIMUTHAL_CUDA_UNAVAILABLE_REASON)

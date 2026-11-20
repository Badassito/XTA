"""Ordered LTA role-layer composition oracle retained outside the production package."""

from __future__ import annotations

from typing import Sequence

from XTA.lta_outputs import LtaLayerRecord, LtaRecompositionOp


def compose_terminal_union(layers: Sequence[LtaLayerRecord]) -> object:
    """Compose ordered binary layers into the always-created terminal union.

    ``union`` ORs an additive layer, ``select`` replaces the current complete
    checkpoint, ``subtract_from_previous_checkpoint`` clears the supplied mask,
    and ``none`` is diagnostic-only.  The returned volume is a new contiguous
    uint8 array and never aliases a component layer.
    """

    try:
        import numpy as np
    except Exception as exc:  # pragma: no cover - required project dependency
        raise RuntimeError("NumPy is required to compose LTA role layers") from exc

    current = None
    expected_shape = None
    seen_ids: set[str] = set()
    for layer in tuple(layers):
        if not isinstance(layer, LtaLayerRecord):
            raise TypeError("layers must contain LtaLayerRecord values")
        if layer.layer_id in seen_ids:
            raise ValueError(f"duplicate LTA layer_id {layer.layer_id!r}")
        seen_ids.add(layer.layer_id)
        op = LtaRecompositionOp.coerce(layer.recomposition_op)
        if op is LtaRecompositionOp.NONE:
            continue
        raw = np.asarray(layer.volume)
        if raw.ndim != 3:
            raise ValueError(
                f"LTA layer {layer.layer_id!r} must have 3D (t,Y,X) shape; got {raw.shape}"
            )
        binary = np.ascontiguousarray(raw != 0, dtype=np.uint8)
        shape = tuple(int(value) for value in binary.shape)
        if expected_shape is None:
            expected_shape = shape
        elif shape != expected_shape:
            raise ValueError(
                f"LTA layer {layer.layer_id!r} shape {shape} does not match {expected_shape}"
            )

        if op is LtaRecompositionOp.SELECT:
            current = binary.copy()
        elif op is LtaRecompositionOp.UNION:
            if current is None:
                current = np.zeros(shape, dtype=np.uint8)
            np.bitwise_or(current, binary, out=current)
        elif op is LtaRecompositionOp.SUBTRACT_FROM_PREVIOUS_CHECKPOINT:
            if current is None:
                raise ValueError(
                    f"LTA layer {layer.layer_id!r} cannot subtract before a union/checkpoint"
                )
            current[binary != 0] = np.uint8(0)
        else:  # pragma: no cover - enum exhaustiveness
            raise AssertionError(op)

    if current is None:
        raise ValueError("at least one composable LTA layer is required")
    return np.ascontiguousarray(current, dtype=np.uint8)

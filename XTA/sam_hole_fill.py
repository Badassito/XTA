"""Opt-in SAM topology treatment at output and recurrent-memory boundaries."""

from __future__ import annotations

import functools
import inspect


def fill_sam_mask_logits_inplace(logits):
    """Fill enclosed 4-connected background without changing existing logits.

    Work on each object's final two spatial dimensions independently.  Only new
    foreground receives a positive logit; scores and existing foreground retain
    their original values.  The host implementation matches detector hole fill.
    """
    import numpy as np
    import torch
    from .lta_postprocessing import fill_binary_mask_holes_2d

    if logits is None or logits.ndim < 2 or logits.numel() == 0:
        return logits
    binary = (logits.detach() > 0).to(device='cpu').numpy()
    added = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape[-2:]
    for source, target in zip(binary.reshape(-1, height, width), added.reshape(-1, height, width)):
        target[:] = fill_binary_mask_holes_2d(source) & ~source
    if bool(added.any()):
        holes = torch.from_numpy(added).to(device=logits.device)
        logits.masked_fill_(holes, 10.0)
    return logits


def configure_sam_hole_fill(predictor, *, enabled: bool) -> dict[str, object]:
    """Install treatment before SAM encodes a prediction for its next frame.

    SDK ``fill_hole_area`` acts only on resized output, after recurrent memory
    encoding.  These worker-local hooks additionally treat memory input and
    output at their actual resolutions, using the same unbounded hole definition.
    """
    tracker = predictor.model.tracker
    model = getattr(tracker, 'model', tracker)
    # Keep the SDK's independent size-limited output treatment disabled.
    for target in (tracker, model):
        if hasattr(target, 'fill_hole_area'):
            target.fill_hole_area = 0
    if not enabled:
        return dict(enabled=False, sdk_output_hole_fill_area=0, memory_hook=False)
    if getattr(model, '_xta_hole_fill_installed', False):
        return dict(enabled=True, sdk_output_hole_fill_area=0, memory_hook=True)

    def wrap(name, parameter):
        original = getattr(model, name, None)
        if not callable(original):
            raise RuntimeError(f'opt-in SAM hole fill requires {name}')
        signature = inspect.signature(original)
        if parameter not in signature.parameters:
            raise RuntimeError(f'opt-in SAM hole fill requires {name}.{parameter}')

        @functools.wraps(original)
        def filled(*args, **kwargs):
            bound = signature.bind(*args, **kwargs)
            fill_sam_mask_logits_inplace(bound.arguments[parameter])
            return original(*bound.args, **bound.kwargs)

        setattr(model, name, filled)

    wrap('_encode_new_memory', 'pred_masks_high_res')
    wrap('_get_orig_video_res_output', 'any_res_masks')
    model._xta_hole_fill_installed = True
    return dict(enabled=True, sdk_output_hole_fill_area=0, memory_hook=True,
                algorithm='enclosed_2d_background_components', background_connectivity=4)

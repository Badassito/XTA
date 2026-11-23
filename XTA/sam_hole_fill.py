"""Opt-in SAM topology treatment at output and recurrent-memory boundaries."""

from __future__ import annotations

import functools
import inspect


def fill_sam_mask_logits_inplace(logits):
    """Fill enclosed 4-connected background without changing existing logits.

    Work on each object's final two spatial dimensions independently.  Only new
    foreground receives a positive logit; scores and existing foreground retain
    their original values. Both paths use the detector's connectivity rule.
    """
    import numpy as np
    import torch
    from .lta_postprocessing import fill_binary_mask_holes_2d

    if logits is None or logits.ndim < 2 or logits.numel() == 0:
        return logits
    if logits.is_cuda:
        try:
            import cupy as cp
            from cupyx.scipy import ndimage as cpx_ndi
        except ImportError:
            pass
        else:
            from ._prediction_cleanup import _device_enclosed_background
            from .geometry import _cupy_external_stream

            device = logits.device

            def allocate(size):
                buffer = torch.empty(max(1, int(size)), dtype=torch.uint8, device=device)
                memory = cp.cuda.UnownedMemory(buffer.data_ptr(), buffer.numel(), buffer, device.index)
                return cp.cuda.MemoryPointer(memory, 0)

            try:
                stream = torch.cuda.current_stream(device)
                with cp.cuda.Device(device.index), _cupy_external_stream(cp, stream), cp.cuda.using_allocator(allocate):
                    background = ~(logits.detach() > 0).reshape(-1, *logits.shape[-2:])
                    holes = _device_enclosed_background(cp, cpx_ndi, cp.from_dlpack(background))
                    device_holes = torch.from_dlpack(holes).reshape(logits.shape)
            except (cp.cuda.memory.OutOfMemoryError, cp.cuda.runtime.CUDARuntimeError,
                    cp.cuda.compiler.CompileException, torch.cuda.OutOfMemoryError) as exc:
                # Preserve potential device faults before attempting a host copy.
                if isinstance(exc, cp.cuda.runtime.CUDARuntimeError) and exc.status != 2:  # cudaErrorMemoryAllocation
                    raise
                background = holes = device_holes = None
            else:
                logits.masked_fill_(device_holes, 10.0)
                return logits
    host_logits = logits.to(device='cpu')
    binary = (host_logits.detach() > 0).numpy()
    added = np.zeros_like(binary, dtype=bool)
    height, width = binary.shape[-2:]
    for source, target in zip(binary.reshape(-1, height, width), added.reshape(-1, height, width)):
        target[:] = fill_binary_mask_holes_2d(source) & ~source
    if bool(added.any()):
        host_logits.masked_fill_(torch.from_numpy(added), 10.0)
        if host_logits is not logits:
            logits.copy_(host_logits)
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

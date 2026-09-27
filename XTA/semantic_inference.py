"""Binary YOLO semantic-logit decoding and native confidence accumulation."""

from __future__ import annotations

from typing import Callable, Optional, Sequence, Tuple

import numpy as np

from ._deps import cv2


def semantic_foreground_probability(logits: object, *, output_size: int) -> object:
    """Decode raw torch logits without discarding foreground confidence."""
    import torch  # type: ignore
    import torch.nn.functional as F  # type: ignore

    if isinstance(logits, (tuple, list)):
        if not logits:
            raise ValueError('Semantic backend returned no outputs')
        logits = logits[0]
    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise ValueError(
            'Semantic inference requires raw [B,C,H,W] logits; this export may have '
            'an in-graph argmax that discards confidence'
        )
    channels = int(logits.shape[1])
    if channels not in (1, 2):
        raise ValueError(
            f'Binary semantic inference requires one foreground logit or two '
            f'background/foreground logits; model has {channels} channels'
        )
    if not logits.dtype.is_floating_point:
        raise ValueError('Semantic inference requires floating-point logits before argmax')
    value = logits.float()
    if tuple(value.shape[-2:]) != (int(output_size), int(output_size)):
        value = F.interpolate(
            value, size=(int(output_size), int(output_size)),
            mode='bilinear', align_corners=False,
        )
    return value[:, 0].sigmoid() if channels == 1 else value.softmax(dim=1)[:, 1]


def semantic_foreground_probability_numpy(
    outputs: Sequence[np.ndarray], *, batch_size: int, output_size: int,
) -> np.ndarray:
    """Decode an OpenVINO semantic logits output to [B,H,W] probabilities."""
    if len(outputs) != 1:
        raise ValueError(f'Semantic OpenVINO export must expose one logits output; got {len(outputs)}')
    logits = np.asarray(outputs[0])
    if logits.ndim != 4 or int(logits.shape[0]) != int(batch_size):
        raise ValueError(
            f'Semantic OpenVINO output must be [B,C,H,W] logits with B={batch_size}; '
            f'got {tuple(logits.shape)}. Class-map exports discard confidence.'
        )
    channels = int(logits.shape[1])
    if channels not in (1, 2):
        raise ValueError(f'Binary semantic OpenVINO output needs 1 or 2 channels; got {channels}')
    if not np.issubdtype(logits.dtype, np.floating):
        raise ValueError('Semantic OpenVINO output must contain floating-point logits')
    result = np.empty((int(batch_size), int(output_size), int(output_size)), dtype=np.float32)
    for index in range(int(batch_size)):
        frame = np.asarray(logits[index], dtype=np.float32)
        if tuple(frame.shape[-2:]) != (int(output_size), int(output_size)):
            frame = np.stack([
                cv2.resize(channel, (int(output_size), int(output_size)), interpolation=cv2.INTER_LINEAR)
                for channel in frame
            ])
        if channels == 1:
            result[index] = 1.0 / (1.0 + np.exp(-np.clip(frame[0], -80.0, 80.0)))
        else:
            difference = np.clip(frame[0] - frame[1], -80.0, 80.0)
            result[index] = 1.0 / (1.0 + np.exp(difference))
    return result


def accumulate_semantic_probability_frame(
    probability: np.ndarray,
    *,
    frame_index: int,
    conf_threshold: float,
    view_union_mm: np.ndarray,
    view_confmap_mm: Optional[np.ndarray],
    M_out_to_native: np.ndarray,
    native_h: int,
    native_w: int,
    restore_planes: Optional[Callable] = None,
) -> Tuple[int, int]:
    """Warp foreground and confidence planes into the existing binary union contract."""
    prob = np.asarray(probability, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f'Semantic probability plane must be 2-D; got {prob.shape}')
    mask = (prob >= float(conf_threshold)).astype(np.uint8)
    conf = np.where(mask > 0, np.clip(np.rint(prob * 255.0), 0, 255), 0).astype(np.uint8)
    if restore_planes is not None:
        mask, conf = restore_planes([mask, conf])
    native_mask = cv2.warpAffine(
        np.asarray(mask, dtype=np.uint8), M_out_to_native,
        dsize=(int(native_w), int(native_h)), flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT, borderValue=0,
    ) > 0
    if not np.any(native_mask):
        return 0, 0
    view_union_mm[int(frame_index)] |= native_mask.astype(np.uint8)
    if view_confmap_mm is not None:
        native_conf = cv2.warpAffine(
            np.asarray(conf, dtype=np.uint8), M_out_to_native,
            dsize=(int(native_w), int(native_h)), flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT, borderValue=0,
        )
        conf_slice = view_confmap_mm[int(frame_index)]
        np.maximum(conf_slice, np.where(native_mask, native_conf, 0).astype(np.uint8), out=conf_slice)
    return 1, 1

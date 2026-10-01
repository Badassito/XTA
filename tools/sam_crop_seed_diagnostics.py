"""Research-only CPU proxy for the pinned SAM mask-resampling chain.

This measures raster changes, not tracker accuracy, neural information retained,
or future predictions. The SDK echoes the original native seed at its anchor;
that echo does not expose the conditioning mask used inside the model.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np


MODEL_IMAGE_SIDE = 1008
MASK_INPUT_SIDE = 1152
LOW_MASK_SIDE = 288
DEFAULT_MAX_NATIVE_PIXELS = 16_777_216


def native_seed_sha256(seed: np.ndarray) -> str:
    """Fingerprint the untouched native Boolean raster and its dimensions."""
    digest = hashlib.sha256(json.dumps(list(seed.shape), separators=(",", ":")).encode())
    digest.update(np.packbits(np.asarray(seed, dtype=bool).reshape(-1)).tobytes())
    return digest.hexdigest()


def _overlap(expected: np.ndarray, projected: np.ndarray) -> dict[str, Any]:
    import cv2

    intersection = int(np.count_nonzero(expected & projected))
    false_positive = int(np.count_nonzero(projected & ~expected))
    false_negative = int(np.count_nonzero(expected & ~projected))
    reference_pixels = int(np.count_nonzero(expected))
    projected_pixels = int(np.count_nonzero(projected))
    union = intersection + false_positive + false_negative
    reference_count, reference_labels = cv2.connectedComponents(expected.astype(np.uint8), connectivity=8)
    projected_count, _ = cv2.connectedComponents(projected.astype(np.uint8), connectivity=8)
    touched = np.unique(reference_labels[expected & projected])
    touched = touched[touched > 0]
    return {
        "intersection_pixels": intersection, "false_positive_pixels": false_positive,
        "false_negative_pixels": false_negative, "reference_pixels": reference_pixels,
        "projected_pixels": projected_pixels,
        "iou": 1.0 if not union else intersection / union,
        "dice": 2 * intersection / (reference_pixels + projected_pixels),
        "recall": intersection / reference_pixels,
        "precision": None if not projected_pixels else intersection / projected_pixels,
        "reference_components_8": int(reference_count) - 1,
        "projected_components_8": int(projected_count) - 1,
        "reference_components_without_projected_overlap": int(reference_count) - 1 - len(touched),
        "changed_native_pixel_fraction": (false_positive + false_negative) / expected.size,
    }


def compute_seed_resampling_diagnostics(
    native_seed: object, *, conditioning_dtype: str = "float32",
    max_native_pixels: int = DEFAULT_MAX_NATIVE_PIXELS,
) -> tuple[dict[str, Any], np.ndarray]:
    """Return a JSON-safe receipt and read-only native-size Boolean proxy.

    Float32 follows the installed operator geometry on CPU. ``bfloat16`` also
    emulates the listed BF16 rounding boundaries, but computes interpolation in
    float32 because CPU antialiased bilinear lacks BF16 support. Neither mode
    claims GPU kernel equivalence or models attention, object pointers, memory
    encoding, or future tracking. No SDK model or CUDA context is constructed.
    """

    import torch
    import torch.nn.functional as functional

    seed = np.asarray(native_seed)
    if seed.dtype != np.bool_ or seed.ndim != 2 or min(seed.shape) < 1:
        raise ValueError("native_seed must be a positive HxW Boolean raster")
    if not bool(seed.any()):
        raise ValueError("empty native seed has no conditioning prompt; report unavailable coverage")
    if isinstance(max_native_pixels, bool) or int(max_native_pixels) < 1 or seed.size > int(max_native_pixels):
        raise ValueError("native seed exceeds the bounded CPU diagnostic pixel budget")
    if conditioning_dtype not in {"float32", "bfloat16"}:
        raise ValueError("conditioning_dtype must be float32 or bfloat16")
    original_hash = native_seed_sha256(seed)
    with torch.inference_mode():
        native = torch.from_numpy(np.array(seed, dtype=np.float32, copy=True))[None, None]
        if tuple(seed.shape) == (MASK_INPUT_SIDE, MASK_INPUT_SIDE):
            model_mask = native
        else:
            model_mask = functional.interpolate(
                native, size=(MASK_INPUT_SIDE, MASK_INPUT_SIDE), mode="bilinear",
                align_corners=False, antialias=True,
            )
        first_restore = functional.interpolate(
            model_mask, size=seed.shape, mode="bilinear", align_corners=False,
        )
        first_projected = first_restore[0, 0].numpy() > 0.5
        if conditioning_dtype == "bfloat16":
            # The pinned _use_mask_as_output casts to the backbone dtype, then
            # performs these two scalar operations separately in that dtype.
            high_logits = model_mask.to(torch.bfloat16) * 20.0 - 10.0
            low_logits = functional.interpolate(
                high_logits.float(), size=(LOW_MASK_SIDE, LOW_MASK_SIDE),
                mode="bilinear", align_corners=False, antialias=True,
            ).to(torch.bfloat16)
            restored = functional.interpolate(
                low_logits.float(), size=seed.shape, mode="bilinear", align_corners=False,
            ).to(torch.bfloat16).float()
        else:
            high_logits = model_mask * 20.0 - 10.0
            low_logits = functional.interpolate(
                high_logits, size=(LOW_MASK_SIDE, LOW_MASK_SIDE),
                mode="bilinear", align_corners=False, antialias=True,
            )
            restored = functional.interpolate(
                low_logits, size=seed.shape, mode="bilinear", align_corners=False,
            )
        projected = restored[0, 0].numpy() > 0.0
        receipt = {
            "schema": "xta.sam-seed-resampling-proxy/1", "research_only": True,
            "scope": "CPU resampling-only proxy; not an actual tracker result",
            "interpretation": (
                "Raster overlap under the listed operators does not measure retained neural features, "
                "memory conditioning, future prediction quality, or prove tracker information loss."
            ),
            "native_shape_yx": list(seed.shape), "native_seed_sha256": original_hash,
            "native_seed_foreground_pixels": int(np.count_nonzero(seed)),
            "model_rgb_image_side": MODEL_IMAGE_SIDE,
            "mask_conditioning_side": MASK_INPUT_SIDE, "low_mask_side": LOW_MASK_SIDE,
            "native_rgb_crop_resize": "direct independent-axis stretch to 1008x1008; no letterbox",
            "conditioning_dtype": conditioning_dtype,
            "interpolation_compute_dtype": "float32_cpu",
            "bfloat16_quantization_emulated": conditioning_dtype == "bfloat16",
            "gpu_kernel_equivalence_claimed": False,
            "native_anchor_output": "SDK overwrites the native anchor with the original seed",
            "operators": [
                "native float32 mask -> 1152x1152 bilinear, align_corners=False, antialias=True",
                "cast conditioning dtype; multiply 20; subtract 10",
                "1152x1152 logits -> 288x288 bilinear, align_corners=False, antialias=True",
                "288x288 logits -> native HxW bilinear, align_corners=False; threshold >0",
            ],
            "source_symbols": [
                "sam3.model.video_tracking_multiplex_demo.VideoTrackingMultiplexDemo.add_new_masks",
                "sam3.model.video_tracking_multiplex.VideoTrackingMultiplex._use_mask_as_output",
                "sam3.model.video_tracking_multiplex_demo.VideoTrackingMultiplexDemo._get_orig_video_res_output",
            ],
            "first_resize_roundtrip_threshold_0_5": _overlap(seed, first_projected),
            "low_grid_native_proxy_threshold_0": _overlap(seed, projected),
            "fractional_conditioning_pixels": int(torch.count_nonzero((model_mask > 0) & (model_mask < 1))),
            "conditioning_value_sum": float(model_mask.sum(dtype=torch.float64)),
        }
    if native_seed_sha256(seed) != original_hash:
        raise RuntimeError("CPU diagnostic unexpectedly changed the native seed")
    projected.setflags(write=False)
    return receipt, projected


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=Path, required=True, help="Untouched Boolean .npy or .npz native seed")
    parser.add_argument("--seed-key", default="seed", help="Array key when --seed is .npz")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-proxy-mask", type=Path, help="Optional native-size Boolean .npy proxy")
    parser.add_argument("--conditioning-dtype", choices=("float32", "bfloat16"), default="float32")
    parser.add_argument("--max-native-pixels", type=int, default=DEFAULT_MAX_NATIVE_PIXELS)
    args = parser.parse_args(argv)
    loaded = np.load(args.seed, allow_pickle=False)
    try:
        seed = loaded[args.seed_key].copy() if isinstance(loaded, np.lib.npyio.NpzFile) else loaded
        receipt, projected = compute_seed_resampling_diagnostics(
            seed, conditioning_dtype=args.conditioning_dtype, max_native_pixels=args.max_native_pixels,
        )
    finally:
        if isinstance(loaded, np.lib.npyio.NpzFile):
            loaded.close()
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(receipt, indent=2, sort_keys=True), encoding="utf-8")
    if args.output_proxy_mask is not None:
        args.output_proxy_mask.parent.mkdir(parents=True, exist_ok=True)
        np.save(args.output_proxy_mask, projected, allow_pickle=False)
    print(json.dumps({"receipt": str(args.output_json), "native_shape": list(seed.shape),
                      "low_grid_proxy": receipt["low_grid_native_proxy_threshold_0"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

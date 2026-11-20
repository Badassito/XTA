"""Pinned SAM 3.1 visual features for the mask-only LTA tracker bridge.

The normal multiplex preparation runs grounding merely to retain its visual
backbone output. Mask-only LTA does not consume detections. This adapter calls
the same visual trunk and both tracker necks, preserving the BF16 cast before
the tracker projections used by the pinned single-rank grounding path.

Installed SAM source is never patched. Unsupported custom models retain their
original feature bridge; failures inside the supported path are propagated.
"""

from __future__ import annotations

from collections.abc import Mapping, MutableMapping
import math
import operator
import threading
from typing import Any

from .env_flags import env_flag


TRACKER_FEATURE_POLICY = "sam3_1_visual_backbone_only_exact_bf16"
TRACKER_FEATURE_AUDIT_KEY = "lta_tracker_feature_preparation"
_FRAME_REUSE_KEY = "_lta_tracker_prepared_frame_identity"
_IMAGE_ID_CACHE_KEY = "_lta_tracker_verified_image_ids"
_STAGING_SLOTS = 2
_STAGING = threading.local()


def async_frame_io_enabled() -> bool:
    """``YOLO_TTA_SAM_ASYNC_FRAME_IO=0`` restores the original synchronous per-frame transfers."""
    return env_flag("YOLO_TTA_SAM_ASYNC_FRAME_IO", True)


def _tensor_version(tensor: Any) -> int | None:
    """In-place version counter, or None for inference tensors (which do not track one)."""
    return None if tensor.is_inference() else int(tensor._version)


def _cache_entry_current(entry: Any, tensor: Any) -> bool:
    return entry is not None and entry[0] is tensor and entry[1] == _tensor_version(tensor)


def _frame_image_id(state: MutableMapping[str, Any], find_inputs: Any, frame_index: int) -> int:
    """Return one frame's ``img_ids`` value without a per-frame device sync.

    The SDK moves every ``FindStage.img_ids`` to CUDA, so ``.item()`` blocks the
    host until all queued GPU work drains, once per tracked frame. All of a
    session's ids are read with one transfer instead. Each cached value is bound
    to its exact tensor object and, where tracked, its in-place version counter;
    a replaced or mutated tensor is read again directly. The SDK creates these
    ids as inference tensors, which have no version counter, so
    ``verify_cached_frame_image_ids`` re-reads them together before the session's
    results are returned; a changed id still fails the session.
    """
    tensor = find_inputs[frame_index].img_ids
    if not async_frame_io_enabled():
        return int(tensor.reshape(-1)[0].item())
    cache = state.get(_IMAGE_ID_CACHE_KEY)
    entry = None if not isinstance(cache, dict) else cache.get(frame_index)
    if _cache_entry_current(entry, tensor):
        return entry[2]
    import torch

    if not isinstance(cache, dict):
        cache = {}
        state[_IMAGE_ID_CACHE_KEY] = cache
        singles = [(index, stage.img_ids) for index, stage in enumerate(find_inputs)
                   if isinstance(getattr(stage, "img_ids", None), torch.Tensor) and stage.img_ids.numel() == 1
                   and stage.img_ids.device == tensor.device and stage.img_ids.dtype == tensor.dtype]
        if singles:
            values = torch.cat([ids.reshape(-1) for _index, ids in singles]).cpu().tolist()
            for (index, ids), value in zip(singles, values):
                cache[index] = (ids, _tensor_version(ids), int(value))
        entry = cache.get(frame_index)
        if _cache_entry_current(entry, tensor):
            return entry[2]
    value = int(tensor.reshape(-1)[0].item())
    cache[frame_index] = (tensor, _tensor_version(tensor), value)
    return value


def verify_cached_frame_image_ids(state: MutableMapping[str, Any]) -> None:
    """Re-read every cached frame id with one transfer; raise if any changed."""
    cache = state.get(_IMAGE_ID_CACHE_KEY)
    if not isinstance(cache, dict) or not cache:
        return
    import torch

    items = sorted(cache.items())
    groups: dict[tuple[object, object], list[tuple[int, tuple[Any, Any, int]]]] = {}
    for index, entry in items:
        groups.setdefault((entry[0].device, entry[0].dtype), []).append((index, entry))
    for members in groups.values():
        values = torch.cat([entry[0].reshape(-1)[:1] for _index, entry in members]).cpu().tolist()
        for (index, entry), value in zip(members, values):
            if int(value) != entry[2] or entry[0].numel() != 1:
                raise RuntimeError("pinned SAM frame identity differs from the requested local frame")


def _upload_frame_image(original_image: Any, device: Any, torch: Any) -> Any:
    """Copy one CPU-offloaded frame to ``device`` as float32 without draining the GPU.

    A pageable host-to-device copy first waits for all queued GPU work. The frame
    is instead copied into one of two per-process pinned staging buffers and sent
    asynchronously; the fp16-to-fp32 conversion then runs on the device. That
    conversion is exact, so the backbone input is bit-identical to
    ``original_image.to(dtype=float32, device=device)``. A staging slot is reused
    only after its previous transfer's event completes.
    """
    target = torch.device(device)
    if (not async_frame_io_enabled() or original_image.device.type != "cpu" or target.type != "cuda"
            or not original_image.is_contiguous() or not torch.cuda.is_available()):
        return original_image.to(dtype=torch.float32, device=target)
    key = (tuple(original_image.shape), original_image.dtype, str(target))
    buffers = getattr(_STAGING, "buffers", None)
    if buffers is None:
        buffers = _STAGING.buffers = {}
    slots = buffers.get(key)
    if slots is None:
        slots = [[torch.empty(original_image.shape, dtype=original_image.dtype, pin_memory=True), None]
                 for _ in range(_STAGING_SLOTS)]
        slots.append(0)  # next slot index
        buffers[key] = slots
    index = slots[-1]
    slots[-1] = (index + 1) % _STAGING_SLOTS
    slot = slots[index]
    if slot[1] is not None:
        slot[1].synchronize()
    slot[0].copy_(original_image)
    on_device = torch.empty(original_image.shape, dtype=original_image.dtype, device=target)
    on_device.copy_(slot[0], non_blocking=True)
    event = torch.cuda.Event()
    event.record(torch.cuda.current_stream(target))
    slot[1] = event
    return on_device.to(dtype=torch.float32)


def _class_identity(value: object) -> tuple[str, str]:
    return type(value).__module__, type(value).__name__


def _unsupported_reason(model: object) -> str | None:
    if _class_identity(model) != (
        "sam3.model.sam3_multiplex_tracking", "Sam3MultiplexTrackingWithInteractivity"
    ):
        return "custom_model"
    detector = getattr(model, "detector", None)
    if _class_identity(detector) != (
        "sam3.model.sam3_multiplex_detector", "Sam3MultiplexDetector"
    ):
        return "custom_detector"
    if _class_identity(getattr(detector, "backbone", None)) != (
        "sam3.model.vl_combiner", "SAM3VLBackboneTri"
    ):
        return "custom_backbone"
    if (
        getattr(model, "world_size", None) != 1
        or getattr(detector, "world_size", None) != 1
        or getattr(model, "rank", None) != 0
        or getattr(detector, "rank", None) != 0
    ):
        return "distributed_world"
    if (
        not bool(getattr(model, "is_multiplex", False))
        or not bool(getattr(detector, "is_multiplex", False))
        or not bool(getattr(detector, "gather_backbone_out", False))
    ):
        return "custom_feature_layout"
    return None


def _canonical_position_contract(model: object) -> tuple[object, ...] | None:
    """Qualify the inspected pinned SDK's spatial-only positional encoder.

    The worker has already verified the exact SAM package tree at startup.
    ``PositionEmbeddingSine.forward`` uses only spatial shape, device and its
    fixed scalar parameters; TriViTDetNeck casts the result to branch dtype.
    Custom encoders, overridden methods and compiled wrappers keep their
    ordinary per-frame positions rather than assuming that same invariant.
    """
    detector = getattr(model, 'detector', None)
    backbone = getattr(detector, 'backbone', None)
    neck = getattr(backbone, 'vision_backbone', None)
    position = getattr(neck, 'position_encoding', None)
    if (_class_identity(neck) != ('sam3.model.necks', 'Sam3TriViTDetNeck')
            or _class_identity(position) != ('sam3.model.position_encoding', 'PositionEmbeddingSine')):
        return None
    for owner, name, module, qualified_name in (
        (backbone, 'forward_image', 'sam3.model.vl_combiner', 'SAM3VLBackboneTri.forward_image'),
        (backbone, '_forward_image_tri_no_act_ckpt', 'sam3.model.vl_combiner', 'SAM3VLBackboneTri._forward_image_tri_no_act_ckpt'),
        (neck, 'forward', 'sam3.model.necks', 'Sam3TriViTDetNeck.forward'),
        (position, 'forward', 'sam3.model.position_encoding', 'PositionEmbeddingSine.forward'),
    ):
        function = getattr(getattr(owner, name, None), '__func__', None)
        if (function is not getattr(type(owner), name, None)
                or getattr(function, '__module__', '') != module
                or getattr(function, '__qualname__', '') != qualified_name):
            return None
    if bool(getattr(neck, 'training', True)) or bool(getattr(position, 'training', True)):
        return None
    parameters = tuple(getattr(position, name, None) for name in
                       ('num_pos_feats', 'temperature', 'normalize', 'scale'))
    if (isinstance(parameters[0], bool) or not isinstance(parameters[0], int)
            or parameters[0] <= 0 or not isinstance(parameters[2], bool)
            or any(isinstance(value, bool) or not isinstance(value, (int, float))
                   or not math.isfinite(value) or value <= 0 for value in (parameters[1], parameters[3]))):
        return None
    return (id(neck), id(position), *parameters)


def _record_preparation(
    state: MutableMapping[str, Any], *, fallback_reason: str | None, cache_hit: bool = False,
    shared_cache_hit: bool = False,
) -> dict[str, object]:
    audit = state.setdefault(TRACKER_FEATURE_AUDIT_KEY, {
        "policy": TRACKER_FEATURE_POLICY,
        "feature_only_preparations": 0,
        "feature_only_cache_hits": 0,
        "shared_feature_cache_hits": 0,
        "shared_feature_cache_misses": 0,
        "fallback_preparations": 0,
        "fallback_reasons": {},
        "fpn_preprojection_dtype": "bfloat16",
        "sam3_detection_neck_requested": False,
        "interactive_and_propagation_necks_requested": True,
    })
    if not isinstance(audit, MutableMapping) or audit.get("policy") != TRACKER_FEATURE_POLICY:
        raise RuntimeError("LTA tracker feature audit state is incompatible")
    if fallback_reason is None:
        if shared_cache_hit:
            audit["shared_feature_cache_hits"] = int(audit.get("shared_feature_cache_hits", 0)) + 1
        elif cache_hit:
            audit["feature_only_cache_hits"] = int(audit.get("feature_only_cache_hits", 0)) + 1
        else:
            audit["feature_only_preparations"] += 1
    else:
        audit["fallback_preparations"] += 1
        reasons = audit["fallback_reasons"]
        reasons[fallback_reason] = int(reasons.get(fallback_reason, 0)) + 1
    return {
        "policy": TRACKER_FEATURE_POLICY if fallback_reason is None else "original_feature_bridge",
        "fallback_reason": fallback_reason,
    }


def _image_identity(image: object) -> tuple[object, ...]:
    """Identify the fixed normalized frame without hashing/copying its pixels."""
    try:
        version = image._version
    except RuntimeError:
        # Pinned inference-mode loader tensors have no version counter. Their
        # pixels are immutable for the fixed session's entire input lifetime.
        version = None
    return (
        image.data_ptr(), tuple(image.shape), tuple(image.stride()),
        image.dtype, image.device, version,
    )


def _precision_identity(torch) -> tuple[object, ...]:
    """Avoid reusing features across a changed inherited precision boundary."""
    return (
        torch.is_inference_mode_enabled(), torch.is_grad_enabled(),
        torch.is_autocast_enabled("cpu"), torch.get_autocast_dtype("cpu"),
        torch.is_autocast_enabled("cuda"), torch.get_autocast_dtype("cuda"),
        torch.backends.cuda.matmul.allow_tf32, torch.backends.cudnn.allow_tf32,
    )


def prepare_tracker_frame_features(
    model: object,
    state: MutableMapping[str, Any],
    frame_idx: int,
    reverse: bool,
    *,
    feature_cache=None,
    cache_frame_identity=None,
) -> dict[str, object]:
    """Populate or reuse the exact one-frame tracker cache without grounding.

    The caller retains its existing inference-mode/autocast context. The
    pinned Sam3MultiplexPredictorWrapper enters CUDA BF16 autocast for its
    process lifetime; the original preparation methods add no new context.
    The frame image stored in the cache is the original normalized image object; only
    the backbone input is promoted to float32 on the detector's device, just
    as in ``Sam3Image._get_img_feats``. Cache pruning follows the original
    directional adjacent-frame rule exactly. An already-prepared prompt frame
    is reused within the same fixed session and inherited precision context.
    Optional worker-owned reuse needs an exact immutable frame identity; it
    shares visual features only, pairing them with the current session image.
    """

    if not isinstance(state, MutableMapping):
        raise TypeError("tracker inference state must be a mutable mapping")
    if isinstance(frame_idx, bool):
        raise TypeError("frame_idx must be an integer")
    frame_index = operator.index(frame_idx)
    if frame_index < 0:
        raise ValueError("frame_idx must be nonnegative")
    if not isinstance(reverse, bool):
        raise TypeError("reverse must be bool")
    if feature_cache is not None and not callable(cache_frame_identity):
        raise TypeError('Shared feature reuse requires a callable cache_frame_identity')
    reason = _unsupported_reason(model)
    if reason is not None:
        original = getattr(model, "_prepare_backbone_feats", None)
        if not callable(original):
            raise RuntimeError("SAM model exposes no original shared feature bridge")
        original(state, frame_index, reverse=reverse)
        return _record_preparation(state, fallback_reason=reason)

    import torch

    detector = model.detector
    backbone = detector.backbone
    tracker = model.tracker
    if bool(getattr(model, "training", False)) or bool(getattr(backbone, "training", False)):
        raise RuntimeError("LTA tracker feature preparation requires evaluation mode")
    if frame_index >= int(state["num_frames"]):
        raise ValueError("frame_idx lies outside the tracker input sequence")
    cache = state["feature_cache"]
    if not isinstance(cache, MutableMapping):
        raise RuntimeError("SAM feature_cache must be a mutable mapping")
    input_batch = state["input_batch"]
    image_batch = input_batch.img_batch
    if not hasattr(image_batch, "tensors") or getattr(image_batch, "mask", None) is not None:
        raise RuntimeError("pinned SAM frame batch must use unmasked NestedTensor storage")
    # Local LTA sessions have one fixed FindStage per frame. Streaming/circular
    # remapping would need its own qualification and must not silently select
    # another image when this adapter is active.
    find_input = input_batch.find_inputs[frame_index]
    image_ids = find_input.img_ids
    if image_ids.numel() != 1 or _frame_image_id(state, input_batch.find_inputs, frame_index) != frame_index:
        raise RuntimeError("pinned SAM frame identity differs from the requested local frame")
    original_image = image_batch.tensors[frame_index]
    if not isinstance(original_image, torch.Tensor) or original_image.ndim != 3:
        raise RuntimeError("pinned SAM frame image must be a CHW tensor")
    # Mask injection prepares the prompt, and the first propagation step asks
    # for that same frame again. Reuse only an entry created by this adapter in
    # this fixed session. Worker-owned visual reuse is handled separately below;
    # tracker state always remains local to the current session.
    identity = _image_identity(original_image)
    precision = _precision_identity(torch)
    previous = state.get(_FRAME_REUSE_KEY)
    cached_entry = cache.get(frame_index)
    if (
        isinstance(previous, Mapping)
        and previous.get("model") is model
        and previous.get("detector") is detector
        and previous.get("backbone") is backbone
        and previous.get("tracker") is tracker
        and previous.get("input_batch") is input_batch
        and previous.get("image_batch") is image_batch
        and previous.get("frame_index") == frame_index
        and previous.get("image_identity") == identity
        and previous.get("detector_device") == str(detector.device)
        and previous.get("precision") == precision
        and cached_entry is previous.get("entry")
    ):
        cache.pop(frame_index + 1 if reverse else frame_index - 1, None)
        return _record_preparation(state, fallback_reason=None, cache_hit=True)
    shared_key = None
    prepared = None
    position_contract = None
    if feature_cache is not None:
        from .lta_sam import PINNED_SAM_PACKAGE_TREE_SHA256
        if getattr(feature_cache, 'position_source_identity', '') == PINNED_SAM_PACKAGE_TREE_SHA256:
            position_contract = _canonical_position_contract(model)
        frame_identity = cache_frame_identity(frame_index)
        if not isinstance(frame_identity, tuple) or not frame_identity:
            raise ValueError('Shared feature frame identity must be a nonempty immutable tuple')
        hash(frame_identity)
        shared_key = (
            frame_identity, TRACKER_FEATURE_POLICY, str(detector.device),
            tuple(original_image.shape), tuple(original_image.stride()),
            original_image.dtype, str(original_image.device), precision,
            position_contract,
        )
        prepared = feature_cache.get(shared_key, model=model)
    shared_hit = prepared is not None
    if not shared_hit:
        image = _upload_frame_image(original_image, detector.device, torch).unsqueeze(0)
        features = backbone.forward_image(
            image,
            need_sam3_out=False,
            need_interactive_out=True,
            need_propagation_out=True,
        )
        if not isinstance(features, Mapping):
            raise RuntimeError("SAM visual backbone returned a non-mapping feature set")
        prepared = {}
    # Preserve pinned cache construction order: interactive projections first,
    # then propagation projections. No grounding output contributes to either.
    for key, decoder in (() if shared_hit else (
        ("interactive", tracker.interactive_sam_mask_decoder),
        ("sam2_backbone_out", tracker.sam_mask_decoder),
    )):
        branch = features.get(key)
        if not isinstance(branch, Mapping) or branch.get("vision_mask") is not None:
            raise RuntimeError(f"SAM visual backbone returned incompatible {key} features")
        pyramid = tuple(branch.get("backbone_fpn", ()))
        positions = branch.get("vision_pos_enc")
        if len(pyramid) != 3 or not isinstance(positions, (tuple, list)) or len(positions) != 3:
            raise RuntimeError(f"SAM {key} features require exactly three pyramid levels")
        for level, (feature, position) in enumerate(zip(pyramid, positions)):
            tensor = getattr(feature, "tensors", None)
            if (
                not isinstance(tensor, torch.Tensor)
                or tensor.ndim != 4
                or tensor.shape[0] != 1
                or getattr(feature, "mask", None) is not None
                or not isinstance(position, torch.Tensor)
                or position.ndim != 4
                or position.shape[0] != 1
                or tensor.shape[-2:] != position.shape[-2:]
            ):
                raise RuntimeError(f"SAM {key} pyramid level {level} has incompatible geometry")
        # Grounding's _build_multigpu_buffer_next_chunk applies this BF16 cast
        # even with one rank and FP32 model storage. Keep it before decoder
        # projections; casting after them can change the tracker features.
        projected = [feature.tensors.to(dtype=torch.bfloat16) for feature in pyramid]
        projected[0] = decoder.conv_s0(projected[0])
        projected[1] = decoder.conv_s1(projected[1])
        prepared[key] = {
            "vision_features": projected[-1],
            "vision_mask": None,
            "vision_pos_enc": positions,
            "backbone_fpn": [
                type(feature)(tensor, None) for feature, tensor in zip(pyramid, projected)
            ],
        }
    if feature_cache is not None and not shared_hit:
        if position_contract is not None:
            prepared = feature_cache.canonicalize_positions(
                prepared, model=model,
                salt=(TRACKER_FEATURE_POLICY, str(detector.device), precision, position_contract),
            )
        # Neither the worker cache nor an active session should retain the
        # redundant SDK position tensors or unprojected pyramid after reuse.
        features = branch = positions = pyramid = feature = position = tensor = None
        feature_cache.put(shared_key, prepared, model=model)
    cache[frame_index] = (original_image, prepared)
    cache.pop(frame_index + 1 if reverse else frame_index - 1, None)
    # One identity receipt replaces the previous receipt; it retains at most
    # the same single prepared frame as the normal pinned cache.
    state[_FRAME_REUSE_KEY] = {
        "model": model, "detector": detector, "backbone": backbone,
        "tracker": tracker, "input_batch": input_batch, "image_batch": image_batch,
        "frame_index": frame_index, "image_identity": identity,
        "detector_device": str(detector.device), "precision": precision,
        "entry": cache[frame_index],
    }
    result = _record_preparation(state, fallback_reason=None, shared_cache_hit=shared_hit)
    if feature_cache is not None and not shared_hit:
        audit = state[TRACKER_FEATURE_AUDIT_KEY]
        audit['shared_feature_cache_misses'] = int(audit.get('shared_feature_cache_misses', 0)) + 1
    return result


__all__ = (
    "TRACKER_FEATURE_AUDIT_KEY",
    "TRACKER_FEATURE_POLICY",
    "prepare_tracker_frame_features",
)

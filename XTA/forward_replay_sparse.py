"""Sparse, exact replay packets for the qualified one-class direct YOLO decoder.

Only anchors that can pass the sweep's lowest gate are retained. Prototype
values are kept on a conservative union of candidate crop boxes; the decoder
still receives its original full shape, dtype and strides. The crop policy is
qualified only for consumers that crop each instance before spatial union.
No files, budgets, CUDA context, or device selection are owned by this module.
"""
from __future__ import annotations

import json
import math

import numpy as np

SCHEMA = 'xta.instance_forward_sparse/1'
_MAX_RECONSTRUCT_BYTES = 512 * 1024**2
_HEADER_LIMIT = 8192


def _torch(value):
    return hasattr(value, 'detach') and hasattr(value, 'device')


def _dtype_name(value):
    return str(value.dtype).removeprefix('torch.')


def _strides(value):
    return tuple(int(v) for v in value.stride()) if _torch(value) else tuple(int(v // value.itemsize) for v in value.strides)


def _dense_layout(shape, strides):
    """Admit contiguous axis permutations; reject holes, overlap and negatives."""
    if len(shape) != len(strides) or any(s < 0 for s in strides):
        return False
    expected = 1
    for stride, size in sorted((s, n) for n, s in zip(shape, strides) if n > 1):
        if stride != expected:
            return False
        expected *= size
    return True


def _bytes(value):
    return int(value.numel()) * int(value.element_size()) if _torch(value) else int(value.nbytes)


def _all(value):
    return bool(value.all().item()) if _torch(value) else bool(np.all(value))


def _header(metadata, like):
    raw = json.dumps(metadata, sort_keys=True, separators=(',', ':'), allow_nan=False).encode('ascii')
    array = np.frombuffer(raw, np.uint8).copy()
    if _torch(like):
        import torch
        return torch.from_numpy(array).to(device=like.device)
    return array


def _metadata(packet):
    if not isinstance(packet, (tuple, list)) or len(packet) != 5:
        raise ValueError('Not a sparse instance packet')
    first = packet[0]
    if getattr(first, 'ndim', -1) != 1 or _dtype_name(first) != 'uint8' or not 0 < int(first.shape[0]) <= _HEADER_LIMIT:
        raise ValueError('Invalid sparse instance header')
    array = first.detach().cpu().numpy() if _torch(first) else np.asarray(first)
    try:
        metadata = json.loads(array.tobytes())
    except (UnicodeError, ValueError) as exc:
        raise ValueError('Invalid sparse instance metadata') from exc
    if not isinstance(metadata, dict) or metadata.get('schema') != SCHEMA:
        raise ValueError('Not a sparse instance packet')
    return metadata


def is_sparse_instance_packet(packet):
    """Identify the numeric envelope; only its small header may transfer to CPU."""
    try:
        _metadata(packet)
        return True
    except (AttributeError, TypeError, ValueError, RuntimeError):
        return False


def _pack_torch(head, proto, floor, input_hw, support):
    import torch
    b, channels, anchors = map(int, head.shape)
    _, masks, ph, pw = map(int, proto.shape)
    # CUDA compaction compares fp32-converted scores with an fp32 gate. Torch's
    # synchronized fallback can round a scalar gate to head dtype first. Keep
    # the union of both admissible sets, including equality for strict gates.
    gate = min(float(np.float32(floor / 100.)), float(np.asarray(floor / 100., dtype=_dtype_name(head))))
    eligible = head[:, 4, :].to(torch.float32) >= gate
    if floor == 0 and _all(eligible):
        return None  # dense floor-zero case: avoid full head gather/proto workspace
    positions = torch.nonzero(eligible, as_tuple=False)
    flat = positions[:, 0] * anchors + positions[:, 1]
    rows = head[positions[:, 0], :, positions[:, 1]].clone()
    empty_ids = torch.empty(0, dtype=torch.int64, device=head.device)
    empty_values = torch.empty((0, masks), dtype=proto.dtype, device=proto.device)
    if not positions.shape[0]:
        return flat, rows, empty_ids, empty_values, 'empty'
    boxes = rows[:, :4].to(torch.float32)
    if not _all(torch.isfinite(boxes)) or not _all(boxes[:, 2:] >= 0):
        return None
    if support == 'full':
        return flat, rows, empty_ids, proto.clone(), 'full'
    # One-cell halo covers the float32 crop arithmetic's boundary rounding.
    # Both CUDA and Torch crop on GLOBAL prototype coordinates; no domain resize.
    scale = boxes.new_tensor((pw / input_hw[1], ph / input_hw[0]))
    low_edge = (boxes[:, :2] - .5 * boxes[:, 2:]) * scale
    high_edge = (boxes[:, :2] + .5 * boxes[:, 2:]) * scale
    if not _all(torch.isfinite(low_edge) & torch.isfinite(high_edge)):
        return None
    # Clip in floating point BEFORE conversion: very large finite coordinates
    # must not overflow int64 and accidentally become a near-origin crop.
    lower = torch.floor(low_edge).clamp(-1, max(ph, pw) + 1).to(torch.int64) - 1
    upper = torch.ceil(high_edge).clamp(-1, max(ph, pw) + 1).to(torch.int64) + 1
    lower[:, 0].clamp_(0, pw); lower[:, 1].clamp_(0, ph)
    upper[:, 0].clamp_(0, pw); upper[:, 1].clamp_(0, ph)
    valid = (upper > lower).all(dim=1)
    lower, upper, batch = lower[valid], upper[valid], positions[valid, 0]
    diff = torch.zeros(b * (ph + 1) * (pw + 1), dtype=torch.int32, device=proto.device)
    x0, y0, x1, y1 = lower[:, 0], lower[:, 1], upper[:, 0], upper[:, 1]
    offsets = torch.cat(((batch * (ph + 1) + y0) * (pw + 1) + x0,
                         (batch * (ph + 1) + y1) * (pw + 1) + x1,
                         (batch * (ph + 1) + y0) * (pw + 1) + x1,
                         (batch * (ph + 1) + y1) * (pw + 1) + x0))
    signs = torch.cat((torch.ones(2 * batch.numel(), dtype=torch.int32, device=proto.device),
                       -torch.ones(2 * batch.numel(), dtype=torch.int32, device=proto.device)))
    diff.scatter_add_(0, offsets, signs)
    covered = diff.reshape(b, ph + 1, pw + 1).cumsum(1, dtype=torch.int32).cumsum(2, dtype=torch.int32)[:, :ph, :pw] > 0
    pixels = torch.nonzero(covered.reshape(-1), as_tuple=False).reshape(-1)
    pb = torch.div(pixels, ph * pw, rounding_mode='floor')
    rem = pixels.remainder(ph * pw)
    py, px = torch.div(rem, pw, rounding_mode='floor'), rem.remainder(pw)
    values = proto[pb, :, py, px].clone()
    return flat, rows, pixels, values, 'candidate_boxes'


def _pack_numpy(head, proto, floor, input_hw, support):
    b, channels, anchors = head.shape
    _, masks, ph, pw = proto.shape
    gate = min(float(np.float32(floor / 100.)), float(np.asarray(floor / 100., dtype=head.dtype)))
    eligible = head[:, 4, :].astype(np.float32) >= gate
    if floor == 0 and np.all(eligible):
        return None
    positions = np.argwhere(eligible)
    flat = positions[:, 0] * anchors + positions[:, 1]
    rows = head[positions[:, 0], :, positions[:, 1]].copy()
    if not len(flat):
        return flat, rows, np.empty(0, np.int64), np.empty((0, masks), proto.dtype), 'empty'
    boxes = rows[:, :4].astype(np.float32)
    if not np.all(np.isfinite(boxes)) or not np.all(boxes[:, 2:] >= 0):
        return None
    if support == 'full':
        return flat, rows, np.empty(0, np.int64), proto.copy(), 'full'
    scale = np.asarray((pw / input_hw[1], ph / input_hw[0]), np.float32)
    with np.errstate(over='ignore', invalid='ignore'):
        low_edge = (boxes[:, :2] - np.float32(.5) * boxes[:, 2:]) * scale
        high_edge = (boxes[:, :2] + np.float32(.5) * boxes[:, 2:]) * scale
    if not np.all(np.isfinite(low_edge) & np.isfinite(high_edge)):
        return None
    lower = np.clip(np.floor(low_edge), -1, max(ph, pw) + 1).astype(np.int64) - 1
    upper = np.clip(np.ceil(high_edge), -1, max(ph, pw) + 1).astype(np.int64) + 1
    lower = np.clip(lower, 0, [pw, ph]); upper = np.clip(upper, 0, [pw, ph])
    valid = np.all(upper > lower, axis=1)
    lower, upper, batch = lower[valid], upper[valid], positions[valid, 0]
    diff = np.zeros((b, ph + 1, pw + 1), np.int32)
    np.add.at(diff, (batch, lower[:, 1], lower[:, 0]), 1)
    np.add.at(diff, (batch, upper[:, 1], upper[:, 0]), 1)
    np.add.at(diff, (batch, lower[:, 1], upper[:, 0]), -1)
    np.add.at(diff, (batch, upper[:, 1], lower[:, 0]), -1)
    covered = diff.cumsum(1, dtype=np.int32).cumsum(2, dtype=np.int32)[:, :ph, :pw] > 0
    pixels = np.flatnonzero(covered)
    pb, rem = np.divmod(pixels, ph * pw); py, px = np.divmod(rem, pw)
    return flat, rows, pixels, proto[pb, :, py, px].copy(), 'candidate_boxes'


def pack_instance_packet(head, proto, *, floor_hundredths, input_shape_hw, proto_support='candidate_boxes'):
    """Gather a numeric sparse envelope, or return None for ordinary dense replay.

    Torch inputs stay on their current device through selection and gathering.
    Supported values are fp16/fp32, a one-class [B,5+M,A] head and [B,M,H,W]
    prototypes, with contiguous/permuted-dense layouts. Callers must qualify
    crop-before-union consumers; use ``full`` when the proto read domain differs.
    """
    try:
        if type(floor_hundredths) is not int or not 0 <= floor_hundredths <= 100:
            return None
        hw = tuple(input_shape_hw)
        if len(hw) != 2 or any(type(v) is not int or v <= 0 for v in hw):
            return None
        if proto_support not in ('candidate_boxes', 'full') or _torch(head) != _torch(proto):
            return None
        if head.ndim != 3 or proto.ndim != 4:
            return None
        hs, ps = tuple(map(int, head.shape)), tuple(map(int, proto.shape))
        if any(n <= 0 for n in hs + ps) or hs[0] != ps[0] or hs[1] != 5 + ps[1] or hs[2] >= 2**30:
            return None
        if any(_dtype_name(v) not in ('float16', 'float32') for v in (head, proto)):
            return None
        strides = [_strides(v) for v in (head, proto)]
        if not all(_dense_layout(shape, stride) for shape, stride in zip((hs, ps), strides)):
            return None
        if _bytes(head) + _bytes(proto) > _MAX_RECONSTRUCT_BYTES:
            return None
        if _torch(head) and head.device != proto.device:
            return None
        encoded = (_pack_torch if _torch(head) else _pack_numpy)(head, proto, floor_hundredths, hw, proto_support)
        if encoded is None:
            return None
        anchors, rows, pixels, values, mode = encoded
        metadata = dict(schema=SCHEMA, head_shape=hs, proto_shape=ps, head_strides=strides[0],
                        proto_strides=strides[1], head_dtype=_dtype_name(head), proto_dtype=_dtype_name(proto),
                        floor_hundredths=floor_hundredths, input_shape_hw=hw, proto_support=mode)
        packet = (_header(metadata, head), anchors, rows, pixels, values)
        if sum(map(_bytes, packet)) >= _bytes(head) + _bytes(proto):
            return None
        return packet
    except (AttributeError, IndexError, OverflowError, RuntimeError, TypeError, ValueError):
        return None


def _validated(metadata, packet):
    shapes, strides, dtypes = [], [], []
    for prefix, rank in (('head', 3), ('proto', 4)):
        shape, stride = metadata.get(prefix + '_shape'), metadata.get(prefix + '_strides')
        dtype = metadata.get(prefix + '_dtype')
        if (not isinstance(shape, list) or len(shape) != rank or any(type(v) is not int or v <= 0 for v in shape)
                or not isinstance(stride, list) or any(type(v) is not int for v in stride)
                or dtype not in ('float16', 'float32') or not _dense_layout(shape, stride)):
            raise ValueError('Invalid sparse instance layout')
        shapes.append(tuple(shape)); strides.append(tuple(stride)); dtypes.append(dtype)
    hs, ps = shapes
    if hs[0] != ps[0] or hs[1] != 5 + ps[1] or sum(math.prod(s) * np.dtype(d).itemsize for s, d in zip(shapes, dtypes)) > _MAX_RECONSTRUCT_BYTES:
        raise ValueError('Invalid sparse instance dimensions')
    _, indices, rows, pixels, values = packet
    if any(_torch(v) != _torch(packet[0]) for v in packet):
        raise ValueError('Mixed sparse instance containers')
    if _torch(packet[0]) and any(v.device != packet[0].device for v in packet):
        raise ValueError('Mixed sparse instance devices')
    for ids, limit in ((indices, hs[0] * hs[2]), (pixels, ps[0] * ps[2] * ps[3])):
        if ids.ndim != 1 or _dtype_name(ids) != 'int64' or not _all((ids >= 0) & (ids < limit)) or not _all(ids[1:] > ids[:-1]):
            raise ValueError('Invalid sparse instance indices')
    if tuple(rows.shape) != (int(indices.shape[0]), hs[1]) or _dtype_name(rows) != dtypes[0]:
        raise ValueError('Invalid sparse instance head rows')
    mode = metadata.get('proto_support')
    expected = ps if mode == 'full' else (int(pixels.shape[0]), ps[1])
    if mode not in ('empty', 'full', 'candidate_boxes') or tuple(values.shape) != expected or _dtype_name(values) != dtypes[1]:
        raise ValueError('Invalid sparse instance prototype values')
    if mode == 'empty' and (indices.shape[0] or pixels.shape[0]):
        raise ValueError('Invalid empty sparse instance packet')
    if mode == 'full' and pixels.shape[0]:
        raise ValueError('Invalid full sparse instance packet')
    return shapes, strides, dtypes


def unpack_instance_packet(packet, *, device=None):
    """Restore full decoder geometry/strides; optional device uploads packed host data only."""
    metadata = _metadata(packet)
    shapes, strides, dtypes = _validated(metadata, packet)
    _, indices, rows, pixels, values = packet
    hs, ps = shapes
    host_upload = device is not None and not _torch(packet[0])
    if host_upload:
        import torch
        # Header and all structural/index checks were completed on the host.
        # Transfer only sparse numeric data; never upload and reread the header.
        indices, rows, pixels, values = (torch.from_numpy(v).to(device=device) for v in (indices, rows, pixels, values))
    if host_upload or _torch(packet[0]):
        import torch
        head = torch.empty_strided(hs, strides[0], dtype=getattr(torch, dtypes[0]), device=rows.device)
        proto = torch.empty_strided(ps, strides[1], dtype=getattr(torch, dtypes[1]), device=values.device)
        head.zero_(); proto.zero_(); head[:, 4, :] = -float('inf')
        batch = torch.div(indices, hs[2], rounding_mode='floor')
        head[batch, :, indices.remainder(hs[2])] = rows
        if metadata['proto_support'] == 'full':
            proto.copy_(values)
        else:
            pb = torch.div(pixels, ps[2] * ps[3], rounding_mode='floor'); rem = pixels.remainder(ps[2] * ps[3])
            proto[pb, :, torch.div(rem, ps[3], rounding_mode='floor'), rem.remainder(ps[3])] = values
    else:
        def allocate(shape, stride, dtype):
            owner = np.empty(math.prod(shape), dtype=dtype)
            return np.ndarray(shape, dtype=dtype, buffer=owner, strides=tuple(v * owner.itemsize for v in stride))
        head, proto = [allocate(s, st, dt) for s, st, dt in zip(shapes, strides, dtypes)]
        head.fill(0); proto.fill(0); head[:, 4, :] = -np.inf
        batch, anchor = np.divmod(indices, hs[2]); head[batch, :, anchor] = rows
        if metadata['proto_support'] == 'full':
            proto[...] = values
        else:
            pb, rem = np.divmod(pixels, ps[2] * ps[3]); py, px = np.divmod(rem, ps[3]); proto[pb, :, py, px] = values
    return head, proto




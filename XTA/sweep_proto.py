"""Disjoint candidate bins for XTA's one-class CUDA retina decoder."""
from __future__ import annotations

from functools import lru_cache
import os


@lru_cache(maxsize=32)
def _gates(cutoffs, device, pid):
    import torch

    return torch.tensor([cent / 100 for cent in cutoffs], dtype=torch.float32, device=device)


def _candidate_bins(head, cutoffs):
    """Pack accepted anchors in ascending score order, using the scalar CUDA >= gate."""
    import torch

    gates = _gates(cutoffs, head.device, os.getpid())
    scores = head[4].float()
    accepted = (scores >= gates[0]).nonzero(as_tuple=True)[0]
    if accepted.numel() == 0:
        return None, [0] * (len(cutoffs) + 1)
    sorted_scores, order = scores.index_select(0, accepted).sort()
    offsets = torch.searchsorted(sorted_scores, gates).tolist()
    offsets.append(accepted.numel())
    return head.index_select(1, accepted.index_select(0, order)), offsets


def consume_sibling_bins(sink, result_index, head, proto, image):
    """Consume exact sibling masks while this frame's raw prototypes are still resident.

    Decode every accepted instance in only one bin. Union and positive-score
    maxima commute across the independently decoded bins, even at overlapping
    box edges. Only the accumulated planes and the current bin are resident.
    """
    from . import inference

    cutoffs = tuple(sink.cutoffs)
    # At up to four siblings, the original direct decoder was faster than
    # binning the real 1024/2048 heads. Both routes consume the same forward.
    if (len(cutoffs) <= 4 or not getattr(head, 'is_cuda', False)
            or head.ndim != 2 or proto.ndim != 3 or head.shape[0] != 5 + proto.shape[0]):
        return False
    import torch

    try:
        packed, offsets = _candidate_bins(head, cutoffs)
    except torch.OutOfMemoryError:
        return False  # The existing per-cut decoder uses a different workspace.
    total = offsets[-1]
    bins = sum(lo != hi for lo, hi in zip(offsets, offsets[1:]))
    sink.record_proto_work(total, sum(total - lo for lo in offsets[:-1]), bins)
    union = scores = None
    payload = inference.GpuFlattenedRetinaPayload(union_gpu=None, conf_gpu=None, instance_count=0)
    proto = proto.contiguous()
    previous_cutoff = None
    for index in range(len(cutoffs)-1, -1, -1):
        cent = cutoffs[index]
        lo, hi = offsets[index:index+2]
        if hi > lo:
            # The caller consumes each output before reusing its live backend bindings.
            bin_payload = inference._build_direct_device_compacted_payload(
                packed[:, lo:hi], proto, image, cent / 100,
                own_inputs=False,
            )
            if bin_payload is None:
                sink.disable("CUDA retina decoding is unavailable for sibling cutoffs")
                return True
            bin_union, bin_scores = bin_payload.union_gpu, bin_payload.conf_gpu
            if union is None:
                union, scores = bin_union, bin_scores
            else:
                torch.maximum(union, bin_union, out=union)
                if scores is not None:
                    torch.maximum(scores, bin_scores, out=scores)
            payload = inference._direct_payload_from_planes(
                union, scores, count=total-lo, refs=(bin_payload,),
            )
            sink.record_detector_work(mask_decodes=1)
        elif (previous_cutoff is not None and hasattr(sink, 'consume_same_result')
              and sink.consume_same_result(result_index, cent, previous_cutoff)):
            previous_cutoff = cent
            continue
        sink.consume_result(result_index, cent, payload)
        previous_cutoff = cent
    return True

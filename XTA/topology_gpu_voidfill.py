"""Bounded GPU background labeling with exact reconciliation between slabs."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from typing import Dict, List, Sequence, Tuple

import numpy as np

from .config import GIB
from .inference import _try_import_cupy_ndimage
from .runtime import runtime_telemetry


def prepare_gpu_void_fill(
    mask: np.ndarray,
    labels_store: np.ndarray,
    *,
    connectivity: int,
    device_ids: Sequence[int],
) -> Tuple[List[Tuple[int, int]], List[np.ndarray], Dict[str, object]]:
    """Return slab LUTs without modifying mask; a failed preparation is discardable."""
    from .topology import (
        _UnionFind,
        _adjacent_gid_pair_codes,
        _adjacent_xy_offsets_for_3d_connectivity,
    )

    modules = _try_import_cupy_ndimage()
    if modules is None:
        raise RuntimeError("CuPy ndimage and a CUDA device are unavailable")
    cp, ndi = modules
    depth, height, width = (int(v) for v in mask.shape)
    plane = height * width
    # CuPy's int32 local label indices must cover the complete slab.
    max_slices = min(64, depth, ((1 << 31) - 1) // max(1, plane))
    admitted = []
    for device_id in dict.fromkeys(int(v) for v in device_ids):
        with cp.cuda.Device(device_id):
            free_bytes, _ = cp.cuda.Device(device_id).mem_info
        slices = min(max_slices, max(0, (int(free_bytes) - GIB) // max(1, plane * 24)))
        if slices > 0:
            admitted.append((device_id, slices))
    if not admitted:
        raise RuntimeError("No selected CUDA device has sufficient void-fill workspace")
    block = min(slices for _, slices in admitted)
    slabs = [(z, min(depth, z + block)) for z in range(0, depth, block)]
    devices = [device for device, _ in admitted[:len(slabs)]]
    assignments = {device: slabs[rank::len(devices)] for rank, device in enumerate(devices)}
    coordinates = np.indices((3, 3, 3), dtype=np.int8) - 1
    structure = np.sum(np.abs(coordinates), axis=0) <= {6: 1, 18: 2, 26: 3}[connectivity]

    def label_device(device_id: int):
        results = []
        with cp.cuda.Device(device_id):
            try:
                device_structure = cp.asarray(structure)
                for z0, z1 in assignments[device_id]:
                    background = cp.asarray(np.asarray(mask[z0:z1])) == 0
                    labels, count = ndi.label(background, structure=device_structure, output=cp.int32)
                    del background
                    # Transfer directly into the admitted RAM/memmap workspace.
                    labels.get(out=np.asarray(labels_store[z0:z1]).view(np.int32))
                    faces = [labels[:, 0, :].ravel(), labels[:, -1, :].ravel(),
                             labels[:, :, 0].ravel(), labels[:, :, -1].ravel()]
                    if z0 == 0:
                        faces.append(labels[0].ravel())
                    if z1 == depth:
                        faces.append(labels[-1].ravel())
                    boundary_ids = cp.asnumpy(cp.unique(cp.concatenate(faces)))
                    results.append((z0, z1, int(count), boundary_ids))
                    del labels, faces, boundary_ids
                del device_structure
                return results
            finally:
                # Each coordinator owns its selected devices for this stage.
                cp.get_default_memory_pool().free_all_blocks()

    with runtime_telemetry().span("topology.void_fill.gpu_label"):
        with ThreadPoolExecutor(max_workers=len(devices), thread_name_prefix="void-fill-gpu") as executor:
            futures = [executor.submit(label_device, device) for device in devices]
            records = sorted(record for future in futures for record in future.result())

    with runtime_telemetry().span("topology.void_fill.reconcile"):
        uf = _UnionFind()
        offsets = []
        for z0, z1, count, boundary_ids in records:
            ids = uf.new_ids(count)
            offset = int(ids[0]) - 1 if count else 0
            offsets.append(offset)
            for local_id in boundary_ids:
                if int(local_id) > 0:
                    uf.mark_boundary(offset + int(local_id))
        neighbors = _adjacent_xy_offsets_for_3d_connectivity(connectivity)
        for index in range(1, len(records)):
            z0 = records[index][0]
            pairs = _adjacent_gid_pair_codes(
                labels_store[z0 - 1], labels_store[z0], neighbors,
                prev_offset=offsets[index - 1], curr_offset=offsets[index],
            )
            uf.union_pair_codes(pairs)
        roots = uf.root_map()
        lookups = []
        for (_, _, count, _), offset in zip(records, offsets):
            lookup = np.zeros(count + 1, dtype=np.uint8)
            lookup[1:] = ~uf.touches_boundary[roots[offset + 1:offset + count + 1]]
            lookups.append(lookup)
    return slabs, lookups, {"backend": "cuda", "devices": devices, "chunks": len(slabs),
                            "block_slices": block, "fallback_reason": None}

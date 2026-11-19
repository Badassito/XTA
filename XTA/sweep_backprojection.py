"""Share categorical backprojection of certified native cutoff masks for one view."""
from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from contextlib import closing, contextmanager
from dataclasses import replace
import errno
from itertools import pairwise
import math
import mmap
import operator
import os
from pathlib import Path
import shutil
import tempfile
import threading
import time
import uuid

import numpy as np

from ._deps import _numba, cv2
from .cylindrical_cuda_projection import RadialEncodedSlice
from .cuda_d1 import _nrrd_layer_key, _nrrd_layer_name
from .geometry import physical_view_name
from .interpolation import (
    CTILE_INDEX_DTYPE, INTERNAL_PACKED_CVOL_FORMAT, IncrementalRawBBoxMaskStoreWriter, NrrdLayerRef,
)

# Projected sibling stores live on scratch until each later-cut child consumes
# them. Row packbits keep them 8x smaller than uint8 crops (the D1 floor layers
# use the same terminal format) and every downstream reader decodes both.
_SIBLING_STORE_FORMAT = INTERNAL_PACKED_CVOL_FORMAT
from .sweep_projection import sweep_rank_projection_reader


class SweepProjectionUnsupported(Exception):
    """The family can still use the original independent scalar projection."""


def _shape(value, name):
    try:
        result = tuple(operator.index(n) for n in value)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must contain three positive integer dimensions") from exc
    if len(result) != 3 or min(result) < 1:
        raise ValueError(f"{name} must contain three positive integer dimensions")
    return result


def _backing_shape(view, native, target):
    if view.family == "orthogonal":
        axes = {"transverse": (0,1,2), "sagittal": (1,0,2), "coronal": (1,2,0)}
        base = physical_view_name(view)
        if base not in axes:
            raise SweepProjectionUnsupported(f"Unknown Cartesian cutoff projection orientation {base!r}")
        # The scalar writer restores Cartesian layers after axis permutation.
        return tuple(native[i] for i in axes[base])
    if view.family in ("tilted", "azimuthal", "radial", "spherical"):
        return target
    raise SweepProjectionUnsupported(f"Cutoff projection does not support view family {view.family!r}")


def _crops(cent, z, native_hw, read_mask_plane, read_mask_crops):
    if read_mask_crops is not None:
        yield from read_mask_crops(cent, z)
        return
    plane = np.asarray(read_mask_plane(cent, z))
    if plane.shape != native_hw or plane.dtype not in (np.dtype(np.uint8), np.dtype(bool)):
        raise ValueError("Native cutoff reader returned an invalid mask plane shape or dtype")
    x, y, w, h = cv2.boundingRect(plane.view(np.uint8))
    if w and h:
        yield y, y+h, x, x+w, plane[y:y+h,x:x+w]


@_numba.njit(nogil=True, cache=True)
def _merge_rank_crop(plane, mask, top, left, level, check_previous):
    populated = gap = False
    for row in range(mask.shape[0]):
        for col in range(mask.shape[1]):
            value = mask[row,col]
            if value > 1:
                raise ValueError("Native cutoff masks must be binary")
            if value:
                populated = True
                if check_previous and plane[top+row,left+col] < level-1:
                    gap = True
                plane[top+row,left+col] = level
    return populated, gap


@contextmanager
def _rank_input(cutoffs, shape, directory, read_mask_plane, read_mask_crops, budget):
    """Construct one sparse-file rank backing and certify adjacent mask inclusion."""
    from .sweep_view_decode import store_rank_region
    backing = tempfile.TemporaryFile(dir=directory)
    try:
        ranks = None
        native_hw = shape[1:]
        plane = None
        boxes = np.zeros((shape[0],4), dtype=np.int64)
        present = set()
        breaks = set()
        for z in range(shape[0]):
            top, bottom, left, right = native_hw[0], 0, native_hw[1], 0
            frames = ([read_mask_crops(cent, z) for cent in cutoffs] if read_mask_crops is not None
                      else [None]*len(cutoffs))
            if any(getattr(crops, "wants_rank_plane", False) for crops in frames):
                # Native journal frames merge every level in one nogil call. A
                # level's gap test depends only on earlier frames and on itself,
                # so all tests are known before the frame starts.
                if plane is None:
                    plane = np.zeros(native_hw, np.uint8)
                merged = frames[0].merge_rank_frame(frames, plane, [
                    level > 1 and level-1 not in breaks for level in range(1, len(cutoffs)+1)
                ]) if hasattr(frames[0], "merge_rank_frame") else None
                if merged is not None:
                    populated, gaps, (y0, y1, x0, x1) = merged
                    for level, cent in enumerate(cutoffs, 1):
                        if gaps[level-1]:
                            breaks.add(level-1)
                        if populated[level-1]:
                            present.add(cent)
                    if any(populated):
                        top, bottom, left, right = y0, y1, x0, x1
            for level, cent in enumerate(cutoffs, 1):
                crops = (frames[level-1] if read_mask_crops is not None
                         else _crops(cent,z,native_hw,read_mask_plane,None))
                if getattr(crops, "wants_rank_plane", False):
                    # A native frame the fused call declined merges this level's
                    # rows in one nogil call per payload, exactly as the per-crop
                    # loop below would; any row it declines is still yielded to it.
                    populated, gap, (y0, y1, x0, x1) = crops.merge_rank_level(
                        plane, level, level > 1 and level-1 not in breaks)
                    if gap:
                        breaks.add(level-1)
                    if populated:
                        present.add(cent)
                        top, bottom = min(top,y0), max(bottom,y1)
                        left, right = min(left,x0), max(right,x1)
                for y0, y1, x0, x1, raw in crops:
                    y0, y1, x0, x1 = map(operator.index, (y0,y1,x0,x1))
                    mask = np.asarray(raw)
                    if not (0 <= y0 <= y1 <= native_hw[0] and 0 <= x0 <= x1 <= native_hw[1]
                            and mask.shape == (y1-y0,x1-x0)
                            and mask.dtype in (np.dtype(np.uint8), np.dtype(bool))):
                        raise ValueError("Invalid native cutoff mask crop")
                    if not mask.size:
                        continue
                    if plane is None:
                        plane = np.zeros(native_hw, np.uint8)
                    # Other contributions to this cutoff may already have assigned
                    # this level. Only a value below the *previous* level is a gap.
                    populated, gap = _merge_rank_crop(plane, mask.view(np.uint8), y0, x0, level,
                                                       level > 1 and level-1 not in breaks)
                    if not populated:
                        continue
                    present.add(cent)
                    if gap:
                        breaks.add(level-1)
                    top, bottom = min(top,y0), max(bottom,y1)
                    left, right = min(left,x0), max(right,x1)
            if bottom:
                if ranks is None:
                    ranks = _map_native_ranks(backing, shape, budget)
                boxes[z] = (top,bottom,left,right)
                budget.add_native_crop(z, shape, top, bottom, left, right)
                # ranks[z][region] = plane[region]; plane[region] = 0, in one nogil call.
                store_rank_region(np.asarray(ranks[z]), plane, top, bottom, left, right)
        del plane
        yield ranks, boxes, present, sorted(breaks)
    finally:
        # The reader releases plan/stream state before returning here. Keep the
        # array mapping owned by NumPy until any exception/quarantine references
        # are released rather than explicitly closing live ndarray buffers.
        backing.close()
        budget.release_native()


def _map_native_ranks(backing, shape, budget):
    length = math.prod(shape)
    if os.name == "nt":
        # The Windows CRT's truncate can materialize a file even on NTFS.
        # Mark the new file sparse, extend with a single tail byte, then map
        # it without truncation. On a filesystem lacking sparse-file support
        # the full logical backing must be admitted before extension.
        import ctypes
        import msvcrt
        returned = ctypes.c_ulong()
        handle = ctypes.c_void_p(msvcrt.get_osfhandle(backing.fileno()))
        sparse = bool(ctypes.windll.kernel32.DeviceIoControl(
            handle, 0x000900C4, None, 0, None, 0, ctypes.byref(returned), None))
        budget.start_native_file(length, sparse=sparse)
        backing.seek(length-1)
        backing.write(b"\0")
        backing.flush()
    else:
        budget.start_native_file(length, sparse=True)
        backing.truncate(length)
    return np.memmap(backing, mode="r+", shape=shape, dtype=np.uint8)


class _BackingBudget:
    """Fund actual sparse writes; the full native mmap extent is only an address space."""

    def __init__(self, directory, index_bytes, limit, host_budget):
        from .runtime import path_is_memory_backed
        self.directory = directory
        self.limit = limit
        self.index_bytes = index_bytes
        self.host_budget = host_budget
        self.memory_backed = path_is_memory_backed(directory)
        self.native_bytes = 0
        self.peak_native_bytes = 0
        self.dense_native = False
        self.output_bytes = 0
        self.polls = 0
        self.until_poll = 0
        self._fund(index_bytes)

    def _fund(self, nbytes):
        if (self.limit is not None and
                self.index_bytes + self.native_bytes + self.output_bytes > self.limit):
            raise SweepProjectionUnsupported("Cutoff projections exceed the view's private scratch allowance")
        # Do not statvfs or read cgroups per slice. Reserve small chunks against
        # *current* free space and preserve the same disk floor as the native journals.
        if nbytes > self.until_poll or self.polls == 0:
            headroom = shutil.disk_usage(self.directory).free - 64*1024**2
            if self.memory_backed:
                from .workspace import available_anon_work_bytes
                headroom = min(headroom, available_anon_work_bytes() - self.host_budget - 64*1024**2)
            if nbytes > headroom:
                raise SweepProjectionUnsupported("Insufficient live disk or tmpfs headroom for cutoff projection")
            self.until_poll = min(headroom, max(nbytes, 64*1024**2))
            self.polls += 1
        self.until_poll -= nbytes

    def add_native_crop(self, z, shape, top, bottom, left, right):
        # Only touched pages of this new sparse backing need funding. A span
        # across adjacent rows often reuses pages, hence the tighter of two bounds.
        if self.dense_native:
            return
        page = mmap.ALLOCATIONGRANULARITY
        h, w = bottom - top, right - left
        start = (z * shape[1] + top) * shape[2] + left
        span = (h-1)*shape[2] + w
        covered = min(((start % page + span + page-1)//page)*page,
                      h*((w + 2*page-2)//page)*page)
        self.native_bytes += covered
        self.peak_native_bytes = max(self.peak_native_bytes, self.native_bytes)
        self._fund(covered)

    def start_native_file(self, size, *, sparse):
        self.dense_native = not sparse
        self.native_bytes = size if not sparse else (mmap.ALLOCATIONGRANULARITY if os.name == "nt" else 0)
        self.peak_native_bytes = max(self.peak_native_bytes, self.native_bytes)
        self._fund(self.native_bytes)

    def release_native(self):
        self.native_bytes = 0
        self.dense_native = False
        self.until_poll = 0

    def add(self, nbytes):
        self.output_bytes += nbytes
        self._fund(nbytes)


def _ref(view, model_name, shape, key, path, stats):
    return NrrdLayerRef(key=key, name=_nrrd_layer_name(view=view, source="fullframe", mask_kind="yolo",
        pass_index=0, stage="pre_interpolation"), path=path, shape=shape, dtype="uint8",
        storage_format=_SIBLING_STORE_FORMAT, model_name=model_name, view_name=view.name,
        physical_view_name=physical_view_name(view), aug_id=view.tta_aug_id,
        angle_deg=float(view.tta_angle_deg), view_family=view.family,
        source="fullframe", mask_kind="yolo", pass_index=0, stage="pre_interpolation",
        description="Cleaned full-frame YOLO mask before interpolation bridges.",
        segment_extent_ijk=tuple(stats["segment_extent_ijk"]), segment_extent_shape_tyx=shape,
        segment_extent_source="raw_bbox_cvol_index")


@_numba.njit(nogil=True, cache=True)
def _cutoff_extents(region, level_count):
    """Measure each rank once, then form all cumulative tight boxes and counts."""
    height, width = region.shape
    stats = np.zeros((level_count + 1, 5), np.int64)
    stats[:, 0] = height
    stats[:, 2] = width
    for y in range(height):
        for x in range(width):
            rank = region[y, x]
            if rank:
                if rank > level_count:
                    raise ValueError("Projected value exceeds its declared cutoff rank")
                stats[rank, 0] = min(stats[rank, 0], y)
                stats[rank, 1] = max(stats[rank, 1], y + 1)
                stats[rank, 2] = min(stats[rank, 2], x)
                stats[rank, 3] = max(stats[rank, 3], x + 1)
                stats[rank, 4] += 1
    for i in range(level_count-1, 0, -1):
        stats[i, 0] = min(stats[i, 0], stats[i+1, 0])
        stats[i, 1] = max(stats[i, 1], stats[i+1, 1])
        stats[i, 2] = min(stats[i, 2], stats[i+1, 2])
        stats[i, 3] = max(stats[i, 3], stats[i+1, 3])
        stats[i, 4] += stats[i+1, 4]
    return stats


def _ordered_planes(project, z0, z1, lanes):
    """At most lanes live outputs, including the one held by the ordered consumer."""
    if lanes <= 1:
        for z in range(z0, z1):
            yield z, project(z)
        return
    pool = ThreadPoolExecutor(max_workers=lanes, thread_name_prefix="sweep-project")
    queue = deque()
    try:
        indices = iter(range(z0, z1))
        for _ in range(min(lanes, z1-z0)):
            z = next(indices)
            queue.append((z,pool.submit(project,z)))
        while queue:
            z, future = queue.popleft()
            yield z, future.result()
            del future
            next_z = next(indices,None)
            if next_z is not None:
                queue.append((next_z,pool.submit(project,next_z)))
    finally:
        pool.shutdown(wait=True, cancel_futures=True)


@_numba.njit(nogil=True, cache=True)
def _pack_rank_level(region, y0, y1, x0, x1, level):
    """Row-wise little-endian packbits of ``region[y0:y1, x0:x1] >= level``.

    Matches ``np.packbits(..., axis=1, bitorder="little")``: zero tail bits per
    row. Compiled without the GIL so projection lanes encode in parallel.
    """
    height = y1 - y0
    width = x1 - x0
    out = np.zeros((height, (width + 7) // 8), np.uint8)
    for row in range(height):
        source_row = y0 + row
        for col in range(width):
            if region[source_row, x0 + col] >= level:
                out[row, col >> 3] |= np.uint8(1 << (col & 7))
    return out


def _encode_rank_plane(z, output, read_crop, bounds, shape, level_count, active):
    """Validate, measure and pack one projected plane on its projection lane.

    Returns (read counters, [(cutoff, record, payload), ...]). The ordered
    consumer then only appends the already encoded blocks, so per-plane
    scanning and packing no longer serialize a whole publication.
    """
    reads = {"rank_crop_reads": 0, "empty_crop_reads": 0, "rank_dense_reads": 0,
             "rank_bbox_scan_pixels": 0, "rank_statistics_pixels": 0}
    if read_crop is not None:
        reads["rank_crop_reads"] = 1
        if output is None:
            reads["empty_crop_reads"] = 1
            return reads, ()
        top, bottom, left, right, region = output
        if (not (0 <= top < bottom <= shape[1] and 0 <= left < right <= shape[2])
                or region.dtype != np.uint8 or region.shape != (bottom-top, right-left)):
            raise ValueError("Invalid sparse projected rank crop")
    else:
        reads["rank_dense_reads"] = 1
        top, left, bottom, right = bounds[1], bounds[2], bounds[4], bounds[5]
        region = output[top:bottom,left:right]
        reads["rank_bbox_scan_pixels"] = int(region.size)
        if not region.size:
            return reads, ()
        x, y, w, h = cv2.boundingRect(region)
        top, left = top+y, left+x
        region = region[y:y+h,x:x+w]
    reads["rank_statistics_pixels"] = int(region.size)
    if not region.size:
        return reads, ()
    stats = _cutoff_extents(region, level_count)
    pieces = []
    for level, cent in active:
        y0, y1, x0, x1, count = map(int, stats[level])
        if not count:
            break
        # The stock store consumes the certified tight box and foreground count
        # without finding them again for every cutoff.
        payload = _pack_rank_level(region, y0, y1, x0, x1, level)
        record = RadialEncodedSlice(z, top+y0, top+y1, left+x0, left+x1, count, 0, int(payload.size))
        pieces.append((cent, record, payload.reshape(-1)))
    return reads, pieces


def _project_chain(ranks, boxes, cutoffs, levels, present, view, shape, work, products,
                   published, model_name, key, memory_bytes, disk, result, diagnostics, workers):
    active = [(level, cent) for level, cent in zip(levels,cutoffs) if cent in present]
    if not active:
        return
    writers = {}
    next_z = {cent:0 for _,cent in active}
    reads = {"rank_crop_reads":0, "empty_crop_reads":0, "rank_dense_reads":0,
             "rank_bbox_scan_pixels":0, "rank_statistics_pixels":0}
    try:
        for _, cent in active:
            writers[cent] = IncrementalRawBBoxMaskStoreWriter(shape=shape, format_name=_SIBLING_STORE_FORMAT,
                store_dir=products / f"conf_{cent:03d}.cvol", desc=f"Sweep {model_name}/{view.name} at {cent:02d}",
                extra_meta={"nrrd_layer_key":key, "projection_output_shape_tyx":list(shape)})
        level_count = max(levels)
        # Compile once before lanes start rather than racing the first calls.
        _pack_rank_level(np.zeros((1, 1), np.uint8), 0, 1, 0, 1, 1)
        with sweep_rank_projection_reader(ranks, view, shape, work, level_count=level_count,
                memory_bytes=memory_bytes, known_native_boxes=boxes, source_max_rank=active[-1][0],
                workers=workers) as project:
            z0,z1 = getattr(project,"known_z_bounds",(0,shape[0]))
            bounds = getattr(project,"known_output_bbox_tyx",(0,0,0,*shape))
            reader_lanes = max(1, int(getattr(project, "parallel_workers", 1)))
            if getattr(project, "serialized_device_reader", False):
                # A CUDA reader is one serialized device lane while scanning and
                # packing run on the host: encode on a few threads so the GPU
                # lease is not held for serial host work. A CPU reader that
                # funded a single lane keeps one (its budget allows no more).
                encode_lanes = min(workers, 4)
            else:
                encode_lanes = reader_lanes
            lanes = min(workers, encode_lanes, max(1,z1-z0))
            # Each in-flight plane holds its projected crop plus one packed
            # payload per cutoff until the ordered commit; keep that within the
            # projection credit for long cutoff grids.
            held_per_plane = math.prod(shape[1:]) * (1 + (len(active) + 7) // 8)
            lanes = max(1, min(lanes, int(memory_bytes) // max(1, 2 * held_per_plane)))
            read_crop = getattr(project, "read_crop", None)
            source = read_crop or project
            ordered_reads = None
            if reader_lanes == 1 and lanes > 1:
                # A single serialized reader caches one projected block: hand
                # it the planes strictly in z order (a turnstile), so concurrent
                # encode threads never make it re-project a block out of order.
                # After its first failure no later plane reaches the reader.
                ordered_reads = threading.Condition()
                turn = [z0]
                reader_failure = []

            def read(z):
                if ordered_reads is None:
                    return source(z)
                with ordered_reads:
                    while turn[0] != z:
                        ordered_reads.wait()
                    failed = reader_failure[0] if reader_failure else None
                if failed is not None:
                    with ordered_reads:
                        turn[0] = z + 1
                        ordered_reads.notify_all()
                    raise RuntimeError("Sweep projection reader failed on an earlier plane") from failed
                try:
                    return source(z)
                except BaseException as error:
                    with ordered_reads:
                        reader_failure.append(error)
                    raise
                finally:
                    with ordered_reads:
                        turn[0] = z + 1
                        ordered_reads.notify_all()

            def encode(z):
                return _encode_rank_plane(z, read(z), read_crop, bounds, shape, level_count, active)

            with closing(_ordered_planes(encode,z0,z1,lanes)) as planes:
                for z, (delta, pieces) in planes:
                    for name, value in delta.items():
                        reads[name] += value
                    for cent, record, payload in pieces:
                        disk.add(int(record.size))
                        writer = writers[cent]
                        if next_z[cent] < z:
                            writer.consume_empty_range(next_z[cent], z-next_z[cent])
                        writer.consume_encoded_block(z, (record,), payload, packed=True)
                        next_z[cent] = z+1
                    del pieces
            info = getattr(project,"projection_diagnostics",None)
            diagnostics.append({**(info() if info else {}), **reads, "execution_lanes":lanes,
                                "nominal_output_voxels":math.prod(shape)})
        for _, cent in active:
            writer = writers[cent]
            if next_z[cent] < shape[0]:
                writer.consume_empty_range(next_z[cent], shape[0]-next_z[cent])
            stats = writer.finalize()
            result[cent] = _ref(view,model_name,shape,key,published / f"conf_{cent:03d}.cvol",stats)
    except BaseException as error:
        for writer in writers.values():
            writer.abort(error)
            writer.discard()
        raise


def project_sweep_view(*, view, model_name, cutoffs_hundredths, native_shape, output_shape,
                       work_dir, read_mask_plane=None, read_mask_crops=None, workers=1,
                       memory_bytes=256*1024**2, max_workspace_bytes=None, metrics=None,
                       retain_empty_orthogonal=False):
    """Project the independently cleaned masks for a physical view into scalar-compatible stores.

    The optional crop reader yields overlapping (y0,y1,x0,x1,binary) contributions.
    At least one reader is required. If the family is nonnested its maximal adjacent
    nested runs are projected independently, exactly retaining its disjoint memberships.
    Caller owns the returned paths until all downstream readers have joined.
    """
    if read_mask_plane is None and read_mask_crops is None:
        raise ValueError("Sweep projection requires a native mask reader")
    if operator.index(workers) < 1:
        raise ValueError("Sweep projection worker count must be positive")
    if metrics is not None and not isinstance(metrics, dict):
        raise TypeError("Sweep projection metrics must be a dictionary")
    raw_cuts = tuple(cutoffs_hundredths)
    if any(isinstance(c, (bool, np.bool_)) for c in raw_cuts):
        raise ValueError("A cutoff must be integer hundredths")
    cutoffs = tuple(operator.index(c) for c in raw_cuts)
    if not cutoffs or tuple(sorted(set(cutoffs))) != cutoffs or not 0 <= cutoffs[0] <= cutoffs[-1] <= 100:
        raise ValueError("Sweep projection cutoff hundredths must be unique, ascending and in [0,100]")
    if getattr(view,"augmentation_base_view",""):
        raise SweepProjectionUnsupported("Policy augmentation views require their scalar stage and empty-pass contract")
    native = _shape(native_shape,"native_shape")
    target = _shape(output_shape,"output_shape")
    shape = _backing_shape(view,native,target)
    if retain_empty_orthogonal and view.family == "orthogonal":
        from .d1_orthogonal_coverage import orthogonal_coverage_supported
        if not orthogonal_coverage_supported(view, native, target):
            raise SweepProjectionUnsupported("Geometry is outside the Cartesian D1 categorical coverage contract")
        # D1 owns the original source grid, including temporal contraction.
        # Its per-axis tables are those of the terminal categorical restore.
        shape = target
    if native[0] != int(view.num_slices):
        raise ValueError("Native cutoff frame count differs from the view descriptor")
    index_bytes = len(cutoffs)*shape[0]*(CTILE_INDEX_DTYPE.itemsize+1)
    budget = operator.index(memory_bytes)
    writer_space = min(1024**2,math.prod(shape[1:])) + 2*math.prod(shape[1:]) + index_bytes
    if budget < 4*math.prod(native[1:]) + 32*native[0] or budget <= writer_space:
        raise SweepProjectionUnsupported("Cutoff projection native or output planes exceed the host workspace allowance")
    if max_workspace_bytes is not None and operator.index(max_workspace_bytes) < 0:
        raise ValueError("max_workspace_bytes must be nonnegative")
    work = Path(work_dir).resolve()
    work.mkdir(parents=True,exist_ok=True)
    disk = _BackingBudget(work, index_bytes, max_workspace_bytes, budget)
    key = _nrrd_layer_key(view_name=view.name,source="fullframe",mask_kind="yolo",pass_index=0,stage="pre_interpolation")
    published = work / ("projected-cuts-" + uuid.uuid4().hex)
    result = dict.fromkeys(cutoffs)
    diagnostics = []
    timings = {"rank_input_seconds": 0.0, "projection_seconds": 0.0}
    try:
        with tempfile.TemporaryDirectory(prefix=".xta-project-",dir=work) as stage:
            temporary = Path(stage)
            products = temporary/"products"
            products.mkdir()
            started = time.perf_counter()
            with _rank_input(cutoffs,native,work,read_mask_plane,read_mask_crops,disk) as (ranks,boxes,present,breaks):
                timings["rank_input_seconds"] += time.perf_counter() - started
                started = time.perf_counter()
                boundaries = [0,*breaks,len(cutoffs)]
                start = boundaries[-2]
                # The last nested group remains exact in the global highest-level
                # image, even when earlier groups contain holes in membership.
                _project_chain(ranks,boxes,cutoffs[start:],range(start+1,len(cutoffs)+1),present,view,shape,
                    temporary,products,published,model_name,key,budget-writer_space,disk,result,diagnostics,workers)
                timings["projection_seconds"] += time.perf_counter() - started
            del ranks
            for first, stop in pairwise(boundaries[:-1]):
                cuts = cutoffs[first:stop]
                started = time.perf_counter()
                with _rank_input(cuts,native,work,read_mask_plane,read_mask_crops,disk) as (ranks,boxes,used,again):
                    timings["rank_input_seconds"] += time.perf_counter() - started
                    started = time.perf_counter()
                    if again or used != present.intersection(cuts):
                        raise RuntimeError("Native sweep masks changed after the view was sealed")
                    _project_chain(ranks,boxes,cuts,range(1,len(cuts)+1),used,view,shape,temporary,products,
                        published,model_name,key,budget-writer_space,disk,result,diagnostics,workers)
                    timings["projection_seconds"] += time.perf_counter() - started
                del ranks
            if retain_empty_orthogonal and view.family == "orthogonal":
                # The scalar D1 owner publishes its Cartesian source-space slot
                # even when every voxel is zero. The ordinary parent-prepare
                # route and the curved families have different empty contracts.
                from .cuda_d1 import _d1_finalize_bitset_layer
                for cent in cutoffs:
                    if cent not in present:
                        name = f"conf_{cent:03d}.cvol"
                        ref = _d1_finalize_bitset_layer(words=None, proven_empty=True,
                            output_shape=target, store_dir=products/name,
                            model_name=model_name, view=view)["d1_layer_ref"]
                        result[cent] = replace(ref, path=published/name,
                            description="Empty Cartesian direct YOLO layer on the scalar D1 source grid.")
            if any(ref is not None for ref in result.values()):
                products.rename(published)
    except MemoryError as error:
        raise SweepProjectionUnsupported("Cutoff projection geometry exceeds its admitted workspace") from error
    except OSError as error:
        if error.errno in (errno.ENOSPC, getattr(errno,"EDQUOT",-1)):
            raise SweepProjectionUnsupported("Cutoff projection private scratch is full") from error
        raise
    if metrics is not None:
        metrics.update(timings)
        metrics.update(cutoffs_hundredths=list(cutoffs), native_nonempty_cutoffs=sorted(present),
                       nested_chain_count=len(boundaries)-1, geometry_passes=len(diagnostics),
                       native_rank_backing_bytes=math.prod(native) if present else 0,
                       output_payload_bytes=disk.output_bytes,
                       native_sparse_page_upper_bound=disk.peak_native_bytes,
                       capacity_polls=disk.polls, scratch_memory_backed=disk.memory_backed,
                       requested_workers=int(workers), projectors=diagnostics)
    return result

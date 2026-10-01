"""Fixed, original-seed-only SAM crop tiling with recoverable halo evidence.

The production recipe matches the qualified native diagnostic: 1008-pixel
contexts, at least 128 pixels on each internal side of midpoint-owned cores.
There is no padding, crop growth, predicted seed handoff, or neighbor fallback.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
from typing import Mapping

import numpy as np

TILE_MAX = 1008
HALO = 128
STRIDE = TILE_MAX - 2 * HALO
MAX_TILES_PER_RUN = 256
MAX_TRACKER_JOBS = 20_000
MAX_SCOPE_TILES = 20_000
MAX_ASSEMBLY_BYTES = 32 * 1024**3
MAX_ACTIVE_PARENT_ASSEMBLIES = 16
SCHEMA = "xta.sam_crop_tiling/1"
TILE_EVIDENCE_SCHEMA = "xta.sam_run_tiles/1"


def resolve_sam_crop_mode(value=None):
    from .config import resolve_sam_crop_mode as resolve_captured_mode
    return (resolve_captured_mode() if value is None else
            resolve_captured_mode({"YOLO_TTA_SAM_CROP_MODE": value}))


def tiling_recipe():
    return dict(schema=SCHEMA, tile_max=TILE_MAX, halo=HALO, stride=STRIDE,
        ownership="adjacent_overlap_midpoints", no_seed_handoff=True,
        extra_context_outside_whole_crop=0)


def axis_windows(start, stop):
    start, stop = int(start), int(stop)
    if start < 0 or stop <= start:
        raise ValueError("SAM tile axis bounds must be positive exclusive intervals")
    width = min(stop - start, TILE_MAX)
    last = stop - width
    starts = list(range(start, last + 1, STRIDE))
    if starts[-1] != last:
        starts.append(last)
    seams = [(a + width + b) // 2 for a, b in zip(starts, starts[1:])]
    boundaries = [start, *seams, stop]
    return tuple((position, position + width, boundaries[index], boundaries[index + 1])
                 for index, position in enumerate(starts))


@dataclass(frozen=True)
class SamCropTile:
    tile_id: str
    crop_bbox_yx: tuple[int, int, int, int]
    ownership_bbox_yx: tuple[int, int, int, int]
    seed_foreground: int = 0

    @property
    def attempted(self):
        return self.seed_foreground > 0


@dataclass(frozen=True)
class SamTiledTrackerJob:
    original_run_index: int
    original_run: object = field(compare=False, repr=False)
    tile: SamCropTile
    run_id: str


def tile_grid(whole_crop):
    y0, x0, y1, x1 = map(int, whole_crop)
    ys, xs = axis_windows(y0, y1), axis_windows(x0, x1)
    if len(ys) * len(xs) > MAX_TILES_PER_RUN:
        raise MemoryError("SAM original hypothesis exceeds the bounded tile inventory")
    tiles = tuple(SamCropTile(f"tile_r{row:02d}_c{column:02d}",
        (y[0], x[0], y[1], x[1]), (y[2], x[2], y[3], x[3]))
        for row, y in enumerate(ys) for column, x in enumerate(xs))
    # Cartesian axis owners partition the original rectangle exactly. These
    # checks protect structural geometry; no quality policy can waive them.
    if sum((t.ownership_bbox_yx[2] - t.ownership_bbox_yx[0]) *
           (t.ownership_bbox_yx[3] - t.ownership_bbox_yx[1]) for t in tiles) != (y1-y0)*(x1-x0):
        raise ValueError("SAM tile ownership does not cover its original crop")
    for tile in tiles:
        a, b, c, d = tile.crop_bbox_yx
        e, f, g, h = tile.ownership_bbox_yx
        if not (y0 <= a <= e < g <= c <= y1 and x0 <= b <= f < h <= d <= x1):
            raise ValueError("SAM tile owner is outside its fixed context")
    return tiles


def clipped_seed_mask(run, observations, crop):
    y0, x0, y1, x1 = crop
    mask = np.zeros((y1-y0, x1-x0), dtype=bool)
    for identifier in run.seed_ids:
        observation = observations[str(identifier)]
        if int(observation.frame_index) != int(run.expected_frames[0]):
            raise ValueError("SAM original seed frame differs from its declared tile job")
        a, b, c, d = observation.bbox_yx
        cy0, cx0, cy1, cx1 = max(a, y0), max(b, x0), min(c, y1), min(d, x1)
        if cy0 < cy1 and cx0 < cx1:
            mask[cy0-y0:cy1-y0, cx0-x0:cx1-x0] |= observation.mask_crop[cy0-a:cy1-a, cx0-b:cx1-b]
    return mask


def prepare_tiled_jobs(runs, groups, observations):
    from dataclasses import replace
    jobs, inventory, assembly_bytes, total_tiles = [], {}, 0, 0
    for index, run in enumerate(runs):
        group = groups[str(run.group_id)]
        y0, x0, y1, x1 = group.context_bbox_yx
        assembly_bytes += 2 * len(run.expected_frames) * (y1-y0) * (x1-x0)
        if assembly_bytes > MAX_ASSEMBLY_BYTES:
            raise MemoryError("SAM tiled assembly exceeds its bounded staging disk budget")
        tiles = []
        for tile in tile_grid(group.context_bbox_yx):
            seed = clipped_seed_mask(run, observations, tile.crop_bbox_yx)
            tile = replace(tile, seed_foreground=int(np.count_nonzero(seed)))
            tiles.append(tile)
            if tile.attempted:
                jobs.append(SamTiledTrackerJob(index, run, tile,
                    f"{run.run_id}__{tile.tile_id}"))
                if len(jobs) > MAX_TRACKER_JOBS:
                    raise MemoryError("SAM tiled scope exceeds its bounded tracker job inventory")
        if not any(tile.attempted for tile in tiles):
            raise ValueError("SAM original hypothesis has no original seed in its complete tile cover")
        inventory[index] = tuple(tiles)
        total_tiles += len(tiles)
        if total_tiles > MAX_SCOPE_TILES:
            raise MemoryError("SAM scope exceeds its bounded complete tile inventory")
    identity = hashlib.sha256(json.dumps(dict(recipe=tiling_recipe(), runs=[
        dict(run_id=run.run_id, expected_frames=list(run.expected_frames),
             tiles=[dict(tile_id=t.tile_id, crop_bbox_yx=t.crop_bbox_yx,
                         ownership_bbox_yx=t.ownership_bbox_yx, seed_foreground=t.seed_foreground)
                    for t in inventory[index]]) for index, run in enumerate(runs)]),
        sort_keys=True).encode()).hexdigest()
    return tuple(jobs), MappingProxyType(inventory), identity, assembly_bytes


def tile_descriptor(run, tile):
    return dict(group_id=str(run.group_id), parent_run_id=str(run.run_id),
        tile_id=tile.tile_id, crop_bbox_yx=list(tile.crop_bbox_yx),
        ownership_bbox_yx=list(tile.ownership_bbox_yx),
        expected_frames=list(run.expected_frames), injected_frames=[int(run.expected_frames[0])],
        seed_ids=list(run.seed_ids), direction=int(run.direction), attempted=tile.attempted,
        seed_foreground=int(tile.seed_foreground), complete=False,
        status="not_attempted_empty_original_seed" if not tile.attempted else "not_attempted")


class TiledRunAssembly:
    """File-backed fixed owners; halo arrays are streamed to the evidence writer."""
    def __init__(self, run, group, tiles, directory):
        self.run, self.group, self.tiles = run, group, tuple(tiles)
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        y0, x0, y1, x1 = group.context_bbox_yx
        self.shape = (len(run.expected_frames), y1-y0, x1-x0)
        self.frame_indices = {int(frame): index for index, frame in enumerate(run.expected_frames)}
        self.raw_path, self.available_path = self.directory / "owned.bool.dat", self.directory / "available.bool.dat"
        self.raw = np.memmap(self.raw_path, mode="w+", dtype=bool, shape=self.shape)
        self.available = np.memmap(self.available_path, mode="w+", dtype=bool, shape=self.shape)
        self.finished, self.receipts = set(), {}

    @property
    def ready(self):
        return len(self.finished) == sum(tile.attempted for tile in self.tiles)

    def consume(self, job, result):
        tile = job.tile
        if tile.tile_id in self.finished:
            raise ValueError("Duplicate SAM tile ownership result")
        y0, x0, _, _ = self.group.context_bbox_yx
        a, b, c, d = tile.crop_bbox_yx
        e, f, g, h = tile.ownership_bbox_yx
        frames = {int(frame): np.asarray(mask) for frame, mask in result.frames.items()}
        if set(frames) - set(self.frame_indices):
            raise ValueError("SAM tile returned a frame outside its original hypothesis")
        for frame, mask in frames.items():
            if mask.shape != (c-a, d-b) or not np.isin(mask, (0, 1)).all():
                raise ValueError("SAM tile transfer shape/binary support differs from its context")
            z = self.frame_indices[frame]
            self.raw[z, e-y0:g-y0, f-x0:h-x0] = mask[e-a:g-a, f-b:h-b]
            self.available[z, e-y0:g-y0, f-x0:h-x0] = True
        descriptor = tile_descriptor(self.run, tile)
        receipt = dict(result.receipt or {})
        complete = set(frames) == set(self.frame_indices) and bool(receipt.get("coverage_complete", True))
        valid = bool(receipt.get("prediction_valid", True))
        descriptor.update(complete=complete, structurally_valid=valid,
            observed_frames=sorted(frames), status="infrastructure_invalid" if not valid else
                "generated_complete" if complete else "generated_incomplete",
            tracker_scores=dict(result.tracker_scores or {}),
            observation_status=dict(result.observation_status or {}), runtime_receipt=receipt)
        self.receipts[tile.tile_id] = descriptor
        self.finished.add(tile.tile_id)
        return descriptor, frames

    def result(self):
        from types import SimpleNamespace
        attempted = [self.receipts[tile.tile_id] for tile in self.tiles if tile.attempted and tile.tile_id in self.receipts]
        return SimpleNamespace(frames={frame: self.raw[index] for frame, index in self.frame_indices.items()},
            tracker_scores={frame: None for frame in self.frame_indices}, observation_status={},
            receipt=dict(schema="xta.sam_tiled_parent_run/1", run_id=str(self.run.run_id),
                prediction_valid=all(v["structurally_valid"] for v in attempted),
                coverage_complete=self.ready and all(v["complete"] for v in attempted),
                tracker_score_semantics="No aggregate probability; independent child scores retained in tile evidence",
                sam_crop_mode="tiled", tile_count=len(self.tiles), attempted_tiles=len(attempted),
                unknown_unseeded_tile_ids=[tile.tile_id for tile in self.tiles if not tile.attempted]))

    def availability(self):
        return {frame: self.available[index] for frame, index in self.frame_indices.items()}

    def close(self):
        for array in (self.raw, self.available):
            array.flush()
            array._mmap.close()
        for path in (self.raw_path, self.available_path):
            path.unlink(missing_ok=True)
        try:
            self.directory.rmdir()
        except OSError:
            pass

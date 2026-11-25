"""Reduce bounded LTA proposals and settle each annotated interval once."""

from collections import Counter
from pathlib import Path

import numpy as np

from .lta_tracklets import blend_directional_masks
from .lta_union_artifacts import iter_union_crops, reduce_union_artifact_into_view


class DirectionalHandoff:
    """Keep forward masks in the view and opposing masks as packed rows on disk."""

    def __init__(self, scratch, view_union, domains):
        self.root = Path(scratch).resolve()
        self.view = view_union
        self.domains = dict(domains)
        self.lineage_anchors = {}
        anchors = sorted(domains)
        intervals = [
            (left, right)
            for left, right in zip(anchors, anchors[1:])
            if right - left > 1 and domains[left].frame_stop == right
            and domains[right].frame_start == left + 1
        ]
        self.frames = {
            frame: (left, right, index)
            for index, (frame, left, right) in enumerate(
                (frame, left, right) for left, right in intervals for frame in range(left + 1, right)
            )
        }
        self.observed = np.zeros(len(self.frames), dtype=np.uint8)
        self.visit_events = set()
        self.path = self.root / "opposing_interval_masks.bits"
        self.opposing = None
        if self.frames:
            _, height, width = self.view.shape
            with self.path.open("xb") as stream:
                stream.truncate(len(self.frames) * height * ((width + 7) // 8))
            try:
                self.opposing = np.memmap(
                    self.path, dtype=np.uint8, mode="r+",
                    shape=(len(self.frames), height, (width + 7) // 8),
                )
            except BaseException:
                self.path.unlink()
                raise

    def bind_seed_inventory(self, inventory):
        for seeds in inventory.values():
            for seed in seeds:
                anchor = int(seed.frame_index)
                if anchor not in self.domains:
                    raise RuntimeError("LTA seed has no annotated interval")
                if self.lineage_anchors.setdefault(seed.lineage, anchor) != anchor:
                    raise RuntimeError("LTA lineage has conflicting authoritative anchors")

    def domain_for_lineage(self, lineage):
        try:
            return self.domains[self.lineage_anchors[lineage]]
        except KeyError as exc:
            raise RuntimeError("LTA relay lineage has no authoritative interval") from exc

    def _record_visit(self, start, stop, anchor_frame, rect):
        # Record interval endpoints only. The number of events scales with
        # dispatched windows/crops, with no dense observation volume.
        for side, first, last in ((0, max(start, anchor_frame + 1), stop),
                                  (1, start, min(stop, anchor_frame))):
            if first >= last:
                continue
            self.visit_events.add((first, last, side, rect))
            for frame in range(first, last):
                interval = self.frames.get(frame)
                if interval is not None:
                    self.observed[interval[2]] |= 1 << side

    def _write_crop(self, frame, top, left, crop, anchor_frame):
        interval = self.frames.get(frame)
        if interval is None or anchor_frame == interval[0]:
            target = self.view[frame, top:top + crop.shape[0], left:left + crop.shape[1]]
            target |= crop
            if interval is not None:
                self.observed[interval[2]] |= 1
            return
        index = interval[2]
        start, bits = divmod(left, 8)
        aligned = np.pad(crop, ((0, 0), (bits, 0))) if bits else crop
        packed = np.packbits(aligned, axis=1, bitorder="little")
        self.opposing[index, top:top + crop.shape[0], start:start + packed.shape[1]] |= packed
        self.observed[index] |= 2

    def reduce(self, descriptor, *, tile_xyxy, frame_start, frame_stop, anchor_frame, windows):
        from .lta_execution import _artifact_ownership_path

        path = Path(descriptor["path"]).resolve(strict=True)
        if not _artifact_ownership_path(path).is_relative_to(_artifact_ownership_path(self.root)):
            raise RuntimeError("LTA worker union is outside the run scratch directory")
        domain = self.domains[anchor_frame]
        if not domain.frame_start <= frame_start < frame_stop <= domain.frame_stop:
            raise RuntimeError("LTA worker output crossed an annotated slice")
        if not self.frames:
            return reduce_union_artifact_into_view(
                descriptor, view_union=self.view, tile_xyxy=tile_xyxy,
                frame_start=frame_start, frame_stop=frame_stop,
            )
        visits = set()
        carried_prompts = set()
        for receipt in windows:
            prompt = receipt.get("window", {}).get("prompt_frame", anchor_frame)
            if prompt != anchor_frame:
                carried_prompts.add(prompt)
            for first, last in receipt.get("adapter", {}).get("model_visited_frame_ranges", ()):
                start, stop = max(frame_start, int(first)), min(frame_stop, int(last))
                if prompt != anchor_frame and start <= prompt < stop:
                    # The prompt was injected from a previous result. Its
                    # carried foreground is real, but a new crop does not
                    # infer background around that seed on the prompt frame.
                    visits.update(((start, prompt), (prompt + 1, stop)))
                else:
                    visits.add((start, stop))
        legacy_active = set()

        def observe(frame, top, left, crop):
            self._write_crop(frame, top, left, crop, anchor_frame)
            if frame in self.frames and frame not in carried_prompts and not any(first <= frame < last for first, last in visits):
                # A nonempty prediction proves this tile ran even for adapters
                # that omit visited-frame telemetry. Omitted masks prove nothing.
                legacy_active.add(frame)

        receipt = reduce_union_artifact_into_view(
            descriptor, view_union=self.view, tile_xyxy=tile_xyxy, frame_start=frame_start,
            frame_stop=frame_stop, observe=observe,
        )
        visits.update((frame, frame + 1) for frame in legacy_active)
        for first, last in visits:
            self._record_visit(first, last, anchor_frame, tuple(tile_xyxy))
        return receipt

    def compose(self, *, observe=None):
        if self.opposing is None:
            return
        _, height, width = self.view.shape
        events = sorted(
            (frame, change, side, rect)
            for first, last, side, rect in self.visit_events
            for frame, change in ((first, 1), (last, -1))
        )
        next_event = 0
        rectangles = [Counter(), Counter()]
        spatial = [np.zeros((height, width), dtype=np.bool_) for _ in range(2)]
        for frame, (left, right, index) in self.frames.items():
            flags = int(self.observed[index])
            if flags < 2:
                continue
            backward = np.unpackbits(self.opposing[index], axis=1, count=width, bitorder="little")
            forward = self.view[frame]
            if flags == 2:
                forward[:] = backward
                continue
            while next_event < len(events) and events[next_event][0] <= frame:
                _, change, side, rect = events[next_event]
                rectangles[side][rect] += change
                if rectangles[side][rect] == 0:
                    del rectangles[side][rect]
                next_event += 1
            for footprint, regions in zip(spatial, rectangles):
                footprint.fill(False)
                for x0, y0, x1, y1 in regions:
                    footprint[y0:y1, x0:x1] = True
            spatial[0] &= spatial[1]
            jointly_visited = spatial[0]
            combined = forward | backward
            common_foreground = int(np.count_nonzero(combined & jointly_visited))
            if common_foreground:
                difference = int(np.count_nonzero((forward ^ backward) & jointly_visited))
                if observe is not None:
                    observe(frame, difference, common_foreground)
                if difference:
                    _blend_shared_regions(
                        combined, forward, backward, jointly_visited, (frame - left) / (right - left),
                    )
            forward[:] = combined

    def close(self):
        if self.opposing is not None:
            from .runtime import close_memmap_array_without_flush

            close_memmap_array_without_flush(self.opposing)
            self.opposing = None
            self.path.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        self.close()


def _blend_shared_regions(combined, forward, backward, jointly_visited, fraction):
    """Paint only jointly visited regions, isolating disconnected footprints."""
    import cv2

    if np.all(jointly_visited):
        combined[:] = blend_directional_masks(forward, backward, fraction)
        return
    # Components clipped by a missing part of either observation may represent
    # different structures. Preserve them and only morph fully observed ones.
    interior = cv2.erode(jointly_visited.view(np.uint8), np.ones((3, 3), dtype=np.uint8),
                         borderType=cv2.BORDER_CONSTANT, borderValue=1).view(np.bool_)
    count, labels, boxes, _ = cv2.connectedComponentsWithStats(jointly_visited.view(np.uint8), connectivity=8)
    for index in range(1, count):
        x, y, width, height, area = map(int, boxes[index])
        region = np.s_[y:y + height, x:x + width]
        left, right = forward[region], backward[region]
        if area == width * height:
            combined[region] = blend_directional_masks(left, right, fraction, matchable=interior[region])
            continue
        shared = labels[region] == index
        blended = blend_directional_masks(left & shared, right & shared, fraction, matchable=interior[region])
        np.copyto(combined[region], blended, where=shared)

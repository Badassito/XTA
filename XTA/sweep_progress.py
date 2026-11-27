"""Small live counters for the view-major sweep's single producer process."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from pathlib import Path


class FanoutProgress:
    """Publish a replaceable heartbeat without scanning packets, masks or output files.

    Call observe_batch with aggregated worker counters at result handoff. The
    actual network-call count excludes warmup, padding within a batch, and
    subsequent cutoff-specific mask decoding. View keys include the model when
    the same view is used by multiple models.
    """

    def __init__(self, root, cutoffs, *, total_views=None, interval=5.0):
        self.path = Path(root) / "fanout_progress.json"
        self.cutoffs = tuple(int(cent) for cent in cutoffs)
        self._bits = {cent: 1 << i for i, cent in enumerate(self.cutoffs)}
        self._branches = {cent: {"native_views_ready": 0, "native_frames_ready": 0,
                                 "views_projected": 0, "views_published": 0,
                                 "published_layers": 0} for cent in self.cutoffs}
        self._seen = {stage: {} for stage in ("native_views_ready", "views_projected", "views_published")}
        self._work = {"rendered_frames": 0, "render_batches": 0, "forward_calls": 0,
                      "mask_decodes": 0, "geometry_passes": 0}
        self._projectors = {}
        self._lock = threading.Lock()
        self._interval = float(interval)
        self._last_write = -float("inf")
        self._warned = False
        self._status = "in_progress"
        self._details = {}
        self._total_views = total_views
        self._sweep_id = os.environ.get("XTA_YOLO_SWEEP_ID")
        self._last_view = None
        self._write(force=True)

    def set_total_views(self, count):
        with self._lock:
            self._total_views = int(count)
            self._write(force=True)

    def _write(self, *, force=False):
        now = time.monotonic()
        if not force and now - self._last_write < self._interval:
            return
        self._last_write = now
        snapshot = {
            **self._details,
            "schema": "xta.view_major_progress/1",
            "sweep_id": self._sweep_id,
            "status": self._status,
            "updated_unix_seconds": time.time(),
            "planned_views": self._total_views,
            "last_view": self._last_view,
            "work": {**self._details.get("work", {}), **self._work},
            "sibling_projectors": self._projectors,
            "cutoffs": [{"cutoff_hundredths": cent, **self._branches[cent]} for cent in self.cutoffs],
        }
        # The parent owns this fresh output path. Unlike a final run receipt a
        # heartbeat need not fsync the output filesystem or stop useful work.
        tmp = self.path.with_suffix(".json.tmp")
        try:
            tmp.write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
            os.replace(tmp, self.path)
        except OSError as exc:
            if not self._warned:
                print(f"Could not update sweep progress {self.path}: {exc}", file=sys.stderr, flush=True)
                self._warned = True

    def observe_batch(self, *, rendered_frames=0, render_batches=0, forward_calls=0, mask_decodes=0):
        with self._lock:
            self._work["rendered_frames"] += rendered_frames
            self._work["render_batches"] += render_batches
            self._work["forward_calls"] += forward_calls
            self._work["mask_decodes"] += mask_decodes
            self._write()

    def _mark(self, stage, view_key, cutoffs):
        key = str(view_key)
        prior = self._seen[stage].get(key, 0)
        new = []
        for cent in self.cutoffs if cutoffs is None else cutoffs:
            bit = self._bits[cent]
            if not (prior & bit):
                self._branches[cent][stage] += 1
                prior |= bit
                new.append(cent)
        self._seen[stage][key] = prior
        self._last_view = key
        return new

    def view_ready(self, view_key, *, nframes, cutoffs=None):
        with self._lock:
            for cent in self._mark("native_views_ready", view_key, cutoffs):
                self._branches[cent]["native_frames_ready"] += nframes
            self._write()

    def view_projected(self, view_key, *, cutoffs, geometry_passes=1, projectors=()):
        with self._lock:
            if self._mark("views_projected", view_key, cutoffs):
                self._work["geometry_passes"] += geometry_passes
                for result in projectors:
                    backend = str(result.get("backend", "unclassified"))
                    totals = self._projectors.setdefault(backend, {"passes": 0})
                    totals["passes"] += 1
                    for field in ("rank_crop_reads", "empty_crop_reads", "rank_dense_reads",
                                  "rank_bbox_scan_pixels", "rank_statistics_pixels",
                                  "nominal_output_voxels", "kernel_calls", "dense_d2h_bytes"):
                        if field in result:
                            totals[field] = totals.get(field, 0) + int(result[field])
                    device_lanes = result.get("lane_diagnostics", ())
                    device_timing = device_lanes[0].get("radial_device_timing") if device_lanes else None
                    for name, timing in (("radial_device_timing", device_timing),
                                         ("radial_host_timing", result.get("radial_host_timing"))):
                        if timing is None:
                            continue
                        summary = totals.setdefault(name, {"passes": 0})
                        summary["passes"] += 1
                        for field, value in timing.items():
                            prior = summary.get(field, 0)
                            summary[field] = max(prior, value) if field.startswith("max_") else prior + value
                    reason = result.get("cuda_reason")
                    if (reason and not backend.startswith("cuda")
                            and reason != "empty_output_certificate"):
                        # Why the GPU path declined: an intended CPU route or a lost lease.
                        declines = totals.setdefault("cuda_declines", {})
                        declines[str(reason)] = declines.get(str(reason), 0) + 1
            self._write()

    def view_published(self, view_key, cutoff, *, layers=1):
        with self._lock:
            first = self._branches[cutoff]["views_published"] == 0
            if self._mark("views_published", view_key, (cutoff,)):
                self._branches[cutoff]["published_layers"] += layers
            self._write(force=first)

    def finish(self, status, **details):
        """Close the producer's stage. Full family completion belongs to manifest.json."""
        with self._lock:
            self._status = status
            self._details = details
            self._write(force=True)

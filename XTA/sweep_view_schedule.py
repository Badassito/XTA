"""Handoff from detector ranges to independently finalized confidence branches."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import CancelledError, ThreadPoolExecutor
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import os
import threading
import time


def _positive_env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw.strip().lower() in ('', 'auto'):
        return int(default)
    value = int(raw)
    if value < 1:
        raise ValueError(f'{name} must be a positive integer or auto')
    return value


def sweep_publication_concurrency():
    """Resolve (publishers, projection lanes per publisher) for early sibling publication.

    Sibling publication must keep pace with inference, otherwise the scheduler
    has to throttle the GPUs. Each publisher owns one view at a time; its
    projection runs on its own CPU lanes. The defaults use about half of the
    visible CPUs for projection and leave the rest to NRRD encoding and the
    floor pipeline.
    """
    from .workspace import _cpu_count
    cores = max(1, int(_cpu_count()))
    # Publications mostly wait (GPU leases, NRRD barriers, journal reads) and
    # their compute stages run without the GIL, so more publishers with fewer
    # lanes each keep pace better than a few wide ones: job 168526 kept 8
    # publishers saturated while using only 5-40 of 128 cores.
    publishers = _positive_env_int('XTA_YOLO_SWEEP_PUBLISHERS', max(2, min(16, cores // 8)))
    lanes = _positive_env_int('XTA_YOLO_SWEEP_PROJECTION_LANES',
                              max(1, min(8, cores // (2 * publishers))))
    return publishers, lanes


class SweepViewSchedule:
    """Consume native-funded view tickets and publish the confidence family."""

    def __init__(self, args, *, cutoff, tiles, save_options, interpolation, gpu, cpu,
                 reconciliation_mode=None):
        self.role = os.environ.get('XTA_YOLO_SWEEP_NATIVE_ROLE') if cutoff is not None else None
        self.cent = cutoff
        self.cutoffs = ()
        self.progress = None
        self.reason = None
        self.eager_projection = False
        self.produced = 0
        self.consumed = 0
        self.scalar = 0
        self.failures = Counter()
        self.ranges = defaultdict(list)
        self.identities = {}
        self.work = Counter()
        self._pending = set()
        self._executor = None
        self.native_frames = 0
        self.native_bytes = 0
        self.projected_views = 0
        self.publishers = 0
        self.publish_backlog_limit = 0
        self.publish_backlog_peak = 0
        self._publish_lock = threading.Lock()
        self._publish_failure = None
        self._cancelled = False
        self._closed = False
        if self.role is None:
            return
        if self.role not in ('produce', 'consume'):
            raise ValueError('Invalid native confidence fanout role')
        self.cutoffs = tuple(map(int, os.environ['XTA_YOLO_SWEEP_CUTOFFS_HUNDREDTHS'].split(',')))
        if (self.cutoffs != tuple(sorted(set(self.cutoffs))) or len(self.cutoffs) < 2
                or self.cutoffs[0] < 0 or self.cutoffs[-1] > 100 or cutoff not in self.cutoffs
                or (self.role == 'produce') != (cutoff == self.cutoffs[0])):
            raise ValueError('Native confidence fanout cutoff identity is inconsistent')
        if gpu and cpu:
            self.reason = 'hybrid backend selection can change scalar detector precision and ownership'
        elif tiles:
            self.reason = 'tile masks require cutoff-specific parent and bridge gates'
        elif args.augmentation_ratio > 1:
            self.reason = 'external augmentation requires per-task inverse mapping'
        elif args.min_conf or args.min_radius or 'yolo' in (args.enable_2D_hole_fill or ()):
            self.reason = 'native detector cleanup requires the scalar route'
        elif (gpu and args.gpu_batch != 1) or (cpu and args.cpu_batch != 1):
            self.reason = 'batched seam and padding ownership require the scalar route'
        elif args.capture_component_replay:
            self.reason = 'requested component capture requires the scalar route'
        # Confidence reconciliation requires the scalar tail's source-aligned
        # observed evidence. Its native detector products still share normally.
        self.eager_projection = bool(self.reason is None and reconciliation_mode != 'confidence'
                                     and ('nrrd' in save_options or 'xtareview' in save_options)
                                     and not interpolation.enabled and not interpolation.extrapolation_enabled)
        if self.producer:
            self.publishers = sweep_publication_concurrency()[0] if self.eager_projection else 1
            self.publish_backlog_limit = _positive_env_int(
                'XTA_YOLO_SWEEP_PUBLISH_BACKLOG', max(64, 16 * self.publishers))
        if self.role == 'produce':
            from .sweep_progress import FanoutProgress
            self.progress = FanoutProgress(Path(os.environ['XTA_YOLO_SWEEP_OUTPUT_ROOT']), self.cutoffs)

    @property
    def producer(self):
        return self.role == 'produce' and self.reason is None

    def bind_views(self, views, artifacts, args):
        if self.progress is not None:
            self.progress.set_total_views(len(views))
        if not self.role or self.reason:
            return
        # Reuse source/model identities captured at scalar startup. Never read
        # the acquisition, checkpoint or per-range pixels to build these keys.
        from .unification.tta_manifest import comparable_artifact_identities

        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                             allow_nan=False).encode('utf-8')).hexdigest()
        common = digest([comparable_artifact_identities(artifacts), {key: value for key, value in vars(args).items()
            if key not in {'conf','output','temp','capture_component_replay'}}])
        self.identities = {str(view.name): digest([common, asdict(view)]) for view in views}

    def record_task(self, model, view, stats):
        item = stats.get('sweep_native_fanout')
        if item is None:
            item = {'status': 'unsupported', 'reason': 'worker route has no native multi-cut decoder'}
        counters = item.get('counters', {})
        with self._publish_lock:
            self.ranges[model, str(view.name)].append(item)
            for key in ('rendered_frames', 'render_batches', 'forward_calls', 'mask_decodes',
                        'identical_native_crops_reused', 'anchors_once', 'scalar_anchor_entries',
                        'active_bins', 'native_crop_bytes', 'scalar_owned_floor_frames',
                        'floor_count_d2h_bytes', 'native_packet_aliases', 'bulk_empty_frames',
                        'empty_range_writes'):
                if key in counters:
                    self.work[key] += int(counters[key])

    def producer_view_ready(self, model, view, shape, publication, *, retain_empty_orthogonal=False):
        """Enqueue an already-funded view; the future settles its publication use.

        Admission owns the parent ticket and byte charges before dispatch.
        Success means all actor readers are closed and any eager payload
        retirement has finished. A sealed native view that still serves
        sibling consumers remains charged to the separate journal quota.
        """
        with self._publish_lock:
            if not self.producer or self._closed or self._cancelled:
                raise RuntimeError('Sweep publication owner is not accepting views')
            if self._publish_failure is not None:
                raise self._publish_failure
            if self.eager_projection and publication is None:
                raise RuntimeError('Missing early confidence view publication owner')
            if self._executor is None:
                self._executor = ThreadPoolExecutor(max_workers=self.publishers,
                                                    thread_name_prefix='sweep-view-publish')
            receipts = self.ranges.pop((model, str(view.name)))
            future = self._executor.submit(self._publish, publication, model, view, tuple(shape),
                                           receipts, retain_empty_orthogonal)
            self._pending.add(future)
            self.publish_backlog_peak = max(self.publish_backlog_peak, len(self._pending))
        future.add_done_callback(self._forget_publication)
        return future

    def _forget_publication(self, future):
        with self._publish_lock:
            self._pending.remove(future)

    def cancel(self, reason=None):
        """Run-failure teardown hook: never start the queued publication backlog."""
        with self._publish_lock:
            self._cancelled = True
            pending = tuple(self._pending)
        for future in pending:
            future.cancel()

    def _check_publication(self, view):
        with self._publish_lock:
            if self._publish_failure is not None:
                raise RuntimeError(
                    f'Skipped {view.name}: earlier sweep view publication failed: {self._publish_failure}'
                ) from self._publish_failure
            if self._cancelled:
                raise CancelledError(f'Sweep view publication cancelled for {view.name}')

    def _publish(self, publication, model, view, shape, receipts, retain_empty_orthogonal):
        from . import sweep_views

        self._check_publication(view)
        started = time.perf_counter()
        try:
            counters = Counter()
            for receipt in receipts:
                counters.update({key: int(receipt.get('counters', {}).get(key, 0)) for key in
                    ('rendered_frames', 'render_batches', 'forward_calls', 'mask_decodes')})
            self.progress.observe_batch(**counters)
            decision = sweep_views.seal_view_family(model, str(view.name), shape, self.cutoffs,
                receipts=receipts, identity=self.identities.get(str(view.name)))
            if decision['status'] == 'fallback':
                with self._publish_lock:
                    self.failures[str(decision.get('reason') or 'unsupported worker route')] += 1
                return 'fallback'
            if decision['status'] != 'ready':
                raise RuntimeError(f'Unrecognized native view decision: {decision["status"]}')
            with self._publish_lock:
                self.produced += 1
                self.native_frames += int(view.num_slices)
                self.native_bytes += int(decision.get('stored_bytes', 0))
            self.progress.view_ready(f'{model}/{view.name}', nframes=int(view.num_slices))
            if self.eager_projection:
                self._check_publication(view)
                if self._publish_view(publication, model, view, shape, retain_empty_orthogonal):
                    return 'projected'
            return 'native'
        except BaseException as error:
            with self._publish_lock:
                if self._publish_failure is None:
                    self._publish_failure = error
                elif error is not self._publish_failure and error.__cause__ is not self._publish_failure:
                    add_note = getattr(self._publish_failure, 'add_note', None)
                    if add_note is not None:
                        add_note(f'Another sweep view publication also failed: {error!r}')
            raise
        finally:
            try:
                from .runtime import runtime_telemetry
                telemetry = runtime_telemetry()
                telemetry.add('sweep.publication.view_seconds', time.perf_counter() - started)
                telemetry.add('sweep.publication.views', 1)
                telemetry.add(f'sweep.publication.{getattr(view, "family", "unknown")}_seconds',
                              time.perf_counter() - started)
            except Exception:
                pass

    def _publish_view(self, publication, model, view, shape, retain_empty_orthogonal):
        from . import sweep_views
        identity = self.identities.get(str(view.name))
        with sweep_views.open_sibling_views(model, str(view.name), self.cutoffs,
                                            native_shape=tuple(shape), identity=identity) as family:
            readers = family.readers
            projected = publication.publish_view_family(view, native_shape=tuple(shape),
                read_mask_plane=lambda c,z: readers[c].read_mask_plane(z),
                read_score_plane=lambda c,z: readers[c].read_score_plane(z),
                read_mask_crops=family, read_native_crops=family.native_crops,
                native_journal_identity=identity,
                counts_by_cutoff={int(c): int(n) for c,n in family.metadata['prediction_counts'].items()},
                retain_empty_orthogonal=retain_empty_orthogonal,
                primary_already_published=True)
        if projected is None:
            return False
        sweep_views.retire_view_payloads(model, str(view.name), self.cutoffs,
                                        identity=self.identities.get(str(view.name)))
        with self._publish_lock:
            self.projected_views += 1
        return True

    def consumer_source(self, model, view, shape, publication):
        """Missing promised data is an integrity failure. An explicit fallback can re-run."""
        if self.role != 'consume' or self.reason:
            return None
        from .sweep_views import SweepViewIntegrityError, view_family_status
        state = view_family_status(model, str(view.name), self.cent, native_shape=tuple(shape),
                                   identity=self.identities.get(str(view.name)))
        if state == 'projected':
            if self.eager_projection and publication is not None and publication.has_published_view(view):
                return 'projected'
            raise SweepViewIntegrityError(f'Projected confidence view lost its publication receipt: {view.name}')
        if state == 'ready':
            return 'native'
        if state != 'fallback':
            raise SweepViewIntegrityError(f'Unexpected native view state for {view.name}: {state}')
        self.failures['producer recorded scalar fallback'] += 1
        return None

    def partition(self, model, views, shape_for, publication):
        detector_views, projected, native = [], deque(), deque()
        for view in views:
            if int(view.augmentation_pass) != 0:
                continue
            source = None
            if self.role == 'consume' and self.reason is None:
                source = self.consumer_source(model, view, shape_for(view), publication)
            if source is None:
                detector_views.append(view)
            elif source == 'projected':
                projected.append((view, source))
            else:
                native.append((view, source))
        fully_projected = bool(projected) and not detector_views and not native
        projected.extend(native)
        return detector_views, projected, fully_projected

    def close(self):
        """Join the publication actor from terminal processing or teardown."""
        with self._publish_lock:
            if self._closed:
                return
            self._closed = True
            executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True)
        if not self._cancelled and self._publish_failure is not None:
            raise self._publish_failure

    def summary(self, total_views):
        reason = [self.reason] if self.reason else sorted(self.failures)
        result = dict(role=self.role, validated=True, produced_views=self.produced,
                      consumed_views=self.consumed, scalar_views=total_views - (self.produced + self.consumed),
                      target_count=len(self.cutoffs), fallback_reasons=reason,
                      native_frames=self.native_frames, stored_native_bytes=self.native_bytes,
                      native_views_retired_after_projection=self.projected_views, work=dict(self.work))
        if self.eager_projection and self.producer:
            result['early_publication'] = dict(
                publishers=self.publishers, backlog_limit=self.publish_backlog_limit,
                backlog_peak=self.publish_backlog_peak)
        return result

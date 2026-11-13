"""Handoff from detector ranges to independently finalized confidence branches."""
from __future__ import annotations

from collections import Counter, defaultdict, deque
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import ExitStack
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
    """Own native per-view coverage decisions and bounded, early branch publication."""

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
        self.pending = deque()
        self.executor = None
        self.native_frames = 0
        self.native_bytes = 0
        self.projected_views = 0
        self.publishers = 0
        self.publish_backlog_limit = 0
        self.publish_backlog_peak = 0
        self.publish_wait_seconds = 0.0
        self.publish_waits = 0
        self.journal_high_water = 0.0
        self.journal_peak_fraction = 0.0
        self.journal_unretirable_peak_bytes = 0
        self._pending_journal_bytes = {}
        self._publish_lock = threading.Lock()
        self._publish_failure = None
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
        elif 'images' in save_options or args.capture_component_replay:
            self.reason = 'requested render or component capture requires the scalar route'
        # Confidence reconciliation requires the scalar tail's source-aligned
        # observed evidence. Its native detector products still share normally.
        self.eager_projection = bool(self.reason is None and reconciliation_mode != 'confidence'
                                     and 'nrrd' in save_options
                                     and not interpolation.enabled and not interpolation.extrapolation_enabled)
        if self.eager_projection:
            # Fail configuration errors at startup, never after inference began.
            self.publishers, _ = sweep_publication_concurrency()
            self.publish_backlog_limit = _positive_env_int(
                'XTA_YOLO_SWEEP_PUBLISH_BACKLOG', max(64, 16 * self.publishers))
            self.journal_high_water = float(os.environ.get('XTA_YOLO_SWEEP_JOURNAL_HIGH_WATER', '0.75'))
            if not 0.0 < self.journal_high_water <= 1.0:
                raise ValueError('XTA_YOLO_SWEEP_JOURNAL_HIGH_WATER must be in (0, 1]')
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
        def digest(value):
            return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                             allow_nan=False).encode('utf-8')).hexdigest()
        common = digest([artifacts, {key: value for key, value in vars(args).items()
            if key not in {'conf','output','temp','capture_component_replay'}}])
        self.identities = {str(view.name): digest([common, asdict(view)]) for view in views}

    def record_task(self, model, view, stats):
        item = stats.get('sweep_native_fanout')
        if item is None:
            item = {'status': 'unsupported', 'reason': 'worker route has no native multi-cut decoder'}
        self.ranges[model, str(view.name)].append(item)
        counters = item.get('counters', {})
        for key in ('rendered_frames', 'render_batches', 'forward_calls', 'mask_decodes',
                    'identical_native_crops_reused', 'anchors_once', 'scalar_anchor_entries',
                    'active_bins', 'native_crop_bytes', 'scalar_owned_floor_frames',
                    'floor_count_d2h_bytes', 'native_packet_aliases', 'bulk_empty_frames',
                    'empty_range_writes'):
            if key in counters:
                self.work[key] += int(counters[key])
        if self.progress is not None and counters:
            self.progress.observe_batch(**{key: int(counters.get(key, 0)) for key in
                ('rendered_frames', 'render_batches', 'forward_calls', 'mask_decodes')})

    def producer_view_ready(self, model, view, shape, publication, *, retain_empty_orthogonal=False):
        from . import sweep_views
        receipts = self.ranges.pop((model, str(view.name)))
        decision = sweep_views.seal_view_family(model, str(view.name), tuple(shape), self.cutoffs,
            receipts=receipts, identity=self.identities.get(str(view.name)))
        if decision['status'] != 'ready':
            if decision['status'] != 'fallback':
                raise RuntimeError(f'Unrecognized native view decision: {decision["status"]}')
            self.failures[str(decision.get('reason') or 'unsupported worker route')] += 1
            return
        self.produced += 1
        self.native_frames += int(view.num_slices)
        self.native_bytes += int(decision.get('stored_bytes', 0))
        self.progress.view_ready(f'{model}/{view.name}', nframes=int(view.num_slices))
        if self.eager_projection:
            if publication is None:
                raise RuntimeError('Missing early confidence view publication owner')
            if self.executor is None:
                self.executor = ThreadPoolExecutor(max_workers=self.publishers,
                                                   thread_name_prefix='sweep-view-publish')
            # This runs on the scheduler thread. Waiting here stops every GPU
            # refill, so a publication backlog is absorbed by the native
            # journal: views differ by orders of magnitude in publication cost,
            # so a view count is a poor brake. Wait only when the journal nears
            # its quota (workers would otherwise fall back to scalar) or the
            # queue reaches its safety cap, and then only for the first
            # publication to finish.
            self._reap_publications(block=self._publication_backpressure())
            future = self.executor.submit(self._publish, publication, model, view, shape,
                                          decision, retain_empty_orthogonal)
            self._pending_journal_bytes[future] = max(0, int(decision.get('stored_bytes', 0) or 0))
            self.pending.append(future)
            self.publish_backlog_peak = max(self.publish_backlog_peak, len(self.pending))

    def _reap_publications(self, *, block):
        """Surface finished publication errors; optionally wait for one free slot."""
        while self.pending and block:
            started = time.perf_counter()
            done, _ = wait(tuple(self.pending), return_when=FIRST_COMPLETED)
            waited = time.perf_counter() - started
            self.publish_wait_seconds += waited
            self.publish_waits += 1
            try:
                from .runtime import runtime_telemetry
                telemetry = runtime_telemetry()
                telemetry.add('sweep.publication.scheduler_wait_seconds', waited)
                telemetry.add('sweep.publication.scheduler_waits', 1)
            except Exception:
                pass
            finished = [future for future in self.pending if future.done()]
            for future in finished:
                self.pending.remove(future)
                self._pending_journal_bytes.pop(future, None)
            self._raise_publication_errors(finished)
            # Retirement just happened: judge the next wait on fresh usage.
            block = self._publication_backpressure(fresh=True)
        finished = [future for future in self.pending if future.done()]
        for future in finished:
            self.pending.remove(future)
            self._pending_journal_bytes.pop(future, None)
        self._raise_publication_errors(finished)

    def _publication_backpressure(self, *, fresh=False):
        """True when the scheduler must wait for publication to catch up.

        Waiting only helps if pending publications will retire enough journal
        bytes to get below the mark. Bytes they will not retire (in-flight worker
        reservations, payloads of views whose projection fell back) would make
        every view drain the whole backlog, so then the scheduler keeps going
        and workers degrade to the exact scalar route if the quota fills.
        """
        if not self.pending:
            return False
        if len(self.pending) >= self.publish_backlog_limit:
            return True
        try:
            from .sweep_views import native_journal_usage
            used, quota = native_journal_usage(max_age_seconds=0.0 if fresh else 0.5)
        except Exception:
            used, quota = None, 0
        if used is None or quota <= 0:
            # Unknown usage: only the queue cap applies (never a hang).
            return False
        fraction = used / quota
        self.journal_peak_fraction = max(self.journal_peak_fraction, fraction)
        mark = self.journal_high_water * quota
        if used < mark:
            return False
        retirable = sum(self._pending_journal_bytes.get(future, 0) for future in self.pending)
        if used - retirable >= mark:
            self.journal_unretirable_peak_bytes = max(self.journal_unretirable_peak_bytes,
                                                      int(used - retirable))
            return False
        return True

    def _raise_publication_errors(self, futures):
        errors = []
        for future in futures:
            if future.cancelled():
                continue
            error = future.exception()
            if error is not None:
                errors.append(error)
        if not errors:
            return
        root = self._publish_failure
        first = root if any(error is root for error in errors) else errors[0]
        self._fail_publications(first)
        skipped = 0
        for other in errors:
            if other is first:
                continue
            if other.__cause__ is first or other.__cause__ is self._publish_failure:
                skipped += 1
            else:
                first.add_note(f'Another sweep view publication also failed: {other!r}')
        if skipped:
            first.add_note(f'{skipped} queued sweep view publication(s) were skipped after this failure')
        raise first

    def _fail_publications(self, error):
        """Stop queued work after the first failure; running views skip remaining stages."""
        with self._publish_lock:
            if self._publish_failure is None:
                self._publish_failure = error
        for future in self.pending:
            future.cancel()

    def cancel(self, reason=None):
        """Run-failure teardown hook: never start the queued publication backlog."""
        self._fail_publications(RuntimeError(reason or 'TTA run failed'))

    def _publish(self, publication, model, view, shape, decision, retain_empty_orthogonal):
        started = time.perf_counter()
        try:
            return self._publish_view(publication, model, view, shape, decision, retain_empty_orthogonal)
        except BaseException as error:
            # Queued views check this flag and skip their work; the scheduler
            # thread surfaces the error and cancels what has not started.
            with self._publish_lock:
                if self._publish_failure is None:
                    self._publish_failure = error
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

    def _publish_view(self, publication, model, view, shape, decision, retain_empty_orthogonal):
        from . import sweep_views
        if self._publish_failure is not None:
            raise RuntimeError(f'Skipped {view.name}: an earlier sweep view publication failed')                 from self._publish_failure
        with ExitStack() as stack:
            readers = {c: stack.enter_context(sweep_views.open_view(model, str(view.name), c,
                       native_shape=tuple(shape), identity=self.identities.get(str(view.name))))
                       for c in self.cutoffs if c != self.cent}
            projected = publication.publish_view_family(view, native_shape=tuple(shape),
                read_mask_plane=lambda c,z: readers[c].read_mask_plane(z),
                read_score_plane=lambda c,z: readers[c].read_score_plane(z),
                read_mask_crops=lambda c,z: readers[c].iter_mask_crops(z),
                read_native_crops=lambda c,z: readers[c].iter_crops(z),
                counts_by_cutoff={int(c): int(n) for c,n in decision['prediction_counts'].items()},
                retain_empty_orthogonal=retain_empty_orthogonal,
                primary_already_published=True)
        if projected is not None:
            sweep_views.retire_view_payloads(model, str(view.name), self.cutoffs,
                                            identity=self.identities.get(str(view.name)))
            with self._publish_lock:
                self.projected_views += 1

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
        detector_views, stored = [], deque()
        for view in views:
            if int(view.augmentation_pass) != 0:
                continue
            source = None
            if self.role == 'consume' and self.reason is None:
                source = self.consumer_source(model, view, shape_for(view), publication)
            if source is None:
                detector_views.append(view)
            else:
                stored.append((view, source))
        fully_projected = bool(stored) and not detector_views and all(
            source == 'projected' for _, source in stored)
        return detector_views, stored, fully_projected

    def close(self):
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=self._publish_failure is not None)
            self.executor = None
        pending, self.pending = list(self.pending), deque()
        if self._publish_failure is not None:
            # The first error already surfaced (or the run is failing); report
            # any other real failures without masking the root cause.
            pending = [future for future in pending if not future.cancelled()
                       and future.exception() is not None
                       and future.exception().__cause__ is not self._publish_failure]
        self._raise_publication_errors(pending)

    def summary(self, total_views):
        reason = [self.reason] if self.reason else sorted(self.failures)
        result = dict(role=self.role, validated=True, produced_views=self.produced,
                      consumed_views=self.consumed, scalar_views=total_views - (self.produced + self.consumed),
                      target_count=len(self.cutoffs), fallback_reasons=reason,
                      native_frames=self.native_frames, stored_native_bytes=self.native_bytes,
                      native_views_retired_after_projection=self.projected_views, work=dict(self.work))
        if self.publishers:
            result['early_publication'] = dict(
                publishers=self.publishers, backlog_limit=self.publish_backlog_limit,
                backlog_peak=self.publish_backlog_peak,
                journal_high_water=self.journal_high_water,
                journal_peak_fraction=round(self.journal_peak_fraction, 4),
                journal_unretirable_peak_bytes=self.journal_unretirable_peak_bytes,
                scheduler_wait_seconds=round(self.publish_wait_seconds, 3),
                scheduler_waits=self.publish_waits)
        return result

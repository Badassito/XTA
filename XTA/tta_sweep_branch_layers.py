"""Bounded view-major publication through the unchanged scalar NRRD owners.

Each cutoff owns its scalar sink, run identity and output directories. The floor
layer may already be submitted by the ordinary pipeline; other cuts use the same
writer and native/LQ mirror task. A completed-view callback receives actual rows
only after those tasks have published both qualities and released their readers.
"""
from __future__ import annotations

from collections import deque
from collections.abc import Callable, Hashable, Mapping
import operator
import threading
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .interpolation import NrrdLayerRef
    from .outputs import NrrdLayerSink


def _cutoff_identity(value: int) -> int:
    if isinstance(value, bool):
        raise ValueError('Branch cutoff must be integer hundredths in [0,100]')
    try:
        cutoff = operator.index(value)
    except TypeError as error:
        raise ValueError('Branch cutoff must be integer hundredths in [0,100]') from error
    if not 0 <= cutoff <= 100:
        raise ValueError('Branch cutoff must be integer hundredths in [0,100]')
    return cutoff


def scalar_layer_suffix(ref: 'NrrdLayerRef', *, view_token: str | None = None) -> str:
    """Use exactly the scalar component naming policy for a branch reference."""
    from .outputs import nrrd_layer_output_suffix
    return nrrd_layer_output_suffix(
        view_token=ref.view_name if view_token is None else view_token,
        source=ref.source,
        mask_kind=ref.mask_kind,
        pass_index=ref.pass_index,
        interpolation_walk_back_index=ref.interpolation_walk_back_index,
        interpolation_candidate_index=ref.interpolation_candidate_index,
        interpolation_backend=ref.interpolation_backend,
        interpolation_direction=ref.interpolation_direction,
        seed_detector_identity=ref.seed_detector_identity,
        sam_bundle_identity=ref.sam_bundle_identity,
        interpolation_policy_identity=ref.interpolation_policy_identity,
        tile_config_id=ref.tile_config_id,
        tile_acceptance=ref.tile_acceptance,
        stage=ref.stage,
    )


class TtaSweepBranchLayers:
    """Publish one finished view across independent scalar cutoff sinks.

    Sinks are supplied by the main coordinator with their exact branch run IDs,
    native shape and LQ specs. This owner neither switches the process-global sink
    nor reconstructs final Global unions: ordinary scalar tail owners do those.
    """

    def __init__(
        self,
        sinks_by_cutoff: Mapping[int, 'NrrdLayerSink'],
        *,
        primary_cutoff: int,
        max_pending_layers: int = 4,
    ) -> None:
        self.sinks_by_cutoff = {_cutoff_identity(cutoff): sink
                                for cutoff, sink in sinks_by_cutoff.items()}
        self.primary_cutoff = _cutoff_identity(primary_cutoff)
        if self.primary_cutoff in self.sinks_by_cutoff:
            raise ValueError('Primary sink remains owned by the scalar pipeline')
        if len({id(sink) for sink in self.sinks_by_cutoff.values()}) != len(self.sinks_by_cutoff):
            raise ValueError('Each cutoff must own an independent scalar sink')
        if isinstance(max_pending_layers, bool):
            raise ValueError('Pending layer limit must be a positive integer')
        self.max_pending_layers = operator.index(max_pending_layers)
        if self.max_pending_layers <= 0:
            raise ValueError('Pending layer limit must be a positive integer')
        self._lock = threading.Lock()
        self._published_views: set[Hashable] = set()
        self._inflight_views: set[Hashable] = set()
        self._idle = threading.Condition(self._lock)
        self._failure: BaseException | None = None
        self._closed = False

    def _drain_all(self, primary_sink: 'NrrdLayerSink | None' = None) -> None:
        first_error: BaseException | None = None
        for sink in (*self.sinks_by_cutoff.values(), *((primary_sink,) if primary_sink is not None else ())):
            try:
                sink.wait()
            except BaseException as error:
                if first_error is None:
                    first_error = error
        if first_error is not None:
            raise first_error

    def publish_view(
        self,
        view_key: Hashable,
        refs_by_cutoff: Mapping[int, 'NrrdLayerRef | None'],
        *,
        view_token: str,
        completed: Callable[[int, Hashable, tuple[dict, ...]], object],
        primary: tuple['NrrdLayerSink', 'NrrdLayerRef | None'] | None = None,
    ) -> dict[int, tuple[dict, ...]]:
        """Publish/drain native and LQ files before acknowledging this view.

        The optional ``primary=(sink, ref)`` has already been submitted by the
        scalar pipeline; its executor remains owned by that pipeline. ``None``
        preserves ordinary omitted-empty-layer semantics; an
        explicit empty reference is written with stock empty-grid semantics. The
        caller may write confidence evidence concurrently, but records a view's
        receipts only after both this barrier and that evidence have settled.
        """
        with self._lock:
            if self._closed or self._failure is not None:
                raise RuntimeError('Branch layer publisher is closed or failed') from self._failure
            if view_key in self._published_views or view_key in self._inflight_views:
                raise ValueError('Physical view was published more than once')
            refs = {_cutoff_identity(cutoff): ref for cutoff, ref in refs_by_cutoff.items()}
            if set(refs) - set(self.sinks_by_cutoff) or (not refs and primary is None):
                raise ValueError('View references contain missing or unregistered branch cutoffs')
            primary_sink = primary[0] if primary is not None else None
            if primary_sink is not None and any(primary_sink is sink for sink in self.sinks_by_cutoff.values()):
                raise ValueError('Primary sink must be independent of sibling sinks')
            self._inflight_views.add(view_key)
        # Distinct views publish concurrently. Each view still fences all of its
        # own cutoff files before acknowledging, and the sinks serialize their
        # manifests internally.
        pending = deque()
        records_by_cutoff = {}

        def acknowledge(cutoff, ref, sink, path=None):
            record = sink.wait_published_layer(ref, scalar_layer_suffix(ref, view_token=view_token),
                output_path=path) if ref is not None else None
            records = () if record is None else (record,)
            completed(cutoff, view_key, records)
            records_by_cutoff[cutoff] = records

        def finish_oldest():
            cutoff, ref, path = pending.popleft()
            acknowledge(cutoff, ref, self.sinks_by_cutoff[cutoff], path)

        try:
            if primary is not None:
                # Fence the already-enqueued floor without taking ownership of
                # its executor or rewriting the ordinary scalar payload.
                acknowledge(self.primary_cutoff, primary[1], primary_sink)
            for cutoff, ref in refs.items():
                if ref is None:
                    acknowledge(cutoff, None, self.sinks_by_cutoff[cutoff])
                    continue
                if len(pending) >= self.max_pending_layers:
                    finish_oldest()
                with self._lock:
                    if self._failure is not None:
                        raise RuntimeError('Another branch layer publication failed') from self._failure
                path = self.sinks_by_cutoff[cutoff].submit_layer(ref,
                    scalar_layer_suffix(ref, view_token=view_token))
                pending.append((cutoff, ref, path))
            while pending:
                finish_oldest()
            with self._lock:
                self._published_views.add(view_key)
                self._inflight_views.discard(view_key)
                self._idle.notify_all()
            return records_by_cutoff
        except BaseException as error:
            with self._lock:
                if self._failure is None:
                    self._failure = error
                self._inflight_views.discard(view_key)
                self._idle.notify_all()
            try:
                self._drain_all(primary_sink)
            except BaseException as drain_error:
                if drain_error is not error and callable(getattr(error, 'add_note', None)):
                    error.add_note(f'Branch layer drain also failed: {drain_error}')
            raise

    def _wait_idle_locked(self) -> None:
        while self._inflight_views:
            self._idle.wait()

    def wait(self) -> None:
        """Settle every cutoff, including after an earlier publication failure."""
        with self._lock:
            self._wait_idle_locked()
            self._drain_all()

    def shutdown(self) -> None:
        """Drain all writers before shutting down every scalar sink executor."""
        with self._lock:
            self._wait_idle_locked()
            if self._closed:
                return
            first_error: BaseException | None = None
            try:
                self._drain_all()
            except BaseException as error:
                first_error = error
            for sink in self.sinks_by_cutoff.values():
                try:
                    sink.shutdown()
                except BaseException as error:
                    if first_error is None:
                        first_error = error
            self._closed = True
            if first_error is not None:
                raise first_error


__all__ = ['TtaSweepBranchLayers', 'scalar_layer_suffix']

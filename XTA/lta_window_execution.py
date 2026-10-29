"""Shared bounded-window dispatch, reduction, and ordered commit for LTA.

The production frontier and the generation reference backend supply only the
relay continuation after a settled graph. Window scheduling and artifact
ownership stay in this single engine.
"""

from __future__ import annotations

import hashlib
from contextlib import ExitStack, nullcontext
from dataclasses import dataclass
from pathlib import Path
import time
from typing import Callable, Mapping, Sequence, TYPE_CHECKING

from .lta_propagation import read_seed_artifact, write_seed_artifact
from .lta_scheduler import LtaSessionWork, LtaViewAffinityScheduler
from .lta_workers import LtaWorkerPool
from .lta_runtime import LtaRuntimeViewPlan
from .lta_rendering import LtaPhysicalViewCacheRef

if TYPE_CHECKING:
    from .lta_execution import _PlannedChain


@dataclass(frozen=True)
class WindowExecution:
    """Run inputs shared by window execution and its relay continuation."""

    scheduler: LtaViewAffinityScheduler
    pool: LtaWorkerPool
    view_plan: LtaRuntimeViewPlan
    cache_ref: LtaPhysicalViewCacheRef
    view_union: object
    temp_root: Path
    conf: float
    empty_frame_limit: int | None
    worker_task_timeout: float
    trace: object | None = None
    enable_2D_hole_fill: bool = False


@dataclass(frozen=True)
class SettledGeneration:
    """Borrowed state valid while the continuation handles a settled graph.

    A continuation may retire the prior coverage snapshot before frontier work.
    Otherwise the engine refreshes that snapshot when it registers the next
    window graph. The engine retains coverage and worker resource ownership.
    """

    records: Sequence[Mapping[str, object]]
    generation: int
    first_plan_order: int
    windowed: bool
    coverage: object
    snapshots: ExitStack
    prior_coverage_path: Path
    worker_audit: dict[str, object]
    admission_audit: dict[str, object]
    dispatched: list[dict[str, object]]


@dataclass(frozen=True)
class RelayContinuation:
    """Next generation and chains, or a terminal generation when chains is None."""

    generation: int
    chains: Sequence[_PlannedChain] | None = None


def drive_window_generations(
    *,
    execution: WindowExecution,
    initial: Sequence[_PlannedChain],
    continue_relays: Callable[[WindowExecution, SettledGeneration], RelayContinuation],
    admission_policy: str,
    relay_fan_in: str,
):
    """Own coverage resources for one bounded-window execution."""
    from .lta_coverage import LtaCoverageLedger

    if execution.scheduler.helper_queue_order != "head":
        raise ValueError("bounded production dispatch requires helper_queue_order='head'")
    with ExitStack() as resources:
        coverage = LtaCoverageLedger(
            Path(execution.temp_root) / "lineage_coverage.sqlite3",
            frame_count=execution.view_plan.frame_count,
        )
        resources.callback(coverage.close)
        snapshots = resources.enter_context(ExitStack())
        return _drive_windows_with_coverage(
            execution=execution, initial=initial, continue_relays=continue_relays,
            admission_policy=admission_policy, relay_fan_in=relay_fan_in,
            coverage=coverage, snapshots=snapshots,
        )


def _drive_windows_with_coverage(
    *, execution: WindowExecution, initial: Sequence[_PlannedChain],
    continue_relays: Callable[[WindowExecution, SettledGeneration], RelayContinuation],
    admission_policy: str, relay_fan_in: str,
    coverage: object, snapshots: ExitStack,
) -> tuple[int, Mapping[int, int], tuple[Mapping[str, object], ...], Mapping[str, object]]:
    from . import lta_execution as coordinator

    scheduler = execution.scheduler
    pool = execution.pool
    view_union = execution.view_union
    temp_root = execution.temp_root
    worker_task_timeout = execution.worker_task_timeout
    trace = execution.trace
    if scheduler.helper_queue_order != "head":
        raise ValueError("bounded production dispatch requires helper_queue_order='head'")
    payload_by_work: dict[str, Mapping[str, object]] = {
        chain.work.work_id: chain.payload for chain in initial
    }
    children: dict[str, list[str]] = {}
    chain_state: dict[str, dict[str, object]] = {}
    windowed = bool(initial and "chain_work_id" in initial[0].payload)

    def register_tasks(tasks: Sequence[_PlannedChain]) -> None:
        for task in tasks:
            payload_by_work[task.work.work_id] = dict(task.payload)
            trace_path = getattr(trace, "path", None)
            if trace_path is not None:
                payload_by_work[task.work.work_id]["worker_trace_root"] = str(Path(trace_path).parent)
            chain_id = str(task.payload.get("chain_work_id", task.work.work_id))
            state = chain_state.setdefault(chain_id, {
                "remaining": 0, "observations": {}, "legacy_relays": [],
                "windowed": "chain_work_id" in task.payload,
            })
            state["remaining"] = int(state["remaining"]) + 1
            for predecessor in task.work.dependency_work_ids:
                children.setdefault(predecessor, []).append(task.work.work_id)

    register_tasks(initial)
    active: dict[str, object] = {}
    active_started: dict[str, float] = {}
    reduced_work_ids: set[str] = set()
    dispatched: list[dict[str, object]] = []
    worker_audit = coordinator._new_worker_audit()
    worker_audit["task_count"] = 0
    worker_audit["skipped_window_count"] = 0
    worker_audit["workers_per_gpu"] = scheduler.workers_per_device
    admission_audit: dict[str, object] = {
        "policy": admission_policy,
        "spatial_seed_candidates": 0, "spatial_seed_handoffs": 0,
        "repeated_seed_revisions": 0, "temporal_seed_handoffs": 0,
        "temporal_window_handoffs": 0, "coalesced_generations": [],
    }
    worker_audit["relay_admission"] = admission_audit
    for ready in tuple(getattr(pool, "ready_events", ())):
        coordinator._accumulate_worker_ready_audit(worker_audit, ready)
    authoritative_active_coverage: dict[
        tuple[str, int, str], tuple[tuple[int, int], ...]
    ] = {}
    relay_records: dict[int, list[dict[str, object]]] = {0: []}
    generation_observations: dict[tuple[str, int], dict[tuple[str, int], dict[str, object]]] = {}
    raw_relay_endpoints = 0
    prior_coverage_path = Path(temp_root) / "lineage_coverage_prior.sqlite3"
    prior_coverage = coverage.snapshot(prior_coverage_path)
    snapshots.callback(prior_coverage.close)
    generation = 0
    next_plan_order = max(chain.work.plan_order for chain in initial) + 1
    view_key = initial[0].work.view
    scheduler.mark_projection_ready(
        view_key,
        device_id=scheduler.owner_for_view(view_key),
    )
    tasks_root = temp_root / "tasks"
    last_progress = time.monotonic()

    def progress(*, force: bool = False) -> None:
        nonlocal last_progress
        now = time.monotonic()
        if not force and now - last_progress < 10.0:
            return
        counts = scheduler.queue_counts()
        reason = "waiting_for_window_results" if counts["active"] else "ready_work" if counts["ready"] else "generation_fan_in"
        if trace is not None:
            trace.event("scheduler_state", generation=generation, reason=reason, **counts)
            trace.flush()
            print(f"LTA window DAG: generation={generation} ready={counts['ready']} blocked={counts['blocked']} active={counts['active']} reason={reason}", flush=True)
        last_progress = now

    def skip_branch(work_id: str, *, reason: str = "empty_dogfood_boundary") -> None:
        pending = [work_id]
        while pending:
            skipped_id = pending.pop()
            scheduler.skip_pending(skipped_id, coordinator._SkippedWindow(reason=reason))
            if reason == "covered_continuation":
                admission_audit["temporal_window_handoffs"] = int(admission_audit["temporal_window_handoffs"]) + 1
            if trace is not None:
                trace.event("window_skipped", work_id=skipped_id, reason=reason)
            pending.extend(children.get(skipped_id, ()))

    def complete_chain_window(work: LtaSessionWork) -> None:
        nonlocal raw_relay_endpoints
        payload = payload_by_work[work.work_id]
        chain_id = str(payload.get("chain_work_id", work.work_id))
        state = chain_state[chain_id]
        state["remaining"] = int(state["remaining"]) - 1
        if state["remaining"]:
            return
        from .lta_relay_episodes import merge_relay_observations_across_chains
        phase = trace.phase("chain_relay_coalescing", chain_work_id=chain_id) if trace is not None else nullcontext()
        with phase:
            target = generation_observations.setdefault((work.tile_config_id, int(work.tile_index)), {})
            merge_relay_observations_across_chains(target, state["observations"])
        raw_count = 2 * sum(len(record.get("episodes", ())) for record in state["observations"].values())
        raw_relay_endpoints += raw_count
        relay_records.setdefault(work.relay_generation, []).extend(state["legacy_relays"])
        if state["windowed"]:
            worker_audit["chain_count"] = int(worker_audit["chain_count"]) + 1
        if trace is not None:
            trace.event("chain_settled", chain_work_id=chain_id, relay_count=0,
                        raw_relay_endpoint_count=raw_count, relay_emission_deferred=True)
        del chain_state[chain_id]

    def emit_generation_relays() -> None:
        from .lta_worker_adapter import _write_relay_artifacts

        emitted_count = 0
        phase = trace.phase("generation_relay_emission", generation=generation,
                            raw_endpoint_count=raw_relay_endpoints) if trace is not None else nullcontext()
        with phase:
            for (config_id, tile_index), observations in sorted(generation_observations.items()):
                emitted = _write_relay_artifacts(
                    observations,
                    output_dir=Path(temp_root) / "generation-relays" / f"generation-{generation:04d}" / coordinator._safe_token(config_id) / f"tile-{tile_index:04d}",
                    source_tile_index=tile_index, generation=generation,
                )
                relay_records.setdefault(generation, []).extend(emitted)
                emitted_count += len(emitted)
        worker_audit["relay_artifact_count"] = int(worker_audit["relay_artifact_count"]) + emitted_count
        summary = {"generation": generation, "raw_endpoint_count": raw_relay_endpoints,
                   "coalesced_endpoint_count": emitted_count,
                   "coalesced_episode_count": sum(len(record.get("episodes", ()))
                       for observations in generation_observations.values() for record in observations.values()),
                   "coverage": coverage.stats()}
        admission_audit["coalesced_generations"].append(summary)
        generation_observations.clear()
        if trace is not None:
            trace.event("generation_relay_summary", **summary)
            trace.flush()

    while True:
        progress()
        coordinator._dispatch_available(
            scheduler,
            pool,
            payload_by_work,
            active,
            active_started,
            dispatched,
            tasks_root=tasks_root,
            trace=trace,
        )
        if active:
            oldest_work_id = min(
                active_started,
                key=lambda work_id: active_started[work_id],
            )
            remaining = (
                float(worker_task_timeout)
                - (time.monotonic() - active_started[oldest_work_id])
            )
            if remaining <= 0.0:
                raise TimeoutError(
                    f"LTA worker task {oldest_work_id!r} exceeded its "
                    f"{float(worker_task_timeout):.1f}-second lease"
                )
            wait_phase = trace.phase("wait_for_window_result", **scheduler.queue_counts()) if trace is not None else nullcontext({})
            with wait_phase as wait_details:
                try:
                    worker_result = pool.wait_result(timeout=min(remaining, 10.0))
                except TimeoutError:
                    worker_result = None
                    wait_details["poll_timeout"] = True
            if worker_result is None:
                progress(force=True)
                if time.monotonic() - active_started[oldest_work_id] >= float(worker_task_timeout):
                    raise TimeoutError(f"LTA worker task {oldest_work_id!r} exceeded its {float(worker_task_timeout):.1f}-second lease")
                continue
            claim = active.pop(worker_result.work_id, None)
            active_started.pop(worker_result.work_id, None)
            if claim is None:
                raise RuntimeError(
                    f"worker returned unclaimed LTA work {worker_result.work_id}"
                )
            if (
                int(worker_result.execution_device_id) != int(claim.execution_device_id)
                or int(getattr(worker_result, "worker_index", 0)) != int(claim.worker_index)
            ):
                raise RuntimeError("worker result does not match the claimed GPU and worker slot")
            # Dense OR into the private view is associative and commutative.
            # Reduce completed work immediately so a slow earlier chain cannot
            # hold every helper idle or retain a full-view file per later task.
            # Audit/relay acknowledgment remains in deterministic plan order.
            phase = trace.phase("window_result_reduction", work_id=worker_result.work_id) if trace is not None else nullcontext()
            with phase:
                manifest = coordinator._load_chain_manifest(worker_result)
                parent_payload = payload_by_work[worker_result.work_id]
                if list(manifest["output_frame_range"]) != [parent_payload["output_frame_start"], parent_payload["output_frame_stop"]]:
                    raise RuntimeError("worker result changed its owned window frame range")
                _relays, union_artifact = coordinator._consume_chain_manifest(manifest, view_union=view_union)
                outgoing = coordinator._verified_dogfood_artifacts(manifest, parent_payload)
            coordinator._unlink_consumed_temp_artifacts((union_artifact,), temp_root=temp_root)
            reduced_work_ids.add(worker_result.work_id)
            empty_children: list[str] = []
            covered_children: list[str] = []
            used_seed_paths: set[str] = set()
            for child_id in children.get(worker_result.work_id, ()):
                child_payload = dict(payload_by_work[child_id])
                prompt = int(child_payload["window"]["prompt_frame"])
                artifact = outgoing.get(prompt)
                if artifact is None:
                    empty_children.append(child_id)
                    continue
                child_seeds = read_seed_artifact(str(artifact["path"]), expected_sha256=str(artifact["sha256"]))
                remaining_seeds = tuple(
                    seed for seed in child_seeds
                    if not prior_coverage.can_handoff(
                        seed, tile_index=int(child_payload["tile_index"]),
                        direction=str(child_payload["window"]["direction"]),
                    )
                )
                handed_off = len(child_seeds) - len(remaining_seeds)
                admission_audit["temporal_seed_handoffs"] = int(admission_audit["temporal_seed_handoffs"]) + handed_off
                if not remaining_seeds:
                    covered_children.append(child_id)
                    continue
                if handed_off:
                    replacement = write_seed_artifact(
                        Path(worker_result.artifact_path).parent / "continuations" / f"{hashlib.sha256(child_id.encode()).hexdigest()[:20]}.npz",
                        remaining_seeds,
                    )
                    artifact = {"path": str(replacement.path), "sha256": replacement.sha256}
                child_payload["seed_artifact_path"] = str(artifact["path"])
                child_payload["seed_artifact_sha256"] = str(artifact["sha256"])
                payload_by_work[child_id] = child_payload
                used_seed_paths.add(str(artifact["path"]))
            coverage_phase = trace.phase("lineage_coverage_ingest", work_id=worker_result.work_id) if trace is not None else nullcontext()
            with coverage_phase:
                coordinator._ingest_lineage_coverage(manifest, parent_payload, coverage, temp_root=temp_root)
            scheduler.complete(claim, worker_result)
            for child_id in empty_children:
                skip_branch(child_id)
            for child_id in covered_children:
                skip_branch(child_id, reason="covered_continuation")
            unused_seeds = [str(record["path"]) for record in outgoing.values() if str(record["path"]) not in used_seed_paths]
            coordinator._unlink_consumed_temp_artifacts(unused_seeds, temp_root=temp_root)
            # Overlap ordered receipt work with the next GPU session instead
            # of waiting until all committable manifests have been processed.
            coordinator._dispatch_available(
                scheduler, pool, payload_by_work, active, active_started,
                dispatched, tasks_root=tasks_root, trace=trace,
            )

        while True:
            commit = scheduler.claim_committable()
            if commit is None:
                break
            consumed_artifacts: list[Path] = []
            try:
                for _work, result in commit.entries:
                    task_payload = payload_by_work[_work.work_id]
                    if isinstance(result, coordinator._SkippedWindow):
                        worker_audit["skipped_window_count"] = int(worker_audit["skipped_window_count"]) + 1
                        worker_audit["planned_window_count"] = int(worker_audit["planned_window_count"]) + 1
                        worker_audit["halted_window_count"] = int(worker_audit["halted_window_count"]) + 1
                        complete_chain_window(_work)
                        continue
                    if _work.work_id not in reduced_work_ids:
                        raise RuntimeError("LTA work reached commit before dense union reduction")
                    manifest = coordinator._load_chain_manifest(result)
                    if "chain_work_id" in task_payload:
                        manifest = {**manifest, "task_granularity": "window"}
                    coordinator._accumulate_worker_audit(worker_audit, manifest, result)
                    coordinator._accumulate_authoritative_active_coverage(
                        authoritative_active_coverage,
                        _work,
                        manifest,
                    )
                    chain_id = str(task_payload.get("chain_work_id", _work.work_id))
                    state = chain_state[chain_id]
                    observation_artifact = manifest.get("relay_observation_artifact")
                    if observation_artifact is not None:
                        from .lta_relay_episodes import read_relay_observations, merge_relay_observations
                        phase = trace.phase("relay_episode_merge", work_id=_work.work_id) if trace is not None else nullcontext()
                        with phase:
                            observations = read_relay_observations(observation_artifact)
                            merge_relay_observations(state["observations"], observations)
                        consumed_artifacts.append(Path(observation_artifact["path"]))
                        # Legacy chain workers also emit standalone relay files.
                        # Their observation packet is authoritative for global
                        # coalescing, so verify and retire the duplicate files.
                        consumed_artifacts.extend(coordinator._verified_relay_artifact_paths(tuple(manifest.get("relays", ()))))
                    else:
                        state["legacy_relays"].extend(dict(item) for item in manifest.get("relays", ()))
                    input_seed = coordinator._verified_artifact_path(
                        str(task_payload["seed_artifact_path"]),
                        expected_sha256=task_payload["seed_artifact_sha256"],
                        description="consumed worker input-seed artifact",
                    )
                    consumed_artifacts.extend(
                        (
                            Path(result.artifact_path).resolve(strict=True),
                            input_seed,
                        )
                    )
                    complete_chain_window(_work)
            except BaseException:
                scheduler.fail_commit(commit)
                raise
            scheduler.complete_commit(commit)
            coordinator._unlink_consumed_temp_artifacts(
                consumed_artifacts,
                temp_root=temp_root,
            )
            reduced_work_ids.difference_update(work.work_id for work, _result in commit.entries)

        if not scheduler.generation_settled(view_key, generation):
            if not active:
                pool.check_liveness()
                counts = scheduler.queue_counts()
                if counts["ready"] == 0:
                    if trace is not None:
                        trace.event("scheduler_stalled", generation=generation, **counts)
                    raise RuntimeError(
                        f"LTA relay generation {generation} stalled before settlement: "
                        f"no active or ready windows; blocked={counts['blocked']}, "
                        f"completed_awaiting_commit={counts['completed_awaiting_commit']}"
                    )
            continue

        emit_generation_relays()
        continuation = continue_relays(execution, SettledGeneration(
            records=relay_records.get(generation, ()), generation=generation,
            first_plan_order=next_plan_order, windowed=windowed,
            coverage=coverage, snapshots=snapshots,
            prior_coverage_path=prior_coverage_path, worker_audit=worker_audit,
            admission_audit=admission_audit, dispatched=dispatched,
        ))
        if continuation.chains is None:
            generation = continuation.generation
            scheduler.seal_view(view_key)
            break
        planned = continuation.chains
        next_generation = continuation.generation
        accepted_chain_count = len(planned)
        snapshots.close()
        coordinator._unlink_consumed_temp_artifacts((prior_coverage_path,), temp_root=temp_root)
        prior_coverage = coverage.snapshot(prior_coverage_path)
        snapshots.callback(prior_coverage.close)
        generation_observations.clear()
        raw_relay_endpoints = 0
        graph_phase = trace.phase("relay_window_graph", generation=next_generation,
                                  accepted_chains=accepted_chain_count) if trace is not None else nullcontext()
        with graph_phase:
            if windowed:
                planned = coordinator._plan_window_tasks(planned, first_plan_order=next_plan_order)
            scheduler.register_generation(
                view_key,
                next_generation,
                (chain.work for chain in planned),
            )
            scheduler.seal_generation(view_key, next_generation)
            register_tasks(planned)
        if trace is not None:
            trace.event("relay_generation_planned", generation=next_generation,
                        accepted_chains=accepted_chain_count, planned_windows=len(planned),
                        coverage=coverage.stats())
            trace.flush()
        print(f"LTA relay plan: generation={next_generation} accepted_chains={accepted_chain_count} planned_windows={len(planned)}", flush=True)
        relay_records[next_generation] = []
        next_plan_order += len(planned)
        generation = next_generation

    if active:
        raise RuntimeError("LTA fixed point sealed with active worker claims")
    admission_audit["coverage"] = coverage.stats()
    worker_audit["authoritative_active_lineage_tile_count"] = len(
        authoritative_active_coverage
    )
    worker_audit["authoritative_active_range_count"] = sum(
        len(ranges) for ranges in authoritative_active_coverage.values()
    )
    worker_audit["window_graph"] = {
        "planned_work_count": len(scheduler.work) + int(worker_audit.get("canonical_frontier", {}).get("tasks", 0)),
        "dispatched_work_count": len(dispatched),
        "skipped_work_count": int(worker_audit["skipped_window_count"]),
        "dependency_readiness": "verified_window_completion",
        "relay_fan_in": relay_fan_in,
    }
    return (
        generation,
        dict(pool.pids),
        tuple(
            sorted(
                dispatched,
                key=lambda item: (int(item["plan_order"]), str(item["work_id"])),
            )
        ),
        worker_audit,
    )

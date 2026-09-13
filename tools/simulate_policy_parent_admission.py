"""CPU-only event model for bounded policy-group admission fairness.

Geometry and group identity can be checked against the 144736 cluster log. Work
costs are deliberately abstract: results compare assumptions, not cluster time.
This development prototype does not modify or invoke the production scheduler.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from dataclasses import asdict, dataclass
import hashlib
import heapq
import io
import contextlib
import json
import math
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
GIB = 1024 ** 3


@dataclass(frozen=True)
class Group:
    name: str
    family: str
    frames: int
    pass_bytes: int
    processing_shape: tuple[int, int, int] = ()


def cluster_workload(log: Path | None = None) -> tuple[list[Group], dict]:
    """Compile the exact recorded geometry, without models or native canvases."""
    from tools.smoke_import import install_stubs
    install_stubs()  # The geometry compiler needs none of the stubbed image kernels.
    from XTA.config import AzimuthalViewRequest, RadialViewRequest, SphericalViewRequest, TiltedViewGroup
    from XTA.geometry import view_processing_volume_shape
    from XTA.unification.runtime import compile_physical_views

    shape, raster = (2911, 3064, 3022), 3072
    raw = log.read_bytes() if log is not None else None
    text = raw.decode('utf-8', errors='replace') if raw is not None else ''
    if raw is not None:
        match = re.search(r'-> processing shape \(t,Y,X\)=\((\d+),\s*(\d+),\s*(\d+)\)', text)
        if match is None:
            raise ValueError('Log does not contain its processing shape')
        shape = tuple(map(int, match.groups()))
    axes = ('transverse', 'sagittal', 'coronal')
    targets = (*axes, *(f'tilted_{axis}' for axis in axes))
    with contextlib.redirect_stdout(io.StringIO()):
        compiled = compile_physical_views(t_dim=shape[0], height=shape[1], width=shape[2],
            cartesian_views=axes, azimuthal_requests=tuple(AzimuthalViewRequest(v) for v in targets),
            tilted_groups=(TiltedViewGroup(axes, (30.,), ('vertical', 'horizontal')),),
            azimuthal_native_raster=raster, radial_requests=tuple(RadialViewRequest(v) for v in targets),
            radial_patch_size=raster,
            spherical_requests=(SphericalViewRequest('transverse'), SphericalViewRequest('tilted_transverse')),
            spherical_patch_size=raster, sampling_policy='coverage')
    groups = []
    for view in compiled.views:
        processing = tuple(view_processing_volume_shape(view, raster))
        family = ('tilted_azimuthal' if view.name.startswith('azimuthal_tilted') else view.family)
        groups.append(Group(view.name, family, int(view.num_slices), math.prod(processing), processing))
    by_name = {group.name: group for group in groups}
    if raw is not None:
        order = list(dict.fromkeys(name.split('__tta')[0] for name in re.findall(
            r'best/(\S+) baseline union workspace:', text)))
        if set(order) != set(by_name):
            raise ValueError('Log groups differ from the supported all-family, 30-degree coverage preset')
        frame_match = re.search(r'(\d+) source frame\(s\)/--angle', text)
        if frame_match is None or int(frame_match.group(1)) != sum(group.frames for group in groups):
            raise ValueError('Compiled source-frame count does not match the log')
        groups = [by_name[name] for name in order]
    return groups, dict(preset='all_families_30_degree_coverage', shape_tyx=shape, raster=raster,
        group_count=len(groups), rendered_frames=sum(group.frames for group in groups),
        group_counts_by_family=dict(Counter(group.family for group in groups)),
        order=('observed first-admission order; log ordering is not elapsed time' if log else 'geometry compiler order'),
        source_log=str(log) if log else None, source_log_sha256=hashlib.sha256(raw).hexdigest() if raw else None)


def simulate(groups: list[Group], *, policy: str, ratio: int = 4, total_bytes: int = 384 * GIB,
             inference_bytes: int = 128 * GIB, view_limit: int = 4, workers: int = 4,
             post_workers: int = 4, lease_frames: int = 64, queue_depth: int = 2,
             small_group_quota: int = 4, inference_factor: float = 1., postprocess_factor: float = 1.) -> dict:
    """Simulate chunked inference and separately retained postprocess parents.

    Existing groups get dispatch preference, then unopened groups keep input
    order. The cap predicates match current policy admission; detailed production
    task ranking, GPU projection contention and storage throughput are not modeled.
    """
    if policy not in ('current', 'reserve_oversized'):
        raise ValueError('Unknown admission policy')
    if (not groups or len({g.name for g in groups}) != len(groups)
            or any(g.frames <= 0 or g.pass_bytes <= 0 for g in groups)):
        raise ValueError('Workload must contain uniquely named positive groups')
    if min(ratio, total_bytes, inference_bytes, view_limit, workers, post_workers,
           lease_frames, queue_depth, small_group_quota) <= 0:
        raise ValueError('Simulation bounds must be positive')
    if any(not math.isfinite(value) or value <= 0 for value in (inference_factor, postprocess_factor)):
        raise ValueError('Cost factors must be finite and positive')
    needs = [group.pass_bytes * ratio for group in groups]
    if max(needs) > total_bytes:
        raise ValueError('A complete policy group exceeds the hard total dense cap')
    chunks = [deque(min(lease_frames, group.frames - start)
                    for start in range(0, group.frames, lease_frames)) for group in groups]
    remaining = [len(parts) for parts in chunks]
    unopened = set(range(len(groups)))
    active = set()
    gpu_queues = [deque() for _ in range(workers)]
    gpu_running = {}
    post_queue = deque()
    post_running = {}
    events = []
    serial = 0
    now = 0.
    inf_used = post_used = 0
    peak_total = peak_inference = peak_post = peak_groups = 0
    admission_order = []
    starts = {}
    completed_parents = set()
    completed_frames = completed_chunks = 0
    gpu_busy_units = all_idle_units = reservation_drain_units = 0.
    inference_drained_at = None
    reserved = None
    since_oversized = reservations = 0
    oversize_order = [i for i, need in enumerate(needs) if need > inference_bytes]
    events_trace = []

    def push(delay, kind, worker, payload):
        nonlocal serial
        serial += 1
        heapq.heappush(events, (now + max(1e-12, delay), serial, kind, worker, payload))

    def check_bounds():
        nonlocal peak_total, peak_inference, peak_post, peak_groups
        if inf_used + post_used > total_bytes:
            raise AssertionError('Hard total dense cap exceeded')
        if inf_used > inference_bytes and len(active) != 1:
            raise AssertionError('Inference cap exceeded outside the single-group exception')
        if len(active) > view_limit or min(inf_used, post_used) < 0:
            raise AssertionError('Invalid admission accounting')
        peak_total = max(peak_total, inf_used + post_used)
        peak_inference = max(peak_inference, inf_used)
        peak_post = max(peak_post, post_used)
        peak_groups = max(peak_groups, len(active))

    def refresh_reservation():
        nonlocal reserved, reservations
        if policy != 'reserve_oversized' or reserved is not None:
            return
        candidates = [i for i in oversize_order if i in unopened]
        small_pending = any(i in unopened and needs[i] <= inference_bytes for i in range(len(groups)))
        if candidates and (since_oversized >= small_group_quota or not small_pending):
            reserved = candidates[0]
            reservations += 1
            events_trace.append(dict(time_units=now, event='reserve', group=groups[reserved].name))

    def can_open(index):
        if reserved is not None and index != reserved:
            return False
        return (len(active) < view_limit and (not active or inf_used + needs[index] <= inference_bytes)
                and inf_used + post_used + needs[index] <= total_bytes)

    def admit(index):
        nonlocal inf_used, reserved, since_oversized
        if not can_open(index):
            raise AssertionError('Unadmitted group selected')
        unopened.remove(index)
        active.add(index)
        inf_used += needs[index]
        starts[index] = (now, len(admission_order))
        admission_order.append(index)
        events_trace.append(dict(time_units=now, event='admit', group=groups[index].name,
                                dense_bytes=needs[index], retained_postprocess_bytes=post_used))
        if index in oversize_order:
            since_oversized = 0
        else:
            since_oversized += 1
        if reserved == index:
            reserved = None
        check_bounds()
        refresh_reservation()

    def dispatch():
        refresh_reservation()
        while True:
            available = [w for w in range(workers)
                         if len(gpu_queues[w]) + int(w in gpu_running) < queue_depth]
            if not available:
                break
            candidates = [i for i in sorted(active) if chunks[i]]
            if not candidates:
                candidates = [i for i in sorted(unopened) if can_open(i)]
                if not candidates:
                    break
                admit(candidates[0])
            index = candidates[0]
            worker = min(available, key=lambda w: (len(gpu_queues[w]) + int(w in gpu_running), w))
            gpu_queues[worker].append((index, chunks[index].popleft()))
        for worker in range(workers):
            if worker not in gpu_running and gpu_queues[worker]:
                index, count = gpu_queues[worker].popleft()
                gpu_running[worker] = index
                push(count * ratio * inference_factor / 1000., 'inference', worker, (index, count))
        for worker in range(post_workers):
            if worker not in post_running and post_queue:
                index, pass_index = post_queue.popleft()
                post_running[worker] = (index, pass_index)
                # Explicit synthetic density variation, unrelated to measured confidence.
                density = .85 + (int.from_bytes(hashlib.sha256(groups[index].name.encode()).digest()[:2], 'little') % 31) / 100.
                pass_factor = 1. + .05 * pass_index
                delay = groups[index].pass_bytes / GIB * .05 * postprocess_factor * density * pass_factor
                push(delay, 'postprocess', worker, (index, pass_index))

    dispatch()
    while events:
        event_time = events[0][0]
        delta = event_time - now
        gpu_busy_units += delta * len(gpu_running)
        if inference_drained_at is None and not gpu_running:
            all_idle_units += delta
        if reserved is not None and active:
            reservation_drain_units += delta
        now = event_time
        simultaneous = []
        while events and events[0][0] == event_time:
            simultaneous.append(heapq.heappop(events))
        for _, _, kind, worker, payload in simultaneous:
            index, value = payload
            if kind == 'inference':
                del gpu_running[worker]
                remaining[index] -= 1
                completed_chunks += 1
                completed_frames += value * ratio
                if remaining[index] == 0:
                    active.remove(index)
                    inf_used -= needs[index]
                    post_used += needs[index]
                    post_queue.extend((index, pass_index) for pass_index in range(ratio))
                    events_trace.append(dict(time_units=now, event='inference_done', group=groups[index].name))
            else:
                del post_running[worker]
                parent = (index, value)
                if parent in completed_parents:
                    raise AssertionError('A policy parent was retired twice')
                completed_parents.add(parent)
                post_used -= groups[index].pass_bytes
            check_bounds()
        if completed_frames == sum(group.frames for group in groups) * ratio and inference_drained_at is None:
            inference_drained_at = now
        dispatch()
    if (unopened or active or inf_used or post_used or post_queue or gpu_running or post_running
            or any(gpu_queues) or any(chunks) or any(remaining)
            or len(completed_parents) != len(groups) * ratio):
        raise AssertionError('Simulation stranded inference or an independent policy parent')
    return dict(policy=policy, simulated_time_units=now, inference_drained_time_units=inference_drained_at,
        all_gpu_idle_before_inference_drain_units=all_idle_units,
        gpu_busy_worker_units=gpu_busy_units, reservation_drain_time_units=reservation_drain_units,
        reservations=reservations, peak_total_dense_bytes=peak_total, peak_inference_bytes=peak_inference,
        peak_postprocess_bytes=peak_post, peak_inference_groups=peak_groups,
        completed_groups=len(groups), completed_independent_parents=len(completed_parents),
        completed_model_frames=completed_frames, completed_slice_leases=completed_chunks,
        oversized_group_count=len(oversize_order),
        oversized_admissions=[dict(group=groups[index].name, time_units=starts[index][0],
                                   admission_position=starts[index][1], dense_bytes=needs[index]) for index in oversize_order],
        admission_order=[groups[index].name for index in admission_order], trace=events_trace)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--log', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--ratios', type=int, nargs='+', default=[1, 2, 4])
    parser.add_argument('--total-gib', type=float, nargs='+', default=[384])
    parser.add_argument('--inference-gib', type=float, default=128)
    parser.add_argument('--lease-frames', type=int, default=64)
    parser.add_argument('--small-group-quota', type=int, default=4)
    parser.add_argument('--inference-factors', type=float, nargs='+', default=[.5, 1., 2.])
    parser.add_argument('--postprocess-factors', type=float, nargs='+', default=[.25, 1., 4.])
    args = parser.parse_args(argv)
    args.output = args.output.resolve()
    if args.output.is_relative_to(ROOT):
        parser.error('Generated simulations belong in task Scratch, outside the repository')
    groups, metadata = cluster_workload(args.log)
    cases = []
    for total_gib in args.total_gib:
        for ratio in args.ratios:
            for inference_factor in args.inference_factors:
                for postprocess_factor in args.postprocess_factors:
                    settings = dict(ratio=ratio, total_bytes=int(total_gib * GIB),
                        inference_bytes=int(args.inference_gib * GIB), lease_frames=args.lease_frames,
                        small_group_quota=args.small_group_quota, inference_factor=inference_factor,
                        postprocess_factor=postprocess_factor)
                    baseline = simulate(groups, policy='current', **settings)
                    fair = simulate(groups, policy='reserve_oversized', **settings)
                    cases.append(dict(settings=settings, current=baseline, reserve_oversized=fair,
                        synthetic_time_change_percent=100. * (fair['simulated_time_units'] / baseline['simulated_time_units'] - 1.)))
    record = dict(description='Discrete-event development prototype; no production scheduling change.',
        limitations=[
            'Time units and relative costs are synthetic, not measured seconds or a prediction of cluster walltime.',
            'All groups are available at time zero. Existing groups are preferred; unopened groups follow the supplied order.',
            'Cap predicates, grouped ownership and independent pass retirement are modeled; detailed production lease ranking is not.',
            'Four inference workers each queue two leases. Four CPU postprocess workers retire independent passes.',
            'GPU projection contention, model/renderer startup, geometry caches, NRRD sink throughput and final topology are omitted.',
            'Postprocess factors represent uncalibrated density/confidence cost changes; they do not correspond to numeric --conf values.',
            'Inference factors represent uncalibrated policy/model costs; no baseline/superheavy speedup is asserted.',
            'The 384-GiB cap is only the parent window, not a complete process RAM plan.',
        ], workload=metadata, groups=[asdict(group) for group in groups], cases=cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(dict(output=str(args.output), workload=metadata, cases=len(cases),
        synthetic_change_percent_range=[min(case['synthetic_time_change_percent'] for case in cases),
                                        max(case['synthetic_time_change_percent'] for case in cases)]), indent=2))


if __name__ == '__main__':
    main()

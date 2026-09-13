"""Reserve bounded retained-payload RAM before dispatching native shell owners."""
from __future__ import annotations

import os
import math
from pathlib import Path
from .config import GIB
from .runtime import HYBRID_DEFERRED_RESULT_MODE, _register_memfd_owner, memfd_workspace_enabled, workspace_anon_cap_bytes, scratch_dir_is_memory_backed
from .workspace import _env_flag, _env_float, _read_meminfo_bytes, available_anon_work_bytes


def publication_ram_headroom():
    """Use physical/cgroup headroom only; spare swap is not a RAM cache budget."""
    info = _read_meminfo_bytes()
    if 'MemAvailable' not in info and os.name == 'nt':
        try:
            import psutil
            return max(0, int(psutil.virtual_memory().available))
        except (ImportError, OSError, ValueError, AttributeError):
            return 0
    return max(0, min(int(info.get('MemAvailable', 0)), int(available_anon_work_bytes())))


def native_fullframe_dense_reserve(tasks, *, total_dense_limit, min_conf=0.,
                                   dense_tiling=False, nrrd_layers=False,
                                   bounded_retirement=True):
    """Charge admitted native parents separately from packed D1 publication grants.

    Policy groups share the retirement window only when every parent is
    explicitly enrolled in scheduler admission with the same storage contract.
    Legacy file tasks still have no lifetime bound and reserve all canvases.
    This charges file-backed canvases too: a pathname on tmpfs is still RAM,
    and dirty pages on ordinary scratch are not immediately reclaimable.
    """
    parents = {}
    policy_groups = {}
    count = (2 if float(min_conf) > 0 else 1) + (
        (3 if nrrd_layers else 1) if dense_tiling else 0)
    has_policy_groups = False
    for grouped_task in tasks:
        siblings = grouped_task.get('augmentation_pass_tasks') or ()
        has_policy_groups = has_policy_groups or bool(siblings)
        members = (grouped_task, *siblings)
        if bounded_retirement and any(task.get('bounded_parent_admission', False) for task in members):
            if not all(task.get('bounded_parent_admission', False)
                       and task.get('kind') == 'fullframe' for task in members):
                raise ValueError('Native policy memory plan requires admission for every sibling parent')
            if {str(task.get('result_mode', 'file')) for task in members} not in ({'file'}, {'direct_union'}):
                raise ValueError('Native policy memory plan requires a uniform file or shared-parent contract')
        group = {}
        for task in members:
            mode = str(task.get('result_mode', 'file'))
            if task.get('kind') != 'fullframe' or mode not in ('file', 'direct_union', HYBRID_DEFERRED_RESULT_MODE):
                continue
            key = (str(task['model_name']), str(task['view'].name))
            shape = tuple(int(v) for v in task['processing_shape'])
            if len(shape) != 3 or min(shape) <= 0:
                raise ValueError('Native parent memory plan requires a positive 3D processing shape')
            bounded_policy = bool(mode in ('file', 'direct_union') and bounded_retirement
                                  and task.get('bounded_parent_admission', False))
            entry = (mode, math.prod(shape) * count, bounded_policy)
            if bounded_policy and key in group:
                raise ValueError(f'Native policy memory plan repeats a sibling parent: {key}')
            if key in parents and parents[key] != entry:
                raise ValueError(f'Native parent memory plan has inconsistent tasks for {key}')
            parents[key] = entry
            group[key] = entry
        if any(entry[2] for entry in group.values()):
            if not all(entry[2] for entry in group.values()):
                raise ValueError('Native policy memory plan requires admission for every sibling parent')
            group_key = frozenset(group)
            for key in group:
                previous = policy_groups.get(key)
                if previous is not None and previous != group_key:
                    raise ValueError(f'Native policy parent belongs to inconsistent admission groups: {key}')
                policy_groups[key] = group_key
            group_bytes = sum(entry[1] for entry in group.values())
            if group_bytes > int(total_dense_limit):
                raise RuntimeError(
                    f'External-policy parent group requires {group_bytes / GIB:.1f} GiB '
                    f'of simultaneous dense canvases, exceeding the '
                    f'{int(total_dense_limit) / GIB:.1f} GiB dense admission limit. '
                    'All passes of one group must fit together to reuse each rendered batch. '
                    'Reduce --augmentation_ratio or raise YOLO_TTA_DIRECT_UNION_TOTAL_GIB '
                    'only with sufficient physical/cgroup memory headroom.')
    unbounded = sum(size for mode, size, bounded in parents.values() if mode == 'file' and not bounded)
    if unbounded > int(total_dense_limit) and len(parents) > 1:
        advice = (
            'External-policy passes require bounded parent retirement to share a memory window. '
            'Run without retained temporary artifacts, reduce --augmentation_ratio or selected '
            'views/angles, or increase YOLO_TTA_DIRECT_UNION_TOTAL_GIB only with sufficient '
            'real memory headroom. '
            if has_policy_groups else
            'Enable YOLO_TTA_GPU_WORKER_DIRECT_UNION=1 for bounded shared unions, '
            'or reduce the requested native views. '
        )
        raise RuntimeError(
            f'File-mode full-frame unions require {unbounded / GIB:.1f} GiB of retained '
            f'parent canvases, exceeding the {int(total_dense_limit) / GIB:.1f} GiB dense limit. '
            + advice + 'File-mode unions have no parent admission.')
    shared = [size for mode, size, bounded in parents.values() if mode != 'file' or bounded]
    # Preserve the established emergency lane for ordinary shared parents.
    # Policy groups require their complete set of passes to fit the hard window.
    oversized_shared = max((size for mode, size, bounded in parents.values()
                            if mode != 'file' and not bounded), default=0)
    admitted = min(sum(shared), max(int(total_dense_limit), oversized_shared))
    if not bounded_retirement:
        admitted = sum(shared)
    return int(unbounded + admitted)


def policy_parent_memory_plan(tasks, *, requested_dense_limit, available_ram_bytes,
                              source_shape, output_reserve_bytes=0,
                              parent_transient_reserve_bytes=0,
                              worker_buffer_reserve_bytes=0, batch_size=1):
    """Bound policy parents and auxiliary outputs within physical/cgroup RAM.

    Policy full-frame leases partition each parent and cannot be split again.
    Shared workers write directly into admitted parent windows; no second copy
    is retained for normal result masks. File compatibility additionally retains
    at most one window's worth of results until coordinator collection. Explicit
    azimuthal seam files remain charged under both storage contracts.
    Packed support also includes its raw and pending compressed representations;
    its ratio uses actual model-raster versus processing-plane dimensions.
    Caller supplies RAM headroom without swap and the independently admitted
    transient/output/batch bounds. Persistent output on tmpfs needs a separate
    retained-storage budget; it cannot be treated as ordinary disk spill.
    """
    shape = tuple(int(v) for v in source_shape)
    if len(shape) != 3 or min(shape) <= 0:
        raise ValueError('Policy memory plan requires a positive 3D source shape')
    coverage_numerator, coverage_denominator = 0, 1
    results_numerator, results_denominator = 0, 1
    for task in tasks:
        if task.get('kind') != 'fullframe' or not task.get('augmentation_pass_tasks'):
            continue
        group = (task, *task['augmentation_pass_tasks'])
        modes = {str(parent.get('result_mode', 'file')) for parent in group}
        if modes not in ({'file'}, {'direct_union'}):
            raise ValueError('Policy memory plan requires a uniform file or shared-parent contract')
        marked = [bool(parent.get('bounded_parent_admission', False)) for parent in group]
        if ((modes == {'direct_union'} or any(marked)) and not all(marked)):
            raise ValueError('Policy memory plan requires admission for every sibling parent')
        if not all(parent.get('kind') == 'fullframe' for parent in group):
            raise ValueError('Policy parent memory plan cannot combine full-frame and tile outputs')
        parent_plane_bytes = 0
        support_bytes = 0
        slices = max(1, int(task.get('slice_count', task['processing_shape'][0])))
        padding = 0
        if str(getattr(task.get('view'), 'family', '')) == 'azimuthal':
            from .geometry import azimuthal_batch_padding_count
            padding = int(azimuthal_batch_padding_count(task['view'], slices,
                max(1, int(task.get('prediction_batch', batch_size))),
                slice_offset=int(task.get('slice_start', 0))))
        # Shared policy parents receive D2H directly into their disjoint window.
        # Only explicit seam slots need separate files; fallback mode also keeps
        # the normal task-window result until coordinator collection.
        result_slices = padding + (0 if str(task.get('result_mode', 'file')) == 'direct_union' else slices)
        if result_slices * results_denominator > results_numerator * slices:
            results_numerator, results_denominator = result_slices, slices
        for index, parent in enumerate(group):
            plane_shape = tuple(int(v) for v in parent['processing_shape'])
            if len(plane_shape) != 3 or min(plane_shape) <= 0:
                raise ValueError('Policy memory plan requires positive 3D processing shapes')
            # Counting only union canvases is conservative when confidence or
            # tile category canvases make the admitted dense set larger.
            parent_plane_bytes += plane_shape[1] * plane_shape[2]
            settings = parent.get('augmentation_settings', task.get('augmentation_settings'))
            if index and str(getattr(settings, 'coverage', 'packed')) != 'none':
                raster = max(1, int(parent.get('out_size', max(plane_shape[1:]))))
                support_bytes += raster * ((raster + 7) // 8)
        # npz may expand incompressible support slightly. Include both its raw
        # and compressed files; small per-file headers fit the safety margin.
        numerator = 201 * support_bytes * (slices + padding)
        denominator = 100 * parent_plane_bytes * slices
        if numerator * coverage_denominator > coverage_numerator * denominator:
            coverage_numerator, coverage_denominator = numerator, denominator

    available = max(0, int(available_ram_bytes))
    requested = max(0, int(requested_dense_limit))
    safety = min(32 * GIB, max(64 * 1024**2, available // 32))
    topology = 5 * math.prod(shape)
    output = max(0, int(output_reserve_bytes))
    transient = max(0, int(parent_transient_reserve_bytes))
    workers = max(0, int(worker_buffer_reserve_bytes))
    fixed = safety + topology + output + transient + workers
    remaining = max(0, available - fixed)
    denominator = coverage_denominator * results_denominator
    numerator = (denominator + results_numerator * coverage_denominator
                 + coverage_numerator * results_denominator)
    # Individual ceilings below may add two bytes; keep that rounding inside RAM.
    dense = min(requested, max(0, remaining - 2) * denominator // numerator)
    results = (dense * results_numerator + results_denominator - 1) // results_denominator
    coverage = (dense * coverage_numerator + coverage_denominator - 1) // coverage_denominator
    return dict(requested_dense_limit_bytes=requested, available_ram_bytes=available,
        dense_limit_bytes=dense, file_result_reserve_bytes=results,
        coverage_reserve_bytes=coverage, safety_reserve_bytes=safety,
        source_topology_reserve_bytes=topology, output_reserve_bytes=output,
        parent_transient_reserve_bytes=transient, worker_buffer_reserve_bytes=workers,
        fixed_reserve_bytes=fixed, total_reserve_bytes=fixed + dense + results + coverage,
        file_result_ratio_numerator=results_numerator,
        file_result_ratio_denominator=results_denominator,
        coverage_ratio_numerator=coverage_numerator,
        coverage_ratio_denominator=coverage_denominator)


def retained_payload_plan(shapes, available, worker_count, publication_pending, unpack_bytes, *, cap=0, output_reserve_bytes=0, native_dense_reserve_bytes=0):
    """Reserve future topology/union and all admitted host publication bitsets.

    Each selected layer is charged its *worst-case* packed payload for its whole
    lifetime, including layers not started yet. Independent workers cannot each
    promise themselves the same currently-free RAM. Disk fallbacks need no grant.
    """
    shapes = [tuple(map(int, s)) for s in shapes]
    if any(len(s) != 3 or min(s) <= 0 for s in shapes):
        raise ValueError('Publication memory plan requires positive 3D shapes')
    dense = max((t * h * w for t, h, w in shapes), default=0)
    words = ((dense + 31) // 32) * 4
    # Five uint8-volume equivalents reserve the final union and worst-case uint32
    # local label raster. Host bitsets cover every publication credit, plus the
    # currently computing owner on each worker. Other work keeps a fixed margin.
    reserve = 32 * GIB + 5 * dense + max(0, int(worker_count)) * (
        (max(0, int(publication_pending)) + 1) * words
        + 2 * max(1, int(publication_pending)) * max(0, int(unpack_bytes)))
    reserve += max(0, int(output_reserve_bytes))
    reserve += max(0, int(native_dense_reserve_bytes))
    budget = max(0, int(available) - reserve) // 2
    if int(cap) > 0:
        budget = min(budget, int(cap))
    remaining = budget
    grants = []
    for t, h, w in shapes:
        need = t * h * ((w + 7) // 8)
        grant = need if need <= remaining else 0
        grants.append(grant)
        remaining -= grant
    return dict(reserve_bytes=reserve, budget_bytes=budget,
                reserved_bytes=budget - remaining, grants=grants)


def publication_output_reserve(sink, window_bytes, member_bytes):
    """Budget actual sink concurrency, mirror canvases and gzip input/output windows."""
    if sink is None:
        return 0
    mirrors = list(sink.low_quality_specs)
    canvases = sum(int(np_size.output_shape_t_y_x[0]) * int(np_size.output_shape_t_y_x[1]) *
                   int(np_size.output_shape_t_y_x[2]) for np_size in mirrors)
    lanes = 1 + min(4, len(mirrors))
    windows = lanes * 2 * (max(0, int(window_bytes)) + 2 * max(0, int(member_bytes)))
    dense = int(sink.output_shape[0]) * int(sink.output_shape[1]) * int(sink.output_shape[2])
    # Include a worst-case global compressed spool as well as per-writer windows.
    return int(sink.max_workers) * (canvases + windows) + dense + dense // 100 + 1024**2


def plan_native_publication_memory(tasks, *, keep_temp, worker_count, publication_pending, unpack_bytes, output_reserve_bytes=0, native_dense_reserve_bytes=0):
    """Create parent-owned empty memfds; immutable task grants bound their growth."""
    if (keep_temp or not memfd_workspace_enabled() or scratch_dir_is_memory_backed()
            or not _env_flag('YOLO_TTA_PUBLICATION_RAM', True)
            or not _env_flag('YOLO_TTA_PACKED_OWNER_PUBLICATION', True)):
        return None
    by_path = {}
    for task in tasks:
        if (task.get('projection_contract') == 'radial_native_pull_v1'
                and task.get('result_mode') == 'd1_owner'
                and not task.get('d1_view_shadow_required')):
            by_path.setdefault(str(task['d1_store_dir']), []).append(task)
    if not by_path:
        return None
    groups = list(by_path.values())
    shapes = [group[0]['d1_output_shape'] for group in groups]
    for group, shape in zip(groups, shapes):
        owners = {(str(task['model_name']), str(task['view'].name)) for task in group}
        if len(owners) != 1 or any(task['d1_output_shape'] != shape for task in group):
            print('Native publication RAM plan declined ambiguous layer identities; retaining disk backings.', flush=True)
            return None
    cap = max(0, int(_env_float('YOLO_TTA_PUBLICATION_RAM_GIB', 0.) * GIB))
    workspace_cap = int(workspace_anon_cap_bytes())
    if workspace_cap > 0:
        cap = min(cap, workspace_cap) if cap else workspace_cap
    headroom = publication_ram_headroom()
    plan = retained_payload_plan(shapes, headroom, worker_count, publication_pending, unpack_bytes,
                                 cap=cap, output_reserve_bytes=output_reserve_bytes,
                                 native_dense_reserve_bytes=native_dense_reserve_bytes)
    # Small volumes can admit many more layers than a process can hold open.
    # Leave descriptors for inference IPC, readers, codecs and final output.
    try:
        import resource
        soft_limit = int(resource.getrlimit(resource.RLIMIT_NOFILE)[0])
        used_fds = len(os.listdir('/proc/self/fd'))
        descriptor_budget = 256 if soft_limit < 0 else max(0, min(256, (soft_limit-used_fds-64)//2))
    except (ImportError, OSError, ValueError):
        descriptor_budget = 0
    admitted = 0
    for index, (group, grant) in enumerate(zip(groups, plan['grants'])):
        if not grant or admitted >= descriptor_budget:
            plan['grants'][index] = 0
            continue
        fd = None
        try:
            fd = os.memfd_create('xta-packed-publication', flags=getattr(os, 'MFD_CLOEXEC', 0))
            path = Path(group[0]['d1_store_dir']).absolute() / 'chunks.bin'
            _register_memfd_owner(str(path), fd, 'planned packed source contribution')
            backing = f'/proc/{os.getpid()}/fd/{fd}'
            fd = None  # runtime registry now owns it through all downstream consumers
            for task in group:
                task['d1_memory_payload_path'] = backing
                task['d1_memory_payload_limit'] = int(grant)
                task['d1_memory_payload_reserve'] = int(plan['reserve_bytes'])
            admitted += 1
        except OSError as exc:
            plan['grants'][index] = 0
            if fd is not None:
                os.close(fd)
            print(f'Publication RAM backing unavailable; keeping disk payload: {exc}', flush=True)
    plan['reserved_bytes'] = sum(plan['grants'])
    print('Native publication RAM plan: '
          f'physical/cgroup_headroom={headroom / GIB:.1f} GiB, '
          f'future_work_reserve={plan["reserve_bytes"] / GIB:.1f} GiB, '
          f'native_dense_window={int(native_dense_reserve_bytes) / GIB:.1f} GiB, '
          f'retained_budget={plan["budget_bytes"] / GIB:.1f} GiB, '
          f'worst_case_reserved={plan["reserved_bytes"] / GIB:.1f} GiB; '
          f'{admitted}/{len(groups)} layers admitted (fd_budget={descriptor_budget}), disk spill remains available.', flush=True)
    return plan

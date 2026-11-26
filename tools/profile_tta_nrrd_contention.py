"""Exercise native scheduler host dispatch while production NRRD writers run.

This is CPU and disk only. It reuses ``OutputLoad`` from the radial host
contention benchmark, exercises the real libdeflate member codec, and shares one
RuntimeTelemetry instance across the output work. Native queue and receipt
timings come from the native scheduler and its spawned host actors. The default
invocation writes a plan and ``--execute`` enables the large fixture.

Install python-deflate into a task-scoped ``--deps-dir`` under the task's Scratch root
before execution. Outputs and the temporary 256x2048x2048 logical fixture stay
inside that output root. Every completed NRRD's decoded SHA-256 is verified on
its reference grid, so sink layers in the production cropped (segment bounding box)
layout verify against the same fixture hash as a full raster would.
The background ``OutputLoad`` writers deliberately use the full-raster layout.
"""
from __future__ import annotations

import argparse
from contextlib import ExitStack
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import sys
import tempfile
import time
from unittest import mock

from XTA import outputs, runtime
from tools.benchmark_radial_host_contention import OutputLoad
from tools.profile_tta_dispatch import load_trace_specs, run_replay
from tools.nrrd_reference_grid import reference_payload_sha256


def _decoded_nrrd_sha256(path: Path) -> str:
    """SHA-256 of the layer's reference-grid bytes; cropped and full layouts agree."""
    return reference_payload_sha256(path)


def _check_task_scoped_dependency(deps_dir: Path, root: Path) -> None:
    dependency_root = deps_dir.resolve()
    task_root = root.parent
    if task_root != dependency_root and task_root not in dependency_root.parents:
        raise ValueError(f'dependency directory must stay under task root: {dependency_root}')
    if not dependency_root.is_dir():
        raise FileNotFoundError(f'task-scoped dependency directory is missing: {dependency_root}')
    if str(dependency_root) not in sys.path:
        sys.path.insert(0, str(dependency_root))
    if importlib.util.find_spec('deflate') is None:
        raise RuntimeError(
            f'python-deflate is missing from {dependency_root}; install it into that directory'
        )


def plan(*, trace: Path, output_root: Path, deps_dir: Path,
         shape: tuple[int, int, int], load_workers: int,
         sink_jobs: int, actor_workers: int = 4, actor_tickets: int = 8,
         max_tasks: int | None = None,
         pre_replay_warmup_seconds: float = 0.0) -> dict[str, object]:
    if len(shape) != 3 or any(value < 1 for value in shape):
        raise ValueError('shape must be three positive dimensions')
    if (load_workers < 1 or sink_jobs < 0 or actor_workers < 1 or actor_tickets < 1
            or pre_replay_warmup_seconds < 0):
        raise ValueError('invalid worker, ticket, or warmup bound')
    if max_tasks is not None and max_tasks < 1:
        raise ValueError('max_tasks must be positive')
    return {
        'trace': str(trace.resolve()), 'output_root': str(output_root.resolve()),
        'deps_dir': str(deps_dir.resolve()), 'shape_tyx': list(shape),
        'logical_input_bytes': int(shape[0] * shape[1] * shape[2]),
        'load_workers': load_workers, 'sink_jobs': sink_jobs,
        'gzip_workers': 64, 'fill_workers': 32, 'sink_workers': 12,
        'codec': 'libdeflate', 'mirrors': False,
        'nrrd_layer_layout': outputs.NRRD_LAYER_LAYOUT_CROPPED,
        'actor_workers': actor_workers, 'actor_tickets': actor_tickets,
        'pre_replay_warmup_seconds': pre_replay_warmup_seconds,
        'max_tasks': max_tasks,
        'gpu_used': False,
        'limitations': (
            'Spawned actors use the production scheduler socket and logical credits. '
            'They send synthetic scientific completions without CUDA rendering or model inference.'
        ),
    }


def run_combined(*, trace: Path, output_root: Path, deps_dir: Path,
                 shape: tuple[int, int, int] = (256, 2048, 2048),
                 load_workers: int = 8, sink_jobs: int = 12,
                 actor_workers: int = 4, actor_tickets: int = 8,
                 replay_max_seconds: float = 240.0,
                 max_tasks: int | None = None,
                 pre_replay_warmup_seconds: float = 0.0) -> dict[str, object]:
    root = output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    _check_task_scoped_dependency(deps_dir, root)
    workload = plan(trace=trace, output_root=root, deps_dir=deps_dir,
                    shape=shape, load_workers=load_workers,
                    sink_jobs=sink_jobs, actor_workers=actor_workers, actor_tickets=actor_tickets,
                    max_tasks=max_tasks,
                    pre_replay_warmup_seconds=pre_replay_warmup_seconds)
    specs = load_trace_specs(trace)
    if max_tasks is not None:
        specs = specs[:max_tasks]
    environment = {
        'YOLO_TTA_NRRD_MEMBER_CODEC': 'libdeflate',
        'YOLO_TTA_NRRD_GZIP_WORKERS': '64',
        'YOLO_TTA_NRRD_FILL_WORKERS': '32',
        'YOLO_TTA_NRRD_LAYER_SINK_WORKERS': '12',
        'YOLO_TTA_TELEMETRY': '1',
        'YOLO_TTA_TELEMETRY_PATH': str(root / 'nrrd-telemetry.jsonl'),
    }
    result: dict[str, object] = {'plan': workload}
    repo_root = Path(__file__).resolve().parents[1]
    source_paths = {}
    for name in (
            'XTA/runtime.py', 'XTA/native_scheduler.py', 'XTA/outputs.py',
            'native/prepared_scheduler.hpp', 'tools/profile_tta_dispatch.py',
        ):
        module = sys.modules.get(name.removesuffix('.py').replace('/', '.'))
        source_paths[name] = Path(getattr(module, '__file__', repo_root / name)).resolve()
    result['source_paths'] = {name: str(path) for name, path in source_paths.items()}
    result['source_sha256'] = {
        name: hashlib.sha256(path.read_bytes()).hexdigest()
        for name, path in source_paths.items()
    }
    run_error: BaseException | None = None
    with mock.patch.dict(os.environ, environment):
        telemetry = runtime.RuntimeTelemetry()
        with ExitStack() as patches:
            patches.enter_context(mock.patch.object(runtime, '_RUNTIME_TELEMETRY', telemetry))
            codec = outputs._require_nrrd_member_codec()
            if str(codec[0]) != 'libdeflate':
                raise RuntimeError(f'codec fallback invalidates this reproduction: {codec[0]}')
            result['codec_selected'] = str(codec[0])
            with tempfile.TemporaryDirectory(
                    prefix='tta-nrrd-contention-', dir=root,
                    ignore_cleanup_errors=True) as raw:
                workspace = Path(raw).resolve()
                # Check before TemporaryDirectory's recursive cleanup can run.
                if workspace.parent != root:
                    raise RuntimeError(f'temporary workspace escaped output root: {workspace}')
                load = OutputLoad(workspace, load_workers, mirrors=False, shape=shape)
                sink = None
                sink_paths: list[Path] = []
                run_error: BaseException | None = None
                try:
                    load.start()
                    sink = outputs.NrrdLayerSink(
                        nrrd_dir=workspace / 'sink', stem='load',
                        output_shape_tyx=shape, max_workers=12)
                    for index in range(sink_jobs):
                        path = sink.submit_layer(load.ref, f'sink_{index:02d}')
                        if path is None:
                            raise RuntimeError('sink rejected the load layer')
                        sink_paths.append(Path(path))
                    if pre_replay_warmup_seconds:
                        deadline = time.monotonic() + float(pre_replay_warmup_seconds)
                        while time.monotonic() < deadline:
                            time.sleep(min(.1, max(0, deadline - time.monotonic())))
                    result['output_progress_before_replay'] = {
                        'load_writer_counts': list(load.counts),
                        'sink': sink.progress_counts(),
                    }
                    started = time.perf_counter()
                    nrrd_calls_before = int(telemetry.phase_calls.get(
                        'nrrd.compression.libdeflate.compress', 0))
                    try:
                        result['replay'] = run_replay(
                            specs, workers=actor_workers, ticket_capacity=actor_tickets,
                            max_seconds=replay_max_seconds,
                        )
                    finally:
                        result['replay_wall_seconds_with_output'] = time.perf_counter() - started
                        result['output_progress_during_replay'] = {
                            'compression_calls': int(telemetry.phase_calls.get(
                                'nrrd.compression.libdeflate.compress', 0)) - nrrd_calls_before,
                            'load_writer_counts': list(load.counts),
                            'sink': sink.progress_counts(),
                        }
                except BaseException as exc:
                    run_error = exc
                finally:
                    try:
                        load.close()  # also verifies every completed load-writer SHA-256
                    except BaseException as exc:
                        if run_error is None:
                            run_error = exc
                    if sink is not None:
                        try:
                            sink.wait()
                        except BaseException as exc:
                            if run_error is None:
                                run_error = exc
                        finally:
                            sink.shutdown()
                hashes = {path.name: _decoded_nrrd_sha256(path) for path in sink_paths}
                if hashes and any(value != load.expected for value in hashes.values()):
                    raise RuntimeError('sink NRRD differs from the exact fixture hash')
                result['output_verification'] = {
                    'source_sha256': load.expected,
                    'load_writer_completions': list(load.counts),
                    'sink_files_verified': len(hashes),
                    'sink_sha256': hashes,
                    'all_exact': len(hashes) == sink_jobs,
                }
                if run_error is not None:
                    result['replay_error'] = f'{type(run_error).__name__}: {run_error}'
            telemetry.flush(final=True)
    result['temporary_fixture_cleaned'] = not workspace.exists()
    (root / 'combined-report.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    if run_error is not None:
        raise run_error
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--deps-dir', type=Path, required=True)
    parser.add_argument('--shape', type=int, nargs=3, default=(256, 2048, 2048))
    parser.add_argument('--load-workers', type=int, default=8)
    parser.add_argument('--sink-jobs', type=int, default=12)
    parser.add_argument('--actor-workers', type=int, default=4)
    parser.add_argument('--actor-tickets', type=int, default=8)
    parser.add_argument('--replay-max-seconds', type=float, default=240.0)
    parser.add_argument('--max-tasks', type=int)
    parser.add_argument('--pre-replay-warmup-seconds', type=float, default=0.0)
    parser.add_argument('--execute', action='store_true',
                        help='allocate fixture and run CPU/disk contention')
    args = parser.parse_args()
    root = args.output_dir.resolve()
    root.mkdir(parents=True, exist_ok=True)
    args_plan = plan(trace=args.trace, output_root=root, deps_dir=args.deps_dir,
        shape=tuple(args.shape), load_workers=args.load_workers,
        sink_jobs=args.sink_jobs, actor_workers=args.actor_workers, actor_tickets=args.actor_tickets,
        max_tasks=args.max_tasks,
        pre_replay_warmup_seconds=args.pre_replay_warmup_seconds)
    (root / 'plan.json').write_text(json.dumps(args_plan, indent=2), encoding='utf-8')
    if not args.execute:
        print(f'Plan saved to {root / "plan.json"}; add --execute after the GPU comparison finishes.')
        return
    result = run_combined(trace=args.trace, output_root=root, deps_dir=args.deps_dir,
        shape=tuple(args.shape), load_workers=args.load_workers,
        sink_jobs=args.sink_jobs, actor_workers=args.actor_workers, actor_tickets=args.actor_tickets,
        replay_max_seconds=args.replay_max_seconds, max_tasks=args.max_tasks,
        pre_replay_warmup_seconds=args.pre_replay_warmup_seconds)
    print(json.dumps({
        'replay_wall_seconds_with_output': result['replay_wall_seconds_with_output'],
        'all_exact': result['output_verification']['all_exact'],
        'temporary_fixture_cleaned': result['temporary_fixture_cleaned'],
    }))


if __name__ == '__main__':
    main()

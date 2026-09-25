"""Compare semantic occupancy queries with canonical PTA raster classification.

CPU-only local sanity check. It preserves the categorical sampling rules and
reports per-family timings; it does not extrapolate a remote job's completion time.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def canonical_classification(mask, coverage, plan, index):
    import numpy as np
    from XTA import geometry, pta_rendering

    canvas_tiles = [tile for tile in plan.tile_layout if tile.shared_job is None]
    full, canvas = pta_rendering.render_plan_frame_mask_source(
        mask=mask, plan=plan, idx=index, need_canvas=bool(canvas_tiles))
    known, known_canvas = (None, None) if coverage is None else pta_rendering.render_plan_frame_mask_source(
        mask=coverage, plan=plan, idx=index, need_canvas=bool(canvas_tiles))
    result = {'full': bool(np.any(full if known is None else np.logical_and(full, known)))}
    for tile in plan.tile_layout:
        if tile.shared_job is not None:
            plane = geometry.render_categorical_dense_tile_for_job(mask, plan.view.shared_view, tile.shared_job, index)
            support = None if coverage is None else geometry.render_categorical_dense_tile_for_job(
                coverage, plan.view.shared_view, tile.shared_job, index)
        else:
            plane = pta_rendering.resize_centered(
                pta_rendering.extract_padded_tile(canvas, tile.x, tile.y, tile.cfg.tile_size),
                tile.out_w, tile.out_h, pta_rendering.cv2.INTER_NEAREST)
            support = None if coverage is None else pta_rendering.resize_centered(
                pta_rendering.extract_padded_tile(known_canvas, tile.x, tile.y, tile.cfg.tile_size),
                tile.out_w, tile.out_h, pta_rendering.cv2.INTER_NEAREST)
        result[tile.tile_tag] = bool(np.any(plane if support is None else np.logical_and(plane, support)))
    return result


def _heatsoak(seconds, workers):
    import cv2
    import numpy as np

    deadline = time.perf_counter() + seconds
    def work(seed):
        mask = (np.random.default_rng(seed).random((512, 512)) > .8).astype(np.uint8)
        calls = 0
        while time.perf_counter() < deadline:
            cv2.connectedComponents(mask, connectivity=8)
            calls += 1
        return calls
    started = time.perf_counter()
    with ThreadPoolExecutor(workers) as pool:
        calls = sum(pool.map(work, range(workers)))
    return {'seconds': time.perf_counter() - started, 'workers': workers,
            'connected_component_calls': calls}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--imgsz', type=int, default=2048)
    parser.add_argument('--side', type=int, default=2048)
    parser.add_argument('--depth', type=int, default=16)
    parser.add_argument('--frames', type=int, default=3)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--heatsoak-seconds', type=float, default=30)
    parser.add_argument('--cpu-workers', type=int, default=4)
    args = parser.parse_args(argv)
    if min(args.imgsz, args.side, args.depth, args.frames, args.repeats, args.cpu_workers) < 1 or args.heatsoak_seconds < 30:
        parser.error('Dimensions/counts must be positive; CPU heatsoak must be at least 30 seconds')
    args.output.mkdir(parents=True, exist_ok=True)
    import cv2
    import numpy as np
    from XTA import pta
    from XTA.pta_classification import classify_semantic_plan_frame
    from XTA.pta_config import parse_pta_args
    cv2.setNumThreads(1)
    shape = (args.depth, args.side, args.side)
    mask = np.zeros(shape, np.uint8)
    side = args.side
    mask[:, side // 4:3 * side // 4, side // 3:2 * side // 3] = 1
    mask[:, side // 8:side // 8 + 1, :] = 1
    coverage = np.ones(shape, np.uint8)
    coverage[args.depth // 2:] = 0
    cases = (
        ('transverse', ['--enable_cartesian', 'transverse'], 'transverse'),
        ('azimuthal', ['--enable_azimuthal', 'transverse:45'], 'azimuthal'),
        ('radial', ['--enable_radial', 'transverse', '--radial_min_radius', '2'], 'radial'),
        ('spherical', ['--enable_spherical', 'transverse', '--spherical_min_radius', '2'], 'spherical'),
        ('tilted_azimuthal', ['--enable_tilted', 'transverse:15:vertical', '--enable_azimuthal', 'tilted_transverse:45'], 'azimuthal'),
        ('tilted_radial', ['--enable_tilted', 'transverse:15:vertical', '--enable_radial', 'tilted_transverse', '--radial_min_radius', '2'], 'radial'),
        ('tilted_spherical', ['--enable_tilted', 'transverse:15:vertical', '--enable_spherical', 'tilted_transverse', '--spherical_min_radius', '2'], 'spherical'),
    )
    plans = []
    for name, options, family in cases:
        config = parse_pta_args(['--input', 'qualification', '--task', 'semantic', '--imgsz', str(args.imgsz), *options])
        views, _ = pta.compile_v18_pta_views(t_dim=shape[0], h=shape[1], w=shape[2], config=config,
                                            azimuthal_native_raster=args.imgsz)
        view = next(view for view in views if view.family == family or view.shared_view.family == family)
        affine = pta.build_affine(view.src_w, view.src_h, 0.0, view.pad_mode, args.imgsz, shared_view=view.shared_view)
        plan = pta.build_render_plan(view=view, aff=affine, tag=name, out_dir=args.output, stem='fixture',
            tile_configs=(pta.TileConfig(1024, 1024, 's1024_st1024'), pta.TileConfig(1536, 1536, 's1536_st1536')),
            save_overlay=False, imgsz=args.imgsz, label_enabled=True, publish_images=False, publish_labels=False)
        indices = sorted(set(int(x) for x in np.linspace(0, max(0, view.num_slices - 1), args.frames)))
        plans.append((name, plan, indices))
    print(f'CPU heatsoak: {args.heatsoak_seconds:g}s with {args.cpu_workers} workers', flush=True)
    report = {'scope': 'Local CPU classification sanity; no remote ETA extrapolation', 'shape': shape,
              'imgsz': args.imgsz, 'repeats': args.repeats,
              'coverage': 'first half of source slices known, second half ignored',
              'heatsoak': _heatsoak(args.heatsoak_seconds, args.cpu_workers), 'cases': []}
    for name, plan, indices in plans:
        def fast(index):
            result = classify_semantic_plan_frame(mask, coverage, plan, index)
            if result is None:
                raise RuntimeError(f'{name}: occupancy helper unexpectedly declined')
            return result
        reference = [canonical_classification(mask, coverage, plan, index) for index in indices]
        assert [fast(index) for index in indices] == reference, name
        elapsed = {'reference': [], 'occupancy': []}
        for repeat in range(args.repeats):
            for kind in (('reference', 'occupancy') if repeat % 2 == 0 else ('occupancy', 'reference')):
                started = time.perf_counter()
                actual = ([canonical_classification(mask, coverage, plan, index) for index in indices]
                          if kind == 'reference' else [fast(index) for index in indices])
                elapsed[kind].append(time.perf_counter() - started)
                assert actual == reference, (name, kind)
        old, new = (statistics.median(elapsed[key]) for key in ('reference', 'occupancy'))
        row = {'family': name, 'frames': len(indices), 'tiles_per_frame': len(plan.tile_layout),
               'reference_seconds': old, 'occupancy_seconds': new, 'speed_ratio': old / new,
               'foreground_queries': sum(sum(result.values()) for result in reference),
               'exact_decisions': True, 'runs_seconds': elapsed}
        report['cases'].append(row)
        (args.output / 'classification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
        print(f'{name}: {old:.3f}s -> {new:.3f}s ({old/new:.2f}x), exact decisions', flush=True)
    report['status'] = 'complete'
    (args.output / 'classification.json').write_text(json.dumps(report, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

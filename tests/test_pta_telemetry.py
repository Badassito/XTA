from __future__ import annotations

import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta, pta_workers, runtime
from XTA.pta_gpu_publication import GpuPublicationResources
from XTA.pta_config import parse_pta_args
from XTA.pta_scheduler import iter_compatible_work_batches


def test_parent_stages_record_calls_and_errors_without_starting_sampler(monkeypatch, tmp_path):
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '1')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', '0')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_DIR', str(tmp_path))
    telemetry = runtime.RuntimeTelemetry()
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', telemetry)
    monkeypatch.setattr(runtime, 'initialize_runtime_observability',
                        mock.Mock(side_effect=AssertionError('unexpected initialization')))
    with pytest.raises(ValueError, match='No image sequence'):
        pta.load_image_sequence_to_memory([], warnings=pta.WarningLog(), workers=1)
    pta.apply_gaussian_smoothing(np.zeros((2, 2, 2), dtype=np.uint8),
                                 sigma=0, passes=0, warnings=pta.WarningLog())
    pta.drain_render_results(mock.Mock(), {}, until=lambda: True)
    assert telemetry.phase_calls['pta.input.decode_sequence'] == 1
    assert telemetry.counters['pta.input.decode_sequence.errors'] == 1
    assert telemetry.phase_calls['pta.prepare.gaussian'] == 1
    assert telemetry.phase_calls['pta.render.wait'] == 1
    assert telemetry._writer_thread is None


def test_worker_finalizer_is_registered_without_initializing_telemetry():
    with (mock.patch('multiprocessing.util.Finalize') as finalize,
          mock.patch.object(runtime, 'initialize_runtime_observability') as initialize):
        pta_workers._render_worker_initializer()
    finalize.assert_called_once_with(None, pta_workers.shutdown_runtime_observability,
                                     exitpriority=10)
    initialize.assert_not_called()


def test_publication_snapshot_preserves_overlap_and_independent_measurements():
    resource = object.__new__(GpuPublicationResources)
    resource.lock = threading.Lock()
    resource.device = 2
    resource.counts = {'images': 45, 'batches': 3}
    resource.timings = {'policy_call_wall': 2.5, 'host_publication': 1.5}
    resource.closed = True
    resource.gpu_bytes = 2048 * 1024**2
    resource.host_bytes = 512 * 1024**2
    snapshot = resource.snapshot()
    resource.counts['images'] = 46
    resource.timings['host_publication'] = 1.75
    assert snapshot['counts'] == {'images': 45, 'batches': 3}
    assert snapshot['stage_wall_seconds'] == {'policy_call_wall': 2.5, 'host_publication': 1.5}
    assert snapshot['stage_times_overlap'] is True
    assert snapshot['closed'] is True
    assert snapshot['device'] == 2


def test_representative_f3_training_workload_counts_without_rendering(monkeypatch, tmp_path):
    monkeypatch.setenv('YOLO_TTA_TELEMETRY', '0')
    monkeypatch.setenv('YOLO_TTA_TASK_TRACE', '0')
    monkeypatch.setenv('YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER', '0')
    monkeypatch.setattr(runtime, '_RUNTIME_TELEMETRY', runtime.RuntimeTelemetry())
    raw_shape = (1931, 3064, 3022)
    shape = pta.shared_media.compute_cube_resize_shape(*raw_shape)
    assert shape == (2911, 3064, 3022)
    config = parse_pta_args(['--input', str(tmp_path), '--imgsz', '2048',
                             '--enable_cartesian', 'transverse', '--enable_tile',
                             '1024:1024', '1536:1536'])
    views, _ = pta.compile_v18_pta_views(t_dim=shape[0], h=shape[1], w=shape[2],
                                       config=config, azimuthal_native_raster=2048)
    plans = pta.build_plans_for_views(
        views=views, out_dir=tmp_path, stem='F3', save_overlay=False, imgsz=2048,
        label_enabled=True, tile_configs=[pta.TileConfig(t.tile_size, t.tile_stride, t.config_id)
                                        for t in config.tiles],
        center_eligible_frame_indices=(0,), publish_images=False, publish_labels=False)
    assert len(plans) == 1
    plan = plans[0]
    assert (plan.aff.canvas_h, plan.aff.canvas_w) == (3064, 3022)
    assert [sum(tile.cfg.tile_size == size for tile in plan.tile_layout)
            for size in (1024, 1536)] == [9, 4]
    assert all((tile.out_h, tile.out_w) == (2048, 2048) for tile in plan.tile_layout)
    prep = SimpleNamespace(plans=plans, label_enabled=True, src=SimpleNamespace(stem='F3'))
    originals = pta.enumerate_candidates_for_volume(prep)
    for index, candidate in enumerate(originals):
        candidate.source_order = index
    policy = pta.LoadedGpuAugmentation(tmp_path / 'policy.py', '0' * 64,
                                      'build_gpu_augmentation', 'cuda',
                                      mock.Mock(side_effect=AssertionError('no GPU construction')))
    versions, _ = pta.plan_augmented_versions(
        originals, augmentation=policy, augmentation_ratio=5, split_active=False,
        augmented_foregrounds={}, require_augmented_foregrounds=False)
    assert len(originals) == 14
    assert len(versions) == 70  # Ratio includes originals: four extra versions each.
    tasks = pta.build_phase_render_tasks(plans, versions, aug_chunk=4, gpu_batch_size=24)
    assert len(tasks) == 1
    assert len(tasks[0].frames) == 1  # A 70-candidate source frame remains atomic.
    plane = np.broadcast_to(np.uint8(0), (2048, 2048))
    work = [pta._GpuItemWork(candidates, plane, plane, (2048, 2048), 'gray', item_key)
            for item_key, candidates in tasks[0].frames[0].items]
    counts = [sum(len(item.candidates) for item in batch)
              for batch in iter_compatible_work_batches(work, candidate_limit=24)]
    assert counts == [20, 20, 20, 10]

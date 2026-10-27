"""Parent classification merges complete GPU jobs with exact CPU capability fallbacks."""
from __future__ import annotations

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta, pta_publication
from tests.test_pta_fraction_rendering import make_plan


def prepared(tmp_path, *, semantic=False):
    plan = make_plan(tmp_path, 'gray')
    mask = np.zeros((5, 6, 7), np.uint8)
    mask[1:4, 1:5, 1:6] = 1
    coverage = np.ones_like(mask) if semantic else None
    if semantic:
        coverage[2] = 0
    return SimpleNamespace(
        label_enabled=True, semantic_foreground=semantic,
        semantic_coverage_for_render=coverage, mask_for_render=mask,
        plans=[plan], smoothing_stats=[], void_fill_stats=None,
        source_shape=mask.shape, processing_shape=mask.shape,
        foreground_preservation_stats={},
        src=SimpleNamespace(stem='sample', label_source='nrrd',
            yolo_polygons_by_frame={}, labels_by_frame={}),
    )


class CapablePool:
    def __init__(self, truth, *, declined=()):
        self.truth = truth
        self.declined = set(declined)
        self.calls = []

    def classify_foregrounds(self, prep, *, gen, jobs, selected_items=None):
        self.calls.append((gen, list(jobs), selected_items))
        complete = {(prep.plans[p].tag, i) for p, i in jobs if (p, i) not in self.declined}
        truth = {key:value for key,value in self.truth.items() if key[:2] in complete}
        return truth, [job for job in jobs if job in self.declined], {'backend':'test_cuda'}


@pytest.mark.parametrize('semantic', [False, True])
def test_complete_gpu_jobs_skip_cpu_projection_and_match_original_truth(tmp_path, semantic):
    prep = prepared(tmp_path, semantic=semantic)
    expected = pta.classify_original_foregrounds_for_volume(prep, workers=1)
    pool = CapablePool(expected)
    with mock.patch.object(pta, 'render_plan_frame_mask_source', side_effect=AssertionError('GPU job replayed on CPU')), \
         mock.patch.object(pta, 'classify_semantic_plan_frame', side_effect=AssertionError('GPU job replayed on CPU')), \
         mock.patch.object(pta.shared_geometry, 'render_categorical_dense_tile_for_job',
                           side_effect=AssertionError('GPU tile replayed on CPU')):
        actual = pta.classify_original_foregrounds_for_volume(prep, workers=1, gpu_pool=pool, generation=9)
    assert actual == expected
    assert pool.calls[0][0] == 9
    assert len(pool.calls[0][1]) == 5
    assert pool.calls[0][2] is None


@pytest.mark.parametrize('semantic', [False, True])
def test_declined_whole_jobs_use_cpu_without_replaying_completed_jobs(tmp_path, monkeypatch, semantic):
    monkeypatch.setenv('YOLO_TTA_PTA_SEMANTIC_CLASSIFICATION', '0')
    prep = prepared(tmp_path, semantic=semantic)
    expected = pta.classify_original_foregrounds_for_volume(prep, workers=1)
    pool = CapablePool(expected, declined={(0, 1), (0, 3)})
    with mock.patch.object(pta, 'render_plan_frame_mask_source', wraps=pta.render_plan_frame_mask_source) as render:
        actual = pta.classify_original_foregrounds_for_volume(prep, workers=1, gpu_pool=pool)
    assert actual == expected
    assert {call.kwargs['idx'] for call in render.call_args_list} == {1, 3}


def test_selected_tile_only_fraction_jobs_require_only_selected_gpu_keys(tmp_path):
    prep = prepared(tmp_path)
    plan = prep.plans[0]
    selected = [pta.OutputCandidate(order=0, volume_name='sample', parent_view_tag=plan.tag,
        output_tag=plan.tile_layout[1].tile_tag, item_key=plan.tile_layout[1].tile_tag,
        frame_idx=2, is_tile=True, label_enabled=True)]
    expected = pta.classify_original_foregrounds_for_volume(prep, workers=1, selected_candidates=selected)
    pool = CapablePool(expected)
    with mock.patch.object(pta, 'render_plan_frame_mask_source', side_effect=AssertionError('rejected full projected')):
        actual = pta.classify_original_foregrounds_for_volume(
            prep, workers=1, gpu_pool=pool, selected_candidates=selected,
        )
    assert actual == expected
    assert pool.calls == [(0, [(0, 2)], {(plan.tag, 2): {plan.tile_layout[1].tile_tag}})]
    assert all(key[2] != 'full' for key in actual)


@pytest.mark.parametrize('failure', ['missing', 'extra', 'duplicate_decline', 'unknown_decline'])
def test_invalid_gpu_partition_or_incomplete_decisions_fail(tmp_path, failure):
    prep = prepared(tmp_path)
    truth = pta.classify_original_foregrounds_for_volume(prep, workers=1)
    if failure == 'missing':
        truth.pop(next(iter(truth)))
        remaining = []
    elif failure == 'extra':
        truth[('unknown-plan', 0, 'full', 0)] = True
        remaining = []
    else:
        remaining = [(0, 0), (0, 0)] if failure == 'duplicate_decline' else [(3, 0)]
    pool = SimpleNamespace(classify_foregrounds=lambda *a, **kw: (truth, remaining, {}))
    with pytest.raises(RuntimeError, match='GPU foreground classifier'):
        pta.classify_original_foregrounds_for_volume(prep, workers=1, gpu_pool=pool)


def test_gpu_execution_error_propagates_without_cpu_replay(tmp_path):
    prep = prepared(tmp_path)
    pool = SimpleNamespace(classify_foregrounds=mock.Mock(side_effect=RuntimeError('CUDA kernel failed')))
    with mock.patch.object(pta, 'render_plan_frame_mask_source', side_effect=AssertionError('CPU replay after kernel failure')):
        with pytest.raises(RuntimeError, match='CUDA kernel failed'):
            pta.classify_original_foregrounds_for_volume(prep, workers=1, gpu_pool=pool)


def test_all_empty_shortcut_never_enters_gpu(tmp_path):
    prep = prepared(tmp_path)
    prep.mask_for_render[:] = 0
    pool = SimpleNamespace(classify_foregrounds=mock.Mock(side_effect=AssertionError('empty cube sent to GPU')))
    result = pta.classify_original_foregrounds_for_volume(prep, workers=1, gpu_pool=pool)
    assert result and not any(result.values())


def test_predecoded_polygon_jobs_stay_on_geometry_shortcut(tmp_path):
    prep = prepared(tmp_path)
    prep.src.label_source = 'yolo'
    prep.src.yolo_polygons_by_frame = {1: (((.1,.1),(.9,.1),(.9,.9),(.1,.9)),)}
    prep.foreground_preservation_stats['guaranteed_output_foreground_transverse_slices'] = 1
    pool = SimpleNamespace(classify_foregrounds=mock.Mock(side_effect=AssertionError('polygon geometry sent to GPU')))
    expected = pta.classify_original_foregrounds_for_volume(prep, workers=1)
    assert pta.classify_original_foregrounds_for_volume(prep, workers=1, gpu_pool=pool) == expected


class StopAtClassification(Exception):
    pass


@pytest.mark.parametrize('gpu_active,pending_a,pending_b,expected_pool', [
    (True, 0, 0, True), (True, 1, 0, False), (True, 0, 1, False), (False, 0, 0, False),
])
def test_real_parent_planner_uses_gpu_pool_only_when_idle(gpu_active, pending_a, pending_b, expected_pool):
    source = (Path(pta.__file__)).read_text(encoding='utf-8')
    node = next(n for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.FunctionDef) and n.name == '_plan_volume_with_budget')
    # Retain the real nested function, including its nonlocal order counter.
    wrapper = ast.FunctionDef(name='build', args=ast.arguments(posonlyargs=[],args=[],kwonlyargs=[],kw_defaults=[],defaults=[]),
        body=[ast.Assign(targets=[ast.Name(id='order_counter',ctx=ast.Store())],value=ast.Constant(0)),
              node, ast.Return(value=ast.Name(id=node.name,ctx=ast.Load()))], decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),wrapper],type_ignores=[])
    prep = SimpleNamespace(label_enabled=True, src=SimpleNamespace(volume_class='fully_labeled'), plans=[])
    classify = mock.Mock(side_effect=StopAtClassification)
    pool = object()
    namespace = dict(pending_loads={3:(SimpleNamespace(result=lambda:object()),object())},
        runtime_telemetry=lambda:SimpleNamespace(span=lambda *a:nullcontext()),
        _submit_load=lambda *a:None, pending_specs={3:object()}, prepare_loaded_source=lambda *a,**kw:prep,
        args=SimpleNamespace(force=False,enable_3d_void_fill=True,augmentation_execution='offline',
            augmentation_ratio=7.,task='segment',save_semantic=False), out_dir=Path('unused'),
        tile_configs=[],channel_variants=[],volume_allocator=None,topology=SimpleNamespace(cuda_device_ids=(0,1)),
        effective_pipeline_depth=2, dataset_publication_selected=True, fraction=1.,
        enumerate_candidates_for_volume=lambda *a,**kw:[], background_filter_requested=False,
        classify_original_foregrounds_for_volume=classify, augmentation=None, gpu_offline_active=gpu_active,
        render_pool=pool, states={}, progress_by_gen={0:SimpleNamespace(pending_a=pending_a,pending_b=pending_b)})
    exec(compile(ast.fix_missing_locations(module),'<actual PTA classification dispatch>','exec'),namespace)
    with pytest.raises(StopAtClassification):
        namespace['build']()(3, 4)
    assert classify.call_args.kwargs['gpu_pool'] is (pool if expected_pool else None)
    assert classify.call_args.kwargs['generation'] == 3


def test_shared_polygon_helper_preserves_pta_import_api():
    assert pta.mask_has_yolo_polygon is pta_publication.mask_has_yolo_polygon
    assert pta.MASK_POLYGON_FAST_FOREGROUND_MIN_PIXELS == 16

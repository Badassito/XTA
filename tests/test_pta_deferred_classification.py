"""Unrestricted integer-ratio copies classify the already-rendered original."""
from __future__ import annotations

import ast
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path
import pickle
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta, pta_publication as publication, pta_workers as workers
from XTA.pta_dataset import OutputCandidate, plan_augmented_versions
from XTA.pta_augmentation import LoadedAugmentation


def candidate(**changes):
    values = dict(order=0, source_order=0, volume_name='sample', parent_view_tag='transverse',
        output_tag='transverse', item_key='full', frame_idx=0, is_tile=False,
        is_transverse=True, label_enabled=True)
    values.update(changes)
    return OutputCandidate(**values)


def parent_planner(namespace):
    node = next(node for node in ast.walk(ast.parse(Path(pta.__file__).read_text()))
        if isinstance(node, ast.FunctionDef) and node.name == '_plan_volume_with_budget')
    wrapper = ast.FunctionDef(name='build', args=ast.arguments(posonlyargs=[], args=[],
        kwonlyargs=[], kw_defaults=[], defaults=[]), body=[
        ast.Assign(targets=[ast.Name(id='order_counter',ctx=ast.Store())],value=ast.Constant(0)),
        node, ast.Return(value=ast.Name(id=node.name,ctx=ast.Load()))], decorator_list=[])
    module = ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),wrapper],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module), '<actual parent planner>', 'exec'), namespace)
    return namespace['build']()


class StopPlanning(Exception):
    pass


@pytest.mark.parametrize('change,deferred', [
    ({}, True), ({'fraction': .2}, True), ({'ratio': 7.5}, False),
    ({'volume_class': 'partially_labeled'}, False), ({'background_filter': True}, False),
    ({'label_source': 'yolo', 'polygons': {0: (((0.,0.),(1.,0.),(1.,1.)),)}}, False),
    ({'semantic': True}, True),
])
def test_real_parent_skips_projection_only_when_membership_is_class_independent(change, deferred):
    cand = candidate()
    prep = SimpleNamespace(label_enabled=True, semantic_foreground=change.get('semantic',False),
        src=SimpleNamespace(stem='sample', volume_class=change.get('volume_class','fully_labeled'),
            label_source=change.get('label_source','nrrd'), yolo_polygons_by_frame=change.get('polygons',{}),
            labels_by_frame={}, mask_volume=np.ones((1,1,1),np.uint8)),
        plans=[], smoothing_stats=[], void_fill_stats=None, foreground_preservation_stats={})
    classify = mock.Mock(side_effect=StopPlanning)
    budget = mock.Mock(side_effect=StopPlanning)
    namespace = dict(pending_loads={0:(SimpleNamespace(result=lambda: prep.src),workers.WarningLog())},
        runtime_telemetry=lambda:SimpleNamespace(span=lambda *args:nullcontext()),
        _submit_load=lambda *args:None, pending_specs=[object()], prepare_loaded_source=lambda *args,**kwargs:prep,
        args=SimpleNamespace(augmentation_execution='offline',augmentation_ratio=change.get('ratio',7.),task='segment',save_semantic=False,background_percent=1.),
        out_dir=Path('unused'), tile_configs=[],channel_variants=[],volume_allocator=None,
        topology=SimpleNamespace(cuda_device_ids=()),effective_pipeline_depth=1,dataset_publication_selected=True,
        fraction=change.get('fraction',1.),enumerate_candidates_for_volume=lambda *args,**kwargs:[cand],
        background_filter_requested=change.get('background_filter',False),augmentation=object(),
        classify_original_foregrounds_for_volume=classify,gpu_offline_active=False,render_pool=None,progress_by_gen={},states={},
        split_active=False,train_split=None,split_method=None,SplitStats=pta.SplitStats,apply_background_filter=budget)
    with pytest.raises(StopPlanning):
        parent_planner(namespace)(0,1)
    assert prep.src.mask_volume is None
    assert classify.call_count == int(not deferred)
    assert budget.call_count == int(deferred)
    assert cand.foreground_deferred == ('semantic' if change.get('semantic') else 'polygon') if deferred else not cand.foreground_deferred


@pytest.mark.parametrize('mode', ['polygon','semantic'])
@pytest.mark.parametrize('kind', ['empty','point','line','triangle','sixteen'])
def test_source_truth_is_resolved_for_all_versions_before_mutable_policy(mode, kind):
    mask = np.zeros((8,8),np.uint8)
    if kind == 'point': mask[2,2]=1
    if kind == 'line': mask[2,1:7]=1
    if kind == 'triangle': mask[2,2]=mask[3,2]=mask[3,3]=1
    if kind == 'sixteen': mask[1:5,1:5]=1
    expected = bool(np.any(mask)) if mode == 'semantic' else publication.mask_has_yolo_polygon(mask)
    originals = [candidate(augmentation_index=index,foreground_deferred=mode) for index in range(7)]
    publication.resolve_deferred_foregrounds(originals,mask)
    mask[:]=0
    publication.resolve_deferred_foregrounds(originals,mask)
    assert [cand.foreground for cand in originals] == [expected]*7
    assert not any(cand.foreground_deferred for cand in originals)
    counts=workers._deferred_foreground_counts(originals)
    assert counts.original_foreground == int(expected)
    assert counts.original_background == int(not expected)
    assert counts.augmented_foreground == 6*int(expected)
    assert counts.augmented_background == 6*int(not expected)
    assert counts.transverse_frames == ({0} if expected else set())
    assert pickle.loads(pickle.dumps(counts)) == counts


def test_semantic_source_truth_intersects_known_output_support():
    mask=np.eye(4,dtype=np.uint8)
    coverage=1-mask
    candidates=[candidate(foreground_deferred='semantic')]
    publication.resolve_deferred_foregrounds(candidates,mask,coverage)
    assert candidates[0].foreground is False


@pytest.mark.parametrize('mode', ['polygon','semantic'])
@pytest.mark.parametrize('kind', ['empty','point','sixteen'])
def test_cpu_publication_matches_preclassification_drop_rules_and_paths(tmp_path, mode, kind):
    mask=np.zeros((8,8),np.uint8)
    if kind=='point': mask[2,2]=1
    if kind=='sixteen': mask[1:5,1:5]=1
    expected=bool(np.any(mask)) if mode=='semantic' else publication.mask_has_yolo_polygon(mask)
    base=candidate(augmentation_index=1,augmentation_seed=42,augmentation_tag='AbCd1234EfGh5678')
    outcomes=[]
    for name,deferred in [('old',False),('deferred',True)]:
        cand=replace(base,foreground=expected if not deferred else True,foreground_deferred=mode if deferred else '')
        with mock.patch.object(publication,'apply_augmentation_pair',return_value=(np.zeros((8,8),np.uint8),np.zeros_like(mask))):
            outcome=publication.write_selected_candidate_version(cand=cand,image=np.zeros((8,8),np.uint8),
                mask=mask,out_dir=tmp_path/name,split_active=False,image_format='png',png_compression=1,
                jpeg_quality=95,warnings=workers.WarningLog(),augmentation=object(),save_images=False,
                save_labels=mode=='polygon',save_binary=mode=='semantic')
        outcomes.append((outcome, sorted(str(path.relative_to(tmp_path/name)) for path in (tmp_path/name).rglob('*') if path.is_file())))
    assert outcomes[0] == outcomes[1]
    assert outcomes[0][0] == ('flip_dropped' if expected else 'written')


def test_integer_ratio_keeps_names_seeds_order_and_copy_counts_without_foreground_plan(tmp_path):
    policy=LoadedAugmentation(tmp_path/'policy.py','abc','build','test',lambda:None)
    source=[candidate(order=index,source_order=index,frame_idx=index,foreground=bool(index%2)) for index in range(5)]
    known,_=plan_augmented_versions(source,augmentation=policy,augmentation_ratio=7,split_active=False,
        augmented_foregrounds={},require_augmented_foregrounds=False)
    deferred,_=plan_augmented_versions([replace(cand,foreground=True,foreground_deferred='polygon') for cand in source],
        augmentation=policy,augmentation_ratio=7,split_active=False,augmented_foregrounds={},require_augmented_foregrounds=False)
    def identity(cand):
        return (cand.order,cand.source_order,cand.frame_idx,cand.item_key,cand.output_tag,
                cand.augmentation_index,cand.augmentation_seed,cand.augmentation_tag,cand.keep)
    assert [identity(cand) for cand in deferred] == [identity(cand) for cand in known]
    assert len(deferred)==35


@pytest.mark.parametrize('kind', ['empty','point','sixteen'])
def test_cpu_worker_resolves_all_copies_before_the_first_source_mutation(tmp_path, kind):
    from tests.test_pta_fraction_rendering import make_plan
    plan=make_plan(tmp_path,'gray')
    volume=np.zeros((5,6,7),np.uint8)
    mask=np.zeros_like(volume)
    if kind=='point': mask[2,2,2]=1
    if kind=='sixteen': mask[2,1:5,1:5]=1
    source=workers.render_plan_frame_source(volume=volume,mask=mask,plan=plan,idx=2,need_canvas=False)
    expected=publication.mask_has_yolo_polygon(source.mask_full)
    candidates=tuple(candidate(order=index,augmentation_index=index,frame_idx=2,
        foreground_deferred='polygon') for index in range(7))
    task=workers.FrameRenderTask(0,2,(('full',candidates),))
    seen=[]
    def publish(**kwargs):
        seen.append([cand.foreground for cand in candidates])
        kwargs['mask'][:]=0
        return 'flip_dropped' if expected and kwargs['cand'].augmentation_index else 'written'
    with mock.patch.dict(workers._WORKER_STATIC,dict(augmentation=None,out_dir=tmp_path,
            split_active=False,image_format='png',png_compression=1,jpeg_quality=95),clear=True), \
         mock.patch.object(workers,'write_selected_candidate_version',side_effect=publish):
        written,flips,_warnings,_examples,measured=workers.execute_render_task(volume,mask,[plan],task)
    assert seen==[[expected]*7]*7
    assert written==(1 if expected else 7)
    assert flips==({'all':6} if expected else {})
    assert measured.original_foreground==int(expected)
    assert measured.augmented_background==6*int(not expected)


def test_measured_parent_counts_replace_placeholders_and_deduplicate_c1_frames():
    prep=SimpleNamespace(src=SimpleNamespace(stem='sample'),plans=[SimpleNamespace(view=SimpleNamespace(family='transverse'))],
        foreground_preservation_stats={'input_foreground_transverse_slices':2})
    measured=workers.DeferredForegroundCounts(2,1,12,6,{0,2})
    backgrounds=pta.BackgroundFilterStats(foreground_before=21)
    augmentations=pta.AugmentationStats(planned_augmented_copies=18,planned_augmented_foreground=18)
    pta._finalize_deferred_classification(prep,3,backgrounds,augmentations,measured,fraction=1.)
    assert (backgrounds.foreground_before,backgrounds.background_before,backgrounds.background_retained)==(14,7,7)
    assert (augmentations.planned_augmented_foreground,augmentations.planned_augmented_background)==(12,6)
    assert prep.foreground_preservation_stats['classified_output_foreground_transverse_slices']==2
    assert prep.foreground_preservation_stats['retained_output_foreground_transverse_slices']==2
    prep.foreground_preservation_stats['input_foreground_transverse_slices']=3
    with pytest.raises(RuntimeError,match='Foreground transverse invariant'):
        pta._finalize_deferred_classification(prep,3,backgrounds,augmentations,measured,fraction=1.)
    pta._finalize_deferred_classification(prep,3,backgrounds,augmentations,measured,fraction=.2)
    with pytest.raises(RuntimeError,match='every original'):
        pta._finalize_deferred_classification(prep,4,backgrounds,augmentations,measured,fraction=.2)
    augmentations.planned_augmented_copies=19
    with pytest.raises(RuntimeError,match='every augmented'):
        pta._finalize_deferred_classification(prep,3,backgrounds,augmentations,measured,fraction=.2)


def test_count_derived_source_support_reuses_scalar_truth_without_conflating_polygon_class():
    mask=np.zeros((8,8),np.uint8)
    mask[2,2]=1
    coverage=np.zeros_like(mask)
    cand=candidate(foreground_deferred='polygon')
    item=SimpleNamespace(candidates=(cand,),mask=mask,semantic_coverage=None,
        output_size=mask.shape,ready_event=None,categorical_ready_event=None)
    runtime={'torch':None}
    support=workers._resolve_gpu_source_foregrounds(runtime,[item])
    assert cand.foreground is False
    mask[:]=0
    with mock.patch.object(workers.np,'any',side_effect=AssertionError('duplicate support reduction')):
        assert workers._gpu_source_mask_nonempty(runtime,[item],known_support=support)==(True,)
    item.semantic_coverage=coverage
    assert workers._gpu_source_mask_nonempty(runtime,[item],known_support=support)==(False,)
    assert workers._gpu_source_mask_nonempty(runtime,[item],known_support=support,include_coverage=False)==(True,)


@pytest.mark.parametrize('failure', ['prepare','planning'])
def test_failed_planning_retains_source_and_prepared_blocks_for_outer_cleanup(tmp_path,failure):
    source_block=mock.Mock()
    mask_block=mock.Mock()
    source=pta.SourceVolume(input_dir=tmp_path,stem='sample',kind='sequence',image_paths=[],video_path=None,
        labels_by_frame={},segmentation_nrrd_path=None,mask_volume=None,volume_class='fully_labeled',
        label_source='nrrd',input_start_index=None,encoded_indices=(),volume=np.zeros((2,3,4),np.uint8),
        fps=1.,volume_block=source_block)
    prepared=object.__new__(pta.PreparedVolume)
    prepared.src=source
    prepared.shm_blocks=[source_block,mask_block,source_block]
    prepared.volume_for_render=source.volume
    prepared.mask_for_render=np.zeros((2,3,4),np.uint8)
    prepared.semantic_coverage_for_render=None
    prepare=mock.Mock(side_effect=StopPlanning if failure=='prepare' else None,return_value=prepared)
    states={}
    namespace=dict(pending_loads={0:(SimpleNamespace(result=lambda:source),workers.WarningLog())},
        runtime_telemetry=lambda:SimpleNamespace(span=lambda *args:nullcontext()),
        _submit_load=lambda *args:None,pending_specs=[object()],prepare_loaded_source=prepare,
        args=SimpleNamespace(),out_dir=tmp_path,tile_configs=[],channel_variants=[],volume_allocator=None,
        topology=SimpleNamespace(cuda_device_ids=()),effective_pipeline_depth=1,
        enumerate_candidates_for_volume=mock.Mock(side_effect=StopPlanning),dataset_publication_selected=True,
        fraction=1.,states=states)
    with pytest.raises(StopPlanning):
        parent_planner(namespace)(0,1)
    state=states[0]
    assert state['source'] is source
    assert state['prep'] is (None if failure=='prepare' else prepared)
    tree=ast.parse(Path(pta.__file__).read_text())
    selected=[node for node in ast.walk(tree) if isinstance(node,ast.FunctionDef)
        and node.name in {'_release_source_volume','_release_render_state'}]
    cleanup_namespace=dict(np=np,SourceVolume=pta.SourceVolume,PreparedVolume=pta.PreparedVolume,
        VolumeRenderProgress=workers.VolumeRenderProgress,RenderPhaseHandle=workers.RenderPhaseHandle,progress_by_gen={})
    module=ast.Module(body=[ast.ImportFrom(module='__future__',names=[ast.alias(name='annotations')],level=0),*selected],type_ignores=[])
    exec(compile(ast.fix_missing_locations(module),'<actual render state cleanup>','exec'),cleanup_namespace)
    cleanup_namespace['_release_render_state'](state)
    source_block.release.assert_called_once()
    assert mask_block.release.call_count==int(failure=='planning')
    assert source.volume.size==0 and source.volume_block is None
    assert state['source'] is state['prep'] is None
    cleanup_namespace['_release_render_state'](state)
    source_block.release.assert_called_once()


@pytest.mark.skipif(__import__('os').environ.get('XTA_TEST_PTA_GPU_PUBLICATION')!='1',reason='opt-in CUDA qualification')
@pytest.mark.parametrize('mode', ['polygon','semantic'])
@pytest.mark.parametrize('kind', ['empty','point','sixteen'])
def test_cuda_source_truth_and_real_publication_match_preclassified_outputs(tmp_path, mode, kind):
    import torch
    mask=np.zeros((8,8),np.uint8)
    if kind=='point': mask[2,2]=1
    if kind=='sixteen': mask[1:5,1:5]=1
    expected=bool(np.any(mask)) if mode=='semantic' else publication.mask_has_yolo_polygon(mask)
    results=[]
    for name,deferred in [('old',False),('deferred',True)]:
        cand=candidate(augmentation_index=1,augmentation_seed=42,augmentation_tag='AbCd1234EfGh5678',
            foreground=expected if not deferred else True,foreground_deferred=mode if deferred else '')
        source=torch.as_tensor(mask,device='cuda:0')
        item=SimpleNamespace(candidates=(cand,),mask=source,image=torch.zeros((8,8),dtype=torch.uint8,device='cuda:0'),
            semantic_coverage=None,output_size=(8,8),ready_event=None,categorical_ready_event=None)
        runtime={'torch':torch,'device_id':0}
        workers._resolve_gpu_source_foregrounds(runtime,[item])
        assert cand.foreground==expected
        source_nonempty=bool(np.any(mask))
        source.zero_()
        destination=tmp_path/name
        with mock.patch.dict(workers._WORKER_STATIC,dict(out_dir=destination,split_active=False,
                image_format='png',save_images=False,save_labels=mode=='polygon',
                save_binary=mode=='semantic',save_semantic=False),clear=True):
            result=workers._publish_gpu_policy_batch(runtime=runtime,
                batch_images=torch.zeros((1,1,8,8),dtype=torch.uint8,device='cuda:0'),
                batch_masks=source[None],candidates=[cand],output_size=(8,8),channel_kind='gray',
                channel_count=1,local_warnings=workers.WarningLog(),source_mask_nonempty=(source_nonempty,))
        paths=sorted(str(path.relative_to(destination)) for path in destination.rglob('*') if path.is_file())
        results.append((result,paths))
    assert results[0]==results[1]
    assert results[0][0][0]==int(not expected)

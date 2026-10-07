"""The production submission seam transfers ownership before drain waiting."""
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pipeline
from XTA.geometry import ViewInfo
from XTA.interpolation import _DirectUnionBackingLease
from XTA.view_prepare import ViewPrepareLeaseState
from tests.test_terminal_component_refs import _function


def submission(tmp_path, *, ready=False, staged=True, dispatch_error=False):
    view=ViewInfo(name='coronal__tta_a0',physical_view_name='coronal',family='orthogonal',
        num_slices=3,src_h=4,src_w=5,pad_mode='clamp')
    key=('model',view.name)
    mask=np.zeros((3,4,5),np.uint8)
    confidence=np.ones((3,4,5),np.float32)
    leases=ViewPrepareLeaseState({key:_DirectUnionBackingLease(key,480)},
        {key},{key:480},set(),{})
    stager=mock.Mock() if staged else None
    captured=[]
    if stager is not None:
        stager.defer.side_effect=lambda task,required: captured.append((task,required))
    executor=mock.Mock()
    future=Future()
    executor.submit.return_value=future
    dispatch=mock.Mock(side_effect=RuntimeError('injected dispatch failure') if dispatch_error else None)
    namespace=dict(vars(pipeline))
    namespace.update(view_processing_submitted=set(),d1_view_shadow_path_by_parent={},
        baseline_union_by_model_view={key:mask},baseline_confmap_by_model_view={key:confidence},
        baseline_union_paths={key:tmp_path/'mask.dat'},baseline_confmap_paths={key:tmp_path/'confidence.dat'},
        baseline_slice_locks_by_model_view={},view_device_hole_filled_slices={},view_slice_meta={},
        args=SimpleNamespace(imgsz=8,min_conf=.5,min_radius=0,interpolation_distance=5,
            interpolation_walk_back=0,interpolation_candidates=1,interpolation_passes=1,
            interpolation_min_radius=0,interpolation_search_angle=90),
        input_T=3,input_H=4,input_W=5,temp_dir=tmp_path,dense_tiling_active=True,
        parent_transient_admission=mock.Mock(),interpolation_settings=SimpleNamespace(backend='sam',
            extrapolation_enabled=False, extrapolation_distance=0,
            extrapolation_walk_back=1, extrapolation_min_radius=3.),
        sam_context=SimpleNamespace(detector_retirement_ready=ready,shared_detector_devices=('cuda:0',)),sam_parent_staging=stager,
        keep_temp_artifacts=False,parent_slice_postprocess_workers=1,parent_interpolation_task_workers=1,
        component_layers_needed=True,angle_variant_streaming_cleanup_active=False,
        angle_variant_gpu_fastpath_active=False,component_ref_dense_retirement_active=True,
        _publish_parent_mask_ready=mock.Mock(),_submit_component_projection=mock.Mock(),
        _submit_sam_layer_projection=mock.Mock(),
        sam_cpu_prepare_executor=None,
        sam_cpu_futures={'source':None,'runtime':None,'runtime_needed':False},views=(view,),
        _sam_parents_ready=lambda:ready,
        sam_cpu_max_detector_parent=120,
        direct_union_inference_bytes=leases.inference_bytes,direct_union_postprocess_bytes=leases.postprocess_bytes,
        direct_union_total_dense_byte_limit=10000,policy_settings=SimpleNamespace(enabled=False),
        _publish_parent_confidence_retired=mock.Mock(),direct_union_backing_leases=leases.leases,
        view_prepare_leases=leases,parent_postprocess_executor=executor,view_processing_futures={},
        gpu_worker_pending_task_ids=[123],_dispatch_inference_windows=dispatch,
        view_processing_volume_shape=lambda *_args:(3,4,5),_view_uses_interpolation=lambda *_args:True)
    source=Path(pipeline.__file__).read_text(encoding='utf-8')
    _function(source,'_sam_parent_requires_staging',namespace)
    function=_function(source,
        '_submit_view_prepare',namespace)
    return function,namespace,view,key,leases,captured,executor,future,dispatch


def test_shared_sam_submission_defers_before_transient_prepare_and_refills(tmp_path):
    function,ns,view,key,leases,captured,executor,_future,dispatch=submission(tmp_path)
    function('model',view)
    task,required=captured[0]
    assert required==480  # Exact float32 confidence plus future tile/category canvases.
    assert task.union_mm.shape==(3,4,5) and task.confmap_mm.dtype==np.float32
    assert leases.leases[key].phase=='postprocess' and leases.postprocess_bytes[key]==480
    assert key not in ns['baseline_union_by_model_view']
    assert key not in ns['baseline_confmap_by_model_view']
    assert not ns['view_processing_futures']
    assert ns['view_processing_submitted']=={key}
    ns['parent_transient_admission'].reserve.assert_not_called()
    ns['_publish_parent_mask_ready'].assert_not_called()
    executor.submit.assert_not_called()
    dispatch.assert_called_once()


def test_dispatch_failure_after_staging_cannot_restore_or_roll_back_owned_input(tmp_path):
    function,ns,view,key,leases,captured,executor,_future,_dispatch=submission(tmp_path,dispatch_error=True)
    with pytest.raises(RuntimeError,match='injected dispatch failure'):
        function('model',view)
    assert len(captured)==1 and captured[0][0].union_mm is not None
    assert not ns['baseline_union_by_model_view'] and not ns['baseline_confmap_by_model_view']
    assert leases.leases[key].phase=='postprocess'
    assert key in leases.postprocess_views and key not in leases.inference_views
    executor.submit.assert_not_called()


@pytest.mark.parametrize('ready,staged', [(True,True),(False,False)])
def test_ready_or_unstaged_route_preserves_ordinary_prepare(tmp_path,ready,staged):
    function,ns,view,key,_leases,captured,executor,future,dispatch=submission(tmp_path,ready=ready,staged=staged)
    function('model',view)
    assert not captured
    executor.submit.assert_called_once()
    assert ns['view_processing_futures']=={future:key}
    dispatch.assert_called_once()


def test_failed_run_cancels_checkpoint_writer_before_executor_shutdown(tmp_path):
    from XTA.sam_parent_staging import DeferredSamParentQueue
    from XTA.tta_lifecycle import PipelineRunResources
    leases=ViewPrepareLeaseState({},set(),{},set(),{})
    writer=mock.Mock()
    stage=DeferredSamParentQueue(temp_dir=tmp_path/'temp',output_dir=tmp_path/'output',
        checkpoint_executor=writer,prepare_executor=mock.Mock(),leases=leases,
        dense_limit=100,ready=lambda:False)
    writer.shutdown.side_effect=lambda **_kwargs: stage.stop.is_set() or pytest.fail('writer shutdown preceded cancellation')
    resources=PipelineRunResources()
    resources.track_executor(writer)
    resources.track_closeable(stage)
    resources.close(failed=True)
    assert stage.stop.is_set()
    writer.shutdown.assert_called_once_with(wait=True,cancel_futures=True)

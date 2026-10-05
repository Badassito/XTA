"""Slab batching preserves the exact depth resize and canonical affine phase."""
from pathlib import Path
import threading
from unittest import mock

import numpy as np
import pytest

from XTA import geometry
from XTA._deps import cv2
from XTA.media import LazyProcessingCube,resize_volume_to_processing_cube_gray8
from XTA.sam_transverse_cache_rendering import iter_transverse_crop_batches
from tests.test_sam_view_image_cache import context_for,demand,crop_from


@pytest.mark.parametrize('source_shape,out_t', [((3,13,17),8),((17,23,29),7),((7,19,33),12)])
@pytest.mark.parametrize('budget', [8192,32768,256*1024])
def test_variable_crop_widths_chunk_boundaries_and_canonical_phase_match_independent(source_shape,out_t,budget):
    source=np.random.default_rng(17).integers(0,256,size=source_shape,dtype=np.uint8)
    out_shape=(out_t,*source_shape[1:]);affine=np.array([[.83,0.,-.4375],[0.,.83,-.21875]],np.float32)
    inverse=cv2.invertAffineTransform(affine).astype(np.float32)
    requests=[(frame,(1,2,10,13+(frame%2)))for frame in range(out_t)]
    # Full depth resize is only a small independent test oracle, never production.
    resized=cv2.resize(source.reshape(source_shape[0],-1),(source_shape[1]*source_shape[2],out_t),
                       interpolation=cv2.INTER_LINEAR).reshape(out_shape)
    from XTA.sam_canvas_rendering import render_canonical_crop
    view=geometry.get_view_infos(*out_shape,cartesian_views=('transverse',))[0]
    actual=[]
    for batch,stats in iter_transverse_crop_batches(source,out_shape,requests,affine=affine,inverse=inverse,
            canvas_width=23,max_workspace_bytes=budget):
        if stats.get('batches'):
            assert stats['peak_resize_workspace_bytes']<=budget
            assert stats['output_bytes']<=budget//4
            assert stats['native_prepared_pixels']<=budget//4
        for frame,bbox,image in batch:
            if image is None:continue
            expected=render_canonical_crop(resized,view,frame,affine=affine,inverse=inverse,
                output_origin_yx=bbox[:2],output_height=bbox[2]-bbox[0],output_width=bbox[3]-bbox[1],
                output_canvas_width=23)
            np.testing.assert_array_equal(image,expected)
            actual.append(frame)
    assert actual  # The small admitted budgets exercise actual batching, not only fallback.


def test_repeated_frames_resize_each_spatial_slab_once_without_full_cube(tmp_path):
    source=np.random.default_rng(9).integers(0,256,size=(7,17,19),dtype=np.uint8)
    lazy=LazyProcessingCube(source,(13,17,19),tmp_path/'never.dat',workers=1,
        request_path=tmp_path/'request',ready_path=tmp_path/'ready',failed_path=tmp_path/'failed')
    view=geometry.get_view_infos(*lazy.shape,cartesian_views=('transverse',))[0]
    context=context_for(tmp_path,lazy);shape=(13,23,23);frames={frame:(2,3,18,20)for frame in range(1,12)}
    affine,inverse,_=context._canvas_transform(view,shape)
    expected={frame:context._render_demand_crop(view,frame,affine,inverse,output_height=16,
        output_width=17,output_origin_yx=(2,3),output_canvas_width=23)for frame in frames}
    try:
        with mock.patch.object(cv2,'resize',wraps=cv2.resize)as resize:
            reference=context.image_provider(view,shape,demand(shape,frames))
            assert resize.call_count< len(frames)
        assert not lazy.materialized and not (tmp_path/'never.dat').exists()
        for frame,bbox in frames.items():np.testing.assert_array_equal(crop_from(reference,frame,bbox),expected[frame])
    finally:context.close();lazy.close()


def test_far_disjoint_crops_keep_independent_rendering():
    source=np.zeros((3,100,100),np.uint8);identity=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)
    requests=[(0,(1,1,3,3)),(1,(90,90,93,93))]
    result=list(iter_transverse_crop_batches(source,(5,100,100),requests,affine=identity,inverse=identity,
        canvas_width=100,max_workspace_bytes=8192))
    assert all(stats=={'independent_frames':1} and batch[0][2]is None for batch,stats in result)


def test_low_budget_stays_independent_and_cancellation_stops_before_resize():
    source=np.zeros((3,17,19),np.uint8);identity=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)
    requests=[(frame,(1,2,15,17))for frame in range(5)]
    assert all(batch[0][2]is None for batch,_stats in iter_transverse_crop_batches(source,(5,17,19),requests,
        affine=identity,inverse=identity,canvas_width=19,max_workspace_bytes=1024))
    stop=threading.Event();stop.set()
    with mock.patch.object(cv2,'resize',side_effect=AssertionError('resized cancelled demand')):
        with pytest.raises(RuntimeError,match='cancelled'):
            list(iter_transverse_crop_batches(source,(5,17,19),requests,affine=identity,inverse=identity,
                canvas_width=19,max_workspace_bytes=8192,cancel_event=stop))


def test_combined_native_planes_outputs_and_remap_row_can_force_safe_independent_fallback():
    source=np.zeros((3,5,45),np.uint8);identity=np.array([[1.,0.,0.],[0.,1.,0.]],np.float32)
    requests=[(frame,(1,2,2,41))for frame in range(3)]
    batches=list(iter_transverse_crop_batches(source,(7,5,45),requests,affine=identity,inverse=identity,
        canvas_width=45,max_workspace_bytes=3250))
    assert all(batch[0][2]is None for batch,_stats in batches)


def test_already_materialized_context_keeps_original_sampler(tmp_path):
    source=np.arange(3*13*17,dtype=np.uint16).astype(np.uint8).reshape(3,13,17)
    lazy=LazyProcessingCube(source,(7,13,17),tmp_path/'cube.dat',workers=1,
        request_path=tmp_path/'request',ready_path=tmp_path/'ready',failed_path=tmp_path/'failed');lazy.materialize()
    context=context_for(tmp_path,lazy);view=geometry.get_view_infos(*lazy.shape,cartesian_views=('transverse',))[0]
    try:
        with mock.patch.object(context,'_render_demand_crop',wraps=context._render_demand_crop)as render:
            reference=context.image_provider(view,lazy.shape,demand(lazy.shape,{1:(1,2,10,13),3:(1,2,10,13)}))
            assert render.call_count==0 and context.exact_backing_reuses==1
            assert reference.path==lazy.backing_path
    finally:context.close();lazy.close()


def test_batch_budget_is_clamped_to_live_remaining_profile_and_rejects_saved_metadata(tmp_path):
    from XTA import sam_resources
    from XTA.interpolation import _ByteAdmissionPool
    source=np.zeros((3,13,17),np.uint8)
    lazy=LazyProcessingCube(source,(7,13,17),tmp_path/'unused.dat',workers=1,
        request_path=tmp_path/'request',ready_path=tmp_path/'ready',failed_path=tmp_path/'failed')
    context=context_for(tmp_path,lazy);view=geometry.get_view_infos(*lazy.shape,cartesian_views=('transverse',))[0]
    affine,inverse,_=context._canvas_transform(view,lazy.shape)
    records=[(frame,1,2,10,13,position*99)for position,frame in enumerate((1,3,5))]
    addresses={frame:{'native_index':frame,'mirror_u':False}for frame in (1,3,5)}
    try:
        pool=_ByteAdmissionPool(65536,'test')
        with sam_resources.admit_sam_parent_resources(pool,8192,'render',base_allowance_bytes=8192,
                headroom_probe=lambda:65536)as profile:
            context._resource_local.profile=profile
            iterator=context._transverse_batch_iterator(view,lazy.shape,records,addresses,
                affine,inverse,'geometry',297)
            batch,stats=next(iterator)
            assert stats['admitted_workspace_bytes']==8192-297
            assert stats['peak_resize_workspace_bytes']<=8192-297
            iterator.close();del batch
            context._resource_local.profile=profile.metadata()
            with pytest.raises(TypeError,match='live SamResourceProfile'):
                context._transverse_batch_iterator(view,lazy.shape,records,addresses,
                    affine,inverse,'geometry',297)
        context._resource_local.profile=profile
        with pytest.raises(RuntimeError,match='expired'):
            context._transverse_batch_iterator(view,lazy.shape,records,addresses,
                affine,inverse,'geometry',297)
    finally:context.close();lazy.close()

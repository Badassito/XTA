"""Parent/worker full-span protocol and explicit byte refusal; no GPU."""
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from XTA.sam_tracker_runtime import SamInterpolationTracker, materialize_interpolation_image_cache
from tests.test_sam_tracker_runtime import _CompletionPool


@pytest.mark.parametrize('frames,direction',[(31,'forward'),(33,'backward'),(82,'forward'),(82,'backward')])
def test_parent_worker_packet_preserves_long_requested_interval(tmp_path,frames,direction):
    cache=materialize_interpolation_image_cache(np.zeros((frames,15,21),np.uint8),path=tmp_path/'images.dat',
        physical_view_id='transverse',source_identity='controlled-long-session')
    tracker=SamInterpolationTracker(model_path='unused',device_ids=(0,),artifact_root=tmp_path/'staging',source_cache_ref=cache)
    tracker._pool=_CompletionPool()
    seed=np.zeros((9,13),bool)
    seed[2:7,3:9]=True
    request=dict(run_id='long-session',seed_mask=seed,seed_frame=0 if direction=='forward' else frames-1,
                 frame_start=0,frame_stop=frames,direction=direction,crop_xyxy=(4,2,17,11))
    with patch.object(tracker,'start',return_value=tracker):
        results=list(tracker.iter_results((request,),source_cache_ref=cache))
    try:
        assert len(results)==1
        result=results[0][1]
        assert set(result.frames)==set(range(frames))
        assert result.receipt['coverage_complete']
        assert result.frames[0].any() and result.frames[frames-1].any()
        assert result.receipt['session_cpu_admission']['frame_count']==frames
        assert result.receipt['session_cpu_admission']['budget_bytes']==2*1024**3
        assert result.receipt['session_cpu_admission']['cuda_history_bound'].startswith('not_claimed')
    finally:
        for _,result in results:tracker.release_result(result)
        tracker.close()


def test_byte_budget_refuses_before_parent_staging_or_predictor(tmp_path):
    cache=materialize_interpolation_image_cache(np.zeros((31,15,21),np.uint8),path=tmp_path/'images.dat',
        physical_view_id='transverse',source_identity='controlled-byte-refusal')
    tracker=SamInterpolationTracker(model_path='unused',device_ids=(0,),artifact_root=tmp_path/'staging',
        source_cache_ref=cache,session_cpu_budget_bytes=64*1024**2)
    seed=np.ones((9,13),bool)
    request=dict(run_id='refused',seed_mask=seed,seed_frame=0,frame_start=0,frame_stop=31,
                 direction='forward',crop_xyxy=(4,2,17,11))
    with patch.object(tracker,'start',side_effect=AssertionError('No model startup')):
        with pytest.raises(MemoryError,match='not staged or truncated'):
            tracker._prepare_task(request,cache_ref=cache,input_index=0,staging_directories=set())
    assert not list((tmp_path/'staging').glob('run-*'))
    tracker.close()


def test_deferred_refill_waits_until_the_previous_transfer_is_consumed(tmp_path):
    cache=materialize_interpolation_image_cache(np.zeros((3,15,21),np.uint8),path=tmp_path/'images.dat',
        physical_view_id='transverse',source_identity='controlled-refill')
    tracker=SamInterpolationTracker(model_path='unused',device_ids=(0,),artifact_root=tmp_path/'staging',source_cache_ref=cache)
    pool=_CompletionPool()
    tracker._pool=pool
    seed=np.ones((9,13),bool)
    requests=[dict(run_id=str(index),seed_mask=seed,seed_frame=0,frame_start=0,frame_stop=3,
                   direction='forward',crop_xyxy=(4,2,17,11)) for index in range(2)]
    with patch.object(tracker,'start',return_value=tracker):
        stream=tracker.iter_results(requests,source_cache_ref=cache,max_in_flight=1,defer_refill_until_consumed=True)
        _,first=next(stream)
        assert len(pool.submissions)==1
        assert first.receipt['dispatch']['refill_after_consumption'] is True
        tracker.release_result(first)
        del first
        _,second=next(stream)
        assert len(pool.submissions)==2
        tracker.release_result(second)
        del second
        assert list(stream)==[]
    tracker.close()

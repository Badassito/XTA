"""Diagnostic image counters must not serialize unrelated ownership operations."""
import threading

import numpy as np
import pytest

from XTA import geometry
from XTA.sam_integration import SamInterpolationContext
from tests.test_sam_view_image_cache import demand


METRICS = ('render_seconds','rendered_frames','rendered_pixels','exact_backing_reuses',
           'image_cache_reused_pixels','source_materializations','source_materialization_seconds',
           'native_sampling_calls','native_sampling_pixels','canonical_sampling_pixels')


@pytest.fixture
def context(tmp_path):
    value=SamInterpolationContext(model_path='unused',device_ids=(0,),temp_dir=tmp_path/'runtime',
        evidence_root=tmp_path/'evidence',source_volume=np.zeros((7,9,11),np.uint8),
        source_identity='immutable-source')
    try:
        yield value
    finally:
        value.close()


def test_actual_context_allocates_a_distinct_image_metrics_lock(context):
    assert context._image_metrics_lock is not context._lock
    assert context._idle._lock is context._lock


def test_image_metrics_progress_while_real_provider_metadata_owns_registry(context,monkeypatch):
    view=geometry.get_view_infos(7,9,11,cartesian_views=('coronal',))[0]
    shape=(view.num_slices,8,8)
    entered,release,done=threading.Event(),threading.Event(),threading.Event()
    errors=[]
    original=context._canvas_transform
    def paused(*args):
        entered.set()
        if not release.wait(5):raise RuntimeError('metadata release timed out')
        return original(*args)
    monkeypatch.setattr(context,'_canvas_transform',paused)
    def metadata():
        try:context._image_provider(view,shape,demand(shape,{2:(1,1,5,5)}),_cache_only=True)
        except BaseException as error:errors.append(error)
    def metrics():
        try:
            context._add_image_metrics(rendered_frames=1,rendered_pixels=16)
            done.set()
        except BaseException as error:errors.append(error)
    owner=threading.Thread(target=metadata)
    writer=threading.Thread(target=metrics)
    try:
        owner.start();assert entered.wait(3)
        held=context._lock.acquire(blocking=False)
        if held:context._lock.release()
        assert not held,'real metadata transaction must retain registry ownership'
        writer.start()
        assert done.wait(1),'diagnostic image progress waited behind unrelated metadata ownership'
        assert context.rendered_frames==1 and context.rendered_pixels==16
    finally:
        release.set();owner.join(3)
        if writer.ident is not None:writer.join(3)
    assert not errors and not owner.is_alive() and not writer.is_alive()


def test_concurrent_diagnostic_updates_lose_no_increments(context):
    barrier=threading.Barrier(9)
    errors=[]
    def write():
        try:
            barrier.wait(3)
            for _ in range(200):context._add_image_metrics(**dict.fromkeys(METRICS,1))
        except BaseException as error:errors.append(error)
    writers=[threading.Thread(target=write) for _ in range(8)]
    for writer in writers:writer.start()
    barrier.wait(3)
    for writer in writers:writer.join(3)
    assert not errors and not any(writer.is_alive() for writer in writers)
    assert {name:getattr(context,name) for name in METRICS}==dict.fromkeys(METRICS,1600)


@pytest.mark.parametrize('field',('cache_logical_bytes','_image_build_credit_bytes','_active_image_calls'))
def test_image_metric_allowlist_does_not_authorize_owner_updates(context,field):
    before=getattr(context,field)
    try:
        with pytest.raises(ValueError,match='ownership or admission'):
            context._add_image_metrics(rendered_frames=1,**{field:1})
        assert getattr(context,field)==before and context.rendered_frames==0
    finally:
        # Keep the before-fix regression runnable: it deliberately demonstrates
        # that an unchecked method could mutate an ownership sentinel.
        setattr(context,field,before)
        context.rendered_frames=0

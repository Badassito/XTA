"""Lazy source warmup stays independent of the immutable checkpoint drain."""
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import media, pipeline, sam_resources
from XTA.config import GIB
from XTA.interpolation import _ByteAdmissionPool
from tests.test_sam_parent_staging_pipeline import submission
from tests.test_terminal_component_refs import _function


@pytest.mark.parametrize('failure', [False, True])
def test_runtime_warmup_gates_parent_resume_and_preserves_startup_error(failure):
    state = {'runtime': None, 'runtime_needed': False}
    future = Future()
    executor = mock.Mock()
    executor.submit.return_value = future
    context = SimpleNamespace(detector_retirement_ready=False, prepare_runtime=mock.Mock())
    pool = _ByteAdmissionPool(GIB, 'test')
    ns = dict(vars(pipeline), sam_cpu_futures=state, sam_cpu_prepare_executor=executor,
              sam_context=context, parent_transient_admission=pool)
    source = Path(pipeline.__file__).read_text()
    ready = _function(source, '_sam_parents_ready', ns)
    warm = _function(source, '_maybe_prepare_sam_runtime', ns)
    warm()
    assert not ready()
    context.detector_retirement_ready = True
    warm()
    assert ready()  # Known-empty work does not load models.
    executor.submit.assert_not_called()
    state['runtime_needed'] = True
    assert not ready()
    warm()
    warm()
    executor.submit.assert_called_once_with(context.prepare_runtime, pool)
    assert not ready() and pool.in_use == 0
    if failure:
        error = RuntimeError('predictor startup failed')
        future.set_exception(error)
        with pytest.raises(RuntimeError) as caught:
            ready()
        assert caught.value is error
        with pytest.raises(RuntimeError):
            warm()
    else:
        future.set_result(None)
        assert ready()
        warm()
    executor.submit.assert_called_once()


@pytest.mark.parametrize('metadata', [None, [False, False, False], [False, True, False]])
def test_completed_parent_foreground_gates_runtime_without_scanning_pixels(tmp_path, metadata):
    submit, ns, view, key, _leases, captured, executor, _future, _dispatch = submission(
        tmp_path, ready=True)
    ns['dense_tiling_active'] = False
    if metadata is not None:
        ns['view_slice_meta'][key] = dict(valid=True, slice_any=np.array(metadata))
    _function(Path(pipeline.__file__).read_text(), '_sam_parents_ready', ns)
    submit('model', view)
    needed = metadata is None or any(metadata)
    assert ns['sam_cpu_futures']['runtime_needed'] is needed
    assert bool(captured) is needed
    assert executor.submit.called is not needed


@pytest.mark.parametrize('failed', [False, True])
def test_parent_checkpoint_drains_before_optional_mutating_cpu_prepare(tmp_path, failed):
    fn, ns, view, key, leases, captured, _exec, _fut, _dispatch = submission(tmp_path)
    ns['dense_tiling_active'] = False
    ns['sam_parent_staging'].owns_ram_first_parent.return_value = True
    ns['sam_cpu_prepare_executor'] = mock.Mock()
    original = ns['baseline_union_by_model_view'][key]
    if failed:
        ns['sam_parent_staging'].defer.side_effect = RuntimeError('controlled defer rejection')
        with pytest.raises(RuntimeError, match='controlled defer rejection'):
            fn('model', view)
        assert leases.leases[key].phase == 'inference'
        assert ns['baseline_union_by_model_view'][key] is original
        ns['_dispatch_inference_windows'].assert_not_called()
    else:
        fn('model', view)
        assert captured and captured[0][0].union_mm is original
        assert not hasattr(captured[0][0], 'cpu_prepare_future')
        assert leases.leases[key].phase == 'postprocess'
    ns['sam_cpu_prepare_executor'].submit.assert_not_called()
    assert not original.any()


@pytest.mark.parametrize('low_headroom,unready',[ (False,False),(True,False),(False,True)])
def test_existing_lazy_cube_warms_without_parent_or_gpu_owner(tmp_path,monkeypatch,low_headroom,unready):
    decoded=np.arange(3*7*9,dtype=np.uint8).reshape(3,7,9)
    source=media.LazyProcessingCube(decoded,(5,7,9),tmp_path/'cube.dat',workers=1,
        request_path=tmp_path/'request',ready_path=tmp_path/'ready',failed_path=tmp_path/'failed')
    if unready:
        readiness=media.VolumeReadiness(3)
        media.register_volume_readiness(decoded,readiness)
    state={'source':None,'parent':None}
    context=SimpleNamespace(source_volume=source,detector_retirement_ready=False,_add_image_metrics=mock.Mock())
    monkeypatch.setattr(sam_resources,'physical_sam_headroom',lambda:0 if low_headroom else 128*GIB)
    with ThreadPoolExecutor(max_workers=1) as executor:
        ns=dict(vars(pipeline))
        ns.update(sam_cpu_futures=state,sam_cpu_prepare_executor=executor,sam_context=context,
            views=(SimpleNamespace(family='tilted'),),direct_union_inference_bytes={},
            direct_union_postprocess_bytes={},direct_union_total_dense_byte_limit=GIB,
            parent_transient_admission=_ByteAdmissionPool(GIB,'test'))
        warm=_function(Path(pipeline.__file__).read_text(),'_maybe_prepare_sam_cpu_source',ns)
        warm()
        if low_headroom or unready:
            assert state['source'] is None and not source.materialized
            context._add_image_metrics.assert_not_called()
        else:
            assert state['source'].result(timeout=5) is None  # Future owns no dense result alias.
            assert source.materialized
            warm()
            context._add_image_metrics.assert_called_once()
            actual=np.array(source,copy=True)
            expected=media.resize_volume_to_processing_cube_gray8(decoded,(5,7,9),tmp_path/'reference.dat',
                workers=1,prefer_memory=False)
            np.testing.assert_array_equal(actual,expected)
            from XTA.runtime import close_memmap_array_without_flush
            close_memmap_array_without_flush(expected)
    source.close()


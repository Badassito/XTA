"""Native CPU fallback publication and typed retired-CUDA admission checks."""
from pathlib import Path
from unittest import mock
import numpy as np
import pytest

from XTA import backprojection as bp, geometry as g, runtime
from XTA.interpolation import (
    CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT,
    IncrementalRawBBoxMaskStoreWriter, RawBBoxMaskStore,
)


@pytest.fixture(autouse=True)
def cpu_only(monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT', '0')
    monkeypatch.setenv('YOLO_TTA_GPU_BACKPROJECT_RESIDENT', '0')


def view_and_source():
    tilted = g._build_tilted_view_infos(5,7,9, tilt_views=('transverse',),
        tilt_angles=(30.,), tilt_directions=('horizontal',))[0]
    view = g._build_azimuthal_view_info(5,7,9, base_view='transverse',
        azimuth_angle=30., azimuthal_native_raster=0, request_token='test', tilted_source=tilted)
    rng = np.random.default_rng(109)
    source = (rng.random((view.num_slices,view.src_h,view.src_w)) < .2).astype(np.uint8)
    return view, source


@pytest.mark.parametrize('encoding', [CVOL_FORMAT, INTERNAL_PACKED_CVOL_FORMAT])
def test_correct_cpu_fallback_publishes_raw_and_packed_stores_without_gpu_lease(tmp_path, encoding):
    view, source = view_and_source()
    expected = bp.backproject_azimuthal_volume_to_volume(source,view,tmp_path/'dense.dat','dense',
        out_shape_tyx=(8,11,14),reserve_bytes=0)
    original = source.copy()
    store_path = tmp_path/'store'
    writer = IncrementalRawBBoxMaskStoreWriter(store_dir=store_path,shape=(8,11,14),format_name=encoding,desc='native coverage')
    try:
        def publish(z, block):
            writer.consume(z,block)
        with mock.patch.object(bp,'_try_acquire_main_process_gpu_stage',side_effect=AssertionError('GPU admission')):
            result = bp.backproject_azimuthal_volume_to_volume(source,view,tmp_path/'sink.dat','sink',
                out_shape_tyx=(8,11,14),reserve_bytes=0,sink_only=True,projection_block_callback=publish)
        writer.finalize()
    except BaseException:
        writer.discard()
        raise
    store = RawBBoxMaskStore.open(store_path,mmap_payload=True)
    try:
        actual = np.stack([store.decode_slice(z) for z in range(8)])
    finally:
        store.close()
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(source,original)
    assert result.shape == (8,11,14)
    assert not (tmp_path/'sink.dat').exists()
    runtime.close_memmap_array(expected)


def test_sink_failure_preserves_original_exception_without_cuda_replay(tmp_path):
    view,source = view_and_source()
    original=source.copy()
    failure=RuntimeError('downstream publication failed')
    def publish(z,block):
        raise failure
    with mock.patch.object(bp,'_try_acquire_main_process_gpu_stage',side_effect=AssertionError('GPU admission')):
        with pytest.raises(RuntimeError) as caught:
            bp.backproject_azimuthal_volume_to_volume(source,view,tmp_path/'sink.dat','sink',
                out_shape_tyx=(8,11,14),reserve_bytes=0,sink_only=True,projection_block_callback=publish)
    assert caught.value is failure
    np.testing.assert_array_equal(source,original)
    assert not (tmp_path/'sink.dat').exists()


def test_direct_retired_sink_entry_uses_actual_source_not_rounded_scatter_callback(tmp_path):
    view,source=view_and_source()
    actual=np.zeros((8,11,14),np.uint8)
    def publish(z,block): actual[z:z+len(block)]=block
    result=bp._project_tilted_azimuthal_sink(source,view,actual.shape,
        lambda frame: (_ for _ in ()).throw(AssertionError('old scatter callback')),
        5,tmp_path/'sink.dat','sink',workers=1,prefer_memory=True,reserve_bytes=0,
        known_row_occupancy=None,known_slice_bboxes=None,callback=publish)
    expected=bp.backproject_azimuthal_volume_to_volume(source,view,tmp_path/'dense.dat','dense',
        out_shape_tyx=actual.shape,reserve_bytes=0)
    np.testing.assert_array_equal(actual,expected)
    assert result.shape==actual.shape
    runtime.close_memmap_array(expected)

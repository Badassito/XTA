"""Planned compact crops preserve exact production intensity bytes."""
from types import SimpleNamespace

import numpy as np
import pytest

from XTA.geometry import ViewInfo
from XTA.lta_rendering import LtaPhysicalViewCacheRef, render_native_tile_window
from XTA.media import LazyProcessingCube, resize_volume_to_processing_cube_gray8
from XTA.runtime import close_memmap_array
from XTA.sam_integration import SamInterpolationContext


def view(shape):
    return ViewInfo(name='transverse__tta_a0', physical_view_name='transverse',
        family='orthogonal', num_slices=shape[0], src_h=shape[1], src_w=shape[2],
        pad_mode='clamp', tta_angle_deg=0.)


def owner(root, source):
    return SamInterpolationContext(model_path='unused', device_ids=('cuda:0',),
        temp_dir=root, evidence_root=root/'evidence', source_volume=source,
        source_identity='source-pixels')


@pytest.mark.parametrize('native_shape,canvas', [((4, 9, 13), (4, 9, 13)),
    ((4, 9, 13), (4, 6, 6)), ((4, 11, 17), (4, 8, 8))])
def test_compact_demand_pixels_match_full_canonical_cache(tmp_path, native_shape, canvas):
    source = np.random.default_rng(3).integers(0, 256, native_shape, dtype=np.uint8)
    context = owner(tmp_path, source)
    geometry = view(native_shape)
    full_ref = context.image_provider(geometry, canvas)
    full = np.array(full_ref.open(), copy=True)
    bbox = (1, 2, canvas[1]-1, canvas[2]-1)
    compact = context.image_provider(geometry, canvas,
        SimpleNamespace(frame_crop_bounds={1: bbox, 3: bbox}))
    assert compact.size_bytes == 2*(bbox[2]-bbox[0])*(bbox[3]-bbox[1])
    assert compact.shape == canvas
    assert compact.frame_crops
    restored = LtaPhysicalViewCacheRef.from_payload(compact.payload())
    for frame in (1, 3):
        image = render_native_tile_window(restored, frame_start=frame, frame_stop=frame+1,
            tile_xyxy=(bbox[1], bbox[0], bbox[3], bbox[2]))[0]
        np.testing.assert_array_equal(np.array(image)[:, :, 0],
            full[frame, bbox[0]:bbox[2], bbox[1]:bbox[3]])
    with pytest.raises(ValueError, match='tracking frame'):
        render_native_tile_window(restored, frame_start=2, frame_stop=3,
            tile_xyxy=(bbox[1], bbox[0], bbox[3], bbox[2]))
    context.close()


@pytest.mark.parametrize('input_shape,output_shape,streaming', [
    ((4, 8, 8), (6, 8, 8), False),
    ((4, 7, 9), (6, 8, 8), False),
    ((4, 7, 9), (6, 8, 8), True),
])
def test_lazy_cube_planned_pixels_match_production_without_materializing_cube(
        tmp_path, input_shape, output_shape, streaming):
    source = np.random.default_rng(8).integers(0, 256, input_shape, dtype=np.uint8)
    expected_map = resize_volume_to_processing_cube_gray8(source, output_shape,
        tmp_path/'expected.dat', prefer_memory=False, workers=1)
    expected = np.array(expected_map, copy=True)
    close_memmap_array(expected_map)
    del expected_map
    proxy = LazyProcessingCube(source, output_shape, tmp_path/'unused_cube.dat', workers=1,
        request_path=tmp_path/'request', ready_path=tmp_path/'ready',
        failed_path=tmp_path/'failed', streaming_backend=streaming)
    context = owner(tmp_path, proxy)
    bbox = (1, 2, 7, 6)
    reference = context.image_provider(view(output_shape), output_shape,
        SimpleNamespace(frame_crop_bounds={0: bbox, 2: bbox, 5: bbox}))
    assert not proxy.materialized
    assert not proxy.backing_path.exists()
    for frame in (0, 2, 5):
        image = render_native_tile_window(reference, frame_start=frame, frame_stop=frame+1,
            tile_xyxy=(bbox[1], bbox[0], bbox[3], bbox[2]))[0]
        # Streaming resize uses the endpoint-aligned general XY path even for
        # T-only changes. These fixtures change XY when streaming is selected.
        np.testing.assert_array_equal(np.array(image)[:, :, 0],
            expected[frame, bbox[0]:bbox[2], bbox[1]:bbox[3]])
    context.close()
    proxy.close()


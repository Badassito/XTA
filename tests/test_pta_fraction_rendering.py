"""A retained tile must not render rejected full images or unrelated tiles."""

from dataclasses import replace
from unittest import mock

import numpy as np
import pytest

from XTA import pta, pta_rendering, pta_workers
from XTA.pta_config import parse_pta_args


def make_plan(tmp_path, channel):
    config = parse_pta_args([
        '--input', 'dataset', '--imgsz', '8', '--enable_cartesian', 'transverse',
        '--channel_format', channel,
    ])
    views, _ = pta.compile_v18_pta_views(
        t_dim=5, h=6, w=7, config=config, azimuthal_native_raster=8,
    )
    view = views[0]
    aff = pta.build_affine(7, 6, 0., view.pad_mode, 8, shared_view=view.shared_view)
    variant = pta.expand_channel_variants([config.channel_format])[0]
    return pta.build_render_plan(
        view=view, aff=aff, tag=view.name, out_dir=tmp_path, stem='sample',
        tile_configs=(pta.TileConfig(4, 3, 's4_st3'),), save_overlay=False,
        imgsz=8, label_enabled=True, publish_images=False, publish_labels=False,
        channel_variant=variant,
    )


@pytest.mark.parametrize('channel', ('gray', 'rgb', 'C3S1'))
def test_canonical_tile_only_source_skips_full_and_preserves_pixels(tmp_path, channel):
    plan = make_plan(tmp_path, channel)
    volume = np.arange(5*6*7, dtype=np.uint8).reshape(5, 6, 7)
    mask = (volume % 7 == 0).astype(np.uint8)
    full = pta_rendering.render_plan_frame_source(volume=volume, mask=mask, plan=plan, idx=2)
    selected = replace(plan, tile_layout=(plan.tile_layout[1],))
    with (
        mock.patch.object(pta_rendering, 'render_channel_formatted_images',
                          side_effect=AssertionError('rejected full intensity rendered')),
        mock.patch.object(pta_rendering, 'render_plan_frame_mask_source',
                          side_effect=AssertionError('rejected full mask rendered')),
        mock.patch.object(pta_rendering, 'render_shared_tile_images',
                          wraps=pta_rendering.render_shared_tile_images) as images,
        mock.patch.object(pta_rendering.shared_geometry, 'render_categorical_dense_tile_for_job',
                          wraps=pta_rendering.shared_geometry.render_categorical_dense_tile_for_job) as masks,
    ):
        source = pta_rendering.render_plan_frame_source(
            volume=volume, mask=mask, plan=selected, idx=2, need_full=False,
        )
    assert source.img_full is source.mask_full is None
    assert source.img_canvas is source.mask_canvas is None
    assert images.call_count == masks.call_count == 1
    tile = selected.tile_layout[0]
    assert set(source.tile_arrays) == {tile.tile_tag}
    actual = pta_workers._derive_item_arrays(source, selected, tile.tile_tag)
    gpu_image, gpu_mask, gpu_shape = pta_workers._derive_gpu_item_source(source, selected, tile.tile_tag)
    for got, expected in zip(actual, full.tile_arrays[tile.tile_tag]):
        np.testing.assert_array_equal(got, expected)
    np.testing.assert_array_equal(gpu_image, actual[0])
    np.testing.assert_array_equal(gpu_mask, actual[1])
    assert gpu_shape == actual[0].shape[:2]
    for derive in (pta_workers._derive_item_arrays, pta_workers._derive_gpu_item_source):
        with pytest.raises(RuntimeError, match='without a rendered full frame'):
            derive(source, selected, 'full')


def test_legacy_tile_keeps_required_canvas_parent(tmp_path):
    plan = make_plan(tmp_path, 'gray')
    tile = replace(plan.tile_layout[0], shared_job=None)
    plan = replace(plan, tile_layout=(tile,))
    volume = np.arange(5*6*7, dtype=np.uint8).reshape(5, 6, 7)
    mask = (volume > 40).astype(np.uint8)
    expected = pta_rendering.render_plan_frame_source(volume=volume, mask=mask, plan=plan, idx=2)
    actual = pta_rendering.render_plan_frame_source(
        volume=volume, mask=mask, plan=plan, idx=2, need_full=False,
    )
    assert actual.img_canvas is not None and actual.mask_canvas is not None
    for got, want in zip(pta_workers._derive_item_arrays(actual, plan, tile.tile_tag),
                         pta_workers._derive_item_arrays(expected, plan, tile.tile_tag)):
        np.testing.assert_array_equal(got, want)


def test_cpu_tile_task_renders_only_accepted_tile(tmp_path):
    plan = make_plan(tmp_path, 'gray')
    tile = plan.tile_layout[1]
    candidate = pta.OutputCandidate(
        order=0, volume_name='sample', parent_view_tag=plan.tag, output_tag=tile.tile_tag,
        item_key=tile.tile_tag, frame_idx=2, is_tile=True, label_enabled=True,
    )
    task = pta_workers.FrameRenderTask(0, 2, ((tile.tile_tag, (candidate,)),))
    volume = np.arange(5*6*7, dtype=np.uint8).reshape(5, 6, 7)
    with (
        mock.patch.dict(pta_workers._WORKER_STATIC, {
            'augmentation': None, 'out_dir': tmp_path, 'split_active': False,
            'image_format': 'png', 'png_compression': 1, 'jpeg_quality': 95,
        }, clear=True),
        mock.patch.object(pta_rendering, 'render_channel_formatted_images',
                          side_effect=AssertionError('rejected full intensity rendered')),
        mock.patch.object(pta_rendering, 'render_plan_frame_mask_source',
                          side_effect=AssertionError('rejected full mask rendered')),
        mock.patch.object(pta_rendering, 'render_shared_tile_images',
                          wraps=pta_rendering.render_shared_tile_images) as render,
        mock.patch.object(pta_workers, 'write_selected_candidate_version', return_value='written') as write,
    ):
        result = pta_workers.execute_render_task(volume, (volume > 40).astype(np.uint8), (plan,), task)
    assert result[0] == 1
    assert render.call_count == write.call_count == 1
    assert render.call_args.kwargs['tile'] is tile
    assert write.call_args.kwargs['cand'] is candidate


def test_nonbatched_gpu_task_skips_rejected_rasters_before_gpu_boundary(tmp_path):
    plan = make_plan(tmp_path, 'gray')
    tile = plan.tile_layout[1]
    candidate = pta.OutputCandidate(
        order=0, volume_name='sample', parent_view_tag=plan.tag, output_tag=tile.tile_tag,
        item_key=tile.tile_tag, frame_idx=2, is_tile=True, label_enabled=True,
    )
    task = pta_workers.FrameRenderTask(0, 2, ((tile.tile_tag, (candidate,)),))
    volume = np.arange(5*6*7, dtype=np.uint8).reshape(5, 6, 7)
    with (
        mock.patch.dict(pta_workers._WORKER_STATIC, {
            'out_dir': tmp_path, 'split_active': False, 'image_format': 'png',
            'png_compression': 1, 'jpeg_quality': 95, 'gpu_batch_size': 1,
        }, clear=True),
        mock.patch.object(pta_workers, '_gpu_runtime_for_worker',
                          return_value={'torch': None, 'policy': None}),
        mock.patch.object(pta_rendering, 'render_channel_formatted_images',
                          side_effect=AssertionError('rejected full intensity rendered')),
        mock.patch.object(pta_rendering, 'render_plan_frame_mask_source',
                          side_effect=AssertionError('rejected full mask rendered')),
        mock.patch.object(pta_workers, '_derive_gpu_item_source',
                          side_effect=RuntimeError('stop before GPU work')) as derive,
    ):
        with pytest.raises(RuntimeError, match='stop before GPU work'):
            pta_workers.execute_gpu_render_task(volume, (volume > 40).astype(np.uint8), (plan,), task)
    source, selected, item_key = derive.call_args.args
    assert source.img_full is source.mask_full is None
    assert set(source.tile_arrays) == {tile.tile_tag}
    assert selected.tile_layout == (tile,)
    assert item_key == tile.tile_tag

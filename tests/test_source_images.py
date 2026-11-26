from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pytest

from XTA import outputs, source_images
from XTA.config import resolve_save_request


@pytest.mark.parametrize('shape', [(4, 16, 24), (2, 8, 12)])
def test_source_images_match_final_label_sequence(tmp_path, shape):
    indices = np.indices(shape)
    frames = (17 * indices[0] + 3 * indices[1] + indices[2]).astype(np.uint8)
    mask = np.zeros(shape, dtype=np.uint8)
    mask[0, 1:6, 1:7] = 1
    images = tmp_path / 'images'
    labels = tmp_path / 'labels'

    assert source_images.write_source_images_from_pattern(
        frames, images / 'scan_%04d.png', workers=2, show_progress=False,
    ) == images
    outputs.write_yolo_labels_from_pattern(
        mask, labels / 'scan_%04d.txt', workers=2, show_progress=False,
    )

    image_paths = sorted(images.glob('*.png'))
    label_paths = sorted(labels.glob('*.txt'))
    assert len(image_paths) == shape[0]
    assert [path.stem for path in image_paths] == [path.stem for path in label_paths]
    for path, expected in zip(image_paths, frames):
        actual = source_images.cv2.imread(str(path), source_images.cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(actual, expected)
    class_id, *coordinates = label_paths[0].read_text().split()
    assert class_id == '0'
    height, width = shape[1:]
    points = np.asarray(coordinates, dtype=float).reshape(-1, 2)
    np.testing.assert_allclose(points.min(axis=0), (1 / width, 1 / height), atol=1e-6)
    np.testing.assert_allclose(points.max(axis=0), (6 / width, 5 / height), atol=1e-6)
    assert label_paths[-1].read_text() == ''


def test_source_image_write_failure_preserves_prior_sequence(tmp_path, monkeypatch):
    pattern = tmp_path / 'images' / 'scan_%04d.png'
    original = np.full((2, 2, 3), 67, dtype=np.uint8)
    source_images.write_source_images_from_pattern(original, pattern, show_progress=False)
    imwrite = source_images.cv2.imwrite

    def fail_second_frame(path, frame):
        return False if Path(path).name.endswith('0002.png') else imwrite(path, frame)

    monkeypatch.setattr(source_images.cv2, 'imwrite', fail_second_frame)
    with pytest.raises(RuntimeError, match='Failed to write source image'):
        source_images.write_source_images_from_pattern(
            np.full_like(original, 143), pattern, workers=1, show_progress=False,
        )

    for path, expected in zip(sorted(pattern.parent.iterdir()), original, strict=True):
        assert path.suffix == '.png'
        np.testing.assert_array_equal(
            source_images.cv2.imread(str(path), source_images.cv2.IMREAD_UNCHANGED), expected,
        )


@pytest.mark.parametrize('native_images', [False, True])
def test_requested_source_image_grids_are_independent(tmp_path, monkeypatch, native_images):
    monkeypatch.setenv('YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN', '0')
    shape = (8, 16, 24)
    intensities = [35, 87, 129, 213]
    frames = np.empty(shape, dtype=np.uint8)
    frames[:] = np.repeat(intensities, 2)[:, None, None]
    mask = np.zeros_like(frames)
    mask[:2, 2:12, 2:12] = 1
    request = resolve_save_request([
        'low_quality:0.50:images,labels',
        'images,labels' if native_images else 'full_quality:labels',
    ])
    (spec,), _ = outputs.resolve_save_grids(request, shape)
    assert spec.output_shape_t_y_x == (4, 8, 12)
    output = tmp_path / 'out'
    scratch = tmp_path / 'scratch'
    with ThreadPoolExecutor(max_workers=2) as pool:
        full, full_futures = outputs.collect_pipeline_output_futures(
            pool, volume_rgb=frames, mask_u8=mask, out_dir=output, stem='scan',
            fps=20, formats=request.full_formats, frame_workers=1, scratch_dir=scratch,
        )
        preview, preview_futures = outputs.collect_low_quality_output_futures(
            pool, volume_gray=frames, mask_u8=mask, out_dir=output, stem='scan',
            fps=20, downbin_specs=[spec], temp_dir=scratch,
        )
        for future in (*full_futures, *preview_futures):
            future.result()

    grid_dir = output / 'low_quality' / spec.token
    assert preview == {
        f'low_quality_{spec.token}_images_dir': grid_dir / 'images',
        f'low_quality_{spec.token}_labels_dir': grid_dir / 'labels',
    }
    image_paths = sorted((grid_dir / 'images').glob('*.png'))
    label_paths = sorted((grid_dir / 'labels').glob('*.txt'))
    assert [p.stem for p in image_paths] == [p.stem for p in label_paths] == [
        f'scan_{index:04d}' for index in range(1, 5)
    ]
    assert label_paths[0].read_text().startswith('0 ')
    assert label_paths[1].read_text() == ''
    for path, intensity in zip(image_paths, intensities, strict=True):
        image = source_images.cv2.imread(str(path), source_images.cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(
            image, np.full(spec.output_shape_t_y_x[1:], intensity, dtype=np.uint8),
        )
    assert ('images_dir' in full) is native_images
    assert (output / 'images').exists() is native_images
    assert not (output / 'nrrd').exists()
    assert not list(output.rglob('*.mkv'))

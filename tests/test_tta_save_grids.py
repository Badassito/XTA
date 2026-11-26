from concurrent.futures import ThreadPoolExecutor
import json
import os
import shutil
import subprocess
import sys

import numpy as np
import pytest

from XTA import outputs
from XTA.cli import run
from XTA.config import resolve_save_request


def test_save_groups_round_trip_and_coalesce_on_resolved_grid():
    request = resolve_save_request([
        'low_quality:0.20:nrrd,xtareview,video', 'full_quality:images,labels,voxel_volume',
        'images', 'low_quality:0.2:binary_images',
    ])
    assert resolve_save_request(request.options) == request
    assert request.full_formats == ('images', 'labels', 'voxel_volume')
    grids, _ = outputs.resolve_save_grids(request, (100, 40, 1))
    assert len(grids) == 1
    assert grids[0].output_shape_t_y_x == (20, 8, 1)
    assert grids[0].formats == ('nrrd', 'xtareview', 'video', 'binary_images')


@pytest.mark.parametrize('group', [
    'low_quality:0.20', 'full_quality:', 'low_quality:0.5:video,', 'low_quality:0:video',
    'low_quality:nan:video', 'low_quality:1.5:video', 'low_quality:0.5:summary',
    'low_quality:0.5:confidence', 'overlay', 'high_quality', 'binary', 'low_quality',
])
def test_invalid_save_groups_fail_before_run(group):
    with pytest.raises(ValueError, match='--save'):
        resolve_save_request(group)


@pytest.mark.parametrize('conf', ['0.25', '0.25:0.25:0.50'])
def test_cpu_only_azimuthal_refused_before_study_creation(tmp_path, conf, capsys):
    output = tmp_path / 'unstarted'
    with pytest.raises(SystemExit) as exc:
        run(['--mode', 'tta', '--device', 'cpu', '--model', 'cpu:detector.xml',
             '--input', 'source.mkv', '--output', str(output), '--conf', conf,
             '--enable_azimuthal', 'transverse:auto'])
    assert exc.value.code == 2
    assert 'OpenVINO cannot execute' in capsys.readouterr().err
    assert not output.exists()


def test_unavailable_native_scheduler_stops_sweep_before_creating_outputs(tmp_path):
    out = tmp_path / 'unstarted'
    env = {**os.environ, 'XTA_NATIVE_SCHEDULER_BINARY': str(tmp_path / 'missing-scheduler.pyd')}
    result = subprocess.run(
        [sys.executable, '-B', '-m', 'XTA', '--mode', 'tta', '--device', 'cpu',
         '--model', 'cpu:detector.xml', '--input', 'source.mkv', '--output', str(out),
         '--conf', '0.25:0.25:0.50'], env=env, capture_output=True, text=True,
    )
    assert result.returncode != 0
    assert 'scheduler' in result.stderr.lower()
    assert not out.exists()


@pytest.mark.skipif(not shutil.which('ffprobe') or not shutil.which('ffmpeg'), reason='FFmpeg required')
def test_selected_scalar_grid_writers_emit_only_requested_products(tmp_path, monkeypatch):
    monkeypatch.setenv('YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN', '0')
    monkeypatch.setenv('YOLO_TTA_FFV1_THREADS', '1')
    shape = (8, 24, 32)
    gray = np.arange(np.prod(shape), dtype=np.uint8).reshape(shape)
    mask = np.zeros(shape, np.uint8)
    mask[:, 6:18, 8:24] = 1
    request = resolve_save_request('low_quality:0.5:video,binary_images,voxel_volume images,labels,binary_video')
    grids, _ = outputs.resolve_save_grids(request, shape)
    preview = grids[0]
    out = tmp_path / 'output'
    out.mkdir()
    with ThreadPoolExecutor(3) as executor:
        native, first = outputs.collect_pipeline_output_futures(
            executor, volume_rgb=gray, mask_u8=mask, out_dir=out, stem='sample', fps=8,
            formats=request.full_formats, frame_workers=1, scratch_dir=tmp_path,
        )
        low, second = outputs.collect_low_quality_output_futures(
            executor, volume_gray=gray, mask_u8=mask, out_dir=out, stem='sample', fps=8,
            downbin_specs=grids, temp_dir=tmp_path, workers=1, show_progress=False,
        )
        for future in first + second:
            future.result()
    assert set(native) == {'images_dir', 'labels_dir', 'binary_video'}
    assert {p.name for p in out.iterdir()} == {'images', 'labels', 'sample_Binary.mkv', 'low_quality'}
    assert {p.stem for p in native['images_dir'].iterdir()} == {p.stem for p in native['labels_dir'].iterdir()}
    pfx = 'low_quality_' + preview.token + '_'
    assert set(low) == {pfx + suffix for suffix in ('overlay', 'binary_tiff_dir', 'voxel_volume')}
    preview_dir = out / 'low_quality' / preview.token
    assert {p.name for p in preview_dir.iterdir()} == {'sample_Overlay.mkv', 'binary_masks', 'sample_VoxelVolume.json'}
    count = json.loads(low[pfx + 'voxel_volume'].read_text())
    frames = list(low[pfx + 'binary_tiff_dir'].glob('*.tiff'))
    assert len(frames) == preview.output_shape_t_y_x[0]
    assert count['voxel_count'] == sum(np.count_nonzero(outputs.tifffile.imread(p)) for p in frames)
    for path, (depth, height, width) in ((native['binary_video'], shape),
                                          (low[pfx + 'overlay'], preview.output_shape_t_y_x)):
        data = subprocess.check_output(['ffprobe', '-v', 'error', '-count_frames', '-show_streams',
                                        '-of', 'json', str(path)])
        stream = json.loads(data)['streams'][0]
        assert (stream['codec_name'], int(stream['nb_read_frames']), stream['height'], stream['width']) == (
            'ffv1', depth, height, width,
        )

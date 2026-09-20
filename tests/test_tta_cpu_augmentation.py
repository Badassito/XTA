"""CPU replay, persistent inference restoration, and independent pass outputs."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from XTA.tta_augmentation_cpu import CpuPolicyAdapter, CpuSpatialReplay, inverse_policy_grid_cpu


class TranslationPolicy:
    tta_replay_contract = 'opencv-affine-elastic-v1'

    def _sample_parameters(self, seed, height, width):
        return {'elastic': False}

    def _forward_matrix(self, params, height, width):
        return [[1., 0., 3.], [0., 1., 0.], [0., 0., 1.]]

    def _elastic_displacement(self, seed, height, width):
        return np.zeros((height, width, 2), np.float32)

    def _apply_intensity_noise(self, image, **kwargs):
        return image


def test_cpu_translation_restores_content_and_marks_cropped_pixels_unknown():
    image = np.zeros((24, 24), np.uint8)
    image[4:12, 5:10] = 220
    image[1, -1] = 255
    adapter = CpuPolicyAdapter(TranslationPolicy())
    output, replays = adapter.apply([image], [13])
    replay = replays[0]
    assert np.array_equal(output[0][4:12, 8:13], image[4:12, 5:10])
    expected = image.copy()
    expected[:, -3:] = 0
    np.testing.assert_array_equal(replay.restore_planes(output)[0], expected)
    assert replay.valid[:, :-3].all() and not replay.valid[:, -3:].any()
    np.testing.assert_array_equal(np.unpackbits(replay.packed_validity(), axis=1)[:, :24], replay.valid)
    assert adapter._replay(13, 24, 24) is replay


@pytest.mark.parametrize('reflection', [False, True])
def test_cpu_elastic_inverse_matches_known_constant_displacement(reflection):
    h, w = 20, 24
    forward = np.eye(3)
    if reflection:
        forward[0, 0], forward[0, 2] = -1., w - 1
    field = np.zeros((h, w, 2), np.float32)
    field[..., 0], field[..., 1] = 2., 3.
    grid, valid = inverse_policy_grid_cpu(forward, field, h, w)
    yy, xx = np.mgrid[:h, :w]
    expected = np.stack((w - 1 - (xx - 2) if reflection else xx - 2, yy - 3), axis=-1)
    np.testing.assert_allclose(grid[valid], expected[valid], atol=1e-5)
    assert 0 < valid.sum() < h * w


def test_cpu_elastic_folds_are_unknown():
    h, w = 24, 24
    field = np.zeros((h, w, 2), np.float32)
    field[..., 0] = -2 * np.arange(w, dtype=np.float32)[None, :] + w - 1
    _grid, valid = inverse_policy_grid_cpu(np.eye(3), field, h, w)
    assert not valid[:, 1:-1].any()


@pytest.mark.parametrize('profile', ['light', 'baseline', 'heavy', 'superheavy'])
def test_cpu_adapter_preserves_shipped_policy_pixels(profile):
    path = Path(__file__).resolve().parents[1] / 'XTA/examples/external_augmentations' / f'CPU_{profile}.py'
    spec = importlib.util.spec_from_file_location(f'test_cpu_policy_{profile}', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    policy = module.build_augmentation()
    adapter = CpuPolicyAdapter(policy)
    image = np.random.default_rng(11).integers(0, 256, (33, 35, 3), dtype=np.uint8)
    for seed in (2, 7, 20):
        policy.set_random_seed(seed)
        expected = policy(image=image, mask=np.zeros(image.shape[:2], np.uint8))['image']
        output, _replay = adapter.apply([image], [seed])
        np.testing.assert_array_equal(output[0], expected)


def test_cpu_policy_requires_explicit_inverse_contract():
    with pytest.raises(TypeError, match='supported inverse contract'):
        CpuPolicyAdapter(lambda **kwargs: kwargs)


def test_cpu_policy_startup_does_not_import_torch_or_albumentations():
    root = Path(__file__).resolve().parents[1]
    subprocess.run([sys.executable, '-c',
        "import sys; from XTA.tta_augmentation_config import resolve_tta_augmentation; "
        "from XTA.tta_augmentation_cpu import worker_cpu_policy; from types import SimpleNamespace; "
        "settings=resolve_tta_augmentation(SimpleNamespace(augmentation="
        "['cpu:XTA/examples/external_augmentations/CPU_light.py'],augmentation_ratio=2),"
        "gpu_devices=(),cpu_enabled=True); worker_cpu_policy(settings); "
        "assert 'torch' not in sys.modules and 'albumentations' not in sys.modules"],
        cwd=root, check=True)


def test_custom_cpu_contract_validates_support_and_retains_input():
    class Custom:
        def apply_tta_batch(self, images, seeds):
            yy, xx = np.mgrid[:5, :5].astype(np.float32)
            grid = np.stack((xx / 2 - 1, yy / 2 - 1), axis=-1)[None]
            grid[0, 0, 0] = np.nan
            images[:] = 100
            return {'images': images, 'inverse_grid': grid, 'valid': np.ones((1, 5, 5), bool)}
    image = np.zeros((5, 5), np.uint8)
    outputs, replays = CpuPolicyAdapter(Custom()).apply([image], [4])
    assert not image.any() and outputs[0].min() == 100
    assert not replays[0].valid[0, 0]
    restored = replays[0].restore_planes([np.ones((5, 5), np.uint8)])[0]
    assert restored.sum() == 24


def test_restore_occurs_before_processing_affine(monkeypatch):
    from XTA import inference
    original = np.zeros((24, 24), np.uint8)
    original[4:12, 5:10] = 1
    augmented, replay = CpuPolicyAdapter(TranslationPolicy()).apply([original * 255], [1])
    monkeypatch.setattr(inference, '_accumulate_cpu_retina_payload_to_prediction_frame',
                        lambda *_args: ((augmented[0] > 0).astype(np.uint8), augmented[0], 1))
    dest, conf = np.zeros((1, 12, 12), np.uint8), np.zeros((1, 12, 12), np.uint8)
    affine = np.asarray([[.5, 0., 0.], [0., .5, 0.]], np.float32)
    count = inference._process_cpu_retina_prediction_frame(
        0, object(), 24, dest, conf, affine, 12, 12, restore_planes=replay[0].restore_planes)
    assert count == (1, 1)
    np.testing.assert_array_equal(dest[0], cv2.warpAffine(original, affine, (12, 12), flags=cv2.INTER_NEAREST))


class BatchSource:
    azimuthal_padding_count = 0

    def __init__(self, images, batch=2):
        self.images, self.batch, self.rendered = images, batch, 0

    def __iter__(self):
        for start in range(0, len(self.images), self.batch):
            self.rendered += 1
            values = self.images[start:start + self.batch]
            values = [*values, *([values[-1]] * (self.batch - len(values)))]
            yield ['frame'] * self.batch, values, [''] * self.batch

    def result_frame_spec(self, index):
        from XTA.geometry import BatchResultFrameSpec
        if index >= len(self.images):
            return None
        return BatchResultFrameSpec(index, index, index + 4, False)


def _exercise_cpu_runtime(tmp_path, monkeypatch, runner, *, shared=False):
    from XTA import tta_augmentation_cpu_runtime as runtime
    from XTA.tta_augmentation_config import TtaAugmentationSettings
    from XTA.augmentation_policy import inspect_augmentation_definition
    monkeypatch.setattr(runtime, 'worker_cpu_policy', lambda _settings: CpuPolicyAdapter(TranslationPolicy()))
    policy_path = tmp_path / 'policy.py'
    policy_path.write_text('def build_augmentation(): pass\n')
    definition = inspect_augmentation_definition(str(policy_path))
    settings = TtaAugmentationSettings(ratio=3, path=str(policy_path),
                                      content_sha256=definition.content_sha256,
                                      export_name='build_augmentation', coverage='packed')
    image = np.zeros((24, 24), np.uint8)
    image[4:12, 5:10] = 255
    source = BatchSource([image.copy() for _ in range(3)])
    # Distinct pass parents are live together; pre-existing slice windows survive.
    targets = []
    for i in range(3):
        parent_shape = (10, 24, 24) if shared else (3, 24, 24)
        parent = np.memmap(tmp_path / f'pass{i}.dat', mode='w+', dtype=np.uint8, shape=parent_shape)
        parent[:] = 7 if shared else 0
        if shared:
            parent[4:7] = 0
        targets.append(parent)
        if not shared and i:
            parent._mmap.close()
    task = dict(view=SimpleNamespace(name='transverse', augmentation_pass=0), kind='fullframe',
                job_id='a0', task_id=7, slice_start=4, slice_count=3, model_name='tiny', out_size=24,
                result_mode='direct_union' if shared else 'file', bounded_parent_admission=shared,
                result_mask_path=str(tmp_path / 'pass0.dat'), processing_shape=(10, 24, 24),
                union_num_slices=10, M_out_to_processing=np.eye(2, 3), augmentation_settings=settings,
                augmentation_support_dir=str(tmp_path / 'support'), result_conf_path=None)
    siblings = [dict(task, view=SimpleNamespace(name=f'transverse_policy{i}', augmentation_pass=i),
                     result_mask_path=str(tmp_path / f'pass{i}.dat')) for i in (1, 2)]
    task['augmentation_pass_tasks'] = siblings
    base = targets[0][4:7] if shared else targets[0]
    result = runtime.predict_cpu_policy_source(runner, source, task=task, cfg=SimpleNamespace(batch=2),
        predict_kwargs=dict(num_frames=3, out_size=24, conf_threshold=.1, view_union_mm=base,
                            view_confmap_mm=None, M_out_to_native=np.eye(2, 3), native_h=24,
                            native_w=24, min_conf=0., min_radius=0.))
    assert source.rendered == 2
    assert result['augmentation_execution']['model_batches'] == 6
    assert result['augmentation_execution']['backend'] == 'cpu'
    assert len(result['augmentation_results']) == 2 and len(result['augmentation_records']) == 2
    for i, target in enumerate(targets):
        if not shared and i:
            target = np.memmap(tmp_path / f'pass{i}.dat', mode='r', dtype=np.uint8, shape=(3, 24, 24))
        window = target[4:7] if shared else target
        np.testing.assert_array_equal(window, np.stack([image > 0] * 3))
        if shared:
            assert (target[:4] == 7).all() and (target[7:] == 7).all()
        target._mmap.close()
    for record in result['augmentation_records']:
        packed = np.load(record['path'])
        assert json.loads(packed['metadata'].item())['backend'] == 'cpu'
        np.testing.assert_array_equal(packed['global_destinations'], [4, 5, 6])
        valid = np.unpackbits(packed['validity_bits'], axis=2)[:, :, :24]
        assert valid[:, :, :-3].all() and not valid[:, :, -3:].any()
    return result


@pytest.mark.parametrize('shared', [False, True])
def test_cpu_runtime_renders_once_and_keeps_independent_pass_outputs(tmp_path, monkeypatch, shared):
    from XTA import workers

    class Runner:
        def infer_source_to_union(self, source, **kwargs):
            for _paths, images, _info in source:
                for i, image in enumerate(images):
                    spec = source.result_frame_spec(i)
                    if spec is None:
                        continue
                    planes = source.restore_prediction_planes(spec, [(image > 0).astype(np.uint8)])
                    kwargs['view_union_mm'][spec.task_index] |= planes[0]
            return dict(prediction_count=kwargs['num_frames'], frames_with_predictions=kwargs['num_frames'],
                        slice_meta=workers._binary_slice_metadata_from_array(kwargs['view_union_mm']))

    result = _exercise_cpu_runtime(tmp_path, monkeypatch, Runner(), shared=shared)
    assert result['prediction_count'] == 3


def test_real_openvino_cpu_augmented_inference(tmp_path, monkeypatch):
    ov = pytest.importorskip('openvino')
    from openvino import opset13 as ops
    from XTA import workers
    inputs = ops.parameter([2, 1, 24, 24], np.float32, name='images')
    proto = ops.subtract(inputs, ops.constant(np.float32(.5)))
    head_array = np.zeros((2, 6, 8), np.float32)
    head_array[:, :4, 0] = [12., 12., 24., 24.]
    head_array[:, 4:, 0] = [.9, 1.]
    model = ov.Model([ops.constant(head_array), proto], [inputs], 'cpu_policy_smoke')
    path = tmp_path / 'model.xml'
    ov.save_model(model, path)
    runner = workers._OpenVinoCpuSegmenter(str(path), imgsz=24, batch=2, input_channels=1,
                requested_precision='fp32', inference_threads=2, physical_cores=2, streams=1, infer_requests=2)
    result = _exercise_cpu_runtime(tmp_path, monkeypatch, runner)
    assert result['prediction_count'] == 3
    assert result['openvino_precision'] == 'fp32'

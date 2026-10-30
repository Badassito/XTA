"""Retire only the native mask reference after successful pipeline preparation."""
import hashlib
from types import SimpleNamespace
import weakref

import numpy as np
import pytest

from XTA import pta
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options


@pytest.mark.parametrize('shared', [False, True])
@pytest.mark.parametrize('resized', [False, True])
def test_pipeline_retires_input_mask_after_prepare_without_retiring_render_owners(
        tmp_path, monkeypatch, shared, resized):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setattr(pta, 'discover_topology', lambda **kwargs: SimpleNamespace(
        cuda_device_ids=(), gpu_cpu_sets=(), worker_cpu_order=(),
        allowed_cpus=(0, 1), summary='CPU ownership regression'))
    source_dir = tmp_path / 'input'
    source_dir.mkdir()
    shape = (4, 8, 8) if resized else (4, 4, 4)
    for index in range(shape[0]):
        assert pta.cv2.imwrite(str(source_dir / f'sample_{index:04d}.png'),
                               np.full(shape[1:], index * 30, np.uint8))
    pta.write_nrrd(np.ones(shape, np.uint8), source_dir / 'sample.nrrd')
    real_prepare = pta.prepare_loaded_source
    real_enumerate = pta.enumerate_candidates_for_volume
    references = []
    checked = []
    retain_native = False
    retained_masks = []
    native_intensity_blocks = []

    def prepare(source, **kwargs):
        reference = weakref.ref(source.mask_volume)
        references.append(reference)
        if retain_native:
            retained_masks.append(source.mask_volume)
        if shared:
            block = pta.SharedBlock(source.volume.nbytes)
            shared_image = block.ndarray(source.volume.shape)
            np.copyto(shared_image, source.volume)
            source.volume, source.volume_block = shared_image, block
            native_intensity_blocks.append(block)
        kwargs['allocator'] = pta.make_volume_allocator(shared)
        result = real_prepare(source, **kwargs)
        # Standalone preparation retains its existing caller-owned input API.
        assert source.mask_volume is reference()
        result._test_mask_digest = hashlib.sha256(result.mask_for_render).hexdigest()
        result._test_image_digest = hashlib.sha256(source.volume).hexdigest()
        return result

    def enumerate_candidates(prepared, **kwargs):
        assert prepared.src.mask_volume is None
        assert (references[-1]() is not None) == (retain_native or (not resized and not shared))
        assert hashlib.sha256(prepared.mask_for_render).hexdigest() == prepared._test_mask_digest
        assert hashlib.sha256(prepared.src.volume).hexdigest() == prepared._test_image_digest
        assert all(not block.released for block in prepared.shm_blocks)
        if shared:
            assert prepared.src.volume_block is native_intensity_blocks[-1]
            assert not prepared.src.volume_block.released
        checked.append(True)
        return real_enumerate(prepared, **kwargs)

    monkeypatch.setattr(pta, 'prepare_loaded_source', prepare)
    monkeypatch.setattr(pta, 'enumerate_candidates_for_volume', enumerate_candidates)
    arguments = ['--input', str(source_dir), '--output', str(tmp_path / 'output'),
                 '--imgsz', '4', '--enable_cartesian', 'transverse', '--save', 'images', 'labels',
                 '--worker_backend', 'thread', '--workers', '2', '--frame_workers', '1',
                 '--jpeg_decode_backend', 'opencv', '--no-topology_aware']
    pta.main(args=build_runtime_options(parse_pta_args(arguments)), argv=arguments)
    assert checked == [True]
    assert all(reference() is None for reference in references)
    assert len(list((tmp_path / 'output' / 'images').glob('*.png'))) == (8 if resized else 4)

    def payloads(output):
        return {str(path.relative_to(output)): path.read_bytes()
                for folder in ('images', 'labels')
                for path in (output / folder).rglob('*') if path.is_file()}

    retired_payloads = payloads(tmp_path / 'output')
    retain_native = True
    arguments[arguments.index('--output') + 1] = str(tmp_path / 'retained-output')
    pta.main(args=build_runtime_options(parse_pta_args(arguments)), argv=arguments)
    assert checked == [True, True]
    assert retired_payloads == payloads(tmp_path / 'retained-output')
    retained_masks.clear()
    assert all(reference() is None for reference in references)
    assert all(block.released for block in native_intensity_blocks)


def test_failure_after_registered_preparation_releases_shared_owners(tmp_path, monkeypatch):
    monkeypatch.setenv('CUDA_VISIBLE_DEVICES', '-1')
    monkeypatch.setattr(pta, 'discover_topology', lambda **kwargs: SimpleNamespace(
        cuda_device_ids=(), gpu_cpu_sets=(), worker_cpu_order=(),
        allowed_cpus=(0, 1), summary='CPU failure ownership regression'))
    source_dir = tmp_path / 'input'
    source_dir.mkdir()
    for index in range(4):
        assert pta.cv2.imwrite(str(source_dir / f'sample_{index:04d}.png'),
                               np.full((8, 8), index * 30, np.uint8))
    pta.write_nrrd(np.ones((4, 8, 8), np.uint8), source_dir / 'sample.nrrd')
    real_prepare = pta.prepare_loaded_source
    blocks, masks, prepared_instances = [], [], []

    def allocate(shape):
        block = pta.SharedBlock(int(np.prod(shape)))
        blocks.append(block)
        return block.ndarray(shape), block

    def prepare(source, **kwargs):
        masks.append(weakref.ref(source.mask_volume))
        image, block = allocate(source.volume.shape)
        np.copyto(image, source.volume)
        source.volume, source.volume_block = image, block
        kwargs['allocator'] = allocate
        prepared = real_prepare(source, **kwargs)
        prepared_instances.append(prepared)
        return prepared

    def fail_after_preparation(prepared, **kwargs):
        assert prepared.src.mask_volume is None
        assert masks[0]() is None
        assert len(blocks) == 3  # Native intensity and both processing cubes.
        assert all(not block.released for block in blocks)
        raise RuntimeError('candidate enumeration failed after preparation')

    monkeypatch.setattr(pta, 'prepare_loaded_source', prepare)
    monkeypatch.setattr(pta, 'enumerate_candidates_for_volume', fail_after_preparation)
    arguments = ['--input', str(source_dir), '--output', str(tmp_path / 'output'),
                 '--imgsz', '4', '--enable_cartesian', 'transverse', '--save', 'images', 'labels',
                 '--worker_backend', 'thread', '--workers', '2', '--frame_workers', '1',
                 '--jpeg_decode_backend', 'opencv', '--no-topology_aware']
    with pytest.raises(RuntimeError, match='candidate enumeration failed after preparation'):
        pta.main(args=build_runtime_options(parse_pta_args(arguments)), argv=arguments)
    assert all(block.released for block in blocks)
    assert masks[0]() is None
    assert prepared_instances[0].src.mask_volume is None
    assert prepared_instances[0].src.volume_block is None
    assert prepared_instances[0].volume_for_render.size == 0
    assert prepared_instances[0].mask_for_render.size == 0

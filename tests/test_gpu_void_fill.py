"""GPU slab reconciliation is exact even when cavities cross slab boundaries."""

from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from scipy import ndimage

from XTA import topology
from XTA.config import GIB
from XTA import topology_gpu_voidfill as gpu


class _Array(np.ndarray):
    def get(self, *, out):
        np.copyto(out, self)


def _fake_cupy(shape, block):
    free = GIB + int(shape[1]) * int(shape[2]) * 24 * block

    class Device:
        def __init__(self, index):
            self.index = index
            self.mem_info = (free, free)

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

    def array(value):
        return np.asarray(value).view(_Array)

    cp = SimpleNamespace(
        cuda=SimpleNamespace(Device=Device), asarray=array, asnumpy=np.asarray,
        int32=np.int32, unique=lambda a: array(np.unique(a)),
        concatenate=lambda a: array(np.concatenate(a)),
        get_default_memory_pool=lambda: SimpleNamespace(free_all_blocks=lambda: None),
    )

    def label(mask, *, structure, output):
        labels, count = ndimage.label(mask, structure=structure, output=output)
        return array(labels), count

    return cp, SimpleNamespace(label=label)


def _run(mask, tmp_path, *, connectivity=6, block=2, devices=(0, 1), disk=False):
    expected = ndimage.binary_fill_holes(
        mask != 0, structure=ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity]),
    )
    actual = mask.copy()
    with mock.patch.dict('os.environ', {'YOLO_TTA_VOIDFILL_CONNECTIVITY': str(connectivity)}), \
         mock.patch.object(gpu, '_try_import_cupy_ndimage', return_value=_fake_cupy(mask.shape, block)), \
         mock.patch.object(topology, 'should_use_in_memory_workspace', return_value=not disk):
        stats = topology.fill_3d_voids_inplace_streaming(
            actual, tmp_path / 'void', connectivity=connectivity, gpu_device_ids=devices,
        )
    np.testing.assert_array_equal(actual != 0, expected)
    np.testing.assert_array_equal(actual[mask != 0], mask[mask != 0])
    assert stats['backend'] == 'cuda'
    assert stats['devices'] == list(devices[:stats['chunks']])
    assert stats['fallback_reason'] is None
    return actual, stats


@pytest.mark.parametrize('connectivity', [6, 18, 26])
@pytest.mark.parametrize('block', [1, 2, 4, 64])
def test_random_slabs_match_scipy(tmp_path, connectivity, block):
    mask = (np.random.default_rng(391).random((13, 12, 11)) < .77).astype(np.uint8)
    _run(mask, tmp_path, connectivity=connectivity, block=block)


@pytest.mark.parametrize('connectivity', [6, 18, 26])
@pytest.mark.parametrize('dimensions', [2, 3])
def test_diagonal_only_exterior_path_across_slabs(tmp_path, connectivity, dimensions):
    mask = np.ones((7, 7, 7), np.uint8)
    for index in range(4):
        mask[index, index, index if dimensions == 3 else 3] = 0
    actual, _ = _run(mask, tmp_path, connectivity=connectivity)
    assert actual[3, 3, 3] == (connectivity < (26 if dimensions == 3 else 18))


@pytest.mark.parametrize('value', [0, 1, 255])
def test_uniform_and_original_foreground_values(tmp_path, value):
    _run(np.full((7, 5, 6), value, np.uint8), tmp_path)


def test_slab_faces_are_not_global_boundaries_and_memmap_is_cleaned(tmp_path):
    mask = np.ones((9, 8, 7), np.uint8)
    mask[2:7, 2:6, 2:5] = 0
    actual, _ = _run(mask, tmp_path, disk=True)
    assert np.all(actual)
    assert not list(tmp_path.glob('*.dat'))


def test_gpu_failure_does_not_mutate_mask_before_cpu_fallback(tmp_path):
    mask = np.ones((7, 7, 7), np.uint8)
    mask[2:5, 2:5, 2:5] = 0
    actual = mask.copy()

    def fail_before_apply(observed_mask, labels, **kwargs):
        labels[:2] = 123456
        np.testing.assert_array_equal(observed_mask, mask)
        raise RuntimeError('synthetic CUDA failure')

    with mock.patch.object(gpu, 'prepare_gpu_void_fill', side_effect=fail_before_apply):
        stats = topology.fill_3d_voids_inplace_streaming(actual, tmp_path / 'void', gpu_device_ids=(0, 1))
    np.testing.assert_array_equal(actual, ndimage.binary_fill_holes(mask))
    assert stats['backend'] == 'cpu'
    assert 'synthetic CUDA failure' in stats['fallback_reason']


def test_partial_gpu_labels_are_discarded_on_label_failure(tmp_path):
    mask = np.ones((7, 7, 7), np.uint8)
    mask[2:5, 2:5, 2:5] = 0
    original = mask.copy()
    cp, ndi = _fake_cupy(mask.shape, block=2)
    label = ndi.label
    calls = []

    def fail_second(*args, **kwargs):
        np.testing.assert_array_equal(mask, original)
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError('failure after first successful slab')
        return label(*args, **kwargs)

    ndi.label = fail_second
    with mock.patch.object(gpu, '_try_import_cupy_ndimage', return_value=(cp, ndi)):
        stats = topology.fill_3d_voids_inplace_streaming(mask, tmp_path / 'void', gpu_device_ids=(0,))
    assert len(calls) == 2
    assert stats['backend'] == 'cpu'
    np.testing.assert_array_equal(mask, ndimage.binary_fill_holes(original))


def test_no_device_and_invalid_connectivity_do_not_enter_gpu(tmp_path):
    mask = np.ones((3, 3, 3), np.uint8)
    with mock.patch.object(gpu, 'prepare_gpu_void_fill') as prepare:
        stats = topology.fill_3d_voids_inplace_streaming(mask, tmp_path / 'void')
        assert stats['backend'] == 'cpu'
        prepare.assert_not_called()
        with mock.patch.dict('os.environ', {'YOLO_TTA_VOIDFILL_CONNECTIVITY': '7'}):
            with pytest.raises(ValueError, match='6, 18, or 26'):
                topology.fill_3d_voids_inplace_streaming(mask, tmp_path / 'void', gpu_device_ids=(0,))
        prepare.assert_not_called()


@pytest.mark.skipif(__import__('os').environ.get('YOLO_TTA_RUN_GPU_TESTS') != '1', reason='explicit GPU qualification')
@pytest.mark.parametrize('connectivity', [6, 18, 26])
def test_real_gpu_connectivity(tmp_path, connectivity):
    mask = (np.random.default_rng(178).random((130, 17, 19)) < .8).astype(np.uint8)
    expected = ndimage.binary_fill_holes(
        mask, structure=ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity]),
    )
    with mock.patch.dict('os.environ', {'YOLO_TTA_VOIDFILL_CONNECTIVITY': str(connectivity)}):
        stats = topology.fill_3d_voids_inplace_streaming(mask, tmp_path / 'void', gpu_device_ids=(0,))
    assert stats['backend'] == 'cuda', stats
    assert stats['chunks'] >= 3
    np.testing.assert_array_equal(mask, expected)

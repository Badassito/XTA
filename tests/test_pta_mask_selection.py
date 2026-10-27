"""Whole-mask selection reuses snapshots; subsets retain exact planes/order."""
from unittest import mock

import numpy as np
import pytest

from XTA import pta_workers as workers
from XTA.pta_dataset import OutputCandidate
from tests.test_pta_gpu_publication import FakeTensor, FakeTorch


class TracedTensor(FakeTensor):
    def wrap(self, values):
        return TracedTensor(self.torch, values, self.name)

    def index_select(self, axis, indices):
        self.torch.selections.append((axis, tuple(indices)))
        return self.wrap(np.take(self.array, indices, axis=axis))

    def to(self, device):
        return self.wrap(self.array.copy() if device == 'cpu' else self.array.astype(device))

    def __gt__(self, value):
        return self.wrap(self.array > value)

    def __eq__(self, value):
        return self.wrap(self.array == value)

    def __mul__(self, value):
        return self.wrap(self.array * value.array)

    def masked_fill_(self, mask, value):
        self.array[mask.array] = value
        return self


@pytest.mark.parametrize('combined', [False, True])
@pytest.mark.parametrize('subset', [False, True])
def test_full_and_partial_mask_coverage_selection(tmp_path, combined, subset):
    torch = FakeTorch()
    torch.selections = []
    if combined:
        torch.Tensor = TracedTensor
    masks = np.zeros((3, 8, 8), np.uint8)
    masks[0, 1:4, 1:4] = 1
    masks[1, 2:6, 2:6] = 255
    masks[2, 4:7, 4:7] = 1
    coverage = np.ones_like(masks)
    coverage[:, :2] = 0
    candidates = [OutputCandidate(
        order=index, volume_name='sample', parent_view_tag='Transverse',
        output_tag='Transverse', item_key='full', frame_idx=index, is_tile=False,
        label_enabled=not subset or index != 1,
    ) for index in range(3)]
    binary, semantic = [], []
    with mock.patch.dict(workers._WORKER_STATIC, {
        'out_dir': tmp_path, 'split_active': False, 'image_format': 'png',
        'save_images': False, 'save_labels': False, 'save_binary': not combined,
        'save_semantic': True,
    }, clear=True), mock.patch.object(workers, 'publish_binary_mask_payloads',
                                     side_effect=lambda payloads: binary.extend(payloads)), \
            mock.patch.object(workers, 'publish_semantic_mask_payloads',
                              side_effect=lambda payloads, **_kw: semantic.extend(payloads)):
        result = workers._publish_gpu_policy_batch(
            runtime={'torch': torch, 'device_id': 0},
            batch_images=TracedTensor(torch, np.zeros((3, 1, 8, 8), np.uint8)),
            batch_masks=TracedTensor(torch, masks), candidates=candidates, output_size=(8, 8),
            channel_kind='gray', local_warnings=workers.WarningLog(),
            source_mask_nonempty=(True,) * 3,
            semantic_coverage_masks=TracedTensor(torch, coverage),
        )
    assert result == (3, {})
    selected = [0, 2] if subset else [0, 1, 2]
    assert torch.selections == ([(0, (0, 2))] * 2 if subset else [])
    for offset, index in enumerate(selected):
        if combined:
            expected = (masks[index] > 0).astype(np.uint8)
            expected[coverage[index] == 0] = 255
            np.testing.assert_array_equal(semantic[offset][1], expected)
            assert semantic[offset][2] is None
        else:
            np.testing.assert_array_equal(binary[offset][1], masks[index])
            np.testing.assert_array_equal(semantic[offset][1], masks[index])
            np.testing.assert_array_equal(semantic[offset][2], coverage[index])

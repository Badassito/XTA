"""Scratch label and binary archives retire only after their mapped readers finish."""

from __future__ import annotations

import gc
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import cuda_d1, finalization, runtime, topology


@pytest.mark.parametrize('keep_n', (1, 2))
@pytest.mark.parametrize('keep_temp', (False, True))
def test_keep_objects_retires_only_unkept_disk_labels_after_both_decision_paths(
    tmp_path: Path, keep_n: int, keep_temp: bool,
) -> None:
    mask = np.zeros((2, 7, 7), dtype=np.uint8)
    mask[0, 1:3, 1:3] = 1
    mask[1, 4:6, 4:6] = 1
    with mock.patch.object(topology, 'interpolation_sparse_labels_enabled', return_value=False):
        stats = finalization.apply_keep_largest_objects_inplace(
            mask, keep_n, tmp_path, keep_temp=keep_temp, prefer_memory=False, workers=1,
        )
    label_path = tmp_path / 'keep_objects' / 'final_keep_objects.fg_labels.u16.dat'
    if keep_temp:
        assert label_path.exists()
    else:
        runtime.wait_for_retired_memmap_unlinks(path=label_path)
        assert not label_path.exists()
    assert stats['num_objects'] == 2
    assert int(mask.sum()) == (4 if keep_n == 1 else 8)


@pytest.mark.parametrize('archive', (False, True))
def test_temporary_raw_volume_waits_for_view_before_delete(tmp_path: Path, archive: bool) -> None:
    raw_path = tmp_path / 'accumulator.dat'
    owner = np.memmap(raw_path, mode='w+', dtype=np.uint8, shape=(2, 3, 4))
    owner[:] = 1
    retained = np.asarray(owner)[0]
    archived: list[Path] = []

    def write_archive(volume, destination, **_kwargs):
        assert int(np.asarray(volume).sum()) == 24
        archived.append(Path(destination))
        Path(destination).mkdir()

    with mock.patch.object(cuda_d1, 'write_raw_bbox_mask_store', side_effect=write_archive):
        cuda_d1.archive_or_delete_binary_volume_storage(
            owner, keep_temp=archive, workers=1, desc='test scratch accumulator',
        )
    assert raw_path.exists()
    assert int(retained.sum()) == 12
    assert len(archived) == int(archive)
    del retained, owner
    gc.collect()
    runtime.wait_for_retired_memmap_unlinks(path=raw_path)
    assert not raw_path.exists()
    if archive:
        assert archived[0].is_dir()


def test_keep_temp_retains_raw_volume_when_archive_fails(tmp_path: Path) -> None:
    raw_path = tmp_path / 'retained.dat'
    owner = np.memmap(raw_path, mode='w+', dtype=np.uint8, shape=(4,))
    owner[:] = 3
    with mock.patch.object(cuda_d1, 'write_raw_bbox_mask_store', side_effect=OSError('archive failed')):
        cuda_d1.archive_or_delete_binary_volume_storage(
            owner, keep_temp=True, workers=1, desc='retained scratch accumulator',
        )
    del owner
    gc.collect()
    assert raw_path.read_bytes() == bytes([3]) * 4

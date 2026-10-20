"""Opt-in GPU execution check for the automatic selected-SAM publication route."""
from contextlib import closing

import numpy as np
import pytest

from XTA import assembly
from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT, RawBBoxMaskStore, write_raw_bbox_mask_store
from XTA.sam_sparse_shell_projection import project_shell_sparse_store
from tests.test_sam_sparse_shell_gpu_parity import (
    CASES, SOURCE_SHAPE, dense_store, pytestmark, real_gpu_owner, require_gpu_stage,
)


@pytest.mark.parametrize('view', [v for v in CASES if v.family == 'spherical' and not v.spherical_tilted_source], ids=lambda v: v.name)
@pytest.mark.parametrize('shape', [SOURCE_SHAPE, (9, 11, 13)])
def test_automatic_sam_spherical_publication_uses_cuda(tmp_path, monkeypatch, view, shape):
    data = np.zeros((view.num_slices, view.src_h, view.src_w), np.uint8)
    data[:, 1:-1, 1:-1] = 1
    source = tmp_path / 'input.cvol'
    write_raw_bbox_mask_store(data, source, format_name=INTERNAL_PACKED_CVOL_FORMAT, workers=1)
    oracle = project_shell_sparse_store(source, view, tmp_path / 'cpu.cvol', out_shape_tyx=shape, workers=1)
    monkeypatch.delenv('YOLO_TTA_GPU_SPHERICAL_BACKPROJECT', raising=False)
    monkeypatch.setattr('XTA.sam_sparse_shell_projection.project_shell_sparse_store',
                        lambda *a, **k: pytest.fail('Automatic CUDA route chose CPU-only adapter'))
    calls = require_gpu_stage(monkeypatch, view)
    work = tmp_path / 'work'
    work.mkdir()
    previous = assembly.final_source_output_shape()
    assembly.set_final_source_output_shape(shape)
    try:
        backing = assembly._sam_directional_source_backing(source, view=view, stage='sam_selected_forward',
            has_foreground=True, model_name='automatic-gpu', source='fullframe', pass_index=1,
            tile_config_id='', workers=1, _work_root=work)
        assert calls and sum(call['count'] for call in calls) == shape[0]
        # This unchanged CUDA kernel has an exact contract on these small lattice fixtures.
        np.testing.assert_array_equal(dense_store(backing['path']), dense_store(oracle['path']))
        with closing(RawBBoxMaskStore.open(source, mmap_payload=True)) as store:
            np.testing.assert_array_equal(np.stack([store.decode_slice(z) for z in range(len(data))]), data)
    finally:
        assembly.set_final_source_output_shape(previous)

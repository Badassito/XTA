"""Parallel TTA voting preserves sequential policy input and receipts."""
from dataclasses import replace
from pathlib import Path
from threading import Barrier, get_ident
from types import SimpleNamespace

import numpy as np

from XTA import confidence_evidence, reconciliation_runtime
from XTA.interpolation import (
    INTERNAL_PACKED_CVOL_FORMAT, NrrdLayerRef, RawBBoxMaskStore, write_raw_bbox_mask_store,
)
from XTA.reconciliation_policy import resolve_reconciliation


def test_packed_group_votes_overlap_and_preserve_callback_blocks(tmp_path, monkeypatch):
    shape = (17, 12, 140)
    settings = resolve_reconciliation(SimpleNamespace(
        reconciliation=Path('XTA/examples/external_reconciliation/union.py'),
        reconciliation_memory_mib=128,
    ))
    refs, confidence_by_key = [], {}
    for index, view in enumerate(('transverse', 'transverse', 'sagittal', 'sagittal', 'coronal')):
        key = f'{view}{index}'
        mask = np.zeros(shape, np.uint8)
        mask[::index % 2 + 1, 2+index:10, 121+index:136] = 1
        mask[:, 6, 132] = 0
        if view == 'coronal':
            mask.fill(0)
        scores = np.zeros(shape, np.uint8)
        scores[:, 3:11, 119:134+index] = 210 - 4*index
        scores[:, 7, 130] = 0
        confidence_by_key[key] = confidence_evidence.write_block_confidence_evidence(
            tmp_path / f'{key}.scores', shape, scores.__getitem__, model_name='m', layer_key=key)
        path = tmp_path / f'{key}.cvol'
        write_raw_bbox_mask_store(mask, path, format_name=INTERNAL_PACKED_CVOL_FORMAT)
        refs.append(NrrdLayerRef(key=key, name=key, path=path, shape=shape, model_name='m',
                                view_name=view, physical_view_name=view, view_family='orthogonal',
                                source='fullframe', mask_kind='yolo',
                                storage_format=INTERNAL_PACKED_CVOL_FORMAT))
    monkeypatch.setattr(confidence_evidence, 'lookup_confidence_evidence',
                        lambda ref: confidence_by_key[ref.key])
    coordinator = get_ident()
    calls = []

    def decide(block):
        assert get_ident() == coordinator
        calls.append({key: val.copy() if isinstance(val, np.ndarray) else val
                      for key, val in block.items()})
        return block['anchored'] & (block['score'] >= 1.5)

    def execute(directory, execution_settings=settings):
        calls.clear()
        output, report = reconciliation_runtime.reconcile_tta_layers(
            refs, views=[], source_shape_tyx=shape, processing_shape_tyx=shape,
            policy=dict(mode='confidence', decide=decide), settings=execution_settings,
            output_dir=tmp_path / directory, workspace=tmp_path / (directory + '-workspace'))
        try:
            return output.copy(), report, calls.copy()
        finally:
            output._mmap.close()

    monkeypatch.setattr(reconciliation_runtime, '_cpu_count', lambda: 1)
    expected_mask, expected_report, expected_calls = execute('serial')
    assert len(expected_calls) == 2
    # A first read in each distinct group has to reach the barrier before
    # either can continue, proving the production runtime activates both lanes.
    monkeypatch.setattr(reconciliation_runtime, '_cpu_count', lambda: 8)
    barrier = Barrier(2, timeout=10)
    original = RawBBoxMaskStore.decode_slice_crop

    def read_crop(owner, z, **kwargs):
        assert owner.root.stem != 'coronal4'
        if z == 0 and owner.root.stem in {'transverse0', 'sagittal2'}:
            barrier.wait()
        return original(owner, z, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(RawBBoxMaskStore, 'decode_slice_crop', read_crop)
        mask, report, parallel_calls = execute('parallel')

    execution = report.pop('execution')
    assert execution['concurrent_vote_workers'] == 2
    assert report['group_count'] == 3
    assert report['layers']['m/coronal4']['foreground_voxels'] == 0
    assert report['planned_working_bytes'] <= settings.memory_mib * 1024**2
    assert report['planned_working_bytes'] - execution['parallel_vote_working_bytes'] == expected_report['planned_working_bytes']
    report['planned_working_bytes'] = expected_report['planned_working_bytes']
    assert report == expected_report
    np.testing.assert_array_equal(mask, expected_mask)
    for actual, expected in zip(parallel_calls, expected_calls, strict=True):
        assert actual.keys() == expected.keys()
        for key in actual:
            np.testing.assert_array_equal(actual[key], expected[key])

    constrained_mask, constrained_report, _ = execute('constrained', replace(settings, memory_mib=8))
    assert 'execution' not in constrained_report
    np.testing.assert_array_equal(constrained_mask, expected_mask)

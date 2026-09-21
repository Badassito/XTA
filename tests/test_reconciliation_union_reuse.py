"""Borrowed union identity, lazy evidence, and postprocessing ownership checks."""
from __future__ import annotations

import ast
from dataclasses import replace
import gc
import json
from pathlib import Path
from types import SimpleNamespace
from unittest import mock
import weakref

import numpy as np
import pytest

from XTA.interpolation import NrrdLayerRef
from XTA.reconciliation_policy import resolve_reconciliation
from XTA import reconciliation_runtime as runtime


def settings(tmp_path, mode='union'):
    path = tmp_path / 'policy_source.py'
    value = ("dict(mode='union', grouping='views', decide=lambda block: block['candidate'] & False)"
             if mode == 'custom' else
             f"dict(mode={mode!r}, grouping='views', threshold=.5, min_sources=1, min_prediction_sources=1)")
    path.write_text(f'def build_reconciliation():\n    return {value}\n', encoding='utf-8')
    return resolve_reconciliation(SimpleNamespace(reconciliation=path, reconciliation_memory_mib=8))


def layer(tmp_path, name, shape, **kwargs):
    return NrrdLayerRef(key=name, name=name, path=tmp_path / f'{name}.unopened', shape=shape,
        model_name='model', view_name=name, physical_view_name=name, view_family='orthogonal',
        source=kwargs.pop('source', 'fullframe'), mask_kind=kwargs.pop('mask_kind', 'yolo'), **kwargs)


def run(refs, array, cfg, tmp_path, **kwargs):
    return runtime.reconcile_tta_layers(refs, views=[], source_shape_tyx=array.shape,
        processing_shape_tyx=array.shape, settings=cfg, output_dir=tmp_path / 'published',
        workspace=tmp_path / 'work', assembled_union=array, **kwargs)


@pytest.mark.parametrize('mapped', [False, True])
def test_union_reuses_exact_array_without_opening_components_scores_or_new_map(tmp_path, mapped):
    shape = (3, 5, 7)
    array = np.memmap(tmp_path / 'assembled.dat', mode='w+', dtype=np.uint8, shape=shape) if mapped else np.zeros(shape, np.uint8)
    array[:] = 0
    array[:, 1:4, 2:6] = 1
    original = array.copy()
    refs = [layer(tmp_path, 'prediction', shape),
            layer(tmp_path, 'tile', shape, source='tile', tile_config_id='grid', tile_acceptance='parent_bridge'),
            layer(tmp_path, 'bridge', shape, mask_kind='bridge', pass_index=1),
            layer(tmp_path, 'empty', shape, segment_extent_ijk=(0, -1, 0, -1, 0, -1)),
            layer(tmp_path, 'global', shape, source='global', layer_role='checkpoint', recomposition_op='select')]
    deferred = SimpleNamespace(path=tmp_path / 'native_evidence', shape=shape,
        storage_shape=(5, 4, 9), coordinate_space='native_view_processing',
        reader=mock.Mock(side_effect=AssertionError('source projection must remain deferred')))
    cfg = settings(tmp_path)
    # Missing component files are intentional: successful reuse proves no payload open.
    with (mock.patch.object(runtime, 'RuntimeLayer', side_effect=AssertionError('opened component')),
          mock.patch.object(runtime, 'reconcile', side_effect=AssertionError('rebuilt union')),
          mock.patch.object(runtime.np, 'memmap', side_effect=AssertionError('allocated output map')),
          mock.patch('XTA.confidence_evidence.lookup_confidence_evidence', return_value=deferred)):
        result, report = run(refs + [refs[0]], array, cfg, tmp_path)
    assert result is array
    np.testing.assert_array_equal(array, original)
    assert report['layer_count'] == 4
    assert report['execution']['strategy'] == 'reuse_assembled_union'
    assert report['execution']['new_source_volume_bytes'] == 0
    assert report['execution']['component_payload_reads'] == 0
    assert report['counts']['candidate_voxels'] == report['counts']['retained_voxels'] == int(original.sum())
    assert report['counts']['rejected_voxels'] == 0
    assert report['layers']['model/prediction']['foreground_voxels'] is None
    assert report['layers']['model/empty']['foreground_voxels'] == 0
    assert report['counts']['confidence_known_voxels'] is None
    assert report['layers']['model/tile']['metadata']['tile_acceptance'] == 'parent_bridge'
    assert report['layers']['model/bridge']['role'] == 'bridge'
    assert report['layers']['model/prediction']['metadata']['confidence_coordinate_space'] == 'native_view_processing'
    assert not (tmp_path / 'work').exists()
    assert (tmp_path / 'published/policy.py').read_bytes() == Path(cfg.path).read_bytes()
    assert json.loads((tmp_path / 'published/manifest.json').read_text()) == report
    deferred.reader.assert_not_called()
    # The ordinary postprocessing tail can still modify the same live buffer.
    array[0, 0, 0] = 1
    assert report['counts']['retained_voxels'] == int(original.sum())
    if mapped:
        assert not array._mmap.closed
        array.flush()
        array._mmap.close()


def test_union_counting_is_binary_checked_and_row_bounded(tmp_path):
    array = np.zeros((2, 10, 16), np.uint8)
    array[:, ::2, ::2] = 1
    cfg = replace(settings(tmp_path), memory_mib=64 / 1024**2)
    count = np.count_nonzero
    windows = []
    def counting(value):
        windows.append(value.shape)
        return count(value)
    with mock.patch.object(runtime.np, 'count_nonzero', side_effect=counting):
        result, report = run([], array, cfg, tmp_path)
    assert result is array
    assert windows == [(4, 16), (4, 16), (2, 16)] * 2
    assert report['planned_working_bytes'] == 64
    assert report['counts']['retained_voxels'] == int(array.sum())
    assert report['execution']['count_windows'] == len(windows)
    array[-1, -1, -1] = 2
    with pytest.raises(ValueError, match='binary 0/1'):
        run([], array, cfg, tmp_path)


@pytest.mark.parametrize('mode', ['weighted', 'confidence', 'custom'])
def test_nontrivial_policies_use_component_reconciliation_instead_of_borrowed_union(tmp_path, mode):
    borrowed = np.ones((2, 3, 5), np.uint8)
    mask = np.zeros_like(borrowed)
    mask[:, 1, 1:4] = 1
    ref = layer(tmp_path, 'prediction', mask.shape)
    mask.tofile(ref.path)
    confidence = SimpleNamespace(path=tmp_path / 'scores', shape=mask.shape,
        reader=mock.Mock(return_value=lambda a, b: (np.full_like(mask[a:b], 230), mask[a:b].astype(bool))))
    if mode != 'confidence':
        confidence.shape = (99, 88, 77)  # Native deferred shape is not read or validated.
        confidence.coordinate_space = 'native_view_processing'
        confidence.reader.side_effect = AssertionError('unneeded confidence reader')
    cfg = settings(tmp_path, mode)
    with mock.patch('XTA.confidence_evidence.lookup_confidence_evidence', return_value=confidence):
        result, report = run([ref], borrowed, cfg, tmp_path)
    try:
        assert result is not borrowed
        np.testing.assert_array_equal(result, np.zeros_like(mask) if mode == 'custom' else mask)
        assert borrowed.all()
        assert report.get('execution', {}).get('strategy') != 'reuse_assembled_union'
        assert confidence.reader.call_count == int(mode == 'confidence')
    finally:
        result._mmap.close()


def test_policy_change_and_manifest_failure_never_close_borrowed_mmap(tmp_path):
    array = np.memmap(tmp_path / 'assembled.dat', mode='w+', dtype=np.uint8, shape=(2, 3, 4))
    array[:] = 0
    cfg = settings(tmp_path)
    with mock.patch.object(runtime, '_publish_reconciliation_report', side_effect=OSError('publication failure')):
        with pytest.raises(OSError, match='publication failure'):
            run([], array, cfg, tmp_path)
    assert not array._mmap.closed
    Path(cfg.path).write_text(Path(cfg.path).read_text() + '# changed\n')
    with pytest.raises(RuntimeError, match='policy changed'):
        run([], array, cfg, tmp_path, policy=dict(mode='union'))
    assert not array._mmap.closed
    array._mmap.close()


@pytest.mark.parametrize('fail', [False, True])
def test_confidence_reader_owners_close_on_success_and_vote_failure(tmp_path, fail):
    shape = (2, 3, 4)
    mask = np.ones(shape, np.uint8)
    refs = [layer(tmp_path, name, shape) for name in ('one', 'two')]
    for ref in refs:
        mask.tofile(ref.path)
    readers = []
    class Reader:
        def __init__(self):
            self.closed = 0
            readers.append(self)
        def __call__(self, first, stop):
            assert not self.closed
            return np.full_like(mask[first:stop], 230), np.ones_like(mask[first:stop], bool)
        def close(self):
            self.closed += 1
    confidence = SimpleNamespace(path=tmp_path / 'scores', shape=shape, reader=Reader)
    cfg = settings(tmp_path, 'confidence')
    with mock.patch('XTA.confidence_evidence.lookup_confidence_evidence', return_value=confidence):
        if fail:
            with mock.patch.object(runtime, 'reconcile', side_effect=RuntimeError('vote failure')):
                with pytest.raises(RuntimeError, match='vote failure'):
                    run(refs, mask, cfg, tmp_path)
            assert not (tmp_path / 'work/reconciled_union.u8.dat').exists()
        else:
            result, _ = run(refs, mask, cfg, tmp_path)
            np.testing.assert_array_equal(result, mask)
            result._mmap.close()
    assert len(readers) == 2
    assert [reader.closed for reader in readers] == [1, 1]


@pytest.mark.parametrize('dtype,shape', [(np.float32, (2, 3, 4)), (np.uint8, (1, 3, 4))])
def test_borrowed_union_requires_the_declared_binary_source_grid(tmp_path, dtype, shape):
    array = np.zeros(shape, dtype=dtype)
    with pytest.raises(ValueError, match='source grid'):
        runtime.reconcile_tta_layers([], views=[], source_shape_tyx=(2, 3, 4),
            processing_shape_tyx=(2, 3, 4), settings=settings(tmp_path),
            output_dir=tmp_path / 'published', workspace=tmp_path / 'work', assembled_union=array)
    assert not (tmp_path / 'published/manifest.json').exists()


def _pipeline_reconciliation_block():
    source = Path(__file__).resolve().parents[1] / 'XTA/pipeline.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and any(isinstance(child, ast.ImportFrom) and child.module == 'reconciliation_runtime'
                       and any(alias.name == 'reconcile_tta_layers' for alias in child.names)
                       for child in node.body)]
    assert len(matches) == 1
    return compile(ast.Module(body=matches, type_ignores=[]), '<pipeline-reconciliation-tail>', 'exec')


@pytest.mark.parametrize('reuse', [False, True])
def test_pipeline_tail_updates_all_known_union_owners_and_preserves_reused_buffer(tmp_path, reuse):
    original = np.zeros((2, 3, 4), np.uint8)
    weak_original = weakref.ref(original)
    replacement = original if reuse else np.ones_like(original)
    closed = []
    namespace = dict(__name__='XTA.pipeline', __package__='XTA', reconciliation_settings=SimpleNamespace(enabled=True),
        nrrd_layer_sink=lambda: None, final_union_mm=original, streamed_final_union_mm=original,
        streaming_final_union_holder={'model': original}, model_name='model',
        nrrd_layer_refs=[], inference_views=[], source_output_shape_tyx=original.shape,
        T=2, H=3, W=4, reconciliation_policy={}, out_dir=tmp_path, temp_dir=tmp_path,
        close_memmap_array=lambda value: closed.append(id(value)), print=lambda *args: None)
    saved = runtime.reconcile_tta_layers
    try:
        runtime.reconcile_tta_layers = lambda *args, **kwargs: (replacement, {})
        exec(_pipeline_reconciliation_block(), namespace)
    finally:
        runtime.reconcile_tta_layers = saved
    assert namespace['final_union_mm'] is replacement
    assert namespace['streamed_final_union_mm'] is replacement
    assert namespace['streaming_final_union_holder']['model'] is replacement
    assert namespace['original_union'] is None
    assert len(closed) == int(not reuse)
    if not reuse:
        original = None
        gc.collect()
        assert weak_original() is None

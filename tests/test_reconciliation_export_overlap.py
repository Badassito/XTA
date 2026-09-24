"""Delayed editable-layer exports remain intact while union finalization advances."""
from __future__ import annotations

import ast
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import replace
import gzip
import gc
import json
import os
from pathlib import Path
import threading
import weakref
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import outputs, reconciliation_runtime as runtime
from XTA.interpolation import CVOL_FORMAT, NrrdLayerRef, write_raw_bbox_mask_store
from XTA.reconciliation_policy import load_reconciliation_policy, resolve_reconciliation


@pytest.fixture
def component(tmp_path, monkeypatch):
    def bbox(value):
        y, x = np.nonzero(value)
        return ((int(x.min()), int(y.min()), int(x.max() - x.min() + 1),
                 int(y.max() - y.min() + 1)) if len(x) else (0, 0, 0, 0))
    monkeypatch.setattr('XTA.interpolation.cv2.boundingRect', bbox)
    # Keep this CPU test independent of optional codec discovery and global probes.
    monkeypatch.setattr(outputs, '_require_nrrd_member_codec',
                        lambda **kwargs: ('zlib', 1, lambda data: gzip.compress(data, mtime=0)))
    monkeypatch.setenv('YOLO_TTA_NRRD_LAYER_ZSHARDS', '1')
    array = np.zeros((3, 5, 7), np.uint8)
    array[:, 1:4, 2:6] = 1
    path = tmp_path / 'component.cvol'
    write_raw_bbox_mask_store(array, path, format_name=CVOL_FORMAT, workers=1, desc='overlap')
    ref = NrrdLayerRef(key='component', name='component', path=path, shape=array.shape,
        storage_format=CVOL_FORMAT, model_name='model', view_name='transverse',
        view_family='orthogonal', source='fullframe', mask_kind='yolo')
    return array, ref


def policy_settings(tmp_path, custom=False):
    source = tmp_path / 'selected_policy.py'
    decide = ", decide=lambda block: block['candidate']" if custom else ''
    source.write_text("def build_reconciliation():\n"
                      f"    return dict(mode='union', grouping='views'{decide})\n")
    settings = resolve_reconciliation(SimpleNamespace(reconciliation=source, reconciliation_memory_mib=8))
    return settings, load_reconciliation_policy(settings)


def pipeline_block():
    tree = ast.parse((Path(__file__).resolve().parents[1] / 'XTA/pipeline.py').read_text(encoding='utf-8'))
    matches = [node for node in ast.walk(tree) if isinstance(node, ast.If)
               and any(isinstance(child, ast.ImportFrom) and child.module == 'reconciliation_runtime'
                       and any(alias.name == 'reconcile_tta_layers' for alias in child.names)
                       for child in node.body)]
    assert len(matches) == 1
    return compile(ast.Module(body=matches, type_ignores=[]), '<pipeline-reconciliation-tail>', 'exec')


def execute_pipeline_block(tmp_path, sink, array, ref, cfg, policy):
    namespace = dict(__name__='XTA.pipeline', __package__='XTA', reconciliation_settings=cfg,
        reconciliation_policy=policy, nrrd_layer_sink=lambda: sink,
        runtime_telemetry=lambda: SimpleNamespace(gauge=lambda *args: None),
        final_union_mm=array, streamed_final_union_mm=array,
        streaming_final_union_holder={'model': array}, model_name='model',
        nrrd_layer_refs=[ref], inference_views=[], source_output_shape_tyx=array.shape,
        T=array.shape[0], H=array.shape[1], W=array.shape[2], out_dir=tmp_path, temp_dir=tmp_path,
        close_memmap_array=lambda value: None, print=lambda *args: None)
    exec(pipeline_block(), namespace)
    return namespace


@pytest.mark.parametrize('failure', [False, True])
def test_delayed_independent_export_allows_union_mutation_and_keeps_final_failure_join(
        tmp_path, monkeypatch, component, failure):
    original, ref = component
    assembled = original.copy()
    cfg, policy = policy_settings(tmp_path)
    entered, release = threading.Event(), threading.Event()
    writer = outputs.write_single_layer_nrrd_from_ref
    def delayed(*args, **kwargs):
        entered.set()
        assert release.wait(5), 'test did not release delayed export'
        if failure:
            raise OSError('late component export failure')
        return writer(*args, **kwargs)
    monkeypatch.setattr(outputs, 'write_single_layer_nrrd_from_ref', delayed)
    sink = outputs.NrrdLayerSink(nrrd_dir=tmp_path / 'nrrd', stem='test',
        output_shape_tyx=original.shape, max_workers=1)
    before = {path.name: path.read_bytes() for path in ref.path.iterdir()}
    try:
        destination = sink.submit_layer(ref, 'component')
        assert entered.wait(5)
        assert sink.pending_layer_refs() == (ref,)
        # Run the actual pipeline reconciliation block while the writer cannot
        # finish. It must return the same array without allocating a new union.
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(execute_pipeline_block, tmp_path, sink, assembled, ref, cfg, policy)
            try:
                result = future.result(timeout=3)
                assert result['final_union_mm'] is assembled
                assembled[:] = 0  # in-place final keep_objects/other cleanup
            finally:
                release.set()
        if failure:
            with pytest.raises(RuntimeError, match='Single-layer NRRD writing failed'):
                sink.wait()
            assert not destination.exists()
        else:
            sink.wait()
            payload = destination.read_bytes().split(b'\n\n', 1)[1]
            assert gzip.decompress(payload) == original.tobytes()
            assert sink.write_manifest().is_file()
        assert sink.pending_layer_refs() == ()
        assert {path.name: path.read_bytes() for path in ref.path.iterdir()} == before
    finally:
        release.set()
        sink.shutdown()
    assert not sink._source_refs


@pytest.mark.parametrize('custom', [False, True])
def test_live_alias_or_custom_policy_keeps_export_barrier(tmp_path, monkeypatch, component, custom):
    original, raw_ref = component
    assembled = original.copy()
    ref = raw_ref if custom else replace(raw_ref, storage_format='live_u8', live_array=assembled,
        path=tmp_path / 'never-created.live', segment_extent_ijk=(2, 5, 1, 3, 0, 2),
        segment_extent_shape_tyx=assembled.shape)
    cfg, policy = policy_settings(tmp_path, custom=custom)
    entered, release, wait_called = threading.Event(), threading.Event(), threading.Event()
    observed = []
    def delayed(ref, shape, destination, **kwargs):
        entered.set()
        assert release.wait(5)
        observed.append((original if custom else ref.live_array).copy())
        destination.write_bytes(b'completed test export')
        return destination
    monkeypatch.setattr(outputs, 'write_single_layer_nrrd_from_ref', delayed)
    sink = outputs.NrrdLayerSink(nrrd_dir=tmp_path / 'nrrd', stem='test',
                                output_shape_tyx=assembled.shape, max_workers=1)
    wait = sink.wait
    def observed_wait():
        wait_called.set()
        return wait()
    monkeypatch.setattr(sink, 'wait', observed_wait)
    try:
        sink.submit_layer(ref, 'alias')
        assert entered.wait(5)
        with ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(execute_pipeline_block, tmp_path, sink, assembled, ref, cfg, policy)
            try:
                assert wait_called.wait(3)
                assert not future.done()
            finally:
                release.set()
            result = future.result(timeout=3)
        if custom:
            assert result['final_union_mm'] is not assembled
            result['final_union_mm']._mmap.close()
        assembled[:] = 0
        np.testing.assert_array_equal(observed[0], original)
    finally:
        release.set()
        sink.shutdown()


def test_policy_and_store_guards_conservatively_keep_export_barrier(tmp_path, component):
    original, ref = component
    refs = [ref]
    sink = SimpleNamespace(pending_layer_refs=lambda: tuple(refs))
    def allowed(array=original, policy=None):
        return runtime.union_nrrd_exports_can_overlap(sink, array,
            policy=policy or dict(mode='union'), source_shape_tyx=original.shape)
    assert allowed()
    assert not allowed(policy=dict(mode='weighted'))
    assert not allowed(policy=dict(mode='confidence'))
    assert not allowed(policy=dict(mode='union', decide=lambda block: block['candidate']))
    refs[:] = [replace(ref, storage_format='raw_u8')]
    assert not allowed()
    refs[:] = [replace(ref, live_array=original)]
    assert not allowed()
    refs[:] = [ref]
    external = np.frombuffer(bytearray(original.nbytes), np.uint8).reshape(original.shape)
    assert not allowed(array=external)
    metadata = ref.path / 'meta.json'
    saved = metadata.read_text()
    value = json.loads(saved)
    value['stats']['raw_payload_bytes'] += 1
    metadata.write_text(json.dumps(value))
    assert not allowed()
    metadata.write_text(saved)
    assert allowed()
    with pytest.raises(ValueError, match='Unknown reconciliation policy fields'):
        allowed(policy=dict(mode='union', surprise=True))


def test_mapped_union_backing_alias_is_detected_even_through_hardlink(tmp_path, component):
    original, ref = component
    # A full-canvas all-one store makes its payload a valid same-shape raw map.
    write_raw_bbox_mask_store(np.ones_like(original), ref.path, format_name=CVOL_FORMAT,
                              workers=1, desc='alias store')
    alias = tmp_path / 'union-hardlink.dat'
    os.link(ref.path / 'chunks.bin', alias)
    union = np.memmap(alias, mode='r+', shape=original.shape, dtype=np.uint8)
    independent = np.memmap(tmp_path / 'independent.dat', mode='w+', shape=original.shape, dtype=np.uint8)
    sink = SimpleNamespace(pending_layer_refs=lambda: (ref,))
    try:
        for array, expected in ((union, False), (np.asarray(union), False),
                                (independent, True), (np.asarray(independent), True)):
            assert runtime.union_nrrd_exports_can_overlap(sink, array, policy=dict(mode='union'),
                source_shape_tyx=original.shape) is expected
    finally:
        union._mmap.close()
        independent._mmap.close()


def test_overlap_does_not_bypass_policy_file_identity(tmp_path, component):
    original, ref = component
    cfg, policy = policy_settings(tmp_path)
    sink = SimpleNamespace(pending_layer_refs=lambda: (ref,),
                           wait=mock.Mock(side_effect=AssertionError('independent export should overlap')))
    Path(cfg.path).write_text(Path(cfg.path).read_text() + '# changed\n')
    with pytest.raises(RuntimeError, match='policy changed'):
        execute_pipeline_block(tmp_path, sink, original.copy(), ref, cfg, policy)
    assert not (tmp_path / 'reconciliation/manifest.json').exists()


@pytest.mark.parametrize('immediate,failure', [(True, False), (True, True), (False, False), (False, True)])
def test_finished_export_releases_tracked_source_without_snapshot_or_shutdown(tmp_path, immediate, failure):
    """An executor that does not retain the task isolates source-tracking ownership."""
    future = Future()
    sink = outputs.NrrdLayerSink(nrrd_dir=tmp_path / 'nrrd', stem='test',
                                output_shape_tyx=(2, 3, 4), max_workers=1)
    real_executor = sink.executor
    real_executor.shutdown(wait=True)
    def submit(_task):
        if immediate:
            if failure:
                future.set_exception(OSError('test export failure'))
            else:
                future.set_result(tmp_path / 'completed.nrrd')
        return future
    sink.executor = SimpleNamespace(submit=submit, shutdown=lambda **kwargs: None)
    array = np.ones((2, 3, 4), np.uint8)
    source = NrrdLayerRef(key='live', name='live', path=tmp_path / 'live.placeholder',
                         shape=array.shape, storage_format='live_u8', live_array=array)
    array_ref, layer_ref = weakref.ref(array), weakref.ref(source)
    try:
        # A separate daemon detects the immediate-completion lock regression
        # without making a broken implementation hang the entire test process.
        submitted, errors = threading.Event(), []
        def issue():
            try:
                sink.submit_layer(source, 'live')
            except BaseException as exc:
                errors.append(exc)
            finally:
                submitted.set()
        thread = threading.Thread(target=issue, daemon=True)
        thread.start()
        assert submitted.wait(3), 'completion callback deadlocked under the sink lock'
        thread.join()
        assert not errors
        source = array = None
        if not immediate:
            gc.collect()
            assert layer_ref() is not None and array_ref() is not None
            if failure:
                future.set_exception(OSError('test export failure'))
            else:
                future.set_result(tmp_path / 'completed.nrrd')
        # Never call pending_layer_refs() or shutdown() before this assertion:
        # release must be caused by completion itself, including failed futures.
        gc.collect()
        assert not sink._source_refs
        assert layer_ref() is None and array_ref() is None
        sink._release_completed_source(future)  # idempotent under snapshot races
        if failure:
            with pytest.raises(RuntimeError, match='Single-layer NRRD writing failed'):
                sink.wait()
        else:
            sink.wait()
    finally:
        if submitted.is_set():
            sink.shutdown()

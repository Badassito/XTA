"""View-major mask publication preserves the scalar native/LQ contract."""
from __future__ import annotations

from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import gzip
import json
from pathlib import Path
import threading
from unittest import mock

import numpy as np
import pytest

from XTA import nrrd_compression, outputs
from XTA.interpolation import NrrdLayerRef
from XTA.tta_sweep_branch_layers import TtaSweepBranchLayers, scalar_layer_suffix


@pytest.fixture
def scalar_codec(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_NRRD_GPU_MIRROR_TEE', '0')
    monkeypatch.setenv('YOLO_TTA_LOW_QUALITY_GPU_DOWNBIN', '0')
    with mock.patch.object(nrrd_compression, '_require_nrrd_member_codec',
                           return_value=('zlib', 1, gzip.compress)):
        yield


def _sink(root, *, run_id='cutoff-run', executor=None):
    spec = outputs.LowQualityDownbinSpec('0.5', '0p5', .5, (2, 3, 4))
    return outputs.NrrdLayerSink(nrrd_dir=root / 'nrrd', stem='case',
        output_shape_tyx=(3, 5, 7), max_workers=1, low_quality_specs=[spec],
        low_quality_root=root / 'low_quality', run_id=run_id, executor=executor)


def _ref(root, view='transverse', *, empty=False):
    mask = np.zeros((3, 5, 7), np.uint8)
    if not empty:
        mask[0, 0, 0] = 1
        mask[1, 1:4, 2:6] = 1
    return NrrdLayerRef(key=view + '_fullframe_yolo', name='layer',
        path=root / (view + '.live'), shape=mask.shape, storage_format='live_u8',
        live_array=mask, model_name='detector', view_name=view,
        physical_view_name=view, aug_id='a0', view_family='orthogonal',
        source='fullframe', mask_kind='yolo')


def test_completed_record_is_exact_scalar_native_and_lq_metadata(tmp_path, scalar_codec):
    sink = _sink(tmp_path)
    ref = _ref(tmp_path)
    suffix = scalar_layer_suffix(ref)
    try:
        path = sink.submit_layer(ref, suffix)
        sink.wait()
        record = sink.published_layer_record(ref, suffix)
        assert record['schema'] == 'xta.nrrd_layer_publication/1'
        assert record['run_id'] == 'cutoff-run'
        assert record['stem'] == 'case'
        assert record['output_shape_tyx'] == [3, 5, 7]
        assert record['native']['filename'] == path.name
        assert record['native']['empty_segment'] is False
        assert record['native']['physical_view_name'] == 'transverse'
        assert record['allocator']['requested_suffix'] == suffix
        assert record['allocator']['occurrence'] == 1
        assert record['files']['native']['size_bytes'] == path.stat().st_size
        assert record['files']['low_quality']['0p5']['size_bytes'] > 0
        assert (tmp_path / 'low_quality' / '0p5' / 'nrrd' / path.name).is_file()
        manifest = sink.write_manifest()
        assert record['native'] == json.loads(manifest.read_text())['layers'][0]
        mirror_manifest = tmp_path / 'low_quality' / '0p5' / 'nrrd' / manifest.name
        assert record['low_quality']['0p5'] == json.loads(mirror_manifest.read_text())['layers'][0]
        record['native']['segment_color_rgb'][0] = -1
        assert sink.published_layer_record(ref, suffix)['native']['segment_color_rgb'][0] >= 0
    finally:
        sink.shutdown()


def test_view_barrier_publishes_native_and_lq_before_next_view_without_floor_rewrite(tmp_path, scalar_codec):
    sinks = {cent: _sink(tmp_path / str(cent), run_id=f'run-{cent}') for cent in [15, 60, 90]}
    publisher = TtaSweepBranchLayers({cent: sink for cent, sink in sinks.items() if cent != 15},
        primary_cutoff=15, max_pending_layers=1)
    callbacks = []
    first = {cent: _ref(tmp_path / str(cent)) if cent != 90 else None for cent in sinks}
    sinks[15].submit_layer(first[15], scalar_layer_suffix(first[15]))

    def completed(cent, view, records):
        for record in records:
            native = record['native']
            assert (sinks[cent].nrrd_dir / native['filename']).is_file()
            for token, row in record['low_quality'].items():
                assert (sinks[cent].low_quality_root / token / 'nrrd' / row['filename']).is_file()
        assert not (sinks[cent].nrrd_dir / 'case_nrrd_manifest.json').exists()
        callbacks.append((cent, view, records))

    try:
        records = publisher.publish_view('transverse', {cent: ref for cent, ref in first.items() if cent != 15},
            view_token='transverse', completed=completed, primary=(sinks[15], first[15]))
        assert len(callbacks) == 3 and records[90] == ()
        assert sinks[15].layer_count() == 1
        assert sinks[60].layer_count() == 1
        assert sinks[90].layer_count() == 0
        assert all(sink.progress_counts()[0] == sink.progress_counts()[1] for sink in sinks.values())

        second = {cent: _ref(tmp_path / str(cent), 'coronal') for cent in sinks}
        sinks[15].submit_layer(second[15], scalar_layer_suffix(second[15]))
        publisher.publish_view('coronal', {cent: ref for cent, ref in second.items() if cent != 15},
            view_token='coronal', completed=completed, primary=(sinks[15], second[15]))
        assert [sink.layer_count() for sink in sinks.values()] == [2, 2, 1]
        assert not sinks[15].pending_layer_refs()
        with pytest.raises(ValueError, match='more than once'):
            publisher.publish_view('coronal', {60: second[60]}, view_token='coronal', completed=completed)
    finally:
        publisher.shutdown()
        # The publisher never owns the pipeline's primary sink executor.
        assert not sinks[15].executor._shutdown
        sinks[15].shutdown()


def test_explicit_empty_ref_has_stock_geometry_and_global_identity(tmp_path, scalar_codec):
    sinks = {15: _sink(tmp_path / 'floor'), 95: _sink(tmp_path / 'high')}
    publisher = TtaSweepBranchLayers({95: sinks[95]}, primary_cutoff=15)
    ref = replace(_ref(tmp_path, empty=True), key='global_final', source='global',
        view_name='global', physical_view_name='', view_family='global',
        mask_kind='union', stage='final_output', layer_role='checkpoint',
        recomposition_op='select', low_quality_recomposition_op='select')
    try:
        records = publisher.publish_view('global', {95: ref}, view_token='Global', completed=lambda *args: None)[95]
        assert len(records) == 1
        record = records[0]
        assert record['native']['suffix'] == 'Global_final_output'
        assert record['native']['empty_segment'] is True
        assert record['native']['output_shape_tyx'] == [3, 5, 7]
        # Cropped default: an empty checkpoint stores one zero voxel at the grid origin.
        assert record['native']['stored_shape_tyx'] == [1, 1, 1]
        assert record['native']['stored_offset_tyx'] == [0, 0, 0]
        assert record['low_quality']['0p5']['stored_shape_tyx'] == [1, 1, 1]
        assert record['native']['recomposition_op'] == 'select'
        assert record['low_quality']['0p5']['empty_segment'] is True
        assert record['low_quality']['0p5']['output_shape_tyx'] == [2, 3, 4]
        assert record['low_quality']['0p5']['recomposition_op'] == 'select'
    finally:
        publisher.shutdown()
        sinks[15].shutdown()


def test_record_requires_writer_completion_and_can_disambiguate_duplicates(tmp_path, scalar_codec):
    entered, release = threading.Event(), threading.Event()
    sink = _sink(tmp_path)
    ref = _ref(tmp_path)
    suffix = scalar_layer_suffix(ref)
    original = outputs.write_layer_nrrd_with_low_quality_mirrors

    def blocked(*args, **kwargs):
        entered.set()
        assert release.wait(10)
        return original(*args, **kwargs)

    try:
        with mock.patch.object(outputs, 'write_layer_nrrd_with_low_quality_mirrors', side_effect=blocked):
            first = sink.submit_layer(ref, suffix)
            assert entered.wait(10)
            with pytest.raises(RuntimeError, match='pending'):
                sink.published_layer_record(ref, suffix)
            release.set()
            sink.wait()
        second = sink.submit_layer(ref, suffix)
        sink.wait()
        with pytest.raises(ValueError, match='ambiguous'):
            sink.published_layer_record(ref, suffix)
        assert sink.published_layer_record(ref, suffix, output_path=first)['allocator']['occurrence'] == 1
        record = sink.published_layer_record(ref, suffix, output_path=second)
        assert record['allocator']['occurrence'] == 2
        assert record['native']['suffix'] == suffix + '_02'
    finally:
        release.set()
        sink.shutdown()


class _PendingSink:
    def __init__(self, counters, *, fail=False):
        self.counters, self.fail = counters, fail
        self.pending = 0
        self.waits = 0
        self.closed = False

    def submit_layer(self, ref, suffix):
        self.pending += 1
        self.counters['pending'] += 1
        self.counters['peak'] = max(self.counters['peak'], self.counters['pending'])
        return Path(suffix)

    def progress_counts(self):
        return 0, self.pending

    def wait(self):
        self.waits += 1
        self.counters['pending'] -= self.pending
        self.pending = 0
        if self.fail:
            raise OSError('injected writer failure')

    def published_layer_record(self, ref, suffix, **kwargs):
        return {'native': {'filename': suffix}, 'low_quality': {}}

    def wait_published_layer(self, ref, suffix, **kwargs):
        self.wait()
        return self.published_layer_record(ref, suffix, **kwargs)

    def shutdown(self):
        self.closed = True


def test_pending_limit_bounds_all_cutoffs_and_failure_drains_every_sink(tmp_path):
    counters = dict(pending=0, peak=0)
    sinks = {cent: _PendingSink(counters) for cent in [10, 20, 30, 40]}
    publisher = TtaSweepBranchLayers({cent: sink for cent, sink in sinks.items() if cent != 10},
        primary_cutoff=10, max_pending_layers=1)
    try:
        publisher.publish_view('view', {cent: _ref(tmp_path) for cent in [20, 30, 40]},
            view_token='transverse', completed=lambda *args: None)
        assert counters['peak'] == 1
        assert counters['pending'] == 0
    finally:
        publisher.shutdown()

    counters = dict(pending=0, peak=0)
    sinks = {cent: _PendingSink(counters, fail=cent == 20) for cent in [10, 20, 30]}
    publisher = TtaSweepBranchLayers({cent: sink for cent, sink in sinks.items() if cent != 10}, primary_cutoff=10)
    with pytest.raises(OSError, match='injected'):
        publisher.publish_view('view', {cent: _ref(tmp_path) for cent in [20, 30]},
            view_token='transverse', completed=lambda *args: None)
    assert all(sinks[cent].waits > 0 and sinks[cent].pending == 0 for cent in [20, 30])
    with pytest.raises(RuntimeError, match='failed'):
        publisher.publish_view('another', {30: _ref(tmp_path)}, view_token='transverse', completed=lambda *args: None)
    with pytest.raises(OSError, match='injected'):
        publisher.shutdown()
    assert all(sinks[cent].closed for cent in [20, 30])
    assert not sinks[10].closed


def test_callback_failure_preserves_error_and_drains_other_cutoffs(tmp_path):
    counters = dict(pending=0, peak=0)
    sinks = {cent: _PendingSink(counters) for cent in [10, 20, 30]}
    publisher = TtaSweepBranchLayers({cent: sink for cent, sink in sinks.items() if cent != 10}, primary_cutoff=10)
    failure = ValueError('journal failed')
    def failed_callback(*args):
        raise failure
    try:
        with pytest.raises(ValueError) as caught:
            publisher.publish_view('view', {cent: _ref(tmp_path) for cent in [20, 30]},
                view_token='transverse', completed=failed_callback)
        assert caught.value is failure
        assert counters['pending'] == 0
        assert all(sinks[cent].waits for cent in [20, 30])
    finally:
        publisher.shutdown()


def test_canonical_scalar_view_token_is_used_instead_of_runtime_ref_name(tmp_path, scalar_codec):
    sink = _sink(tmp_path)
    ref = replace(_ref(tmp_path), view_name='transverse_tta_0_runtime')
    publisher = TtaSweepBranchLayers({80: sink}, primary_cutoff=15)
    try:
        records = publisher.publish_view('view', {80: ref}, view_token='Transverse_TTA_a0',
            completed=lambda *args: None)[80]
        assert records[0]['native']['suffix'] == 'Transverse_TTA_a0_fullframe_yolo'
        assert records[0]['native']['view_name'] == 'transverse_tta_0_runtime'
    finally:
        publisher.shutdown()


def test_branch_payload_geometry_and_metadata_match_independent_scalar_writer(tmp_path, scalar_codec):
    branch = _sink(tmp_path / 'branch')
    baseline = _sink(tmp_path / 'scalar')
    publisher = TtaSweepBranchLayers({80: branch}, primary_cutoff=15)
    small = np.zeros((2, 3, 4), np.uint8)
    small[1, 1:, 1:3] = 1
    ref = replace(_ref(tmp_path), shape=small.shape, live_array=small)
    token = 'Transverse_TTA_a0'
    suffix = scalar_layer_suffix(ref, view_token=token)
    try:
        baseline_path = baseline.submit_layer(ref, suffix)
        expected = baseline.wait_published_layer(ref, suffix)
        actual = publisher.publish_view('view', {80: ref}, view_token=token,
            completed=lambda *args: None)[80][0]
        assert actual['native'] == expected['native']
        assert actual['low_quality'] == expected['low_quality']
        for scalar_path, branch_path in [
            (baseline_path, branch.nrrd_dir / actual['native']['filename']),
            (baseline.low_quality_root / '0p5' / 'nrrd' / expected['low_quality']['0p5']['filename'],
             branch.low_quality_root / '0p5' / 'nrrd' / actual['low_quality']['0p5']['filename']),
        ]:
            expected_header, expected_payload = scalar_path.read_bytes().split(b'\n\n', 1)
            actual_header, actual_payload = branch_path.read_bytes().split(b'\n\n', 1)
            assert actual_header == expected_header
            assert gzip.decompress(actual_payload) == gzip.decompress(expected_payload)
    finally:
        publisher.shutdown()
        baseline.shutdown()


def test_shared_executor_is_bounded_and_survives_individual_sink_shutdown(tmp_path, scalar_codec):
    pool = ThreadPoolExecutor(max_workers=1)
    first = _sink(tmp_path / 'first', executor=pool)
    second = _sink(tmp_path / 'second', executor=pool)
    ref = _ref(tmp_path)
    suffix = scalar_layer_suffix(ref)
    try:
        first.submit_layer(ref, suffix)
        first.shutdown()
        assert not first.pending_layer_refs()
        assert pool.submit(lambda: 'alive').result() == 'alive'
        second.submit_layer(ref, suffix)
        record = second.wait_published_layer(ref, suffix)
        assert (second.nrrd_dir / record['native']['filename']).exists()
        second.shutdown()
        assert len(pool._threads) == 1
    finally:
        pool.shutdown(wait=True)


def test_shared_executor_sink_shutdown_reports_failure_and_releases_readers(tmp_path, scalar_codec):
    pool = ThreadPoolExecutor(max_workers=1)
    sink = _sink(tmp_path, executor=pool)
    ref = _ref(tmp_path)
    try:
        with mock.patch.object(outputs, 'write_layer_nrrd_with_low_quality_mirrors',
                               side_effect=OSError('mirror publication failed')):
            sink.submit_layer(ref, scalar_layer_suffix(ref))
            with pytest.raises(RuntimeError, match='NRRD writing failed'):
                sink.shutdown()
        assert not sink.pending_layer_refs()
        assert pool.submit(lambda: 'alive').result() == 'alive'
    finally:
        pool.shutdown(wait=True)


def test_normal_view_barrier_uses_quiet_single_layer_fences(tmp_path, scalar_codec):
    sink = _sink(tmp_path)
    publisher = TtaSweepBranchLayers({80: sink}, primary_cutoff=15)
    try:
        with mock.patch.object(sink, 'wait', side_effect=AssertionError('historical full sink scan')):
            publisher.publish_view('view', {80: _ref(tmp_path)}, view_token='Transverse_TTA_a0',
                completed=lambda *args: None)
    finally:
        publisher.shutdown()


def test_scalar_observer_runs_once_after_native_and_all_mirrors_are_durable(tmp_path, scalar_codec):
    sink = _sink(tmp_path)
    ref = _ref(tmp_path)
    seen = []
    suffix = scalar_layer_suffix(ref)
    name = outputs.nrrd_layer_file_basename('case', suffix)
    def observe(published_ref):
        assert published_ref is ref
        assert (sink.nrrd_dir / name).is_file()
        assert (sink.low_quality_root / '0p5' / 'nrrd' / name).is_file()
        seen.append(published_ref.key)
    sink.set_layer_publication_observer(observe)
    try:
        sink.submit_layer(ref, suffix)
        sink.wait_published_layer(ref, suffix)
        assert seen == [ref.key]
    finally:
        sink.shutdown()


def test_failed_layer_does_not_notify_success_observer(tmp_path, scalar_codec):
    sink = _sink(tmp_path)
    observer = mock.Mock()
    sink.set_layer_publication_observer(observer)
    ref = _ref(tmp_path)
    try:
        with mock.patch.object(outputs, 'write_layer_nrrd_with_low_quality_mirrors',
                               side_effect=OSError('mirror failed')):
            sink.submit_layer(ref, scalar_layer_suffix(ref))
            with pytest.raises(RuntimeError, match='NRRD writing failed'):
                sink.wait()
        observer.assert_not_called()
    finally:
        sink.shutdown()


def test_observer_failure_propagates_through_existing_scalar_drain(tmp_path, scalar_codec):
    sink = _sink(tmp_path)
    failure = ValueError('floor progress publication failed')
    sink.set_layer_publication_observer(mock.Mock(side_effect=failure))
    ref = _ref(tmp_path)
    try:
        sink.submit_layer(ref, scalar_layer_suffix(ref))
        with pytest.raises(RuntimeError) as caught:
            sink.wait()
        assert caught.value.__cause__ is failure
        assert not sink.pending_layer_refs()
    finally:
        sink.shutdown()

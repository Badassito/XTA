"""Selected layer formats share the established raster and sweep handoff."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from XTA import nrrd_compression, outputs
from XTA.native_review_store import NativeReviewStore
from XTA.reconciliation_io import read_layer_manifest
from XTA.tta_sweep_view_publication import ViewMajorPublication, branch_run_id
from tests.test_nrrd_cropped_layers import SHAPE, _layers
from tests.test_tta_sweep_view_publication import SHAPE as SWEEP_SHAPE, publish


@pytest.fixture
def preview(monkeypatch):
    monkeypatch.setenv('YOLO_TTA_NRRD_GPU_MIRROR_TEE', '0')
    monkeypatch.setenv('YOLO_TTA_NRRD_LAYER_ZSHARDS', '1')
    monkeypatch.delenv('XTA_YOLO_SWEEP_XTAREVIEW_STAGE_ROOT', raising=False)
    codec = nrrd_compression._nrrd_member_codec_spec('zlib')
    monkeypatch.setattr(nrrd_compression, '_require_nrrd_member_codec', lambda **_: codec)
    return outputs.LowQualityDownbinSpec('0.5', '0p5', .5, (6, 15, 16))


def _sink(root, preview, *, full=(), cutoff=.405):
    return outputs.NrrdLayerSink(
        nrrd_dir=root / 'nrrd', stem='case', output_shape_tyx=SHAPE, max_workers=2,
        low_quality_specs=[preview] if preview is not None else [], low_quality_root=root / 'low_quality',
        run_id='run', full_quality_formats=full, cutoff=cutoff, source_path=Path('scan.mkv'),
        output_root=root, review_scratch_dir=root.parent / 'scratch',
    )


def _publish(sink, layers):
    try:
        for name, ref in layers:
            sink.submit_layer(ref, name)
        sink.wait()
    finally:
        sink.shutdown()
    return sink.write_manifest()


@pytest.mark.parametrize('formats', [('xtareview',), ('nrrd', 'xtareview')])
def test_preview_without_native_files_keeps_exact_legacy_raster(tmp_path, preview, formats):
    sources = _layers(tmp_path / 'source')
    baseline = _sink(tmp_path / 'baseline', preview, full=('nrrd',))
    _publish(baseline, sources)
    candidate = _sink(tmp_path / 'candidate', replace(preview, formats=formats))
    assert _publish(candidate, sources) is None

    paths = candidate.published_paths()
    grid = 'low_quality/0p5'
    assert set(paths) == {grid}
    assert set(paths[grid]) == set(formats)
    assert not candidate.nrrd_dir.exists()
    assert bool(list((tmp_path / 'candidate').rglob('*.seg.nrrd'))) == ('nrrd' in formats)
    reference_manifest = baseline.published_paths()[grid]['nrrd']
    with read_layer_manifest(reference_manifest, workspace=tmp_path, include_all_layers=True) as reference:
        originals = {layer.metadata['segment_name']: layer.read_slab(0, reference.shape_tyx[0])
                     for layer in reference}
    with NativeReviewStore(paths[grid]['xtareview']) as store:
        assert store.shape == (6, 15, 16)
        assert store.source_record['confidence_threshold_exact'] == .405
        assert store.source_record['source_stem'] == 'scan'
        assert len(store.layers) == len(originals)
        for lid, layer in store.layers.items():
            actual = store.read_region(lid, store.cutoffs[0], tuple((0, n) for n in store.shape))
            np.testing.assert_array_equal(actual, originals[layer['name']])


def test_sweep_adopts_preview_archive_layer_without_native_files(tmp_path, monkeypatch, preview):
    stage = tmp_path / 'stage'
    monkeypatch.setenv('XTA_YOLO_SWEEP_XTAREVIEW_STAGE_ROOT', str(stage))
    spec = replace(preview, formats=('xtareview',))
    root = tmp_path / 'branch'
    sources = dict(_layers(tmp_path / 'source'))
    detector = replace(sources['live'], stage='pre_interpolation')
    producer = _sink(root, spec, cutoff=.8)
    try:
        layer_token = producer.submit_layer(detector, 'detector')
        receipt = producer.wait_published_layer(detector, 'detector', output_path=layer_token)
    finally:
        producer.shutdown()
    assert receipt['files']['native'] is None
    assert not layer_token.exists()
    assert set(receipt['files']['review']) == {'low_quality/0p5'}

    consumer = _sink(root, spec, cutoff=.8)
    wrong_format = {**receipt, 'files': {**receipt['files'], 'review': {}}}
    with pytest.raises(ValueError, match='review payloads'):
        consumer.adopt_published_layer(detector, wrong_format)
    assert consumer.layer_count() == 0
    assert consumer.adopt_published_layer(detector, receipt) == layer_token
    _publish(consumer, [('global', sources['global'])])
    assert consumer.staged_review_grids() == ('low_quality/0p5',)
    assert consumer.published_paths() == {}
    assert not root.exists()
    with NativeReviewStore(stage / 'conf_080' / 'low_quality' / '0p5.xtareview') as store:
        assert store.cutoffs == (80,)
        assert sorted(layer['role'] for layer in store.layers.values()) == ['direct', 'global']


def test_native_archive_and_preview_nrrd_are_independent(tmp_path, preview):
    sources = _layers(tmp_path / 'sources')
    sink = _sink(tmp_path / 'mixed', replace(preview, formats=('nrrd',)), full=('xtareview',))
    assert _publish(sink, sources) is None
    products = sink.published_paths()
    assert set(products['native']) == {'xtareview'}
    assert set(products['low_quality/0p5']) == {'nrrd'}
    assert not sink.nrrd_dir.exists()
    assert not list((sink.output_root / 'low_quality').rglob('*.xtareview'))
    with NativeReviewStore(products['native']['xtareview']) as store:
        assert store.shape == SHAPE
        region = tuple((0, n) for n in SHAPE)
        mask = store.read_region(store.global_layer_id, store.cutoffs[0], region)
        np.testing.assert_array_equal(mask, dict(sources)['global'].live_array)


def test_eager_sweep_can_deliver_archive_only(tmp_path, monkeypatch, preview):
    from XTA import assembly
    from XTA.geometry import ViewInfo

    family, work = tmp_path / 'family', tmp_path / 'work'
    archive_stage = work / 'archives'
    monkeypatch.setenv('XTA_YOLO_SWEEP_XTAREVIEW_STAGE_ROOT', str(archive_stage))
    monkeypatch.setenv('XTA_YOLO_SWEEP_NATIVE_ROLE', 'produce')
    preview = replace(preview, output_shape_t_y_x=(2, 4, 4), formats=('xtareview',))

    def make_owner(cutoff):
        branch = family / 'confidence_cutoffs' / f'conf_{cutoff:03d}'
        sink = outputs.NrrdLayerSink(
            nrrd_dir=branch / 'nrrd', stem='case', output_shape_tyx=SWEEP_SHAPE, max_workers=2,
            low_quality_specs=[preview], low_quality_root=branch / 'low_quality',
            run_id=branch_run_id('test', cutoff), full_quality_formats=(), cutoff=cutoff/100,
        )
        owner = ViewMajorPublication(
            family_root=family, workspace=work, sweep_id='test', cutoffs_hundredths=(50, 70, 100),
            current_cutoff=cutoff, scalar_sink=sink, source_shape_tyx=SWEEP_SHAPE,
            model_name='detector', task='segment', min_conf=0., confidence_floor_by_backend={'gpu': .5},
        )
        return owner, sink

    view = ViewInfo(name='transverse__tta_a0', num_slices=3, src_h=8, src_w=8,
                    pad_mode='pad', family='orthogonal', physical_view_name='transverse',
                    full_t=3, full_h=8, full_w=8, tta_aug_id='a0')
    owner, floor = make_owner(50)
    try:
        refs = publish(owner, view, score_lanes=False)
        assert refs[70] is not None
        owner.finish_floor()
    finally:
        owner.close()
        floor.shutdown()

    consumer, sink = make_owner(70)
    try:
        assert consumer.adopt_view(view) == (refs[70], 6)
        monkeypatch.setattr(assembly, 'nrrd_layer_sink', lambda: sink)
        mask = np.zeros(SWEEP_SHAPE, np.uint8)
        mask[:, 2:6, 2:6] = 1
        assembly.materialize_nrrd_global_layer(
            mask, model_name='detector', source='global', mask_kind='union',
            stage='final_output_after_all_postprocessing', temp_dir=work, include_empty=True,
            volume_is_immutable=True,
        )
        sink.write_manifest()
        assert not list(family.rglob('*.nrrd'))
        path = archive_stage / 'conf_070' / 'low_quality' / '0p5.xtareview'
        with NativeReviewStore(path) as archive:
            assert len(archive.layers) == 2
            region = tuple((0, n) for n in archive.shape)
            layers = [archive.read_region(lid, 70, region) for lid in archive.layers]
            assert layers[0].any()
            np.testing.assert_array_equal(*layers)
    finally:
        sink.shutdown()
        consumer.close()

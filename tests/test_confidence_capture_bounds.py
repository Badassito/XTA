"""Trusted parent bounds avoid dense reads without changing score blocks."""
from pathlib import Path

import numpy as np
import pytest

from XTA import confidence_evidence
from XTA.confidence_evidence import (_MaskedNativeScoreReader, capture_prediction_confidence,
    configure_confidence_evidence, shutdown_confidence_publication, write_block_confidence_evidence)
from XTA.geometry import TiltedViewGroup, get_view_infos


@pytest.fixture(autouse=True)
def reset():
    configure_confidence_evidence(None,enabled=False)
    yield
    shutdown_confidence_publication(raise_errors=False)
    configure_confidence_evidence(None,enabled=False)


class GuardedScores:
    """Reject any access outside the declared active crop, including whole slices."""
    def __init__(self, values, selection):
        self.values,self.shape,self.size,self.selection = values,values.shape,values.size,selection
        self.reads = 0
    def __getitem__(self, key):
        assert key == self.selection
        self.reads += 1
        return self.values[key]


def sample():
    shape=(7,273,311)
    scores=np.random.default_rng(21).integers(0,256,shape,dtype=np.uint8)
    mask=np.zeros(shape,np.uint8)
    mask[3,117:266,121:295] = 1
    mask[3,150:154,150:154] = 0
    scores[3,125:130,130:135] = 0
    active=np.zeros(shape[0],bool); active[3]=True
    boxes=np.full((shape[0],4),-1,np.int64); boxes[3]=(117,266,121,295)
    return mask,scores,active,boxes


def test_bounds_skip_empty_planes_and_only_read_the_retained_crop(tmp_path):
    mask,scores,active,boxes=sample()
    selection=(3,slice(117,266),slice(121,295))
    guarded_mask,guarded_scores=GuardedScores(mask,selection),GuardedScores(scores,selection)
    reader=_MaskedNativeScoreReader(guarded_mask,guarded_scores,active,boxes)
    actual=write_block_confidence_evidence(tmp_path/'bounded',scores.shape,reader,model_name='m',layer_key='k')
    expected=write_block_confidence_evidence(tmp_path/'dense',scores.shape,
        lambda z:np.where(mask[z],scores[z],np.uint8(0)),model_name='m',layer_key='k')
    for name in ('index.bin','scores.u8.zlib'):
        assert (actual.path/name).read_bytes()==(expected.path/name).read_bytes()
    assert guarded_mask.reads==guarded_scores.reads==1
    assert reader.capture_metrics['empty_slices_skipped']==6
    assert reader.capture_metrics['bounded_input_bytes']==2*149*174


def test_deferred_capture_uses_bounds_and_preserves_unknowns_and_inputs(tmp_path):
    mask,scores,active,boxes=sample()
    original_mask,original_scores=mask.copy(),scores.copy()
    view=get_view_infos(*scores.shape,cartesian_views=('transverse',))[0]
    configure_confidence_evidence(tmp_path/'out',enabled=True,defer_projection=True)
    ref=capture_prediction_confidence(mask,scores,view=view,model_name='m',temp_dir=tmp_path,
        known_slice_any=active,known_slice_bboxes=boxes)
    with ref.native_reader() as reader:
        actual,known=reader(0,scores.shape[0])
    expected=np.where(mask,scores,np.uint8(0))
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(known,expected>0)
    np.testing.assert_array_equal(mask,original_mask)
    np.testing.assert_array_equal(scores,original_scores)


def test_empty_metadata_does_not_touch_input_pixels(tmp_path):
    values=np.ones((3,20,21),np.uint8)
    guarded=GuardedScores(values,None)
    reader=_MaskedNativeScoreReader(guarded,guarded,np.zeros(3,bool),np.full((3,4),-1,np.int64))
    ref=write_block_confidence_evidence(tmp_path/'empty',values.shape,reader,model_name='m',layer_key='k')
    assert ref.metadata['known_voxels']==0 and guarded.reads==0


def test_eager_scores_certify_post_mask_support_without_changing_projection(tmp_path, monkeypatch):
    import XTA.sweep_projection as projection

    shape = (11, 19, 23)
    views = get_view_infos(*shape, cartesian_views=(),
        tilt_groups=(TiltedViewGroup(('transverse',), (23.,), ('vertical',)),),
        azimuthal_views=('transverse', 'tilted_transverse'), azimuthal_azimuth_angles=(45., 45.),
        azimuthal_native_raster=0, radial_views=('transverse',), radial_patch_size=13,
        spherical_views=('transverse',), spherical_patch_size=13)
    for family in ('tilted', 'azimuthal', 'radial', 'spherical'):
        view = next(view for view in views if view.family == family)
        native_shape = (view.num_slices, 13, 13)
        unmasked = np.full(native_shape, 177, np.uint8)
        mask = np.zeros(native_shape, bool)
        mask[1:, 2:8, 3:12] = True
        unmasked[1:, 2:3, 3:11] = 0  # Selected mask pixels can still have unknown scores.
        expected = np.where(mask, unmasked, np.uint8(0))
        expected_boxes = projection._rank_boxes(expected, 255)[0]

        with monkeypatch.context() as disabled:
            disabled.setattr(confidence_evidence, '_eager_rank_score_bounds_supported',
                             lambda *args: False)
            configure_confidence_evidence(tmp_path / family / 'reference', enabled=True)
            previous = capture_prediction_confidence(mask, unmasked.copy(), view=view,
                model_name='model', temp_dir=tmp_path, output_shape=shape)

        def reject_scan(*_):
            pytest.fail('Projected score bounds should already be certified')

        existing_projection = projection.sweep_rank_projection_reader
        seen = []
        def record_bounds(*args, **kwargs):
            seen.append(kwargs['known_native_boxes'])
            return existing_projection(*args, **kwargs)

        configure_confidence_evidence(tmp_path / family / 'certified', enabled=True)
        with monkeypatch.context() as instrumented:
            instrumented.setattr(projection, '_rank_boxes', reject_scan)
            instrumented.setattr(projection, 'sweep_rank_projection_reader', record_bounds)
            actual = capture_prediction_confidence(mask, unmasked, view=view,
                model_name='model', temp_dir=tmp_path, output_shape=shape)
        assert len(seen) == 1
        np.testing.assert_array_equal(seen[0], expected_boxes)
        np.testing.assert_array_equal(unmasked, expected)
        for name in ('index.bin', 'scores.u8.zlib'):
            assert (previous.path / name).read_bytes() == (actual.path / name).read_bytes()


def test_eager_score_bounds_do_not_reorder_overlapping_file_mappings(tmp_path):
    from types import SimpleNamespace

    shape = (3, 5, 7)
    path = tmp_path / 'shared.dat'
    owner = np.memmap(path, dtype=np.uint8, mode='w+', shape=shape)
    alias = np.memmap(path, dtype=np.uint8, mode='r+', shape=shape)
    other = np.memmap(tmp_path / 'other.dat', dtype=np.uint8, mode='w+', shape=shape)
    try:
        view = SimpleNamespace(family='spherical', num_slices=shape[0])
        supports = confidence_evidence._eager_rank_score_bounds_supported
        assert not supports(np.asarray(owner), np.asarray(alias), view)
        assert supports(np.asarray(owner), np.asarray(other), view)
    finally:
        for mapping in (owner, alias, other):
            mapping._mmap.close()


@pytest.mark.parametrize('replacement',[(0,274,0,1),(3,2,0,1),(0,1,-1,2),(0,1,0,312)])
def test_invalid_active_bounds_fail_before_any_read(replacement):
    mask,scores,active,boxes=sample(); boxes[3]=replacement
    with pytest.raises(ValueError,match='native grid'):
        _MaskedNativeScoreReader(mask,scores,active,boxes)

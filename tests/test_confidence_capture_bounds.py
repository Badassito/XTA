"""Trusted parent bounds avoid dense reads without changing score blocks."""
from pathlib import Path

import numpy as np
import pytest

from XTA.confidence_evidence import (_MaskedNativeScoreReader, capture_prediction_confidence,
    configure_confidence_evidence, shutdown_confidence_publication, write_block_confidence_evidence)
from XTA.geometry import get_view_infos


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


@pytest.mark.parametrize('replacement',[(0,274,0,1),(3,2,0,1),(0,1,-1,2),(0,1,0,312)])
def test_invalid_active_bounds_fail_before_any_read(replacement):
    mask,scores,active,boxes=sample(); boxes[3]=replacement
    with pytest.raises(ValueError,match='native grid'):
        _MaskedNativeScoreReader(mask,scores,active,boxes)

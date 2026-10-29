"""Partial PNG loading keeps the encoded lattice and every required channel plane."""
from dataclasses import replace
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from XTA import pta, pta_inputs
from tests.test_pta_multi_input import cpu_only, run, artifacts


def sequence(root,indices=tuple(range(1,13)),labels=(3,9)):
    root.mkdir()
    for index in indices:
        image=np.full((8,8),index*10,np.uint8)
        assert pta.cv2.imwrite(str(root/f'sample_{index:04d}.png'),image)
        if index in labels:
            (root/f'sample_{index:04d}.txt').write_text(
                '0 .1 .1 .9 .1 .9 .9 .1 .9\n' if index==labels[0] else '')
    return root


def spec_for(root):
    return pta.discover_volume_specs(root,force=False,warnings=pta.WarningLog())[0]


def variants(channel):
    return pta.expand_channel_variants(pta.resolve_channel_formats(channel))


@pytest.mark.parametrize('channel',['gray','RGB','C3S2'])
@pytest.mark.parametrize('fraction',['1','.5'])
def test_partial_output_pixels_labels_tiles_and_context_match_full_decode(tmp_path,monkeypatch,channel,fraction):
    source=sequence(tmp_path/'source')
    extra=['--channel_format',channel,'--fraction',fraction,'--enable_tile','4:4']
    with mock.patch.object(pta,'_required_partial_sequence_positions',return_value=None):
        full=run([source],tmp_path/'full',extra=extra)
    reads=[]
    original=pta_inputs._read_gray_image_opencv
    def read(path):
        reads.append(path)
        return original(path)
    monkeypatch.setattr(pta_inputs,'_read_gray_image_opencv',read)
    partial=run([source],tmp_path/'partial',extra=extra)
    for folder in ('images','labels'):
        assert artifacts(tmp_path/'partial',folder)==artifacts(tmp_path/'full',folder)
    assert partial['outputs']['dataset_candidates_processed']==full['outputs']['dataset_candidates_processed']
    assert len(reads)<12
    spec=spec_for(source)
    required=pta._required_partial_sequence_positions(spec,variants(channel),force=False,require_all_frames=False)
    expected={spec.image_paths_by_index[sorted(spec.image_paths_by_index)[i]] for i in {0,*required}}
    assert set(reads)==expected


def test_required_context_uses_encoded_indices_and_preserves_gap_skips(tmp_path):
    source=sequence(tmp_path/'source',indices=(1,2,4,6,7,8),labels=(2,7))
    spec=spec_for(source)
    assert pta._required_partial_sequence_positions(spec,variants('gray'),force=False,require_all_frames=False)==(1,4)
    assert pta._required_partial_sequence_positions(spec,variants('C3S1'),force=False,require_all_frames=False)==(1,3,4,5)
    assert pta._required_partial_sequence_positions(spec,variants('C3S2'),force=False,require_all_frames=False)==(0,1,2,4)


@pytest.mark.parametrize('gate',['force','diagnostics','full','unlabeled','jpeg','no_channels'])
def test_ineligible_loads_retain_all_source_frames(tmp_path,gate):
    spec=spec_for(sequence(tmp_path/'source'))
    kwargs=dict(force=False,require_all_frames=False)
    channels=variants('gray')
    if gate=='force':kwargs['force']=True
    elif gate=='diagnostics':kwargs['require_all_frames']=True
    elif gate=='full':spec=replace(spec,volume_class='fully_labeled')
    elif gate=='unlabeled':spec=replace(spec,label_source='none',volume_class='unlabeled')
    elif gate=='jpeg':spec=replace(spec,image_paths_by_index={i:p.with_suffix('.jpg') for i,p in spec.image_paths_by_index.items()})
    else:channels=None
    assert pta._required_partial_sequence_positions(spec,channels,**kwargs) is None


def test_unused_png_dimensions_are_still_validated(tmp_path):
    source=sequence(tmp_path/'source',indices=(1,2,3,4),labels=(1,3))
    assert pta.cv2.imwrite(str(source/'sample_0004.png'),np.zeros((7,8),np.uint8))
    images=sorted(source.glob('*.png'))
    with pytest.raises(ValueError,match='dimension mismatch'):
        pta.load_image_sequence_to_memory(images,warnings=pta.WarningLog(),workers=2,required_positions=(0,2))


def test_partial_load_keeps_unknown_coverage_and_full_frame_shape(tmp_path):
    root=sequence(tmp_path/'source',indices=(1,2,3,4,5,6),labels=(2,5))
    spec=spec_for(root)
    loaded=pta.load_source_volume_from_spec(spec,warnings=pta.WarningLog(),workers=2,channel_variants=variants('gray'))
    assert loaded.volume.shape==(6,8,8)
    assert loaded.encoded_indices==(1,2,3,4,5,6)
    np.testing.assert_array_equal(loaded.volume[[1,4]],np.asarray([np.full((8,8),20,np.uint8),np.full((8,8),50,np.uint8)]))
    assert not loaded.volume[[2,3,5]].any()
    assert set(loaded.labels_by_frame)=={1,4}

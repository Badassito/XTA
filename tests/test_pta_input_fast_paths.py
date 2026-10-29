"""One-shot input paths preserve masks, geometry, and shared-block ownership."""
import gzip
from pathlib import Path
import threading
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest

from XTA import pta, pta_inputs
from XTA.pta_config import parse_pta_args
from XTA.pta_runtime import build_runtime_options


@pytest.mark.parametrize('encoding',['raw','gzip'])
def test_canonical_uint8_nrrd_avoids_legacy_full_bytearray_path(tmp_path,encoding):
    nrrd=pytest.importorskip('nrrd')
    source=np.arange(5*6*7,dtype=np.uint8).reshape(5,6,7)
    path=tmp_path/'mask.nrrd'
    nrrd.write(str(path),source.transpose(2,1,0),header={'encoding':encoding,'space directions':np.eye(3)})
    warnings=pta.WarningLog()
    with mock.patch.object(nrrd,'read',side_effect=AssertionError('full bytearray reader')):
        actual=pta.load_nrrd_mask_for_volume(path,source.shape,warnings=warnings)
    np.testing.assert_array_equal(actual,source>0)
    assert actual.dtype==np.uint8 and actual.flags.c_contiguous
    assert warnings.counts['segmentation_nrrd_bounded_readinto']==1


@pytest.mark.parametrize('case',['signed','detached','noncanonical'])
def test_other_nrrd_layouts_keep_general_reader_semantics(tmp_path,case):
    nrrd=pytest.importorskip('nrrd')
    source=np.arange(5*6*7,dtype=np.int16).reshape(5,6,7)-30
    path=tmp_path/('mask.nhdr' if case=='detached' else 'mask.nrrd')
    data=source.transpose(2,1,0)
    header={'space directions':np.eye(3)}
    if case=='detached':data=(data>0).astype(np.uint8)
    elif case=='noncanonical':
        data=(source>0).astype(np.uint8)
        header={'space directions':np.asarray([[0,0,1],[0,1,0],[1,0,0]])}
    nrrd.write(str(path),data,header=header)
    with mock.patch.object(nrrd,'read',wraps=nrrd.read) as read:
        actual=pta.load_nrrd_mask_for_volume(path,source.shape,warnings=pta.WarningLog())
    assert read.call_count==1
    np.testing.assert_array_equal(actual,source>0)


def write_payload(path,encoding,payload):
    path.write_bytes((f'NRRD0005\ntype: uint8\ndimension: 3\nsizes: 4 3 2\nencoding: {encoding}\n'
        'space directions: (1,0,0) (0,1,0) (0,0,1)\n\n').encode()+payload)


@pytest.mark.parametrize('case',['short_raw','long_raw','short_gzip','bad_crc','long_gzip'])
def test_fast_nrrd_rejects_corruption_and_wrong_payload_size(tmp_path,case):
    pytest.importorskip('nrrd')
    payload=bytes(range(24))
    encoding='raw' if case.endswith('raw') else 'gzip'
    if case=='short_raw':payload=payload[:-1]
    elif case=='long_raw':payload+=b'X'
    else:
        payload=gzip.compress(payload+(b'X' if case=='long_gzip' else b''))
        if case=='short_gzip':payload=payload[:-4]
        elif case=='bad_crc':payload=payload[:-8]+bytes([payload[-8]^1])+payload[-7:]
    path=tmp_path/'broken.nrrd'
    write_payload(path,encoding,payload)
    with pytest.raises((ValueError,EOFError,OSError)):
        pta.load_nrrd_mask_for_volume(path,(2,3,4),warnings=pta.WarningLog())


def test_fast_nrrd_reads_concatenated_gzip_members(tmp_path):
    pytest.importorskip('nrrd')
    path=tmp_path/'mask.nrrd'
    payload=bytes(range(24))
    write_payload(path,'gzip',gzip.compress(payload[:11])+gzip.compress(payload[11:]))
    actual=pta.load_nrrd_mask_for_volume(path,(2,3,4),warnings=pta.WarningLog())
    np.testing.assert_array_equal(actual,np.frombuffer(payload,np.uint8).reshape(2,3,4)>0)


def test_fast_nrrd_keeps_ambiguous_cube_rejection(tmp_path):
    nrrd=pytest.importorskip('nrrd')
    path=tmp_path/'ambiguous.nrrd'
    nrrd.write(str(path),np.ones((3,3,3),np.uint8))
    with pytest.raises(ValueError,match='ambiguous'):
        pta.load_nrrd_mask_for_volume(path,(3,3,3),warnings=pta.WarningLog())


def video_spec():
    return pta.VolumeInputSpec(input_dir=Path('input'),stem='sample',kind='video',
        image_paths_by_index={},video_path=Path('sample.mkv'),labels_by_index={},
        segmentation_nrrd_path=Path('sample.nrrd'),volume_class='fully_labeled',label_source='nrrd',
        input_start_index=None,encoded_indices=(0,1),frame_count_hint=2,fps_hint=7.,
        video_info_hint={'num_frames':2,'height':3,'width':4,'fps':7.})


def test_video_and_nrrd_decode_overlap_and_reuse_discovered_probe():
    spec=video_spec()
    video_started=threading.Event();mask_started=threading.Event()
    volume=np.zeros((2,3,4),np.uint8);mask=np.ones_like(volume)
    def decode(path,**kwargs):
        assert kwargs['probe_info'] is spec.video_info_hint
        video_started.set()
        assert mask_started.wait(3),'NRRD did not overlap video decode'
        return volume,7.,None
    def load(path,shape,**kwargs):
        assert shape==volume.shape
        mask_started.set()
        assert video_started.wait(3),'video did not overlap NRRD decode'
        return mask
    with mock.patch.object(pta,'ffprobe_info',side_effect=AssertionError('repeat probe')), \
         mock.patch.object(pta,'decode_video_gray8_to_memory',side_effect=decode), \
         mock.patch.object(pta,'load_nrrd_mask_for_volume',side_effect=load):
        assert pta.estimate_spec_resident_bytes(spec)>0
        result=pta.load_source_volume_from_spec(spec,warnings=pta.WarningLog(),workers=2)
    assert result.volume is volume and result.mask_volume is mask and result.fps==7.


def test_failed_parallel_mask_load_releases_decoded_shared_volume():
    block=SimpleNamespace(release=mock.Mock())
    with mock.patch.object(pta,'decode_video_gray8_to_memory',return_value=(np.zeros((2,3,4),np.uint8),7.,block)), \
         mock.patch.object(pta,'load_nrrd_mask_for_volume',side_effect=ValueError('bad mask')):
        with pytest.raises(ValueError,match='bad mask'):
            pta.load_source_volume_from_spec(video_spec(),warnings=pta.WarningLog(),workers=2)
    block.release.assert_called_once()


def test_failed_sequential_mask_load_releases_decoded_shared_volume():
    blocks=[]
    def allocate(shape):
        block=pta.SharedBlock(int(np.prod(shape)))
        blocks.append(block)
        return block.ndarray(shape),block
    def decode(path,*,allocator,**kwargs):
        volume,block=allocator((2,3,4))
        volume.fill(7)
        return volume,7.,block
    try:
        with mock.patch.object(pta,'decode_video_gray8_to_memory',side_effect=decode), \
             mock.patch.object(pta,'load_nrrd_mask_for_volume',side_effect=ValueError('bad mask')):
            with pytest.raises(ValueError,match='bad mask'):
                pta.load_source_volume_from_spec(video_spec(),warnings=pta.WarningLog(),
                    workers=1,allocator=allocate)
        assert len(blocks)==1 and blocks[0].released
    finally:
        for block in blocks:block.release()


def test_failed_video_process_start_releases_its_own_allocation():
    block=SimpleNamespace(release=mock.Mock())
    with mock.patch.object(pta_inputs,'_require_bin'), mock.patch.object(pta_inputs.subprocess,'Popen',side_effect=OSError('spawn failed')):
        with pytest.raises(OSError,match='spawn failed'):
            pta.decode_video_gray8_to_memory(Path('sample.mkv'),warnings=pta.WarningLog(),
                allocator=lambda shape:(np.zeros(shape,np.uint8),block),probe_info=video_spec().video_info_hint)
    block.release.assert_called_once()


def source_volume():
    image=np.arange(2*5*5,dtype=np.uint8).reshape(2,5,5)
    return pta.SourceVolume(input_dir=Path('input'),stem='sample',kind='sequence',image_paths=[],
        video_path=None,labels_by_frame={},segmentation_nrrd_path=Path('sample.nrrd'),
        mask_volume=np.ones_like(image),volume_class='fully_labeled',label_source='nrrd',
        input_start_index=0,encoded_indices=(0,1),volume=image,fps=1.)


def prepare(source,out_dir,allocator):
    config=parse_pta_args(['--input','input','--imgsz','8','--enable_cartesian','transverse','--save','images','labels'])
    args=build_runtime_options(config)
    return pta.prepare_loaded_source(source,args=args,warnings=pta.WarningLog(),workers=2,
        out_dir=out_dir,tile_configs=(),channel_variants=pta.expand_channel_variants(
            pta.resolve_channel_formats(args.channel_format)),write_side_effects=False,allocator=allocator)


def test_preparation_writes_final_shared_cubes_without_workspace_then_recopy(tmp_path):
    blocks=[]
    def allocate(shape):
        block=pta.SharedBlock(int(np.prod(shape)))
        blocks.append(block)
        return block.ndarray(shape),block
    try:
        with mock.patch.object(pta.shared_media,'allocate_workspace_array',side_effect=AssertionError('intermediate cube')):
            result=prepare(source_volume(),tmp_path,allocate)
        assert len(blocks)==2
        assert result.volume_render_block is blocks[0] and result.mask_render_block is blocks[1]
        assert result.shm_blocks==blocks
        assert result.processing_shape==(5,5,5)
        assert np.all(result.mask_for_render)
    finally:
        for block in blocks:block.release()


def test_failed_cube_resize_releases_both_caller_owned_destinations(tmp_path):
    blocks=[]
    def allocate(shape):
        block=pta.SharedBlock(int(np.prod(shape)))
        blocks.append(block)
        return block.ndarray(shape),block
    with mock.patch.object(pta.shared_media,'resize_categorical_volume_to_processing_cube_uint8',
                           side_effect=RuntimeError('resize failed')):
        with pytest.raises(RuntimeError,match='resize failed'):
            prepare(source_volume(),tmp_path,allocate)
    assert len(blocks)==2 and all(block.released for block in blocks)


@pytest.mark.parametrize('failure',['invariant','plans'])
def test_failure_after_cube_resize_releases_new_blocks_but_not_source_owner(tmp_path,failure):
    source=source_volume()
    caller_block=pta.SharedBlock(source.volume.nbytes)
    np.copyto(caller_block.ndarray(source.volume.shape),source.volume)
    source.volume=caller_block.ndarray(source.volume.shape)
    source.volume_block=caller_block
    blocks=[]
    def allocate(shape):
        block=pta.SharedBlock(int(np.prod(shape)))
        blocks.append(block)
        return block.ndarray(shape),block
    name='reassert_foreground_slice_anchors' if failure=='invariant' else 'compile_v18_pta_views'
    try:
        with mock.patch.object(pta,name,side_effect=RuntimeError('post-resize failure')):
            with pytest.raises(RuntimeError,match='post-resize failure'):
                prepare(source,tmp_path,allocate)
        assert len(blocks)==2 and all(block.released for block in blocks)
        assert not caller_block.released
        assert source.volume[1,4,4]==49
    finally:
        for block in blocks:block.release()
        source.volume=None
        caller_block.release()

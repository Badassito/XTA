"""Exact adoption of stock detector NRRD publications before scalar tail work."""
from __future__ import annotations

import copy
from dataclasses import replace
import gc
import gzip
import json
from pathlib import Path
import threading
import weakref

import numpy as np
import pytest

from XTA import outputs, nrrd_compression
from XTA.interpolation import NrrdLayerRef
from XTA.tta_sweep_branch_manifest import layer_payload_identity


@pytest.fixture(autouse=True)
def cpu_writer(monkeypatch):
    # Disable before the writer's conditional torch import/availability probe.
    monkeypatch.setenv('YOLO_TTA_NRRD_GPU_MIRROR_TEE','0')
    monkeypatch.setattr(nrrd_compression,'_require_nrrd_member_codec',
                        lambda *args,**kwargs:('zlib',1,gzip.compress))


def reference(root, name='view', *, empty=False, known_extent=False, mirror=True):
    shape=(4,8,8)
    mask=np.zeros(shape,np.uint8)
    if not empty:mask[1:3,2:6,3:7]=1
    root.mkdir(parents=True,exist_ok=True)
    path=root/f'{name}.dat'
    mask.tofile(path)
    extent=((0,-1,0,-1,0,-1) if empty else (3,6,2,5,1,2)) if known_extent else None
    return NrrdLayerRef(key=name,name=name,path=path,shape=shape,model_name='detector',
        view_name=name,physical_view_name=name,view_family='orthogonal',source='fullframe',mask_kind='yolo',
        stage='pre_interpolation',description='Exact cleaned detector support',mirror_low_quality=mirror,
        segment_extent_ijk=extent,segment_extent_shape_tyx=shape if known_extent else (0,0,0))


def sink(root, *, run_id='branch-run', low=True):
    spec=outputs.LowQualityDownbinSpec('0.5','0p5_4x4x4',.5,(4,4,4))
    return outputs.NrrdLayerSink(nrrd_dir=root/'nrrd',stem='case',output_shape_tyx=(4,8,8),max_workers=1,
        low_quality_specs=(spec,) if low else (),low_quality_root=root/'low_quality' if low else None,
        run_id=run_id)


def published(root, ref, suffix='view_fullframe_yolo', *, low=True):
    source=sink(root,low=low)
    try:
        path=source.submit_layer(ref,suffix)
        record=source.wait_published_layer(ref,suffix,output_path=path)
        manifest=source.write_manifest()
        native=json.loads(manifest.read_text())
        mirrors={spec.token:json.loads((source._lq_nrrd_dir(spec)/manifest.name).read_text())
                 for spec in source.low_quality_specs if record['low_quality']}
    finally:
        source.shutdown()
    return path,record,native,mirrors


@pytest.mark.parametrize('empty,known_extent,low,mirror',[
    (False,False,True,True),(False,True,True,True),(True,False,True,True),
    (True,True,True,True),(False,False,False,True),(False,False,True,False)])
def test_adopt_uses_exact_stock_rows_and_files_without_writing(tmp_path,monkeypatch,empty,known_extent,low,mirror):
    ref=reference(tmp_path/'sources',empty=empty,known_extent=known_extent,mirror=mirror)
    root=tmp_path/'branch'
    path,record,native,mirrors=published(root,ref,low=low)
    payloads=[path,*[root/'low_quality'/token/'nrrd'/path.name for token in record['low_quality']]]
    before={item:layer_payload_identity(item) for item in payloads}
    target=sink(root,low=low)
    def no_payload_write(*args,**kwargs):pytest.fail('Adoption rewrote a payload')
    monkeypatch.setattr(outputs,'write_single_layer_nrrd_from_ref',no_payload_write)
    monkeypatch.setattr(outputs,'write_layer_nrrd_with_low_quality_mirrors',no_payload_write)
    try:
        assert target.adopt_published_layer(ref,record)==path
        assert target.progress_counts()==(1,1)
        assert target.layer_count()==1
        assert target.low_quality_layer_count()==len(record['low_quality'])
        assert target.pending_layer_refs()==()
        target.wait()
        manifest=target.write_manifest()
        assert json.loads(manifest.read_text())==native
        for token,original in mirrors.items():
            assert json.loads((root/'low_quality'/token/'nrrd'/manifest.name).read_text())==original
        assert target.published_layer_record(ref,'view_fullframe_yolo')==record
    finally:
        target.shutdown()
    assert {item:layer_payload_identity(item) for item in payloads}==before


def test_empty_absence_is_a_noop_and_partial_absence_is_invalid(tmp_path):
    target=sink(tmp_path/'branch')
    try:
        assert target.adopt_published_layer(None,None) is None
        assert target.layer_count()==0 and target.progress_counts()==(0,0)
        with pytest.raises(ValueError):target.adopt_published_layer(None,{})
        with pytest.raises(ValueError):target.adopt_published_layer(reference(tmp_path/'source'),None)
    finally:target.shutdown()


@pytest.mark.parametrize('mutation',[
    'run','stem','output_grid','layer_key','backing_grid','stored_grid','empty_type',
    'suffix','filename','color','allocator','mirror','mirror_grid','mirror_downbin','file_identity'])
def test_invalid_record_does_not_partially_register_rows(tmp_path,mutation):
    ref=reference(tmp_path/'source')
    root=tmp_path/'branch'
    _,record,_,_=published(root,ref)
    value=copy.deepcopy(record)
    if mutation=='run':value['run_id']='another-run'
    elif mutation=='stem':value['stem']='another-case'
    elif mutation=='output_grid':value['output_shape_tyx']=[4,8,9]
    elif mutation=='layer_key':value['native']['layer_key']='another-layer'
    elif mutation=='backing_grid':value['native']['backing_shape_tyx']=[4,8,9]
    elif mutation=='stored_grid':value['native']['stored_shape_tyx']=[0,8,8]
    elif mutation=='empty_type':value['native']['empty_segment']=0
    elif mutation=='suffix':value['native']['suffix']='different'
    elif mutation=='filename':value['native']['filename']='../escape.seg.nrrd'
    elif mutation=='color':value['allocator']['color_rgb']=[.01,.02,.03]
    elif mutation=='allocator':value['allocator']['occurrence']=False
    elif mutation=='mirror':value['low_quality']={}
    elif mutation=='mirror_grid':next(iter(value['low_quality'].values()))['output_shape_tyx']=[4,4,5]
    elif mutation=='mirror_downbin':next(iter(value['low_quality'].values()))['downbin_scale']=.25
    elif mutation=='file_identity':value['files']['native']['size_bytes']+=1
    target=sink(root)
    try:
        with pytest.raises(ValueError):target.adopt_published_layer(ref,value)
        assert target.layer_count()==0 and target.low_quality_layer_count()==0
        assert target.progress_counts()==(0,0)
        assert target._suffix_counts=={} and target._segment_colors_in_use==set()
        assert target._physical_names=={}
    finally:target.shutdown()


def test_changed_payload_and_changed_ref_fail_before_adoption(tmp_path):
    ref=reference(tmp_path/'source')
    root=tmp_path/'branch'
    path,record,_,_=published(root,ref)
    target=sink(root)
    try:
        with pytest.raises(ValueError,match='reference identity'):
            target.adopt_published_layer(replace(ref,stage='another_stage'),record)
        path.write_bytes(path.read_bytes()+b'mutated')
        with pytest.raises(ValueError,match='changed after publication'):
            target.adopt_published_layer(ref,record)
        assert target.layer_count()==0
    finally:target.shutdown()


def test_adopted_rows_are_private_and_source_reference_is_not_retained(tmp_path):
    ref=reference(tmp_path/'source')
    root=tmp_path/'branch'
    _,record,_,_=published(root,ref)
    target=sink(root)
    weak=weakref.ref(ref)
    try:
        target.adopt_published_layer(ref,record)
        record['native']['description']='caller mutation'
        next(iter(record['low_quality'].values()))['segment_color_rgb'][0]=.123
        assert target._manifest[0]['description']=='Exact cleaned detector support'
        assert target._lq_manifests['0p5_4x4x4'][0]['segment_color_rgb']!=[.123,*record['native']['segment_color_rgb'][1:]]
        del ref
        gc.collect()
        assert weak() is None
    finally:target.shutdown()


def test_duplicate_suffix_allocator_restores_in_any_adoption_order(tmp_path):
    ref=reference(tmp_path/'source')
    root=tmp_path/'branch'
    source=sink(root)
    try:
        first=source.submit_layer(ref,'same')
        second=source.submit_layer(ref,'same')
        a=source.wait_published_layer(ref,'same',output_path=first)
        b=source.wait_published_layer(ref,'same',output_path=second)
        original_colors=set(source._segment_colors_in_use)
    finally:source.shutdown()
    target=sink(root)
    try:
        target.adopt_published_layer(ref,b)
        target.adopt_published_layer(ref,a)
        assert target._suffix_counts=={'same':2}
        assert target._segment_colors_in_use==original_colors
        next_path=target.submit_layer(reference(tmp_path/'source','later'),'same')
        assert next_path.name=='case_same_03.seg.nrrd'
        target.wait()
        with pytest.raises(ValueError,match='conflicts'):
            target.adopt_published_layer(ref,a)
        assert target.layer_count()==3
    finally:target.shutdown()


def test_adoption_allows_unrelated_pending_fallback_layer(tmp_path,monkeypatch):
    ref=reference(tmp_path/'source')
    root=tmp_path/'branch'
    path,record,_,_=published(root,ref)
    fallback=reference(tmp_path/'source','fallback')
    target=sink(root)
    entered,released=threading.Event(),threading.Event()
    original=outputs.write_layer_nrrd_with_low_quality_mirrors
    def blocked(*args,**kwargs):
        entered.set()
        if not released.wait(10):raise RuntimeError('Test did not release fallback writer')
        return original(*args,**kwargs)
    monkeypatch.setattr(outputs,'write_layer_nrrd_with_low_quality_mirrors',blocked)
    try:
        target.submit_layer(fallback,'fallback')
        assert entered.wait(5)
        assert target.adopt_published_layer(ref,record)==path
        assert target.layer_count()==2 and target.progress_counts()==(1,2)
        released.set()
        target.wait()
        assert target.progress_counts()==(2,2)
    finally:
        released.set()
        target.shutdown()

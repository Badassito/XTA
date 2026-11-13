"""Cutoff-bound edits remain ordinary legacy bundles and reject wrong sources."""
from __future__ import annotations

import copy
import ast
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from types import SimpleNamespace

import numpy as np
import pytest

from tools import prepare_pta_input as prep
from tools import xta_sweep_edits as edits

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'tools'))
import xta_native_filter as nf


def toolbar_contract():
    path = Path(__file__).resolve().parents[1]/'tools/slicer/xta_slicerrc.py'
    tree = ast.parse(path.read_text(encoding='utf-8'))
    names = {'_xta_sweep_scalar','_xta_sweep_review','_xta_sweep_binding','_xta_sweep_grid_rows'}
    body = [node for node in tree.body if isinstance(node,ast.FunctionDef) and node.name in names]
    assert {node.name for node in body} == names
    namespace = dict(vars(edits))
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    return namespace


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def sweep_fixture(tmp_path, native_payload='reference_only'):
    """A two-cutoff review; by default native is reference-only, as the publisher writes it."""
    root = tmp_path/'sweep'
    review_path = root/'review'/'manifest.json'
    grids = []
    arrays = {}
    for grid_id, shape in (('native',(4,8,8)),('low_quality/0p50',(2,4,4))):
        grid = dict(grid_id=grid_id,shape_tyx=list(shape),space='left-posterior-superior',
            directions_xyz=[[1,0,0],[0,1,0],[0,0,1]],origin_xyz=[0,0,0],groups=[],branches=[])
        for cent in (25,75):
            branch_dir = root/'confidence_cutoffs'/f'conf_{cent:03d}'
            run_path = branch_dir/'manifest.json'
            save(run_path,dict(status='complete',confidence_sweep=dict(sweep_id='sweep-fixture',cutoff_hundredths=cent)))
            folder = branch_dir/grid_id/'nrrd'
            folder.mkdir(parents=True)
            low = np.zeros((2,4,4),np.uint8)
            low[:,1:3,1:3] = 1
            if cent == 25:
                low[:,1,3] = 1
            mask = low if grid_id != 'native' else low.repeat(2,0).repeat(2,1).repeat(2,2)
            arrays[grid_id,cent] = mask
            rows, segments = [], []
            for suffix, logical, role in (('Direct','component','direct'),('Global_final_output','global','global')):
                filename = f'sample_{suffix}.seg.nrrd'
                row = dict(filename=filename,segment_name=f'sample_{suffix}',suffix=suffix,
                    model_name='model',layer_key=logical,recomposition_op='select' if role=='global' else 'union')
                rows.append(row)
                nf.write_nrrd(folder/filename,shape,nf.array_slabs(mask,2),row['segment_name'])
                sid = f'{grid_id}:{cent}:{logical}'
                segments.append(dict(row,source_filename=filename,logical_id=logical,segment_id=sid,
                    cutoff_cent=cent,role=role,display_name=f'Renamed {logical} at {cent}',
                    view_name='transverse',mask_kind='yolo' if role=='direct' else 'union',
                    stage='pre_interpolation' if role=='direct' else 'final',layer_role='additive_component' if role=='direct' else 'checkpoint'))
            scalar = dict(layout='one_single_layer_nrrd_per_component',output_shape_tyx=list(shape),layers=rows)
            if grid_id != 'native':
                scalar.update(quality='low_quality',full_quality_output_shape_tyx=[4,8,8])
            path = folder/'sample_nrrd_manifest.json'
            save(path,scalar)
            grid['branches'].append(dict(cutoff_cent=cent,
                branch_dir='../'+branch_dir.relative_to(root).as_posix(),
                run_manifest='../'+run_path.relative_to(root).as_posix(),
                run_manifest_sha256=edits._xta_sweep_digest(run_path),
                source_manifest='../'+path.relative_to(root).as_posix(),
                source_manifest_sha256=edits._xta_sweep_digest(path)))
            grid['groups'].append(dict(path=f'group-{grid_id.replace("/","-")}-{cent}.seg.nrrd',segments=segments))
        if grid_id == 'native' and native_payload == 'reference_only':
            grid.update(payload='reference_only',segments=[row for g in grid['groups'] for row in g['segments']],
                        groups=[],coverage=[])
        elif native_payload is not None:
            grid['payload'] = 'grouped'
        grids.append(grid)
    save(review_path,dict(schema='xta.sweep_review/1',sweep_id='sweep-fixture',grids=grids))
    return review_path, arrays


def independent_fixture(tmp_path, native_payload='reference_only'):
    review,arrays=sweep_fixture(tmp_path,native_payload)
    root=review.parent.parent
    data=edits._xta_sweep_json(review)
    data['source_mode']='independent_scalar'
    names={25:'ordinary-completed-red',75:'another-independent-run'}
    for cent,name in names.items():
        old=root/'confidence_cutoffs'/f'conf_{cent:03d}'
        new=root/name
        old.rename(new)
        save(new/'manifest.json',dict(status='complete',launcher=dict(mode='tta'),
            resolved_configuration=dict(conf=cent/100)))
    for grid in data['grids']:
        for branch in grid['branches']:
            cent=branch['cutoff_cent']
            prefix=f'../confidence_cutoffs/conf_{cent:03d}'
            replacement='../'+names[cent]
            for key in ('branch_dir','run_manifest','source_manifest'):
                branch[key]=branch[key].replace(prefix,replacement)
            branch['run_manifest_sha256']=edits._xta_sweep_digest(root/names[cent]/'manifest.json')
            branch['source_mode']='independent_scalar'
    save(review,data)
    return review,arrays


def legacy_child(root, review, arrays, cent=25):
    grid_id = 'low_quality/0p50'
    api = toolbar_contract()
    data = api['_xta_sweep_review'](review)
    binding = api['_xta_sweep_binding'](data,grid_id,cent,{'component','global'})
    root.mkdir(parents=True)
    low = data['grids'][grid_id]['branches'][cent]
    full = data['grids']['native']['branches'][cent]
    shutil.copyfile(low['path'],root/'source_manifest.json')
    shutil.copyfile(full['path'],root/'source_native_manifest.json')
    shutil.copyfile(low['run_path'],root/'source_run_manifest.json')
    segments = []
    for index, identity in enumerate(binding['logical_segments']):
        mask = arrays[grid_id,cent]
        file = f'{index:05d}.npz'
        np.savez_compressed(root/file,mask=mask,extent_xyz=[0,3,0,3,0,1],ijk_to_ras=prep.REFERENCE_RAS)
        segments.append(dict(name=identity['segment_name'],file=file,sha256=edits._xta_sweep_digest(root/file),
            foreground_voxels=int(mask.sum()),source_filename=identity['source_filename'],
            source_sha256=edits._xta_sweep_digest(low['path'].parent/identity['source_filename']),
            recomposition_op=identity['recomposition_op'],logical_id=identity['logical_id'],
            cutoff_cent=cent,sweep_segment_id=identity['segment_id']))
    bundle = dict(format='xta-slicer-edits-v1',source_manifest_name=low['path'].name,
        source_manifest_sha256=low['sha256'],preview_shape_tyx=[2,4,4],full_quality_shape_tyx=[4,8,8],
        sweep_context=binding,segments=segments)
    save(root/'edits.json',bundle)
    return bundle, low['path'], full['path']


def test_source_binding_accepts_same_names_on_distinct_cuts_and_legacy_consumers(tmp_path):
    review, arrays = sweep_fixture(tmp_path)
    data = toolbar_contract()['_xta_sweep_review'](review)
    assert len(data['records']) == 8
    for cent in (25,75):
        child = tmp_path/f'edits-{cent}'
        bundle,low,full = legacy_child(child,review,arrays,cent)
        receipt = edits.validate_sweep_source_pair(child,bundle,low,full)
        assert receipt['cutoff_cent'] == cent and receipt['logical_layer_count'] == 2
        shape,native,selected,_ = prep.load_bundle(child,'sample')
        assert shape==(2,4,4) and native==(4,8,8) and len(selected)==1
        actual = np.zeros(shape,np.uint8)
        prep.insert_snapshot(actual,child/selected[0]['file'],selected[0],1024**2)
        np.testing.assert_array_equal(actual,arrays['low_quality/0p50',cent])


def test_different_cutoff_native_manifest_rejected_before_any_nrrd_access(tmp_path,monkeypatch):
    review,arrays = sweep_fixture(tmp_path)
    child = tmp_path/'edits'
    bundle,low,_ = legacy_child(child,review,arrays)
    wrong = toolbar_contract()['_xta_sweep_review'](review)['grids']['native']['branches'][75]['path']
    # Scalar manifests really can have identical bytes despite different payloads.
    assert edits._xta_sweep_digest(wrong)==bundle['sweep_context']['native_manifest_sha256']
    def forbidden(*args,**kwargs):
        pytest.fail('Wrong-cutoff transfer reached NRRD header or payload access')
    monkeypatch.setattr(nf,'Nrrd',forbidden)
    args = nf.parser().parse_args(['--edits',str(child),'--low-manifest',str(low),
        '--full-manifest',str(wrong),'--output',str(tmp_path/'out'),'--validate-only'])
    with pytest.raises(ValueError,match='bound cutoff branch'):
        nf.run(args)
    assert not (tmp_path/'out').exists()


@pytest.mark.parametrize('change', ['null','cent','logical','segment','name','grid','shape','duplicate'])
def test_malformed_edit_binding_fails_closed(tmp_path,change):
    review,arrays = sweep_fixture(tmp_path)
    child = tmp_path/'edits'
    bundle,low,full = legacy_child(child,review,arrays)
    bad = copy.deepcopy(bundle)
    if change=='null':bad['sweep_context']=None
    elif change=='cent':bad['segments'][0]['cutoff_cent']=75
    elif change=='logical':bad['segments'][0]['logical_id']='another-layer'
    elif change=='segment':bad['segments'][0]['sweep_segment_id']='another-segment'
    elif change=='name':bad['segments'][0]['name']='Renamed on screen'
    elif change=='grid':bad['sweep_context']['grid_id']='native'
    elif change=='shape':bad['preview_shape_tyx']=[2,4,5]
    elif change=='duplicate':bad['sweep_context']['logical_segments'][1]['logical_id']='component'
    with pytest.raises(ValueError):
        edits.validate_sweep_source_pair(child,bad,low,full)


def test_ordinary_legacy_bundle_retains_its_existing_route():
    assert edits.validate_sweep_source_pair(None,{'format':'xta-slicer-edits-v1'},None,None) is None


def test_bound_native_preview_acceptance_still_precedes_native_payload(tmp_path,monkeypatch):
    review,arrays = sweep_fixture(tmp_path)
    child=tmp_path/'edits'
    bundle,low,full = legacy_child(child,review,arrays)
    original_slabs = nf.Nrrd.slabs
    def slabs(self,*args,**kwargs):
        if self.path.parent == full.parent:
            pytest.fail('Invalid edited preview reached native payload processing')
        return original_slabs(self,*args,**kwargs)
    monkeypatch.setattr(nf.Nrrd,'slabs',slabs)
    # A valid lattice and hashes cannot waive the existing Global/union edit check.
    global_edit = next(row for row in bundle['segments'] if row['logical_id']=='global')
    mask = arrays['low_quality/0p50',25].copy()
    mask[:,1,3]=0
    np.savez_compressed(child/global_edit['file'],mask=mask,extent_xyz=[0,3,0,3,0,1],ijk_to_ras=prep.REFERENCE_RAS)
    global_edit.update(sha256=edits._xta_sweep_digest(child/global_edit['file']),foreground_voxels=int(mask.sum()))
    save(child/'edits.json',bundle)
    args=nf.parser().parse_args(['--edits',str(child),'--low-manifest',str(low),'--full-manifest',str(full),
        '--output',str(tmp_path/'out'),'--scratch',str(tmp_path/'scratch'),'--block-mib','1'])
    with pytest.raises(ValueError,match='Global differs'):
        nf.run(args)
    assert not (tmp_path/'out'/'sample.nrrd').exists()


def test_bound_preview_validation_reports_selected_cutoff(tmp_path):
    review,arrays=sweep_fixture(tmp_path)
    child=tmp_path/'edits'
    _,low,full=legacy_child(child,review,arrays)
    output=tmp_path/'validation'
    args=nf.parser().parse_args(['--edits',str(child),'--low-manifest',str(low),'--full-manifest',str(full),
        '--output',str(output),'--scratch',str(tmp_path/'scratch'),'--validate-only','--block-mib','1'])
    assert nf.run(args)==0
    report=edits._xta_sweep_json(output/'report.json')
    assert report['status']=='validated_preview_only'
    assert report['sweep_edit_binding']['cutoff_cent']==25
    assert report['pta_label_file'] is None


@pytest.mark.parametrize('native_payload',['reference_only','grouped',None])
@pytest.mark.parametrize('cent',[25,75])
def test_bound_normal_native_transfer_publishes_exact_selected_source(tmp_path,cent,native_payload):
    review,arrays=sweep_fixture(tmp_path,native_payload)
    child=tmp_path/'edits'
    _,low,full=legacy_child(child,review,arrays,cent)
    output=tmp_path/'native-output'
    args=nf.parser().parse_args(['--edits',str(child),'--low-manifest',str(low),'--full-manifest',str(full),
        '--output',str(output),'--scratch',str(tmp_path/'scratch'),'--block-mib','1'])
    assert nf.run(args)==0
    np.testing.assert_array_equal(nf.Nrrd(output/'sample.nrrd').array(),arrays['native',cent])
    report=edits._xta_sweep_json(output/'report.json')
    assert report['sweep_edit_binding']['cutoff_cent']==cent
    assert report['status']=='complete'


def test_copy_of_a_different_cut_run_receipt_is_rejected(tmp_path):
    review,arrays=sweep_fixture(tmp_path)
    child=tmp_path/'edits'
    bundle,low,full=legacy_child(child,review,arrays)
    run=edits._xta_sweep_json(child/'source_run_manifest.json')
    run['confidence_sweep']['cutoff_hundredths']=75
    save(child/'source_run_manifest.json',run)
    bundle['sweep_context']['run_manifest_sha256']=edits._xta_sweep_digest(child/'source_run_manifest.json')
    with pytest.raises(ValueError,match='run-manifest identity'):
        edits.validate_sweep_source_pair(child,bundle,low,full)


def test_native_extra_rows_and_different_recomposition_roles_are_preserved(tmp_path):
    review,arrays=sweep_fixture(tmp_path)
    data=edits._xta_sweep_json(review)
    native=data['grids'][0]
    branch=native['branches'][0]
    full=review.parent/branch['source_manifest']
    scalar=edits._xta_sweep_json(full)
    scalar['layers'][0]['recomposition_op']='select'
    scalar['layers'].append(dict(scalar['layers'][0],filename='extra.seg.nrrd',layer_key='extra',segment_name='extra'))
    save(full,scalar)
    branch['source_manifest_sha256']=edits._xta_sweep_digest(full)
    native['segments'][0]['recomposition_op']='select'
    save(review,data)
    child=tmp_path/'edits'
    bundle,low,full=legacy_child(child,review,arrays)
    receipt=edits.validate_sweep_source_pair(child,bundle,low,full)
    assert receipt['logical_layer_count']==2
    assert len(edits._xta_sweep_json(full)['layers'])==3


def test_grouped_selection_requires_explicit_unique_cutoff_and_grid(tmp_path):
    review,arrays=sweep_fixture(tmp_path)
    root=tmp_path/'export'
    child=root/'cuts'/'25'
    legacy_child(child,review,arrays)
    index=dict(schema='xta.sweep_edits/1',sweep_id='sweep-fixture',cuts=[dict(cutoff_cent=25,
        grid_id='low_quality/0p50',directory='cuts/25',edits_sha256=edits._xta_sweep_digest(child/'edits.json'))])
    save(root/'sweep_edits.json',index)
    assert edits.resolve_cut_bundle(root,25)==child
    with pytest.raises(ValueError,match='exactly one'):
        edits.resolve_cut_bundle(root,75)
    index['cuts'].append(dict(index['cuts'][0],grid_id='low_quality/other'))
    save(root/'sweep_edits.json',index)
    with pytest.raises(ValueError,match='exactly one'):
        edits.resolve_cut_bundle(root,25)
    assert edits.resolve_cut_bundle(root,25,grid_id='low_quality/0p50')==child


@pytest.mark.parametrize('native_payload',['reference_only','grouped'])
def test_review_rejects_source_escape_and_duplicate_cut_identity(tmp_path,native_payload):
    review,_=sweep_fixture(tmp_path,native_payload)
    original=edits._xta_sweep_json(review)
    bad=copy.deepcopy(original)
    bad['grids'][0]['branches'][0]['source_manifest']='../../escape.json'
    save(review,bad)
    with pytest.raises(ValueError,match='escapes'):
        toolbar_contract()['_xta_sweep_review'](review)
    for grid in (0,1):
        bad=copy.deepcopy(original)
        rows=(bad['grids'][grid]['segments'] if bad['grids'][grid].get('payload')=='reference_only'
              else bad['grids'][grid]['groups'][0]['segments'])
        rows.append(copy.deepcopy(rows[0]))
        save(review,bad)
        with pytest.raises(ValueError,match='Duplicate'):
            toolbar_contract()['_xta_sweep_review'](review)


@pytest.mark.parametrize('change,match',[
    ('preview_reference','Only a native'),('native_groups','Only a native'),('native_coverage','Only a native'),
    ('missing_rows','identity rows'),('unknown','Unsupported review grid payload'),('grouped_rows','inside groups')])
def test_reference_only_grid_rules_fail_closed(tmp_path,change,match):
    review,_=sweep_fixture(tmp_path)
    data=edits._xta_sweep_json(review)
    native,low=data['grids']
    if change=='preview_reference':
        low.update(payload='reference_only',segments=[r for g in low['groups'] for r in g['segments']],groups=[])
    elif change=='native_groups':native['groups']=[dict(path='native.seg.nrrd',segments=native['segments'])]
    elif change=='native_coverage':native['coverage']=[dict(cutoff_cent=25,path='native/conf_025.label.nrrd')]
    elif change=='missing_rows':del native['segments']
    elif change=='unknown':native['payload']='masks_elsewhere'
    elif change=='grouped_rows':low['segments']=[]
    save(review,data)
    with pytest.raises(ValueError,match=match):
        toolbar_contract()['_xta_sweep_review'](review)


def test_review_without_payload_fields_keeps_grouped_meaning(tmp_path):
    review,_=sweep_fixture(tmp_path,native_payload=None)
    data=edits._xta_sweep_json(review)
    assert all('payload' not in grid for grid in data['grids'])
    loaded=toolbar_contract()['_xta_sweep_review'](review)
    assert {grid:value['payload'] for grid,value in loaded['grids'].items()}=={'native':'grouped','low_quality/0p50':'grouped'}
    assert len(loaded['records'])==8


@pytest.mark.parametrize('cent',[True,1.0,-1,101,'25'])
def test_cent_identity_is_exact_integer(cent):
    with pytest.raises(ValueError):edits._xta_sweep_cent(cent)


class MutableTag:
    def __init__(self,value):self.value=value
    def __str__(self):return self.value


class Matrix:
    def __init__(self):self.values=np.eye(4)
    def GetElement(self,row,column):return self.values[row,column]


class Image:
    def GetExtent(self):return self.extent
    def GetImageToWorldMatrix(self,matrix):matrix.values=self.matrix.copy()
    def GetPointData(self):return SimpleNamespace(GetScalars=lambda:self.mask.reshape(-1))


class Segment:
    def __init__(self,tags):self.tags=tags
    def GetTag(self,key,value):
        if key not in self.tags:return False
        value.value=self.tags[key]
        return True
    def GetName(self):return 'Same renamed display name'


class SegmentationNode:
    def __init__(self,rows,arrays):
        self.rows=rows
        self.arrays=arrays
        self.transform=None
        self.matrix=prep.REFERENCE_RAS.copy()
    def GetName(self):return 'Both cutoffs, duplicate display names'
    def GetParentTransformNode(self):return self.transform
    def GetSegmentation(self):return self
    def GetNumberOfSegments(self):return len(self.rows)
    def GetNthSegmentID(self,index):return list(self.rows)[index]
    def GetSegment(self,sid):return self.rows[sid][0]
    def CreateRepresentation(self,representation):return True
    def GetBinaryLabelmapRepresentation(self,sid,image):
        image.mask=self.arrays[self.rows[sid][1]].copy()
        z,y,x=image.mask.shape
        image.extent=[0,x-1,0,y-1,0,z-1]
        image.matrix=self.matrix.copy()
        return True


def export_api(review,arrays):
    path=Path(__file__).resolve().parents[1]/'tools/slicer/xta_slicerrc.py'
    tree=ast.parse(path.read_text(encoding='utf-8'))
    body=[node for node in tree.body if isinstance(node,ast.FunctionDef)
          and (node.name.startswith('_xta_sweep_') or node.name in {'ExportXTASweepEdits','ExportXTAEdits'})]
    namespace=dict(vars(edits),np=np,os=os,shutil=shutil,tempfile=tempfile,datetime=datetime,timezone=timezone)
    statuses=[]
    vtk=SimpleNamespace(mutable=MutableTag,vtkMatrix4x4=Matrix)
    slicer=SimpleNamespace(vtkOrientedImageData=Image,
        vtkSegmentationConverter=SimpleNamespace(GetSegmentationBinaryLabelmapRepresentationName=lambda:'binary'))
    namespace.update(vtk=vtk,slicer=slicer,qt=SimpleNamespace(),
        numpy_support=SimpleNamespace(vtk_to_numpy=lambda values:np.asarray(values)))
    exec(compile(ast.Module(body=body,type_ignores=[]),str(path),'exec'),namespace)
    data=namespace['_xta_sweep_review'](review)
    rows={}
    # Reverse order and duplicate display names deliberately: only tags establish identity.
    for row in reversed(list(data['records'].values())):
        if not row['grid_id'].startswith('low_quality/'):continue
        tags={'XTA.SweepID':data['sweep_id'],'XTA.SegmentID':row['segment_id'],
              'XTA.LogicalID':row['logical_id'],'XTA.CutoffCent':str(row['cutoff_cent']),
              'XTA.GridID':row['grid_id'],'XTA.Role':row['role'],
              'XTA.Model':row['model_name'],'XTA.LayerKey':row['layer_key']}
        rows[f'local-{len(rows)}']=(Segment(tags),(row['grid_id'],row['cutoff_cent']))
    node=SegmentationNode(rows,arrays)
    slicer.util=SimpleNamespace(getNodesByClass=lambda kind:[node],
        showStatusMessage=lambda *values:statuses.append(values),mainWindow=lambda:None)
    return namespace,node,statuses


@pytest.mark.parametrize('native_payload',['reference_only','grouped',None])
def test_real_export_definitions_create_independent_legacy_children_by_tags(tmp_path,native_payload):
    review,arrays=sweep_fixture(tmp_path,native_payload)
    namespace,node,statuses=export_api(review,arrays)
    original_digest=namespace['_xta_sweep_digest']
    def no_native_payload(path):
        if Path(path).suffix=='.nrrd' and 'native' in Path(path).parts:
            pytest.fail('Slicer export opened a native source NRRD payload')
        return original_digest(path)
    namespace['_xta_sweep_digest']=no_native_payload
    before={key:value.copy() for key,value in arrays.items()}
    output=tmp_path/'export'
    assert namespace['ExportXTAEdits'](review,output)==str(output)
    assert statuses and str(output) in statuses[-1][0]
    assert not (output/'edits.json').exists()
    index=edits._xta_sweep_json(output/'sweep_edits.json')
    assert {row['cutoff_cent'] for row in index['cuts']}=={25,75}
    for cent in (25,75):
        child=edits.resolve_cut_bundle(output,cent)
        bundle=edits._xta_sweep_json(child/'edits.json')
        assert len(bundle['segments'])==2
        assert {row['name'] for row in bundle['segments']}=={'sample_Direct','sample_Global_final_output'}
        assert all(row['cutoff_cent']==cent for row in bundle['segments'])
        low,full,selected,_=prep.load_bundle(child,'sample')
        target=np.zeros(low,np.uint8)
        prep.insert_snapshot(target,child/selected[0]['file'],selected[0],1024**2)
        np.testing.assert_array_equal(target,arrays['low_quality/0p50',cent])
    for key,value in arrays.items():np.testing.assert_array_equal(value,before[key])
    assert len(node.rows)==4


def test_export_one_cutoff_and_portable_selection_without_original_sources(tmp_path):
    review,arrays=sweep_fixture(tmp_path)
    namespace,_,_=export_api(review,arrays)
    output=tmp_path/'export'
    namespace['ExportXTASweepEdits'](review,output,cutoff_cents=[75])
    assert len(edits._xta_sweep_json(output/'sweep_edits.json')['cuts'])==1
    moved=tmp_path/'portable'
    output.rename(moved)
    # A copied bundle can be inspected without access to its source tree. Transfer
    # later requires the exact branch layout and bound source run receipt again.
    review.parent.parent.rename(tmp_path/'offline-sources')
    assert edits.resolve_cut_bundle(moved,75).is_dir()
    with pytest.raises(ValueError,match='exactly one'):edits.resolve_cut_bundle(moved,25)


@pytest.mark.parametrize('problem',['tag','duplicate','transform','rotated','fractional_origin','missing_global'])
def test_export_integrity_failure_publishes_no_bundle(tmp_path,problem):
    review,arrays=sweep_fixture(tmp_path)
    namespace,node,_=export_api(review,arrays)
    first=next(iter(node.rows))
    if problem=='tag':node.rows[first][0].tags['XTA.LogicalID']='wrong'
    elif problem=='duplicate':node.rows['duplicate']=node.rows[first]
    elif problem=='transform':node.transform=object()
    elif problem=='rotated':node.matrix[0,0]=0
    elif problem=='fractional_origin':node.matrix[0,3]=.25
    elif problem=='missing_global':
        node.rows={sid:value for sid,value in node.rows.items() if value[0].tags['XTA.Role']!='global'}
    output=tmp_path/'export'
    with pytest.raises((ValueError,RuntimeError)):
        namespace['ExportXTASweepEdits'](review,output)
    assert not output.exists()
    assert not list(tmp_path.glob('.xta-sweep-edits-*'))


def test_source_tree_relocation_preserves_cut_binding(tmp_path):
    review,arrays=sweep_fixture(tmp_path)
    child=tmp_path/'edits'
    bundle,low,full=legacy_child(child,review,arrays)
    source_root=review.parent.parent
    relative_low,relative_full=low.relative_to(source_root),full.relative_to(source_root)
    relocated=tmp_path/'new-machine'/'sources'
    relocated.parent.mkdir()
    source_root.rename(relocated)
    receipt=edits.validate_sweep_source_pair(child,bundle,relocated/relative_low,relocated/relative_full)
    assert receipt['cutoff_cent']==25


@pytest.mark.parametrize('native_payload',['reference_only','grouped'])
@pytest.mark.parametrize('cent',[25,75])
def test_independent_scalar_export_and_normal_native_transfer(tmp_path,cent,native_payload):
    review,arrays=independent_fixture(tmp_path,native_payload)
    namespace,_,_=export_api(review,arrays)
    output=tmp_path/'edits'
    namespace['ExportXTASweepEdits'](review,output)
    child=edits.resolve_cut_bundle(output,cent)
    bundle=edits._xta_sweep_json(child/'edits.json')
    assert bundle['sweep_context']['source_mode']=='independent_scalar'
    data=namespace['_xta_sweep_review'](review)
    low=data['grids']['low_quality/0p50']['branches'][cent]['path']
    full=data['grids']['native']['branches'][cent]['path']
    wrong=data['grids']['native']['branches'][75 if cent==25 else 25]['path']
    assert edits._xta_sweep_digest(full)==edits._xta_sweep_digest(wrong)
    with pytest.raises(ValueError,match='bound cutoff branch'):
        edits.validate_sweep_source_pair(child,bundle,low,wrong)
    native_output=tmp_path/'native-output'
    args=nf.parser().parse_args(['--edits',str(child),'--low-manifest',str(low),'--full-manifest',str(full),
        '--output',str(native_output),'--scratch',str(tmp_path/'scratch'),'--block-mib','1'])
    assert nf.run(args)==0
    np.testing.assert_array_equal(nf.Nrrd(native_output/'sample.nrrd').array(),arrays['native',cent])


@pytest.mark.parametrize('conf',['0.251','0.25000000000000000000000000000000000000000000000000001',
                                 'NaN','true','null','1.25'])
def test_independent_original_conf_must_match_exact_finite_lattice(tmp_path,conf):
    review,_=independent_fixture(tmp_path)
    data=edits._xta_sweep_json(review)
    run=review.parent/data['grids'][0]['branches'][0]['run_manifest']
    run.write_text('{"status":"complete","launcher":{"mode":"tta"},"resolved_configuration":{"conf":'+conf+'}}',encoding='utf-8')
    for grid in data['grids']:
        grid['branches'][0]['run_manifest_sha256']=edits._xta_sweep_digest(run)
    save(review,data)
    with pytest.raises(ValueError):toolbar_contract()['_xta_sweep_review'](review)


def test_published_reference_only_native_binds_each_cutoff_without_native_payload(tmp_path):
    from tests.test_sweep_publication import scalar
    from XTA.sweep_publication import publish_sweep_review
    branches=[scalar(tmp_path,cent)[0] for cent in (50,80)]
    review=publish_sweep_review(tmp_path,branches,sweep_id='sweep-test',memory_mib=2)
    api=toolbar_contract()
    original_digest=api['_xta_sweep_digest']
    def manifests_only(path):
        if Path(path).name.endswith('.seg.nrrd'):
            pytest.fail('Review binding hashed a scalar mask payload')
        return original_digest(path)
    api['_xta_sweep_digest']=manifests_only
    data=api['_xta_sweep_review'](review)
    assert data['grids']['native']['payload']=='reference_only'
    for cent,branch in branches:
        low_rows=data['grids']['low_quality/half']['by_cut'][cent]
        logical={row['logical_id'] for row in low_rows.values()
                 if row['original'].get('suffix')=='Global_final_output' or row['recomposition_op']=='union'}
        binding=api['_xta_sweep_binding'](data,'low_quality/half',cent,logical)
        native_manifest=branch/'nrrd'/'volume_nrrd_manifest.json'
        assert binding['native_manifest_relative']==native_manifest.relative_to(tmp_path).as_posix()
        assert binding['native_manifest_sha256']==edits._xta_sweep_digest(native_manifest)
        assert binding['branch_dir_relative']==f'confidence_cutoffs/conf_{cent:03d}'
        assert {row['layer_key'] for row in binding['logical_segments']}=={'d','t','b','g'}
        assert all(row['segment_id']==low_rows[row['logical_id']]['segment_id'] for row in binding['logical_segments'])


@pytest.mark.parametrize('field,value',[('status','incomplete'),('mode','pta'),('source_mode','confidence_sweep')])
def test_independent_mode_does_not_waive_run_or_source_identity(tmp_path,field,value):
    review,_=independent_fixture(tmp_path)
    data=edits._xta_sweep_json(review)
    if field=='source_mode':
        data['grids'][0]['branches'][0]['source_mode']=value
    else:
        run=review.parent/data['grids'][0]['branches'][0]['run_manifest']
        receipt=edits._xta_sweep_json(run)
        if field=='mode':receipt['launcher']['mode']=value
        else:receipt[field]=value
        save(run,receipt)
        for grid in data['grids']:grid['branches'][0]['run_manifest_sha256']=edits._xta_sweep_digest(run)
    save(review,data)
    with pytest.raises(ValueError):toolbar_contract()['_xta_sweep_review'](review)

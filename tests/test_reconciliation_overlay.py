"""Source-grid overlay sampling and report-driven comparison rendering."""
from __future__ import annotations

import base64
import gzip
import io
import json
from pathlib import Path
import re
import shutil
import subprocess

import cv2
import numpy as np
import pytest

from tools import overlay_reconciliation as overlay_tool
from tools.overlay_reconciliation import overlay, resize_stream, sampling_plan, selected_masks
from XTA.reconciliation_io import write_seg_nrrd


@pytest.mark.parametrize('source_shape,target_shape', [((11,17,19),(4,7,8)), ((3,7,5),(8,9,10)), ((1,4,6),(1,2,3))])
def test_streaming_gray_matches_authoritative_output_resize(tmp_path, monkeypatch, source_shape, target_shape):
    from XTA import outputs
    from XTA.runtime import close_memmap_array
    rng = np.random.default_rng(930)
    source = rng.integers(0, 256, source_shape, np.uint8)
    actual = np.empty(target_shape, np.uint8)
    monkeypatch.setattr(outputs, '_try_gpu_downbin_volume', lambda *args: False)
    expected = outputs.resize_gray_volume_to_shape(source, target_shape, tmp_path / 'reference.dat', workers=1)
    try:
        envelopes = resize_stream(iter(source), source_shape, actual, [0, target_shape[0]-1])
        np.testing.assert_array_equal(actual, expected)
        _, footprint = sampling_plan(source_shape, target_shape, envelopes)
        interpolation = cv2.INTER_AREA if all(a<=b for a,b in zip(target_shape[1:],source_shape[1:])) else cv2.INTER_LINEAR
        for z, envelope in envelopes.items():
            lo, hi = footprint[z]
            independently_resized = [cv2.resize(source[i], tuple(reversed(target_shape[1:])), interpolation=interpolation)
                                     for i in range(lo, hi)]
            np.testing.assert_array_equal(envelope, np.maximum.reduce(independently_resized))
    finally:
        close_memmap_array(expected)


def test_production_slice_mapping_uses_source_frame_count():
    plan, windows = sampling_plan((1931,3064,3022), (388,612,604), [124,194])
    assert plan[124][:2] == (618,619)
    assert plan[124][2] == pytest.approx(154/387)
    assert plan[194][:2] == (967,968)
    assert plan[194][2] == pytest.approx(191/387)
    assert windows == {124:(617,623),194:(965,971)}


def test_bad_frame_count_and_shape_fail():
    out = np.empty((2,3,4),np.uint8)
    with pytest.raises(ValueError, match='requires'):
        resize_stream(iter(np.zeros((1,3,4),np.uint8)), (2,3,4), out)
    with pytest.raises(ValueError, match='more frames'):
        resize_stream(iter(np.zeros((3,3,4),np.uint8)), (2,3,4), out)
    with pytest.raises(ValueError, match='differs'):
        resize_stream(iter(np.zeros((2,4,3),np.uint8)), (2,3,4), out)


def test_overlay_retains_background_and_handles_only_candidate_voxels():
    source = np.arange(12,dtype=np.uint8).reshape(3,4)
    candidate = np.zeros_like(source); candidate[1,1:3] = 1
    retained = np.zeros_like(source); retained[1,1] = 1
    original = source.copy(), candidate.copy(), retained.copy()
    result = overlay(source,candidate,retained,alpha=1)
    np.testing.assert_array_equal(result[1,1], [45,190,135])
    np.testing.assert_array_equal(result[1,2], [240,85,85])
    np.testing.assert_array_equal(result[0], np.repeat(source[0,:,None],3,axis=1))
    for before,after in zip(original,(source,candidate,retained)):
        np.testing.assert_array_equal(before,after)
    retained[0,0] = 1
    with pytest.raises(ValueError,match='outside'):
        overlay(source,candidate,retained)


def test_saved_mask_slices_decode_axes_and_concatenated_gzip(tmp_path):
    data = np.zeros((12,5,7),np.uint8)
    data[3,1,6] = 1; data[10,4,2] = 1
    header = (b'NRRD0005\ntype: uint8\ndimension: 3\nsizes: 7 5 12\n'
              b'space: left-posterior-superior\nspace directions: (1,0,0) (0,1,0) (0,0,1)\n'
              b'space origin: (0,0,0)\nencoding: gzip\n\n')
    path = tmp_path / 'mask.nrrd'
    raw = data.tobytes()
    path.write_bytes(header + gzip.compress(raw[:117]) + gzip.compress(raw[117:]))
    geometry = dict(space='left-posterior-superior',directions_xyz=np.eye(3),origin_xyz=[0,0,0])
    result = selected_masks(path,data.shape,[3,10],geometry)
    np.testing.assert_array_equal(result[3],data[3])
    np.testing.assert_array_equal(result[10],data[10])
    path.write_bytes(header.replace(b'(0,0,0)',b'(1,0,0)')+gzip.compress(raw))
    with pytest.raises(ValueError,match='geometry differs'):
        selected_masks(path,data.shape,[3],geometry)


CURATED = ('confidence_core_rescue', 'quorum3', 'confidence_anchored',
           'largest_island', 'hybrid_with_fill')


def saved_comparison(tmp_path, names, *, reference='union'):
    shape = (1, 8, 10)
    gray = np.full(shape, 100, np.uint8)
    candidate = np.zeros(shape, np.uint8)
    candidate[:, 1:7, 1:9] = 1
    methods = []
    masks = {reference: candidate}
    for index, name in enumerate(names):
        masks[name] = candidate.copy()
        masks[name][:, 2 + index % 4, 3:6] = 0
    for name, mask in masks.items():
        path = tmp_path / f'{name}.seg.nrrd'
        write_seg_nrrd(path, shape_tyx=shape, read_slab=lambda lo, hi, value=mask: value[lo:hi])
        methods.append(dict(policy_name=name, status='complete', output=str(path),
                            is_union_reference=name == reference))
    # Unavailable confidence methods have no output to open.
    methods.append(dict(policy_name='unavailable', status='unavailable'))
    dataset = dict(dataset='run', methods=methods, reference_geometry=dict(
        space='left-posterior-superior', directions_xyz=np.eye(3).tolist(), origin_xyz=[0, 0, 0]),
        previews=dict(slices=[dict(z=0, union_slice_voxels=int(candidate.sum()), roi_y0_y1_x0_x1=[1, 7, 1, 9])]))
    return dataset, gray, {0: gray[0].copy()}


@pytest.mark.parametrize('names', [CURATED, ('cross_sections', 'largest_island')], ids=['curated', 'historical'])
def test_report_methods_render_without_retired_policy_files_and_preserve_pixels(tmp_path, monkeypatch, names):
    pytest.importorskip('matplotlib')
    image_module = pytest.importorskip('PIL.Image')
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'mpl'))
    dataset, gray, envelopes = saved_comparison(tmp_path, names)
    output = tmp_path / 'render'
    output.mkdir()
    record, = overlay_tool.render_dataset(output, dataset, gray, envelopes)
    assert list(record['layers']) == list(names)
    assert record['overview_methods'] == list(names[:5])
    assert record['roi'] == [1, 7, 1, 9]
    assert record['union_reference'] == 'union'
    assert 'unavailable' not in record['layers']
    with image_module.open(output / 'run' / f'z0_{names[0]}_full.png') as image:
        values = np.asarray(image)
    assert values.shape == (8, 10, 3)
    np.testing.assert_array_equal(values[0, 0], [100, 100, 100])
    np.testing.assert_array_equal(values[1, 1], [79, 134, 113])  # unchanged 38% green
    np.testing.assert_array_equal(values[2, 3], [153, 94, 94])   # unchanged 38% red
    rgba = np.asarray(image_module.open(io.BytesIO(base64.b64decode(record['layers'][names[0]].split(',', 1)[1]))))
    np.testing.assert_array_equal(rgba[1, 1], [45, 190, 135, 255])
    np.testing.assert_array_equal(rgba[2, 3], [240, 85, 85, 255])
    assert rgba[0, 0, 3] == 0
    assert (output / 'run' / 'z0_comparison.png').is_file()
    if 'cross_sections' in names:
        assert record['method_labels']['cross_sections'] == 'Cross sections'


def test_custom_reference_and_union_only_report_are_supported(tmp_path, monkeypatch):
    pytest.importorskip('matplotlib')
    pytest.importorskip('PIL.Image')
    monkeypatch.setenv('MPLCONFIGDIR', str(tmp_path / 'mpl'))
    dataset, gray, envelopes = saved_comparison(tmp_path, (), reference='raw_reference')
    output = tmp_path / 'render'
    output.mkdir()
    record, = overlay_tool.render_dataset(output, dataset, gray, envelopes)
    assert record['union_reference'] == 'raw_reference'
    assert record['layers'] == {} and record['overview_methods'] == []
    assert (output / 'run' / 'z0_comparison.png').is_file()


def test_incomplete_or_ambiguous_union_reports_fail_clearly():
    with pytest.raises(ValueError, match='completed raw-union reference'):
        overlay_tool.comparison_methods(dict(methods=[dict(policy_name='quorum3', status='complete')]))
    with pytest.raises(ValueError, match='Duplicate completed'):
        overlay_tool.comparison_methods(dict(methods=[dict(policy_name='union', status='complete')] * 2))


def test_viewer_selects_report_methods_and_switches_to_historical_and_union_only_records(tmp_path):
    node = shutil.which('node')
    if node is None:
        pytest.skip('Node is needed for the standalone viewer behavior check')
    def record(names):
        return dict(dataset='run', z=0, roi=[1, 7, 1, 9], source='source', slab='slab',
                    layers={name: name for name in names}, source_position=0., source_seconds=0., mask_frames=[0, 0])
    viewer = tmp_path / 'review.html'
    overlay_tool.write_viewer(viewer, [record(CURATED), record(('cross_sections', 'largest_island')), record(())])
    script = re.search(r'<script>(.*)</script>', viewer.read_text(), re.S).group(1)
    script_path = tmp_path / 'viewer.js'
    script_path.write_text(script, encoding='utf-8')
    subprocess.run([node, '--check', str(script_path)], check=True, capture_output=True, text=True)
    harness = tmp_path / 'viewer_test.cjs'
    harness.write_text(r'''
const fs=require('fs'),vm=require('vm'),assert=require('assert');
class Element {
  constructor(id){this.id=id;this.value='';this.children=[];this.hidden=false;this.checked=true;this.style={};}
  append(value){this.children.push(value);if(this.value==='')this.value=value.value;}
  replaceChildren(){this.children=[];this.value='';}
  addEventListener(){}
  getContext(){return {drawImage(){},getImageData(){return {data:new Uint8ClampedArray(4)};},putImageData(){}};}
}
const elements=new Map(),get=id=>{if(!elements.has(id))elements.set(id,new Element(id));return elements.get(id);};
get('sample').value='0';get('alpha').value='38';get('background').value='source';
class Picture {constructor(){this.width=10;this.height=8;}set src(value){assert.notStrictEqual(value,undefined);setImmediate(()=>this.onload());}}
const context={document:{getElementById:get,createElement:tag=>new Element(tag)},Image:Picture};
vm.createContext(context);vm.runInContext(fs.readFileSync(process.argv[2],'utf8'),context);
(async()=>{
 await context.render();
 assert.deepStrictEqual(get('method_a').children.map(o=>o.value),JSON.parse(process.argv[3]));
 assert.strictEqual(get('method_a').value,'confidence_core_rescue');
 assert.strictEqual(get('method_b').value,'quorum3');
 assert.strictEqual(get('alpha').value,'38');
 assert.strictEqual(get('original').width,8);assert.strictEqual(get('original').height,6);
 get('method_a').value='hybrid_with_fill';await context.render();
 assert.strictEqual(get('method_a').value,'hybrid_with_fill');assert.strictEqual(get('title_a').textContent,'Hybrid with fill');
 get('sample').value='1';await context.render();
 assert.deepStrictEqual(get('method_a').children.map(o=>o.value),['cross_sections','largest_island']);
 assert.strictEqual(get('title_a').textContent,'Cross sections');assert.strictEqual(get('title_b').textContent,'Largest island');
 get('sample').value='2';await context.render();
 assert.strictEqual(get('panel_a').hidden,true);assert.strictEqual(get('panel_b').hidden,true);
 assert.strictEqual(get('method_a_control').hidden,true);assert.strictEqual(get('method_b_control').hidden,true);
 assert.strictEqual(get('panels').style.gridTemplateColumns,'repeat(1,minmax(0,1fr))');
 console.log(JSON.stringify({curated:true,historical:true,union_only:true}));
})().catch(error=>{console.error(error);process.exitCode=1;});
''', encoding='utf-8')
    result = subprocess.run([node, str(harness), str(script_path), json.dumps(CURATED)],
                            check=True, capture_output=True, text=True)
    assert json.loads(result.stdout) == dict(curated=True, historical=True, union_only=True)

"""Exact full-domain oracles for zero-margin CCL/dilation optimizations."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from XTA import sam_policy


class Bundle:
    def __init__(self, volume, *, endpoints=None, unrelated=None):
        self.volume = np.asarray(volume, bool)
        t, h, w = self.volume.shape
        self.candidate_reads = 0
        a, b = ('A', 'B')
        source, target = endpoints or (self.volume[0].copy(), self.volume[-1].copy())
        self.masks = {'endpoint:A':source, 'endpoint:B':target}
        for frame in range(t):
            self.masks[f'edge_contract:E:{frame}'] = np.ones((h,w),bool)
            self.masks[f'known_foreground:{frame}'] = np.zeros((h,w),bool)
            self.masks[f'unrelated:{frame}'] = np.zeros((h,w),bool) if unrelated is None else unrelated[frame]
        self.groups = {'G':dict(group_id='G',context_bbox_yx=(0,0,h,w),frame_indices=list(range(t)),
            endpoints=[dict(observation_id=a,frame_index=0),dict(observation_id=b,frame_index=t-1)],
            edges=[dict(edge_id='E',source_id=a,target_id=b)],mask_keys={key:key for key in self.masks})}
        self.runs = {'R':dict(observed_frames=list(range(t)),group_id='G')}
    def group_mask(self, group_id, key):
        return self.masks[key]
    def candidate_mask(self, run_id, frame):
        self.candidate_reads += 1
        return self.volume[frame]


def original_topology(bundle, selected, connectivity):
    """Frozen v25 full-shape algorithm, including IDs and supporting owners."""
    group = bundle.groups['G']
    shape = bundle.volume.shape
    additions = np.zeros(shape,bool)
    for run_id in selected:
        for frame in bundle.runs[run_id]['observed_frames']:
            additions[frame] |= bundle.candidate_mask(run_id,frame)
    endpoints = {row['observation_id']:row for row in group['endpoints']}
    structure = ndi.generate_binary_structure(3,{6:1,18:2,26:3}[connectivity])
    edges = []
    for edge in group['edges']:
        source, target = endpoints[edge['source_id']], endpoints[edge['target_id']]
        lo, hi = sorted((source['frame_index'],target['frame_index']))
        local = additions[lo:hi+1].copy()
        for frame in range(lo,hi+1):
            contract = bundle.group_mask('G',f"edge_contract:{edge['edge_id']}:{frame}")
            local[frame-lo] &= contract
            local[frame-lo] |= bundle.group_mask('G',f'known_foreground:{frame}') & contract
        source_mask = bundle.group_mask('G',f"endpoint:{source['observation_id']}")
        target_mask = bundle.group_mask('G',f"endpoint:{target['observation_id']}")
        local[source['frame_index']-lo] |= source_mask
        local[target['frame_index']-lo] |= target_mask
        labels, _ = ndi.label(local,structure=structure)
        a = set(map(int,np.unique(labels[source['frame_index']-lo][source_mask])))-{0}
        b = set(map(int,np.unique(labels[target['frame_index']-lo][target_mask])))-{0}
        common = a & b
        path = additions[lo:hi+1] & np.isin(labels,list(common)) if common else np.zeros(local.shape,bool)
        owners = []
        for run_id in selected:
            if any(frame in bundle.runs[run_id]['observed_frames'] and
                   np.any(bundle.candidate_mask(run_id,frame)&path[frame-lo]) for frame in range(lo,hi+1)):
                owners.append(run_id)
        edges.append(dict(edge_id=edge['edge_id'],source_id=source['observation_id'],target_id=target['observation_id'],
            connected=bool(common) and bool(path.any()),local_native_interval=[lo,hi],
            supporting_component_labels=sorted(common),supporting_run_ids=sorted(owners),
            local_addition_voxels=int(np.count_nonzero(path))))
    dilated = ndi.binary_dilation(additions,structure=structure)
    count = sum(int(np.count_nonzero(dilated[frame]&bundle.group_mask('G',f'unrelated:{frame}')))
                for frame in group['frame_indices'])
    return dict(connectivity=connectivity,edges=edges,
        all_requested_edges_connected=bool(edges) and all(row['connected'] for row in edges),
        selected_addition_voxels=int(np.count_nonzero(additions)),unintended_contact_voxels=count,
        unintended_contact_status='measured')


@pytest.mark.parametrize('connectivity',(6,18,26))
@pytest.mark.parametrize('shape',((1,1,1),(1,9,11),(7,13,15)))
def test_cropped_labels_keep_exact_full_domain_ids(connectivity,shape):
    rng = np.random.default_rng(734)
    structure = ndi.generate_binary_structure(3,{6:1,18:2,26:3}[connectivity])
    for case in range(12):
        volume = rng.random(shape) < (.05 if case%2 else .3)
        if case%3 == 0:
            volume[:] = False
            if min(shape)>4:
                volume[2:-2,2:-2,2:-2] = rng.random(tuple(n-4 for n in shape)) < .4
        labels, bounds = sam_policy._label_foreground_crop(volume,structure)
        expected, _ = ndi.label(volume,structure=structure)
        if bounds is None:
            assert not expected.any() and labels.size == 0
        else:
            lower, upper = bounds
            np.testing.assert_array_equal(labels,expected[tuple(slice(a,b) for a,b in zip(lower,upper))])


@pytest.mark.parametrize('connectivity',(6,18,26))
@pytest.mark.parametrize('kind',('empty','dense','sparse','interior','crop_edges','spurs'))
def test_topology_receipt_equals_original_full_domain(connectivity,kind):
    rng = np.random.default_rng(938)
    volume = np.zeros((7,19,23),bool)
    if kind=='dense':
        volume[:] = rng.random(volume.shape)<.65
    elif kind=='sparse':
        volume[:] = rng.random(volume.shape)<.04
    elif kind=='interior':
        volume[:,6:13,8:15] = True
        volume[2,2,2] = True
    elif kind=='crop_edges':
        volume[:,:,0] = True
        volume[:,0,:] = True
        volume[-1,-1,-1] = True
    elif kind=='spurs':
        volume[:,8:11,10:13] = True
        volume[3,9,12:23] = True
    unrelated = rng.random(volume.shape)<.2
    bundle = Bundle(volume,unrelated=unrelated)
    expected = original_topology(bundle,['R'],connectivity)
    actual = sam_policy.measure_group_topology(bundle,'G',['R'],connectivity=connectivity)
    assert actual == expected


def test_empty_path_does_not_decode_supporting_candidates_again():
    volume = np.zeros((5,19,23),bool)
    volume[2,9,10] = True
    a, b = np.zeros(volume.shape[1:],bool), np.zeros(volume.shape[1:],bool)
    a[1,1], b[-2,-2] = True, True
    bundle = Bundle(volume,endpoints=(a,b))
    expected = original_topology(bundle,['R'],26)
    old_reads = bundle.candidate_reads
    bundle.candidate_reads = 0
    actual = sam_policy.measure_group_topology(bundle,'G',['R'],connectivity=26)
    assert actual == expected
    assert bundle.candidate_reads == 5 and old_reads > bundle.candidate_reads


def test_empty_additions_skip_dilation_but_retain_measured_status(monkeypatch):
    bundle = Bundle(np.zeros((5,19,23),bool))
    expected = original_topology(bundle,[],26)
    monkeypatch.setattr(sam_policy.ndimage,'binary_dilation',lambda *a,**k:pytest.fail('Empty volume dilated'))
    actual = sam_policy.measure_group_topology(bundle,'G',[],connectivity=26)
    assert actual == expected and actual['unintended_contact_status']=='measured'


def test_fixed_continuation_keeps_labels_and_skips_empty_path_planes():
    volume = np.zeros((5,19,23),bool)
    volume[2,9,10] = True
    endpoint = np.zeros(volume.shape[1:],bool)
    endpoint[9,10] = True
    bundle = Bundle(volume,endpoints=(endpoint,endpoint))
    for frame in range(5):
        bundle.masks[f'known_foreground:{frame}'][9,10] = True
    expected = original_topology(bundle,['R'],26)
    old_reads = bundle.candidate_reads
    bundle.candidate_reads = 0
    actual = sam_policy.measure_group_topology(bundle,'G',['R'],connectivity=26)
    assert actual == expected
    assert actual['edges'][0]['connected'] and actual['edges'][0]['supporting_component_labels']==[1]
    assert bundle.candidate_reads == 6 and old_reads > bundle.candidate_reads


def test_nonzero_label_ids_preserved_with_multiple_fixed_endpoints():
    volume = np.zeros((5,19,23),bool)
    volume[:,4:6,5:7] = True
    volume[:,12:14,15:17] = True
    bundle = Bundle(volume)
    bundle.groups['G']['endpoints'].extend([dict(observation_id='C',frame_index=1),
                                         dict(observation_id='D',frame_index=3)])
    bundle.groups['G']['edges'].append(dict(edge_id='F',source_id='C',target_id='D'))
    second = np.zeros(volume.shape[1:],bool)
    second[12:14,15:17] = True
    bundle.masks['endpoint:C'] = second
    bundle.masks['endpoint:D'] = second
    for frame in range(5):
        bundle.masks[f'edge_contract:F:{frame}'] = np.ones(volume.shape[1:],bool)
    actual = sam_policy.measure_group_topology(bundle,'G',['R'],connectivity=26)
    assert actual == original_topology(bundle,['R'],26)
    assert actual['edges'][1]['supporting_component_labels']==[2]


def test_sparse_bbox_does_not_relax_full_shape_admission():
    bundle = Bundle(np.zeros((5,19,23),bool))
    bundle.volume[:,9,10] = True
    with pytest.raises(MemoryError,match='group memory budget'):
        sam_policy.measure_group_topology(bundle,'G',['R'],max_group_bytes=bundle.volume.size*16-1)

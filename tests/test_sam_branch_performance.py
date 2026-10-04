"""Bounded branch reuse avoids repeat labeling without changing selected pixels."""
from collections import Counter
from copy import deepcopy
from unittest import mock

import numpy as np
import pytest
from scipy import ndimage

from XTA import sam_branch_selection as branch
from XTA.sam_evidence import SamEvidenceWriter, _plain, fingerprint
from XTA.sam_filtering import build_mask_filter
from tests.test_sam_branch_support import bundle_fixture


def multiple_edges(tmp_path):
    original, _ = bundle_fixture(tmp_path/'original', grown=True)
    group = _plain(original.groups['g'])
    group['edges'].append(dict(edge_id='e2', source_id='a', target_id='b'))
    masks = {name: original.group_mask('g', name) for name in group['mask_keys']}
    for name, plane in list(masks.items()):
        if name.startswith(('edge_write:e:', 'edge_contract:e:')):
            masks[name.replace(':e:', ':e2:')] = plane
    with SamEvidenceWriter(tmp_path/'bundle', dict(shape_tyx=[5,13,15])) as writer:
        writer.add_group(group, masks)
        for run_id in ('f', 'r'):
            run = _plain(original.runs[run_id]); run['edge_ids'] = ['e', 'e2']
            writer.add_run(run, {frame: original.raw_mask(run_id, frame) for frame in run['expected_frames']})
        return writer.commit()


def build_counted(bundle, spool_bytes):
    counts = Counter()
    original = branch._uncached_seed_rooted_raw
    def counted(*args, **kwargs):
        counts[args[2]] += 1
        return original(*args, **kwargs)
    with bundle.reader(max_cache_bytes=0) as reader, mock.patch.object(branch, '_uncached_seed_rooted_raw', side_effect=counted):
        radius = build_mask_filter(bundle, enabled=False)
        recipe, diagnostic = branch.build_connected_edge_selection(reader, radius, 'g',
            {'e': ['f','r'], 'e2': ['f','r']}, write_domain='fixed_context', owner_spool_bytes=spool_bytes)
    return recipe, diagnostic, counts


def test_tiny_reader_cache_roots_each_owner_once_per_edge_with_bounded_spool(tmp_path):
    bundle = multiple_edges(tmp_path)
    recipe, _, counts = build_counted(bundle, 1024**2)
    assert set(recipe['edges']) == {'e','e2'}
    assert counts == {'f': 2, 'r': 2}


def test_spool_overflow_recomputes_without_changing_quality_or_packed_pixels(tmp_path):
    bundle = multiple_edges(tmp_path)
    cached, cached_diagnostic, cached_counts = build_counted(bundle, 1024**2)
    fallback, fallback_diagnostic, fallback_counts = build_counted(bundle, 1)
    assert cached == fallback
    assert cached_diagnostic == fallback_diagnostic
    assert fallback_counts == {'f': 4, 'r': 4}
    assert sum(fallback_counts.values()) > sum(cached_counts.values())


def two_groups(tmp_path):
    original, _ = bundle_fixture(tmp_path/'original', grown=True)
    with SamEvidenceWriter(tmp_path/'bundle', dict(shape_tyx=[5,13,30])) as writer:
        for suffix, x in (('', 0), ('2', 15)):
            group = _plain(original.groups['g'])
            group['group_id'] = 'g'+suffix
            group['context_bbox_yx'] = [0,x,13,x+15]
            for endpoint in group['endpoints']:
                endpoint['observation_id'] += suffix
            for edge in group['edges']:
                for field in ('edge_id','source_id','target_id'):
                    edge[field] += suffix
            masks = {}
            for name in original.groups['g']['mask_keys']:
                parts = name.split(':')
                if parts[0] in ('edge_write','edge_contract','endpoint','evaluation','permitted'):
                    parts[1] += suffix
                masks[':'.join(parts)] = original.group_mask('g',name)
            writer.add_group(group,masks)
            for identifier in ('f','r'):
                run = _plain(original.runs[identifier])
                run.update(run_id=identifier+suffix,group_id='g'+suffix,
                    seed_ids=[name+suffix for name in run['seed_ids']],
                    held_out_ids=[name+suffix for name in run['held_out_ids']],edge_ids=['e'+suffix])
                writer.add_run(run,{frame:original.raw_mask(identifier,frame) for frame in run['expected_frames']})
        return writer.commit()


def test_unrelated_recipe_append_reuses_existing_owner_candidate_and_packet(tmp_path):
    bundle = two_groups(tmp_path)
    radius = build_mask_filter(bundle,enabled=False)
    with bundle.reader() as reader:
        one,_ = branch.build_connected_edge_selection(reader,radius,'g',{'e':['f','r']},write_domain='fixed_context')
        two,_ = branch.build_connected_edge_selection(reader,radius,'g2',{'e2':['f2','r2']},write_domain='fixed_context')
        first = reader.filter_snapshot(dict(mask_filter=radius,branch_selection=one))
        mask = reader.effective_candidate_mask('f',2,first)
        computations = reader.stats['effective_candidate_computations']
        combined = branch.merge_branch_selections(bundle,radius,[one,two],write_domain='fixed_context')
        second = reader.filter_snapshot(dict(mask_filter=radius,branch_selection=combined))
        np.testing.assert_array_equal(reader.effective_candidate_mask('f',2,second),mask)
        assert reader.stats['effective_candidate_computations'] == computations


def test_changed_actual_packet_data_cannot_hit_old_candidate_cache(tmp_path):
    bundle = two_groups(tmp_path)
    radius = build_mask_filter(bundle,enabled=False)
    with bundle.reader() as reader:
        recipe,_ = branch.build_connected_edge_selection(reader,radius,'g',{'e':['f','r']},write_domain='fixed_context')
        first = reader.filter_snapshot(dict(mask_filter=radius,branch_selection=recipe))
        reader.effective_candidate_mask('f',2,first)
        corrupted = deepcopy(recipe)
        corrupted['edges']['e']['owner_support']['f']['2']['data'] = 'AAAA'
        # Preserve the claimed packet checksum while resealing the outer spec.
        corrupted['sha256'] = fingerprint({key:value for key,value in corrupted.items() if key != 'sha256'})
        second = reader.filter_snapshot(dict(mask_filter=radius,branch_selection=corrupted))
        import pytest
        with pytest.raises(ValueError,match='checksum'):
            reader.effective_candidate_mask('f',2,second)


def test_direct_only_edges_skip_empty_partial_volume_dilation(tmp_path):
    bundle = multiple_edges(tmp_path)
    radius = build_mask_filter(bundle,enabled=False)
    with bundle.reader() as reader, mock.patch.object(branch.ndimage,'binary_dilation',
            side_effect=AssertionError('No partial owners require no meeting dilation')):
        recipe,_ = branch.build_connected_edge_selection(reader,radius,'g',{'e':['f','r']},write_domain='fixed_context')
    assert set(recipe['edges']) == {'e'}


@pytest.mark.parametrize('connectivity',(1,2,3))
@pytest.mark.parametrize('kind',('sparse','empty','canvas_boundary','diagonal'))
def test_foreground_ccl_crop_preserves_original_scan_labels(connectivity,kind):
    volume=np.zeros((7,19,23),bool)
    if kind=='sparse':
        volume[2:6,4:13,7:19]=np.random.default_rng(123).random((4,9,12))<.12
    elif kind=='canvas_boundary':
        volume[0,:3,:4]=True; volume[6,-2:,-2:]=True
    elif kind=='diagonal':
        for offset in range(4): volume[offset+2,offset+7,offset+9]=True
    structure=ndimage.generate_binary_structure(3,connectivity)
    expected,_=ndimage.label(volume,structure=structure)
    labels,bounds=branch._label_foreground_crop(volume,structure)
    if bounds is None:
        assert not expected.any() and not labels.size
    else:
        lower,upper=bounds
        np.testing.assert_array_equal(labels,expected[tuple(slice(a,b) for a,b in zip(lower,upper))])


def test_sparse_owner_labeling_visits_foreground_bounds_not_full_family_canvas(tmp_path):
    bundle=multiple_edges(tmp_path)
    radius=build_mask_filter(bundle,enabled=False)
    cells=[]; original=branch.ndimage.label
    def observe(volume,*args,**kwargs):
        cells.append(volume.size)
        return original(volume,*args,**kwargs)
    with bundle.reader(max_cache_bytes=0) as reader, mock.patch.object(branch.ndimage,'label',side_effect=observe):
        recipe,_=branch.build_connected_edge_selection(reader,radius,'g',{'e':['f','r']},write_domain='fixed_context')
    assert recipe['edges']
    assert len(cells)==3
    assert all(count<5*13*15 for count in cells)


def test_modern_reselection_replay_reserves_current_branch_workspace_before_selection(tmp_path):
    from XTA.sam_replay import replay_sam_directional_nrrds
    bundle=multiple_edges(tmp_path)
    with mock.patch('XTA.sam_replay.select_sam_proposals',side_effect=RuntimeError('selection reached')):
        with pytest.raises(ValueError,match='bounded group topology'):
            replay_sam_directional_nrrds(bundle,tmp_path/'modern',
                policy={'sam_bridge_policy':{'version':6}},memory_mib=1)
        with pytest.raises(RuntimeError,match='selection reached'):
            replay_sam_directional_nrrds(bundle,tmp_path/'legacy',
                policy={'sam_bridge_policy':{'version':4}},memory_mib=1)

"""Discovery reuse changes neither scientific plans nor bounded refusals."""
from dataclasses import fields, is_dataclass
from collections.abc import Mapping

import numpy as np
import pytest

from XTA import sam_bridge_planning as planning
from XTA.sam_extrapolation_planning import _BinaryBaseline


def frozen(value):
    if isinstance(value, planning._PackedObservationMask):
        value = np.asarray(value)
    if isinstance(value, np.ndarray):
        return value.shape, value.dtype.str, value.tobytes()
    if is_dataclass(value):
        return {field.name:frozen(getattr(value,field.name)) for field in fields(value)
                if field.name not in {'_observation_index_source', 'contract_recipe'}}
    if isinstance(value, Mapping):
        return {key:frozen(item) for key,item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(map(frozen,value))
    return value


def scan(volume, inventory=None, **kwargs):
    return planning._observations(volume, scope=kwargs.pop('scope','target'),
        labels=kwargs.pop('labels',None), lineage=kwargs.pop('lineage',{'stage':'target'}),
        limits=kwargs.pop('limits',planning.SamPlanningLimits()), inventory=inventory, **kwargs)


@pytest.fixture
def volume():
    value=np.zeros((9,32,37),np.uint8)
    value[1:3,4:9,5:10]=1
    value[6:8,5:10,6:11]=1
    value[4,23:27,28:33]=1  # Isolated, without a compatible bridge candidate.
    return value


def test_all_original_components_reused_with_new_scope_and_binary_labels(volume, monkeypatch):
    inventory=planning.SamObservationInventory()
    original=planning.plan_sam_bridges(volume, interpolation_distance=5,
        interpolation_walk_back=0, _observation_inventory=inventory)
    assert len(original.observations)==5
    expected=scan(_BinaryBaseline(volume),scope='new',lineage={'source':'post_interpolation'})
    def forbidden(*args, **kwargs):
        raise AssertionError('Unchanged components must not be relabelled')
    monkeypatch.setattr(planning.ndi,'label',forbidden)
    actual=scan(_BinaryBaseline(volume),inventory,scope='new',lineage={'source':'post_interpolation'})
    assert frozen(actual)==frozen(expected)
    source={obs.frame_index:obs for obs in original.observations}
    assert all(obs.mask_crop is source[obs.frame_index].mask_crop for obs in actual[0])
    assert all(not obs.mask_crop.flags.writeable for obs in actual[0])
    inventory.clear()
    assert not inventory._frames


def test_changed_slice_is_the_only_relabelled_slice(volume,monkeypatch):
    inventory=planning.SamObservationInventory()
    scan(volume,inventory,scope='original')
    changed=volume.copy()
    changed[3,4:9,5:10]=1
    expected=scan(_BinaryBaseline(changed))
    calls=[]
    original=planning.ndi.label
    def label(mask,*args,**kwargs):
        calls.append(mask.copy())
        return original(mask,*args,**kwargs)
    monkeypatch.setattr(planning.ndi,'label',label)
    actual=scan(_BinaryBaseline(changed),inventory)
    assert frozen(actual)==frozen(expected)
    assert len(calls)==1
    np.testing.assert_array_equal(calls[0],changed[3,4:9,5:10]!=0)
    assert next(obs for obs in actual[0] if obs.frame_index==3).bbox_yx==(4,5,9,10)
    assert inventory.snapshot()['reused_frames']==8
    assert inventory.snapshot()['label_calls']==6


def test_touching_labels_recompute_the_binary_union(monkeypatch):
    volume=np.zeros((2,13,17),np.uint8)
    volume[:,3:8,3:7]=2
    volume[:,3:8,7:11]=7
    inventory=planning.SamObservationInventory()
    assert len(scan(volume,inventory)[0])==4
    expected=scan(_BinaryBaseline(volume))
    calls=[]
    original=planning.ndi.label
    def label(*args,**kwargs):
        calls.append(1)
        return original(*args,**kwargs)
    monkeypatch.setattr(planning.ndi,'label',label)
    actual=scan(_BinaryBaseline(volume),inventory)
    assert len(calls)==2 and len(actual[0])==2
    assert frozen(actual)==frozen(expected)


@pytest.mark.parametrize('wrap',[False,True])
def test_complete_bridge_plan_is_identical_on_reuse(volume,wrap):
    inventory=planning.SamObservationInventory()
    options=dict(interpolation_distance=5,interpolation_walk_back=1,wrap_axis=wrap,
        interpolation_min_radius=0,scope_id='scope')
    expected=planning.plan_sam_bridges(volume,**options)
    planning.plan_sam_bridges(volume,_observation_inventory=inventory,**options)
    actual=planning.plan_sam_bridges(volume,_observation_inventory=inventory,**options)
    assert frozen(actual)==frozen(expected)


@pytest.mark.parametrize('max_count,max_bytes',[(2,100000),(100,2200),(100,100000)])
def test_changed_limits_preserve_exact_failure_prefix(volume,max_count,max_bytes):
    inventory=planning.SamObservationInventory()
    scan(volume,inventory)
    limits=planning.SamPlanningLimits(max_observations=max_count,max_observation_bytes=max_bytes)
    expected=scan(_BinaryBaseline(volume),limits=limits)
    actual=scan(_BinaryBaseline(volume),inventory,limits=limits)
    assert frozen(actual)==frozen(expected)
    assert not inventory._active and not inventory._old


def test_growth_evicts_old_future_geometry_without_a_new_refusal(monkeypatch):
    volume=np.zeros((3,24,24),bool)
    volume[:,1:4,1:4]=True
    inventory=planning.SamObservationInventory()
    limits=planning.SamPlanningLimits(max_observation_bytes=3600)
    scan(volume,inventory,limits=limits)
    changed=volume.copy()
    changed[0,1:22,1:22]=True
    expected=scan(changed,limits=limits)
    peaks=[]
    trim=inventory._trim
    def observe(charge):
        trim(charge)
        peaks.append(charge+inventory._old_bytes+inventory._empty_bytes)
    monkeypatch.setattr(inventory,'_trim',observe)
    actual=scan(changed,inventory,limits=limits)
    assert frozen(actual)==frozen(expected)
    assert max(peaks)<=limits.max_observation_bytes


def test_empty_frame_metadata_is_bounded_and_never_refuses_observations():
    volume=np.zeros((30,13,17),bool)
    inventory=planning.SamObservationInventory()
    limits=planning.SamPlanningLimits(max_observation_bytes=1200)
    scan(volume,inventory,limits=limits)
    assert inventory.snapshot()['retained_bytes']<=limits.max_observation_bytes
    volume[-1,2:5,3:7]=True
    expected=scan(volume,limits=limits)
    actual=scan(volume,inventory,limits=limits)
    assert frozen(actual)==frozen(expected)
    assert actual[2] is None
    assert inventory.snapshot()['retained_bytes']<=limits.max_observation_bytes


def test_same_foreground_new_canonical_value_regenerates_ids_and_order():
    volume=np.zeros((1,15,20),np.uint8)
    volume[0,2:4,14:17]=1
    volume[0,3:12,2:4]=1
    inventory=planning.SamObservationInventory()
    old=scan(volume,inventory,scope='before')
    volume*=255
    labels=volume.astype(np.int32)
    expected=scan(volume,labels=labels,scope='after')
    actual=scan(volume,inventory,labels=labels,scope='after')
    assert frozen(actual)==frozen(expected)
    assert [item.component_index for item in actual[0]]==[1,2]
    assert [item.observation_id for item in old[0]]!=[item.observation_id for item in actual[0]]
    assert all(a.mask_crop is b.mask_crop for a,b in zip(old[0],actual[0]))


@pytest.mark.parametrize('label', (float('nan'), np.datetime64('NaT', 'ns')))
@pytest.mark.parametrize('labelled_first', (False, True))
def test_nonreflexive_canonical_label_never_reuses_binary_geometry(label, labelled_first):
    volume = np.zeros((1, 5, 5), np.uint8)
    volume[0, 1:4, 1:4] = 1
    labels = np.full(volume.shape, label)
    inventory = planning.SamObservationInventory()
    if labelled_first:
        assert not scan(volume, inventory, labels=labels)[0]
        expected = scan(_BinaryBaseline(volume))
        actual = scan(_BinaryBaseline(volume), inventory)
        assert len(actual[0]) == 1
    else:
        assert len(scan(_BinaryBaseline(volume), inventory)[0]) == 1
        expected = scan(volume, labels=labels)
        actual = scan(volume, inventory, labels=labels)
        assert not actual[0]
    assert frozen(actual) == frozen(expected)


def test_error_releases_old_inventory_and_concurrent_use_is_rejected(volume):
    inventory=planning.SamObservationInventory()
    scan(volume,inventory)
    class Broken:
        shape=volume.shape
        dtype=volume.dtype
        def __getitem__(self,frame):
            raise RuntimeError('input read failed')
    with pytest.raises(RuntimeError,match='input read failed'):
        scan(Broken(),inventory)
    assert not inventory._old and not inventory._active
    inventory._begin(planning.SamPlanningLimits())
    try:
        with pytest.raises(RuntimeError,match='concurrent'):
            scan(volume,inventory)
        with pytest.raises(RuntimeError,match='active'):
            inventory.clear()
    finally:
        inventory._finish()

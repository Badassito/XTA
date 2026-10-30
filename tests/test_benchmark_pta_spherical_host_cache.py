"""CPU-only checks of independent cache benchmark comparisons and selection."""
import importlib.util
from pathlib import Path
from types import SimpleNamespace


def tool():
    path=Path(__file__).resolve().parents[1]/'tools/benchmark_pta_spherical_host_cache.py'
    spec=importlib.util.spec_from_file_location('pta_spherical_cache_benchmark',path)
    module=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_selection_requires_reused_radius_bands_and_six_keys():
    import pytest
    module=tool()
    views=[SimpleNamespace(family='spherical',shared_view=(key,band)) for band in range(2) for key in range(8)]
    selected=module.choose_key_groups(views,lambda view:view[0],6)
    assert [key for key,_ in selected]==list(range(6))
    assert all(len(group)==2 for _key,group in selected)
    with pytest.raises(ValueError,match='two radius bands'):
        module.choose_key_groups(views[:8],lambda view:view[0],6)
    with pytest.raises(ValueError,match='six distinct'):
        module.choose_key_groups(views,lambda view:view[0],5)


def test_comparison_checks_partitions_candidate_state_and_raw_prejpeg_bytes():
    from copy import deepcopy
    module=tool()
    baseline=dict(files={'a.jpg':'encoded'},policy_partitions=[[[None,7]]],
        candidate_states=[[0,True,True,'']],totals={'written':1},workload={'keys':6},
        raw_render={'source':'pixelhash'},raw_policy=[{'mask':'maskhash'}],
        qualification_partitions=[[[None,7]]],classification_outputs={'image_sha256':'original'})
    candidate=deepcopy(baseline)
    assert all(module.compare_results(baseline,candidate).values())
    candidate['raw_render']['source']='changed-even-if-jpeg-is-equal'
    assert not module.compare_results(baseline,candidate)['exact_raw_render']
    candidate['policy_partitions']=[[[None],[7]]]
    assert not module.compare_results(baseline,candidate)['exact_policy_partitions']
    candidate['classification_outputs']['image_sha256']='changed'
    assert not module.compare_results(baseline,candidate)['exact_classification_outputs']


def test_counter_delta_ignores_boolean_flags_and_retains_new_counters():
    module=tool()
    assert module.counter_delta({'builds':4,'bytes':100},{'builds':7,'bytes':400,'hits':5,'enabled':True})=={'builds':3,'bytes':300,'hits':5}


def test_frozen_tool_uses_actual_scratch_for_outputs_and_gpu_lock():
    module=tool()
    parent=Path('/workspace')
    assert module.scratch_root(parent/'Test')==parent/'Scratch'
    assert module.scratch_root(parent/'Scratch'/'Experiments'/'case'/'candidate-source')==parent/'Scratch'


def test_fixture_arguments_pass_the_production_tilt_validation():
    from XTA.pta_config import parse_pta_args
    config=parse_pta_args(tool().spherical_fixture_arguments(2048))
    assert config.args.imgsz==2048


def test_settled_host_memory_checks_authoritative_union_and_retirement():
    import pytest
    module=tool()
    snapshot=dict(live_bytes=100,budget_bytes=128,peak_live_bytes=120,cache_bytes=100,
        building_bytes=0,borrowed_bytes=0,scratch_bytes=0,quarantined_bytes=0,entries=1)
    module.validate_host_snapshot(snapshot)
    with pytest.raises(AssertionError,match='retirement'):
        module.validate_host_snapshot(snapshot,cleared=True)
    snapshot.update(live_bytes=0,cache_bytes=0,entries=0)
    module.validate_host_snapshot(snapshot,cleared=True)
    snapshot['borrowed_bytes']=1
    with pytest.raises(AssertionError,match='lease'):
        module.validate_host_snapshot(snapshot)

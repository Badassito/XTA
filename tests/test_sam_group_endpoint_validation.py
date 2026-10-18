"""Group-invariant mask checks are shared without weakening run validation."""
from collections import Counter
import hashlib
import json
from types import MappingProxyType
from unittest import mock

import numpy as np
import pytest

from XTA import sam_policy
from XTA.sam_evidence import SamEvidenceBundle, SamEvidenceWriter, _plain, fingerprint
from XTA.sam_mask_reader import SamMaskReader
from tests.reference_backends.sam_infrastructure import _infrastructure as original_infrastructure
from tests.test_sam_branch_performance import multiple_edges, two_groups
from tests.test_sam_cyclic_evidence_replay import bundle_at
from tests.test_sam_tiled_evidence_policy import tiled_bundle


def _many_runs(tmp_path, count=12):
    source = multiple_edges(tmp_path / 'source')
    group = _plain(source.groups['g'])
    with SamEvidenceWriter(tmp_path / 'many', _plain(source.scope)) as writer:
        writer.add_group(group, {name: source.group_mask('g', name) for name in group['mask_keys']})
        for index in range(count):
            run = _plain(source.runs['f' if index % 2 == 0 else 'r'])
            raw = {frame: source.raw_mask(run['run_id'], frame) for frame in run['expected_frames']}
            run['run_id'] = f'run-{index:03}'
            writer.add_run(run, raw)
        return writer.commit()


def test_endpoint_geometry_calls_scale_with_groups_instead_of_runs(tmp_path):
    bundle = _many_runs(tmp_path)
    counts = Counter()
    original = SamMaskReader.group_mask

    def counted(reader, group_id, name):
        if name.startswith(('endpoint:', 'evaluation:')):
            counts[name] += 1
        return original(reader, group_id, name)

    with bundle.reader(max_cache_bytes=0) as reader, mock.patch.object(SamMaskReader, 'group_mask', counted):
        group = reader.groups['g']
        expected = {key: original_infrastructure(reader, group, run) for key, run in reader.runs.items()}
        old = counts.copy()
        counts.clear()
        validation = sam_policy._group_endpoint_validation(reader, group)
        assert validation is not None and validation.errors == ()
        actual = {key: sam_policy._infrastructure(reader, group, run, _group_endpoints=validation)
                  for key, run in reader.runs.items()}
        assert actual == expected
        assert set(old.values()) == {12}
        assert set(counts.values()) == {1}
        assert len(counts) == 2 * len(group['endpoints'])
        # The retained certificate contains only identities and an error tuple.
        assert not any(isinstance(value, np.ndarray) for value in vars(validation).values())


@pytest.mark.parametrize('kind', ('branches', 'two_groups', 'cyclic180', 'cyclic360'))
@pytest.mark.parametrize('cache_bytes', (0, 1024**2))
def test_complete_selection_matches_fresh_checks(tmp_path, kind, cache_bytes):
    if kind.startswith('cyclic'):
        bundle = bundle_at(tmp_path, period=float(kind[6:]))
        policy = {'sam_bridge_policy': 'permissive'}
    else:
        bundle = (multiple_edges if kind == 'branches' else two_groups)(tmp_path)
        policy = {'sam_bridge_policy': {'version': 6, 'component_min_radius': 0.}}
    with mock.patch.object(sam_policy, '_group_endpoint_validation', return_value=None):
        expected = sam_policy.select_sam_proposals(bundle, policy, reader_cache_bytes=cache_bytes)
    actual = sam_policy.select_sam_proposals(bundle, policy, reader_cache_bytes=cache_bytes)
    for key in ('selected_run_ids', 'run_receipts', 'group_receipts', 'branch_selection', 'guarded_rescue'):
        assert actual.get(key) == expected.get(key)


@pytest.mark.parametrize('options', ({}, {'incomplete': True}, {'unknown_eval': True}, {'halo_leak': 'large'}))
def test_tiled_run_dependent_coverage_checks_are_not_reused(tmp_path, options):
    bundle, _ = tiled_bundle(tmp_path, **options)
    with bundle.reader(max_cache_bytes=0) as reader:
        group = next(iter(reader.groups.values()))
        validation = sam_policy._group_endpoint_validation(reader, group)
        assert validation is not None
        run = reader.runs['F']
        expected = original_infrastructure(reader, group, run)
        assert sam_policy._infrastructure(reader, group, run, _group_endpoints=validation) == expected
        if options.get('unknown_eval'):
            assert 'tiled_required_evaluation_coverage_incomplete' in expected
        # The shared verdict cannot waive a changed run's missing terminal data.
        changed = _plain(run)
        terminal = next(ep for ep in group['endpoints'] if ep['observation_id'] in run['held_out_ids'])
        changed['raw_mask_keys'].pop(str(terminal['frame_index']))
        changed['observed_frames'].remove(terminal['frame_index'])
        assert sam_policy._infrastructure(reader, group, changed, _group_endpoints=validation) == original_infrastructure(reader, group, changed)


@pytest.mark.parametrize('options', ({}, {'incomplete': True}, {'unknown_eval': True}, {'halo_leak': 'large'}))
def test_multi_run_tiled_selection_matches_original_infrastructure(tmp_path, options):
    source, _ = tiled_bundle(tmp_path / 'source', **options)
    group = next(iter(source.groups.values()))
    with SamEvidenceWriter(tmp_path / 'two-tiled', _plain(source.scope)) as writer:
        writer.add_group(_plain(group), {name: source.group_mask(group['group_id'], name) for name in group['mask_keys']})
        for identifier in ('F', 'F2'):
            run = _plain(source.runs['F'])
            for tile in run['tile_evidence']:
                raw = {frame: source.tile_raw_mask('F', tile['tile_id'], frame) for frame in tile['observed_frames']}
                writer.add_run_tile(identifier, tile, raw)
            run['run_id'] = identifier
            writer.add_run(run, {frame: source.raw_mask('F', frame) for frame in run['observed_frames']},
                availability_masks={frame: source.availability_mask('F', frame) for frame in run['observed_frames']})
        bundle = writer.commit()
    with mock.patch.object(sam_policy, '_infrastructure',
            lambda *args, **kwargs: original_infrastructure(*args)), \
            mock.patch.object(sam_policy, '_group_endpoint_validation', return_value=None):
        expected = sam_policy.select_sam_proposals(bundle, reader_cache_bytes=0)
    actual = sam_policy.select_sam_proposals(bundle, reader_cache_bytes=0)
    for key in ('selected_run_ids', 'run_receipts', 'group_receipts', 'guarded_rescue'):
        assert actual[key] == expected[key]


@pytest.mark.parametrize('binding', ('scope', 'groups', 'runs', 'records', 'missing_sources', 'missing_record_source'))
def test_rebound_metadata_falls_back_before_and_after_census(tmp_path, binding):
    bundle = multiple_edges(tmp_path)
    with bundle.reader(max_cache_bytes=0) as reader:
        validation = sam_policy._group_endpoint_validation(reader, reader.groups['g'])
        assert validation is not None
        if binding == 'missing_sources':
            del bundle._frame_geometry_sources
        elif binding == 'missing_record_source':
            del bundle._mask_records_source
        else:
            setattr(bundle, binding, _plain(getattr(bundle, binding)))
        group = reader.groups['g']
        assert sam_policy._group_endpoint_validation(reader, group) is None
        with mock.patch.object(sam_policy.np, 'array_equal', wraps=np.array_equal) as equality:
            actual = sam_policy._infrastructure(reader, group, reader.runs['f'], _group_endpoints=validation)
            assert equality.call_count == len(group['endpoints'])
        assert actual == original_infrastructure(reader, group, reader.runs['f'])


def test_mutable_records_inside_replacement_mapping_are_not_certified(tmp_path):
    bundle = multiple_edges(tmp_path)
    bundle.records = MappingProxyType(_plain(bundle.records))
    with bundle.reader(max_cache_bytes=0) as reader:
        assert sam_policy._group_endpoint_validation(reader, reader.groups['g']) is None


@pytest.mark.parametrize('replace_before_census', (False, True))
def test_mutable_backing_of_replacement_proxy_cannot_reuse_verdict(tmp_path, replace_before_census):
    bundle = multiple_edges(tmp_path)
    assert bundle._mask_records_source is bundle.records
    with bundle.reader(max_cache_bytes=0) as reader:
        group = reader.groups['g']
        validation = None if replace_before_census else sam_policy._group_endpoint_validation(reader, group)
        if not replace_before_census:
            assert validation is not None
        backing = dict(bundle.records)
        bundle.records = MappingProxyType(backing)
        if replace_before_census:
            validation = sam_policy._group_endpoint_validation(reader, group)
            assert validation is None
        endpoint_id = group['endpoints'][0]['observation_id']
        endpoint_key = group['mask_keys'][f'endpoint:{endpoint_id}']
        evaluation_key = group['mask_keys'][f'evaluation:{endpoint_id}']
        # Every value is an original frozen record, and the outer proxy's
        # identity stays fixed. Its caller-owned backing is nevertheless mutable.
        backing[evaluation_key] = backing[endpoint_key]
        for run in reader.runs.values():
            expected = original_infrastructure(reader, group, run)
            assert 'endpoint_evaluation_region_not_independently_declared' in expected
            assert sam_policy._infrastructure(reader, group, run, _group_endpoints=validation) == expected


def test_custom_reader_and_instance_override_keep_public_fresh_path(tmp_path):
    bundle = multiple_edges(tmp_path)

    class CustomReader(SamMaskReader):
        pass

    with CustomReader(bundle, max_cache_bytes=0) as custom:
        assert sam_policy._group_endpoint_validation(custom, custom.groups['g']) is None
    with bundle.reader(max_cache_bytes=0) as reader:
        reader.group_mask = lambda group_id, name: bundle.group_mask(group_id, name)
        assert sam_policy._group_endpoint_validation(reader, reader.groups['g']) is None


def test_public_measurement_still_checks_geometry_each_call(tmp_path):
    bundle = multiple_edges(tmp_path)
    with bundle.reader(max_cache_bytes=0) as reader:
        with mock.patch.object(sam_policy, '_group_endpoint_validation', side_effect=AssertionError('Public call must be independent')):
            for _ in range(2):
                result = sam_policy.measure_sam_run(reader, 'f')
                assert result['infrastructure_errors'] == original_infrastructure(reader, reader.groups['g'], reader.runs['f'])


def test_certificate_is_bound_to_owning_transaction_and_inherited_by_fork(tmp_path):
    bundle = multiple_edges(tmp_path)
    with bundle.reader(max_cache_bytes=0) as parent:
        group = parent.groups['g']
        validation = sam_policy._group_endpoint_validation(parent, group)
        with parent.fork(max_cache_bytes=0) as child:
            with mock.patch.object(sam_policy.np, 'array_equal', side_effect=AssertionError('Fork must borrow verdict')):
                assert sam_policy._infrastructure(child, group, child.runs['f'], _group_endpoints=validation) == []
        with bundle.reader(max_cache_bytes=0) as independent:
            with mock.patch.object(sam_policy.np, 'array_equal', wraps=np.array_equal) as equality:
                sam_policy._infrastructure(independent, group, independent.runs['f'], _group_endpoints=validation)
                assert equality.call_count == len(group['endpoints'])


@pytest.mark.parametrize('outside', (False, True))
def test_invalid_invariant_verdict_is_propagated_to_every_run(tmp_path, outside):
    bundle = _many_runs(tmp_path)
    index_path, manifest_path = bundle.directory / 'index.json', bundle.directory / 'manifest.json'
    index, manifest = json.loads(index_path.read_text()), json.loads(manifest_path.read_text())
    group = index['groups']['g']
    endpoint = group['endpoints'][0]['observation_id']
    group['mask_keys'][f'evaluation:{endpoint}'] = group['mask_keys'][f'endpoint:{endpoint}']
    if outside:
        other = group['endpoints'][1]['observation_id']
        group['mask_keys'][f'evaluation:{endpoint}'] = group['mask_keys'][f'endpoint:{other}']
    encoded = json.dumps(index).encode()
    index_path.write_bytes(encoded)
    manifest['files']['index.json'].update(bytes=len(encoded), sha256=hashlib.sha256(encoded).hexdigest())
    manifest['evidence_fingerprint'] = fingerprint({key: value for key, value in manifest.items() if key != 'evidence_fingerprint'})
    manifest_path.write_text(json.dumps(manifest))
    bundle = SamEvidenceBundle.open(bundle.directory)
    with bundle.reader(max_cache_bytes=0) as reader:
        group = reader.groups['g']
        validation = sam_policy._group_endpoint_validation(reader, group)
        assert validation.errors == ('endpoint_evaluation_region_not_independently_declared',)
        for run in reader.runs.values():
            assert sam_policy._infrastructure(reader, group, run, _group_endpoints=validation) == original_infrastructure(reader, group, run)


def test_cached_endpoint_verdict_cannot_waive_payload_integrity(tmp_path):
    bundle = multiple_edges(tmp_path)
    with pytest.raises(ValueError, match='changed'):
        with bundle.reader(max_cache_bytes=1024**2) as reader:
            group = reader.groups['g']
            validation = sam_policy._group_endpoint_validation(reader, group)
            path = bundle.directory / 'masks.bin'
            data = path.read_bytes()
            path.write_bytes(bytes([data[0] ^ 1]) + data[1:])
            assert validation is not None

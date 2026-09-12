"""A completed run must publish every planned independent policy output group."""
from __future__ import annotations

import hashlib
import json

import pytest

from tools.check_tta_augmentation_run import check


def _fixture(root):
    support = root / 'augmentation_support'
    support.mkdir()
    policy = b'# fixture\n'
    (support / 'policy.py').write_bytes(policy)
    nrrd = root / 'nrrd'
    nrrd.mkdir()
    base = 'transverse__tta_a0'
    groups = [{'kind': kind, 'view': base, 'tile_config_id': config}
              for kind, config in [('fullframe', ''), ('tile', 'cfg1'), ('tile', 'cfg2')]]
    manifest = {'ratio': 3, 'content_sha256': hashlib.sha256(policy).hexdigest(),
                'coverage': 'none', 'coverage_records': [], 'planned_output_groups': groups,
                'execution_records': [dict(group, task_id=i, job_id=f'job{i}', model_name='fixture',
                    slice_start=0, slice_count=2, pass_count=3, source_render_replays=0,
                    rendered_batches=2, model_batches=6) for i, group in enumerate(groups)]}
    layers = []
    for group in groups:
        for p in range(3):
            view = base + (f'__policy_{p:03d}' if p else '')
            filename = f"{view}_{group['kind']}_{group['tile_config_id']}.seg.nrrd"
            (nrrd / filename).write_bytes(b'checker only validates manifests and file presence')
            layers.append({'view_name': view, 'filename': filename, 'source': group['kind'],
                           'tile_config_id': group['tile_config_id'], 'mask_kind': 'yolo', 'pass_index': 0})
    _write(root, manifest, layers)
    return manifest, layers


def _write(root, manifest, layers):
    (root / 'augmentation_manifest.json').write_text(json.dumps(manifest))
    (root / 'nrrd' / 'fixture_nrrd_manifest.json').write_text(json.dumps({'layers': layers}))


def test_complete_planned_groups_pass(tmp_path):
    _fixture(tmp_path)
    report = check(tmp_path)
    assert report['fullframe_groups'] == 1
    assert report['tile_groups'] == 2
    assert report['total_nrrds'] == 9


@pytest.mark.parametrize('missing', ['fullframe', 'cfg1', 'cfg2'])
def test_wholly_missing_published_group_fails(tmp_path, missing):
    manifest, layers = _fixture(tmp_path)
    layers = [layer for layer in layers if layer['source'] != missing and layer['tile_config_id'] != missing]
    _write(tmp_path, manifest, layers)
    with pytest.raises(AssertionError, match='published/planned output group mismatch'):
        check(tmp_path)


def test_missing_execution_and_publication_still_fails_against_plan(tmp_path):
    manifest, layers = _fixture(tmp_path)
    manifest['execution_records'] = [row for row in manifest['execution_records'] if row['tile_config_id'] != 'cfg2']
    layers = [row for row in layers if row['tile_config_id'] != 'cfg2']
    _write(tmp_path, manifest, layers)
    with pytest.raises(AssertionError, match='execution/planning group mismatch'):
        check(tmp_path)


def test_partially_missing_policy_pass_fails(tmp_path):
    manifest, layers = _fixture(tmp_path)
    _write(tmp_path, manifest, layers[:-1])
    with pytest.raises(AssertionError, match='missing tile passes'):
        check(tmp_path)


def test_legacy_manifest_requires_rerun(tmp_path):
    manifest, layers = _fixture(tmp_path)
    del manifest['planned_output_groups']
    _write(tmp_path, manifest, layers)
    with pytest.raises(AssertionError, match='rerun with updated augmentation manifests'):
        check(tmp_path)


def test_no_nrrd_mode_still_checks_execution_against_plan(tmp_path):
    manifest, _ = _fixture(tmp_path)
    (tmp_path / 'nrrd' / 'fixture_nrrd_manifest.json').unlink()
    assert check(tmp_path, require_nrrd=False)['inference_tasks'] == 3
    manifest['execution_records'].pop()
    (tmp_path / 'augmentation_manifest.json').write_text(json.dumps(manifest))
    with pytest.raises(AssertionError, match='execution/planning group mismatch'):
        check(tmp_path, require_nrrd=False)


def test_unplanned_published_group_fails(tmp_path):
    manifest, layers = _fixture(tmp_path)
    manifest['planned_output_groups'] = manifest['planned_output_groups'][:-1]
    manifest['execution_records'] = manifest['execution_records'][:-1]
    _write(tmp_path, manifest, layers)
    with pytest.raises(AssertionError, match='published/planned output group mismatch'):
        check(tmp_path)

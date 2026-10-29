"""Small, CPU-only fixtures for reusable PTA diagnostic entry points."""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import cv2
import numpy as np
import pytest

from tools import audit_pta_f3_metadata as metadata
from tools import audit_pta_smoke_labels as smoke
from tools import inspect_pta_f3_global as global_inspection
from tools import inspect_pta_f3_provenance as provenance
from tools import pta_audit_common as common


ROOT = Path(__file__).resolve().parents[1]


def files_under(path):
    return {str(p.relative_to(path)): p.read_bytes() for p in path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('module', ['audit_pta_smoke_labels', 'audit_pta_f3_metadata',
                                   'inspect_pta_f3_global', 'inspect_pta_f3_provenance'])
def test_import_does_not_read_datasets_write_evidence_or_load_image_libraries(module, tmp_path):
    # A new interpreter rules out modules cached by another test hiding side effects.
    code = """
import importlib, pathlib, sys
from unittest import mock
with mock.patch.object(pathlib.Path, 'read_text', side_effect=AssertionError('import read dataset')), \
     mock.patch.object(pathlib.Path, 'open', side_effect=AssertionError('import opened dataset')), \
     mock.patch.object(pathlib.Path, 'write_text', side_effect=AssertionError('import wrote evidence')):
    importlib.import_module(sys.argv[1])
assert not {'cv2', 'numpy', 'PIL'} & sys.modules.keys()
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT), PYTHONDONTWRITEBYTECODE='1')
    subprocess.run([sys.executable, '-B', '-c', code, 'tools.' + module],
                   cwd=tmp_path, env=env, check=True, capture_output=True, text=True)
    assert files_under(tmp_path) == {}


@pytest.fixture
def smoke_input(tmp_path):
    source = tmp_path / 'smoke-input'
    (source / 'images').mkdir(parents=True)
    (source / 'labels').mkdir()
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    assert cv2.imwrite(str(source / 'images/F2_example.png'), image)
    # Duplicate polygon must remain filled when masks are unioned individually.
    square = '0 .25 .25 .5 .25 .5 .5 .25 .5\n'
    (source / 'labels/F2_example.txt').write_text(
        square * 2 + '0 .75 .75 .875 .75 .875 .875 .75 .875\n', encoding='utf-8')
    assert cv2.imwrite(str(source / 'images/F3_empty.png'), np.zeros_like(image))
    (source / 'labels/F3_empty.txt').write_text('', encoding='utf-8')
    return source


def test_smoke_inventory_union_counts_empty_masks_montages_and_read_only_inputs(smoke_input, tmp_path):
    before = files_under(smoke_input)
    output = tmp_path / 'evidence'
    rows = smoke.audit(smoke_input, output, montage_limit=2)
    first, empty = rows
    assert first['name'] == 'F2_example'
    assert first['foreground_pixels'] == 34
    assert first['polygon_count'] == 3
    assert first['components8'] == first['components_le_25'] == 2
    assert first['component_areas_descending'] == [25, 9]
    assert first['pixels_in_components_le_25'] == 34
    assert empty['foreground_pixels'] == empty['components8'] == 0
    assert json.loads((output / 'smoke_label_component_inventory.json').read_text()) == rows
    assert len(list(output.glob('smoke_group_*_montage.png'))) == 2
    assert cv2.imread(str(output / 'smoke_contrast_detail_montage.png')).shape[:2] == (560, 3072)
    assert cv2.imread(str(output / 'F3_empty_contrast_crop_outline.png')).shape[:2] == (16, 16)
    assert files_under(smoke_input) == before


def test_explicit_sample_names_bound_rendering_without_reducing_inventory(smoke_input, tmp_path):
    output = tmp_path / 'selected'
    rows = smoke.audit(smoke_input, output, sample_names=('F3_empty',))
    assert len(rows) == 2
    assert not (output / 'F2_example_outline.png').exists()
    assert (output / 'F3_empty_outline.png').exists()


def test_invalid_polygon_fails_before_generating_evidence(smoke_input, tmp_path):
    (smoke_input / 'labels/F2_example.txt').write_text('0 nan 0 1 0 1 1\n', encoding='utf-8')
    output = tmp_path / 'invalid-evidence'
    with pytest.raises(ValueError, match='finite and normalized'):
        smoke.audit(smoke_input, output)
    assert not output.exists()


@pytest.mark.parametrize('diagnostic', ['smoke', 'metadata', 'global', 'provenance'])
def test_diagnostics_reject_output_inside_repository_before_input_io(diagnostic, tmp_path):
    output = ROOT / 'tools' / 'must-not-create-audit-output'
    missing = tmp_path / 'missing'
    with pytest.raises(ValueError, match='outside the repository'):
        if diagnostic == 'smoke':
            smoke.audit(missing, output)
        elif diagnostic == 'metadata':
            metadata.audit(missing, missing, missing / 'edits.json', output)
        elif diagnostic == 'global':
            global_inspection.inspect(missing / 'edits.json', output)
        else:
            provenance.inspect(missing, missing, missing / 'edits.json', output)
    assert not output.exists()


def test_smoke_rejects_output_under_input_dataset(smoke_input):
    with pytest.raises(ValueError, match='outside the input directory'):
        smoke.audit(smoke_input, smoke_input / 'evidence')
    assert not (smoke_input / 'evidence').exists()


@pytest.fixture
def receipts(tmp_path):
    native, source, edited = (tmp_path / name for name in ('native', 'input', 'edited'))
    for directory in (native, source, edited):
        directory.mkdir()
    payload = b'NRRD0005\ntype: uint8\ndimension: 3\nsizes: 2 2 2\n\n' + bytes(range(8))
    for directory in (native, source):
        (directory / 'sample.nrrd').write_bytes(payload)
    report = dict(tool_version='fixture', status='complete', created_utc='2026-10-27T00:00:00Z',
                  shape_tyx=[8, 8, 8], preview_shape_tyx=[4, 4, 4],
                  preview_manifest_sha256='preview', full_manifest_sha256='full',
                  initial_global_voxels=32, final_global_voxels=24,
                  unresolved_components=0, auto_resolved_components=1, auto_removed_voxels=8,
                  union_cleanup={}, warnings=[], pta_label_file='sample.nrrd',
                  layers=[dict(conflicting_voxels_preserved=2, unobserved_components_preserved=3)])
    (native / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    np.savez(edited / 'global.npz', packed=np.array([3], dtype=np.uint8))
    edits = dict(segments=[dict(name='Global', file='global.npz', recomposition_op='select', foreground_voxels=2)],
                 preview_shape_tyx=[4, 4, 4], full_quality_shape_tyx=[8, 8, 8],
                 source_manifest_sha256='original', notes=[])
    edits_path = edited / 'edits.json'
    edits_path.write_text(json.dumps(edits), encoding='utf-8')
    return native, source, edits_path


def test_metadata_and_inspectors_preserve_calculations_and_input_bytes(receipts, tmp_path):
    native, source, edits_path = receipts
    before = {p: files_under(p) for p in (native, source, edits_path.parent)}
    result = metadata.audit(*receipts, tmp_path / 'metadata-evidence')
    assert result['input_and_old_native_bytes_equal_sha256'] is True
    assert result['native_conflicting_voxels_preserved'] == 2
    assert result['native_unobserved_components_preserved'] == 3
    assert result['count_comparison'] == dict(exact_full_to_preview_grid_voxel_ratio=8,
        preview_global_voxels_times_exact_grid_ratio=16,
        native_output_change_vs_scaled_preview_percent=50,
        native_output_change_vs_initial_native_percent=-25)
    inspection = global_inspection.inspect(edits_path, tmp_path / 'global-evidence')
    array = inspection['global_segments'][0]['arrays'][0]
    assert array == dict(key='packed', shape=[1], dtype='uint8', preview='[3]')
    inspection = provenance.inspect(*receipts, tmp_path / 'provenance-evidence')
    assert inspection['foreground_sum'] == 2
    assert inspection['conflicting_total'] == 2
    assert inspection['unobserved_components'] == 3
    assert inspection['headers'][0]['header_lines'][0] == 'NRRD0005'
    assert inspection['headers'][0]['bytes'] == (native / 'sample.nrrd').stat().st_size
    assert all(files_under(p) == value for p, value in before.items())


def test_metadata_zero_foreground_has_explicit_undefined_percentages(receipts, tmp_path):
    native, _, edits_path = receipts
    report = json.loads((native / 'report.json').read_text())
    report['initial_global_voxels'] = 0
    (native / 'report.json').write_text(json.dumps(report), encoding='utf-8')
    edits = json.loads(edits_path.read_text())
    edits['segments'][0]['foreground_voxels'] = 0
    edits_path.write_text(json.dumps(edits), encoding='utf-8')
    comparison = metadata.audit(*receipts, tmp_path / 'zero-evidence')['count_comparison']
    assert comparison['native_output_change_vs_scaled_preview_percent'] is None
    assert comparison['native_output_change_vs_initial_native_percent'] is None


@pytest.mark.parametrize('name', ['../sample.nrrd', 'subdir/sample.nrrd', 'H:/sample.nrrd'])
def test_receipt_members_cannot_select_another_directory(name, tmp_path):
    with pytest.raises(ValueError, match='plain receipt member'):
        common.receipt_child(tmp_path, name)


def test_smoke_command_line(smoke_input, tmp_path):
    output = tmp_path / 'cli-evidence'
    completed = subprocess.run([sys.executable, '-B', str(ROOT / 'tools/audit_pta_smoke_labels.py'),
                                '--input-root', str(smoke_input), '--output-dir', str(output),
                                '--sample-name', 'F2_example'],
                               cwd=tmp_path, capture_output=True, text=True, check=True)
    assert json.loads(completed.stdout)['images'] == 2
    assert (output / 'smoke_label_component_inventory.json').is_file()


@pytest.mark.parametrize('tool', ['audit_pta_f3_metadata', 'inspect_pta_f3_global', 'inspect_pta_f3_provenance'])
def test_receipt_diagnostic_command_lines(tool, receipts, tmp_path):
    native, source, edits_path = receipts
    output = tmp_path / 'cli-evidence'
    args = ['--edits', str(edits_path), '--output-dir', str(output)]
    if tool != 'inspect_pta_f3_global':
        args += ['--native-dir', str(native), '--input-dir', str(source)]
    completed = subprocess.run([sys.executable, '-B', str(ROOT / 'tools' / (tool + '.py')), *args],
                               cwd=tmp_path, capture_output=True, text=True, check=True)
    assert isinstance(json.loads(completed.stdout), dict)
    assert len(list(output.glob('*.json'))) == 1

"""Live archived SAM outputs preserve masks, retries, and interrupted evidence."""
import json
from contextlib import closing
from pathlib import Path
import tempfile

import numpy as np
import pytest

from XTA.artifact_archive import iter_artifacts, read_artifact, reference, split_reference
from XTA.interpolation import RawBBoxMaskStore
from XTA.sam_evidence import SamEvidenceBundle, load_sam_online_selection
from XTA.sam_interpolation import SamInterpolationInfrastructureError, interpolate_sam_view_volume_pass
from tests.test_sam_adaptive_interpolation import _execute
from tests.test_sam_interpolation import RepeatedSeedTracker, _observations
from tests.test_sam_iterative_extrapolation import _case


@pytest.fixture
def archive_root(tmp_path, monkeypatch):
    staging = tmp_path/'staging'
    staging.mkdir()
    monkeypatch.setattr(tempfile, 'tempdir', str(staging))
    output = tmp_path/'output'
    output.mkdir()
    return Path(reference(output/'sam-artifacts.tar', 'sam_interpolation'))


def _assert_output_has_no_loose_artifacts(root):
    archive, _ = split_reference(root)
    assert {path.name for path in archive.parent.iterdir()} <= {'sam-artifacts.tar', 'sam-artifacts.tar.lock'}


def _decode_components(components, shape):
    result = np.zeros(shape, np.uint8)
    for component in components:
        with closing(RawBBoxMaskStore.open(Path(component['path']), mmap_payload=True)) as store:
            for frame in range(shape[0]):
                decoded = store.decode_slice_crop(frame)
                if decoded is not None:
                    y0,x0,y1,x1,crop = decoded
                    result[frame,y0:y1,x0:x1] |= crop
    return result


def test_interpolation_publishes_evidence_receipt_and_cvol_during_run(archive_root, tmp_path):
    original = _observations()
    merged, stats, components = interpolate_sam_view_volume_pass(original,
        work_dir=archive_root, runtime_work_dir=tmp_path/'runtime', runtime=RepeatedSeedTracker(),
        gap_distance=5, min_radius=0, interpolation_walk_back=0, return_bridge_components=True)
    try:
        bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
        assert bundle.manifest['complete'] and len(bundle.runs) == 2
        selection = load_sam_online_selection(bundle)
        assert selection['selection_identity'] == stats['sam_selection_identity']
        generation = json.loads(read_artifact(bundle.directory.parent/'generation.json'))
        assert generation['sam_selected_runs'] == 2
        np.testing.assert_array_equal(_decode_components(components, original.shape), merged & ~original)
        assert stats['added_voxels'] == 108
        _assert_output_has_no_loose_artifacts(archive_root)
    finally:
        merged._mmap.close()


def test_failed_tracker_preserves_committed_partial_evidence_without_outputs(archive_root):
    with pytest.raises(SamInterpolationInfrastructureError, match='controlled worker failure'):
        interpolate_sam_view_volume_pass(_observations(), work_dir=archive_root,
            runtime=RepeatedSeedTracker(failure=True), gap_distance=5, min_radius=0,
            interpolation_walk_back=0, return_bridge_components=True)
    manifests = list(iter_artifacts(archive_root, 'manifest.json'))
    assert len(manifests) == 1
    assert json.loads(read_artifact(manifests[0]))['complete'] is False
    failures = list(iter_artifacts(archive_root, 'failure.json'))
    assert len(failures) == 1 and json.loads(read_artifact(failures[0]))['complete'] is False
    assert not list(iter_artifacts(archive_root, 'meta.json'))
    _assert_output_has_no_loose_artifacts(archive_root)


def test_failed_cvol_append_preserves_recovery_stage_and_committed_evidence(archive_root, tmp_path, monkeypatch):
    from XTA import artifact_archive
    publish = artifact_archive.publish_directory

    def fail_cvol(staging, destination):
        if str(destination).endswith('.cvol'):
            raise OSError('controlled archive append failure')
        return publish(staging, destination)

    monkeypatch.setattr(artifact_archive, 'publish_directory', fail_cvol)
    with pytest.raises(SamInterpolationInfrastructureError, match='controlled archive append failure'):
        interpolate_sam_view_volume_pass(_observations(), work_dir=archive_root,
            runtime_work_dir=tmp_path/'runtime', runtime=RepeatedSeedTracker(), gap_distance=5,
            min_radius=0, interpolation_walk_back=0, return_bridge_components=True)
    evidence = next(iter_artifacts(archive_root, 'manifest.json'))
    assert json.loads(read_artifact(evidence))['complete'] is True
    assert list(iter_artifacts(archive_root, 'selection.json'))
    assert not list(iter_artifacts(archive_root, 'generation.json'))
    assert not list(iter_artifacts(archive_root, 'meta.json'))
    stages = list((tmp_path/'runtime').rglob('.artifact-stage-*'))
    assert len(stages) == 1
    assert {path.name for path in stages[0].iterdir()} == {'meta.json', 'index.bin', 'chunks.bin'}
    _assert_output_has_no_loose_artifacts(archive_root)


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_interpolation_retry_keeps_initial_and_final_archived_ownership(archive_root, mode):
    original, runtime, providers, merged, stats, components = _execute(archive_root, crop_mode=mode)
    try:
        ledger = stats['sam_crop_retry']
        initial = SamEvidenceBundle.open(ledger['initial_evidence_path'])
        final = SamEvidenceBundle.open(stats['sam_evidence_path'])
        assert initial.evidence_fingerprint == ledger['initial_evidence_fingerprint']
        assert final.evidence_fingerprint == ledger['final_evidence_fingerprint']
        assert set(initial.groups).isdisjoint(final.groups)
        assert len(providers) == 1 and len(runtime.calls) == 4
        assert list(iter_artifacts(archive_root, 'crop_retry.json'))
        np.testing.assert_array_equal(_decode_components(components, original.shape), merged & ~original)
        _assert_output_has_no_loose_artifacts(archive_root)
    finally:
        merged._mmap.close()


@pytest.mark.parametrize('mode', ['whole', 'tiled'])
def test_extrapolation_retries_publish_native_stores_and_original_seed_evidence(archive_root, mode):
    case = _case(archive_root, mode=mode)
    returned, stats, components = case.run()
    assert returned is case.observed
    np.testing.assert_array_equal(returned, case.original)
    np.testing.assert_array_equal(_decode_components(components, returned.shape), case.truth & ~case.original)
    assert len(case.providers) >= 2
    bundle = SamEvidenceBundle.open(stats['sam_evidence_path'])
    assert bundle.manifest['complete']
    selection = json.loads(read_artifact(stats['sam_selection_receipt_path']))
    assert selection['evidence_fingerprint'] == bundle.evidence_fingerprint
    ledger = json.loads(read_artifact(next(iter_artifacts(archive_root, 'crop_retry.json'))))
    assert ledger['status'] == 'complete'
    assert len(ledger['attempt_history'][case.group.group_id]) == len(case.providers)
    _assert_output_has_no_loose_artifacts(archive_root)

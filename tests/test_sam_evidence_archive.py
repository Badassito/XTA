"""Live archive evidence preserves partial runs and exact proposal ownership."""
import json
from pathlib import Path

import numpy as np
import pytest

from XTA.artifact_archive import (append_members, artifact_exists, list_members,
    member_info, open_artifact, read_artifact, reference)
from XTA.sam_evidence import (SamEvidenceBundle, SamEvidenceWriter, _plain,
    export_sam_evidence, iter_selected_planes, load_sam_online_selection)
from XTA.sam_policy import select_sam_proposals
from XTA.sam_replay import replay_sam_directional_nrrds
from tests.test_sam_evidence_import import _bundle, _scope
from tests.test_sam_evidence_policy import fixture_group, fixture_run
from tests.test_sam_replay_tools import _read


def test_long_logical_paths_publish_before_whole_run_finishes(tmp_path):
    archive = tmp_path / 'sam.tar'
    logical = '/'.join(['scope-' + 'x' * 180] * 5) + '/evidence'
    destination = Path(reference(archive, logical))
    group, masks, raw = fixture_group()
    writer = SamEvidenceWriter(destination, _scope())
    stage = writer.staging
    try:
        assert len(str(stage)) < 240
        assert not artifact_exists(destination)
        writer.add_group(group, masks)
        writer.add_run(fixture_run('F', group), raw)
        partial = writer.commit(complete=False)
    finally:
        writer.abort()
    assert not stage.exists()
    assert not partial.manifest['complete']
    assert len(list_members(archive)) == 3
    assert {item.name for item in tmp_path.iterdir()} <= {'sam.tar', 'sam.tar.lock'}
    np.testing.assert_array_equal(partial.raw_mask('F', 2), raw[2])
    append_members(archive, {'other-scope/progress.json': b'{}'})
    partial.assert_unchanged()
    with partial.reader() as reader:
        np.testing.assert_array_equal(reader.raw_mask('F', 2), raw[2])
        append_members(archive, {'other-scope/selection.json': b'{}'})
    assert reader.stats['transaction_complete']
    with pytest.raises(FileExistsError):
        SamEvidenceWriter(destination, _scope())


def test_aborted_scope_exposes_no_members_and_cleans_short_stage(tmp_path):
    archive = tmp_path / 'sam.tar'
    first = _bundle(Path(reference(archive, 'finished/evidence')))
    destination = Path(reference(archive, 'aborted/evidence'))
    with SamEvidenceWriter(destination, _scope()) as writer:
        stage = writer.staging
        group, masks, _ = fixture_group()
        writer.add_group(group, masks)
    assert not stage.exists()
    assert not artifact_exists(destination)
    assert len(list_members(archive)) == 3
    first.assert_unchanged()


@pytest.mark.parametrize('tiled', [False, True])
def test_archive_raw_bytes_and_import_match_directory_bundles(tmp_path, tiled):
    directory = _bundle(tmp_path / 'legacy', tiled=tiled)
    archived = _bundle(Path(reference(tmp_path / 'sam.tar', 'source/evidence')), tiled=tiled)
    assert directory.evidence_fingerprint == archived.evidence_fingerprint
    for name in ('manifest.json', 'index.json', 'masks.bin'):
        assert read_artifact(directory.directory / name) == read_artifact(archived.directory / name)
    copied_path = Path(reference(tmp_path / 'sam.tar', 'retry/evidence'))
    with SamEvidenceWriter(copied_path, _plain(archived.scope)) as writer:
        with writer.import_transaction(archived):
            writer.import_group(archived, 'family')
            writer.import_run(archived, 'family:F')
        copied = writer.commit()
    for key in archived.records:
        np.testing.assert_array_equal(copied.mask(key), directory.mask(key))
    if not tiled:
        expected = directory.raw_crop_boundary_contacts_with_foreground('family:F', 2)
        assert copied.raw_crop_boundary_contacts_with_foreground('family:F', 2) == expected
        with copied.reader() as reader:
            assert reader.raw_crop_boundary_contacts_with_foreground('family:F', 2) == expected
    with open_artifact(archived.directory / 'masks.bin') as stream:
        stream.seek(0, 2)
        assert stream.read(1024) == b''
        with pytest.raises(ValueError, match='bounds'):
            stream.seek(1, 2)


def test_archive_receipts_export_and_model_free_replay(tmp_path):
    archive = tmp_path / 'sam.tar'
    bundle = _bundle(Path(reference(archive, 'online/evidence')),
                     scope=_scope(selection_receipt_required=True))
    with pytest.raises(ValueError, match='retained online selection receipt'):
        load_sam_online_selection(bundle)
    receipt = select_sam_proposals(bundle)
    encoded = json.dumps(receipt).encode('utf-8')
    append_members(archive, {'online/selection.json': encoded})
    assert load_sam_online_selection(bundle)['evidence_fingerprint'] == bundle.evidence_fingerprint
    destination = Path(reference(archive, 'portable'))
    result = export_sam_evidence(bundle, destination)
    assert result['file_count'] == 5
    copied = SamEvidenceBundle.open(destination)
    assert load_sam_online_selection(copied)['policy_hash'] == receipt['policy_hash']
    assert read_artifact(destination / 'online_selection.json') == encoded
    export_sam_evidence(copied, tmp_path / 'directory_export')
    assert (tmp_path / 'directory_export' / 'online_selection.json').read_bytes() == encoded
    copied.assert_unchanged()
    replay = replay_sam_directional_nrrds(copied, tmp_path / 'replay', memory_mib=1)
    forward, _ = _read(tmp_path / 'replay' / replay['layers'][0]['path'])
    selected = {frame: plane for _, frame, plane in iter_selected_planes(copied, receipt)}
    np.testing.assert_array_equal(forward, np.stack([selected[frame] for frame in range(5)]))
    append_members(archive, {'portable/online_selection.json': encoded + b' '}, replace=True)
    with pytest.raises(ValueError, match='checksum'):
        load_sam_online_selection(copied)
    with pytest.raises(ValueError, match='checksum'):
        export_sam_evidence(copied, tmp_path / 'corrupt_export')
    assert not (tmp_path / 'corrupt_export').exists()


def test_archive_payload_corruption_is_detected_on_open_and_transaction_exit(tmp_path):
    archive = tmp_path / 'sam.tar'
    bundle = _bundle(Path(reference(archive, 'source/evidence')))
    entry = member_info(bundle.directory / 'masks.bin')
    with pytest.raises(ValueError, match='checksum|changed'):
        with bundle.reader() as reader:
            reader.raw_mask('family:F', 2)
            with archive.open('r+b') as stream:
                stream.seek(entry['offset'])
                value = stream.read(1)
                stream.seek(entry['offset'])
                stream.write(bytes([value[0] ^ 1]))
    with pytest.raises(ValueError, match='checksum|changed'):
        SamEvidenceBundle.open(bundle.directory)

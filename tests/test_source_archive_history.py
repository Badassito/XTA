"""History fixture lookup is stable, exact, and fails instead of skipping guards."""
import hashlib

import pytest

from tools.source_archive_history import history_archive_path, require_history_archive


def test_canonical_path_does_not_depend_on_old_experiment_sibling_layout(tmp_path):
    path = history_archive_path('job150615', history_root=tmp_path)
    assert path == tmp_path / '2026-10-04-cleanup/release-validation-fixtures/job150615/XTA_v25.0.0_complete_source.zip'
    assert history_archive_path('guarded_rescue', history_root=tmp_path).parent.name == 'guarded_rescue'


def test_missing_archive_fails_closed_instead_of_reducing_test_coverage(tmp_path):
    with pytest.raises(FileNotFoundError, match='missing from History'):
        require_history_archive('throughput', 'a' * 64, history_root=tmp_path)


def test_exact_archive_is_read_without_mutation_and_bad_bytes_are_rejected(tmp_path):
    path = history_archive_path('throughput', history_root=tmp_path)
    path.parent.mkdir(parents=True)
    original = b'exact retained artifact'
    path.write_bytes(original)
    expected = hashlib.sha256(original).hexdigest()
    assert require_history_archive('throughput', expected, history_root=tmp_path) == path
    assert path.read_bytes() == original
    path.write_bytes(original + b'changed')
    with pytest.raises(ValueError, match='independent SHA256 pin'):
        require_history_archive('throughput', expected, history_root=tmp_path)


@pytest.mark.parametrize('name', ['../throughput', 'unknown'])
def test_unknown_names_cannot_escape_fixture_inventory(tmp_path, name):
    with pytest.raises(ValueError, match='Unknown retained release fixture'):
        history_archive_path(name, history_root=tmp_path)


def test_digest_must_be_an_exact_pin(tmp_path):
    with pytest.raises(ValueError, match='exact SHA256 pin'):
        require_history_archive('throughput', 'not-a-pin', history_root=tmp_path)

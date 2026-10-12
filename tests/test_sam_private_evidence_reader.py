"""Private writer reads reuse one descriptor without caching trusted payloads."""
import io
from pathlib import Path

import numpy as np
import pytest

from XTA.sam_evidence import SamEvidenceWriter


def test_staged_reader_reuses_handle_and_observes_appends(tmp_path, monkeypatch):
    opened = []
    original = Path.open
    def recording_open(path, *args, **kwargs):
        result = original(path, *args, **kwargs)
        if path.name == 'masks.bin' and args == ('rb',):
            opened.append(result)
        return result
    monkeypatch.setattr(Path, 'open', recording_open)
    with SamEvidenceWriter(tmp_path/'evidence', {}) as writer:
        rng = np.random.default_rng(156023)
        expected = {}
        for index in range(5):
            mask = rng.random((67, 89)) < .13
            key = str(index)
            writer._put(key, mask, mask.shape)
            expected[key] = mask
            for old_key, old_mask in expected.items():
                np.testing.assert_array_equal(writer._read_staged(old_key), old_mask)
            assert not writer._read_staged(key).flags.writeable
        assert len(opened) == 1 and isinstance(opened[0], io.FileIO)
    assert opened[0].closed and not writer.staging.exists()


@pytest.mark.parametrize('mutation', ['byte', 'truncate', 'replace', 'missing'])
def test_staged_reader_authenticates_after_each_prior_read(tmp_path, monkeypatch, mutation):
    with SamEvidenceWriter(tmp_path/'evidence', {}) as writer:
        mask = np.eye(33, dtype=bool)
        writer._put('mask', mask, mask.shape)
        np.testing.assert_array_equal(writer._read_staged('mask'), mask)
        reader = writer._staged_reader
        path = writer.staging/'masks.bin'
        if mutation in {'replace', 'missing'}:
            replacement = writer.staging/'replacement.bin'
            replacement.write_bytes(path.read_bytes())
            original_stat = Path.stat
            # Windows forbids unlinking open files; emulate the POSIX path result.
            def changed_stat(candidate, *args, **kwargs):
                if candidate == path:
                    if mutation == 'missing':
                        raise FileNotFoundError(str(path))
                    return original_stat(replacement, *args, **kwargs)
                return original_stat(candidate, *args, **kwargs)
            monkeypatch.setattr(Path, 'stat', changed_stat)
        else:
            with path.open('r+b', buffering=0) as change:
                if mutation == 'truncate':
                    change.truncate(0)
                else:
                    byte = change.read(1)
                    change.seek(0)
                    change.write(bytes([byte[0] ^ 1]))
        with pytest.raises((ValueError, FileNotFoundError)):
            writer._read_staged('mask')
    assert reader.closed and not writer.staging.exists()


@pytest.mark.parametrize('failure', [False, True])
def test_commit_closes_private_reader_even_when_publication_rejected(tmp_path, failure):
    with SamEvidenceWriter(tmp_path/'evidence', {}) as writer:
        mask = np.eye(17, dtype=bool)
        writer._put('mask', mask, mask.shape)
        writer._read_staged('mask')
        reader = writer._staged_reader
        if failure:
            writer._packing_failure = True
            with pytest.raises(RuntimeError, match='successful publication'):
                writer.commit()
        else:
            bundle = writer.commit()
            np.testing.assert_array_equal(bundle.mask('mask'), mask)
        assert reader.closed and writer._staged_reader is None

"""Sealed CVOL members retain bounded streaming reads and legacy crop bytes."""
import contextlib
import json

import numpy as np
import pytest

from XTA import artifact_archive as archive
from XTA import interpolation as ip


def make_store(tmp_path, *, packed=False, empty=False):
    source = np.zeros((4, 7, 13), np.uint8)
    if not empty:
        source[1, 1:6:2, 2:12:2] = 1
        source[2, 0, 0] = source[2, -1, -1] = 1
        source[3] = 1
    directory = tmp_path / 'plain.cvol'
    ip.write_raw_bbox_mask_store(source, directory, format_name=(
        ip.INTERNAL_PACKED_CVOL_FORMAT if packed else ip.CVOL_FORMAT))
    return source, directory


def append_store(tmp_path, directory, replacements=None):
    path = tmp_path / 'artifacts.tar'
    logical = 'views/' + 'long_parent_name_' * 25 + '/selected.cvol'
    members = {f'{logical}/{p.name}': p for p in directory.iterdir() if p.is_file()}
    members.update({f'{logical}/{name}': value for name, value in (replacements or {}).items()})
    archive.append_members(path, members)
    return path, archive.reference(path, logical)


@pytest.mark.parametrize('packed', [False, True])
@pytest.mark.parametrize('options', [{}, {'mmap_payload': True}, {'cache_payload_in_ram': True}])
def test_archived_crops_match_plain_and_remain_valid_after_append(tmp_path, packed, options):
    source, directory = make_store(tmp_path, packed=packed)
    path, reference = append_store(tmp_path, directory)
    with contextlib.closing(ip.RawBBoxMaskStore.open(reference, **options)) as stored, \
            contextlib.closing(ip.RawBBoxMaskStore.open(directory, mmap_payload=True)) as plain:
        assert isinstance(stored._chunks_bytes, np.memmap)
        info = archive.member_info(str(reference) + '/chunks.bin')
        assert len(stored._chunks_bytes) == info['bytes']
        assert stored._chunks_bytes.offset == info['offset']
        assert not stored.index.flags.writeable
        archive.append_members(path, {'unrelated/next.bin': b'not a mask' * 256})
        for z in range(source.shape[0]):
            np.testing.assert_array_equal(stored.decode_slice(z), source[z])
            first, second = stored.decode_slice_crop(z), plain.decode_slice_crop(z)
            if first is None:
                assert second is None
            else:
                assert first[:4] == second[:4]
                np.testing.assert_array_equal(first[4], second[4])
            np.testing.assert_array_equal(stored.decode_slice(z, dtype=bool), source[z] != 0)
        actual = np.concatenate([member.reshape(-1) if member is not None else np.zeros(size, np.uint8)
            for size, member in stored.iter_native_sparse_members(0, len(source), member_bytes=91)])
        np.testing.assert_array_equal(actual, source.reshape(-1))
        stored.unlink()
        assert path.is_file()
    with contextlib.closing(ip.RawBBoxMaskStore.open(reference)) as reopened:
        np.testing.assert_array_equal(reopened.decode_slice(3), source[3])


def test_empty_archive_payload_needs_no_mapping(tmp_path):
    source, directory = make_store(tmp_path, empty=True)
    _path, reference = append_store(tmp_path, directory)
    with contextlib.closing(ip.RawBBoxMaskStore.open(reference, mmap_payload=True)) as stored:
        assert stored._chunks_bytes == b''
        for z in range(len(source)):
            np.testing.assert_array_equal(stored.decode_slice(z), source[z])


@pytest.mark.parametrize('change', ['short_index', 'extra_index', 'outside_member',
                                   'declared_payload', 'invalid_kind', 'invalid_shape'])
def test_archive_rejects_invalid_lengths_and_cross_member_offsets(tmp_path, change):
    _source, directory = make_store(tmp_path)
    replacements = {}
    if change in ('short_index', 'extra_index'):
        raw = (directory / 'index.bin').read_bytes()
        replacements['index.bin'] = raw[:-1] if change == 'short_index' else raw + b'\0'
    elif change in ('outside_member', 'invalid_kind'):
        index = np.fromfile(directory / 'index.bin', dtype=ip.CTILE_INDEX_DTYPE)
        if change == 'outside_member':
            index[1]['offset'] = (directory / 'chunks.bin').stat().st_size - 1
        else:
            index[1]['kind'] = 2
        replacements['index.bin'] = index.tobytes()
    else:
        meta = json.loads((directory / 'meta.json').read_text())
        if change == 'declared_payload':
            meta['stats']['raw_payload_bytes'] += 1
        else:
            meta['shape'][0] = -1
        replacements['meta.json'] = json.dumps(meta).encode()
    path, reference = append_store(tmp_path, directory, replacements)
    archive.append_members(path, {'unrelated/bytes.bin': bytes(4096)})
    with pytest.raises(ValueError):
        ip.RawBBoxMaskStore.open(reference)


@pytest.mark.parametrize('member', ['meta.json', 'index.bin', 'chunks.bin'])
def test_archive_verifies_member_hash_before_exposing_payload(tmp_path, member):
    _source, directory = make_store(tmp_path)
    path, reference = append_store(tmp_path, directory)
    info = archive.member_info(str(reference) + '/' + member)
    with path.open('r+b') as handle:
        handle.seek(info['offset'])
        original = handle.read(1)
        handle.seek(info['offset'])
        handle.write(bytes([original[0] ^ 1]))
    with pytest.raises((ValueError, OSError)):
        ip.RawBBoxMaskStore.open(reference)

"""Packed contact census preserves every dense-reader authentication and bit."""
import copy
import hashlib
import io
import json
import tracemalloc
import zlib

import numpy as np
import pytest

from XTA.sam_crop_retry import raw_crop_boundary_contacts, crop_boundary_contacts_from_counts
from XTA.sam_evidence import (SamEvidenceWriter, SamEvidenceBundle, fingerprint, _decode_mask,
    _decode_raw_crop_boundary_contacts)


def _record(mask, *, packed=None, encoded_suffix=b''):
    array = np.asarray(mask, dtype=bool)
    packed = np.packbits(array.reshape(-1), bitorder='little').tobytes() if packed is None else packed
    encoded = zlib.compress(packed) + encoded_suffix
    record = dict(shape=list(array.shape), packed_bytes=len(packed), bytes=len(encoded), offset=0,
        compressed_sha256=hashlib.sha256(encoded).hexdigest(), sha256=hashlib.sha256(packed).hexdigest(),
        foreground=int(np.count_nonzero(array)))
    return encoded, record


def _contacts(encoded, record, *, box=None, canvas=None, budget=64*1024**2):
    height, width = map(int, record['shape'])
    box = (5, 7, 5+height, 7+width) if box is None else box
    canvas = (height+20, width+20) if canvas is None else canvas
    return _decode_raw_crop_boundary_contacts(io.BytesIO(encoded), record, budget, box, canvas)


@pytest.mark.parametrize('shape', [(1, 1), (1, 7), (7, 1), (1, 8193), (8193, 1),
    (3, 5), (5, 3), (9, 13), (17, 31), (101, 137)])
@pytest.mark.parametrize('pattern', ['empty', 'white', 'random', 'one_pixel', 'all_boundaries'])
def test_exact_dense_parity_for_odd_strides_and_every_boundary(shape, pattern):
    mask = np.zeros(shape, bool)
    if pattern == 'white':
        mask[:] = True
    elif pattern == 'random':
        mask[:] = np.random.default_rng(1729).random(shape) > .61
    elif pattern == 'one_pixel':
        mask[-1, -1] = True
    elif pattern == 'all_boundaries':
        mask[0, :] = mask[-1, :] = mask[:, 0] = mask[:, -1] = True
    encoded, record = _record(mask)
    box, canvas = (5, 7, 5+shape[0], 7+shape[1]), (shape[0]+20, shape[1]+20)
    dense = _decode_mask(io.BytesIO(encoded), record, 64*1024**2)
    assert _contacts(encoded, record) == raw_crop_boundary_contacts(dense, box, canvas)


@pytest.mark.parametrize('box,canvas', [((0, 0, 3, 5), (3, 5)), ((0, 7, 3, 12), (10, 12)),
    ((5, 0, 8, 5), (8, 20)), ((5, 7, 8, 12), (20, 20))])
def test_canvas_border_classification_is_exact(box, canvas):
    mask = np.ones((3, 5), bool)
    encoded, record = _record(mask)
    assert _contacts(encoded, record, box=box, canvas=canvas) == raw_crop_boundary_contacts(mask, box, canvas)


def test_padding_bits_are_authenticated_but_excluded_from_foreground_and_edges():
    mask = np.eye(3, 5, dtype=bool)
    packed = bytearray(np.packbits(mask.reshape(-1), bitorder='little').tobytes())
    packed[-1] |= 128  # Padding after valid bit14; legacy decoder deliberately ignores it.
    encoded, record = _record(mask, packed=bytes(packed))
    dense = _decode_mask(io.BytesIO(encoded), record, 1024)
    assert np.array_equal(dense, mask)
    assert _contacts(encoded, record) == raw_crop_boundary_contacts(mask, (5, 7, 8, 12), (23, 25))


@pytest.mark.parametrize('field,value,reason', [('compressed_sha256', '0'*64, 'compressed-mask checksum'),
    ('sha256', '0'*64, 'mask checksum'), ('foreground', 0, 'foreground count'),
    ('packed_bytes', 99, 'packed shape'), ('shape', [0, 5], 'mask shape'),
    ('shape', [2**40, 2**40], 'mask shape'), ('bytes', 10_000, 'compressed mask')])
def test_every_record_authentication_is_kept(field, value, reason):
    mask = np.zeros((3, 5), bool)
    mask[1, 2] = True  # An interior-only corruption must still fail a boundary query.
    encoded, record = _record(mask)
    record[field] = value
    with pytest.raises(ValueError, match=reason):
        _contacts(encoded, record)


def test_complete_compressed_payload_is_authenticated_before_boundary_access():
    encoded, record = _record(np.eye(7, dtype=bool))
    corrupted = bytearray(encoded)
    corrupted[len(encoded)//2] ^= 1
    with pytest.raises(ValueError, match='compressed-mask checksum'):
        _contacts(bytes(corrupted), record)


@pytest.mark.parametrize('mode', ['trailing', 'truncated', 'overlong'])
def test_malformed_or_bomb_zlib_fails_after_matching_compressed_checksum(mode):
    mask = np.eye(3, 5, dtype=bool)
    encoded, record = _record(mask, encoded_suffix=b'forbidden' if mode == 'trailing' else b'')
    if mode == 'truncated':
        encoded = encoded[:-1]
    elif mode == 'overlong':
        encoded = zlib.compress(bytes(1024*1024))  # Bounded decompress must not expand this.
    record.update(bytes=len(encoded), compressed_sha256=hashlib.sha256(encoded).hexdigest())
    with pytest.raises(ValueError, match='Malformed.*payload|compressed mask'):
        _contacts(encoded, record)


def test_valid_payload_cannot_change_the_declared_crop_shape():
    encoded, record = _record(np.ones((3, 5), bool))
    with pytest.raises(ValueError, match='declared crop geometry'):
        _contacts(encoded, record, box=(5, 7, 10, 10))


def _bundle(tmp_path, mask):
    shape = mask.shape
    box = (5, 7, shape[0]+5, shape[1]+7)
    canvas = (shape[0]+20, shape[1]+20)
    seed = np.zeros(shape, bool)
    seed[0, 0] = True
    group = dict(group_id='G', context_bbox_yx=box, frame_indices=[0, 1], complete=True,
        endpoints=[dict(observation_id='A', frame_index=0), dict(observation_id='B', frame_index=1)], edges=[])
    masks = {'endpoint:A': seed, 'endpoint:B': seed, 'evaluation:A': np.ones(shape, bool),
        'evaluation:B': np.ones(shape, bool)}
    for frame in (0, 1):
        masks[f'acceptance:{frame}'] = np.ones(shape, bool)
        masks[f'write:{frame}'] = ~seed if frame else np.zeros(shape, bool)
    run = dict(run_id='R', group_id='G', direction='forward', seed_ids=['A'], held_out_ids=['B'],
        injected_frames=[0], expected_frames=[0, 1], complete=True)
    with SamEvidenceWriter(tmp_path/'evidence', {'shape_tyx': [2, *canvas]}) as writer:
        writer.add_group(group, masks)
        writer.add_run(run, {0: seed, 1: mask})
        return writer.commit(), box, canvas


def test_transaction_uses_packed_query_without_dense_decode_and_detaches_cache(tmp_path, monkeypatch):
    mask = np.random.default_rng(19).random((13, 17)) > .75
    bundle, box, canvas = _bundle(tmp_path, mask)
    expected = raw_crop_boundary_contacts(mask, box, canvas)
    with bundle.reader(max_cache_bytes=4096) as reader:
        monkeypatch.setattr(np, 'unpackbits', lambda *a, **k: pytest.fail('Dense unpack is forbidden for contacts'))
        contacts = reader.raw_crop_boundary_contacts('R', 1, crop_bbox_yx=box, canvas_shape_yx=canvas)
        assert contacts == expected
        contacts['internal_contacts']['top'] = -1
        assert reader.raw_crop_boundary_contacts('R', 1) == expected
        assert reader.stats['packed_boundary_contact_scans'] == 1
        assert reader.stats['mask_decodes'] == 0
        assert reader.stats['peak_cache_bytes'] <= 4096
    assert bundle.raw_crop_boundary_contacts('R', 1) == expected


@pytest.mark.parametrize('foreground', [0, 1])
def test_authenticated_foreground_distinguishes_empty_from_nonempty_interior_with_no_contacts(tmp_path, monkeypatch, foreground):
    mask = np.zeros((5, 7), bool)
    mask[2, 3] = bool(foreground)
    bundle, box, canvas = _bundle(tmp_path, mask)
    with bundle.reader() as reader:
        monkeypatch.setattr(np, 'unpackbits', lambda *a, **k: pytest.fail('Raw empty stopping does not decode a full mask'))
        contacts, count = reader.raw_crop_boundary_contacts_with_foreground('R', 1,
            crop_bbox_yx=box, canvas_shape_yx=canvas)
        assert not any(contacts['internal_contacts'].values())
        assert not any(contacts['canvas_edge_contacts'].values())
        assert count == foreground
        assert reader.raw_crop_boundary_contacts('R', 1) == contacts
        assert reader.stats['packed_boundary_contact_scans'] == 1
        assert reader.stats['mask_decodes'] == 0
    assert bundle.raw_crop_boundary_contacts_with_foreground('R', 1) == (contacts, foreground)


@pytest.mark.parametrize('kwargs', [dict(crop_bbox_yx=(0, 0, 3, 5)), dict(canvas_shape_yx=(99, 99))])
def test_caller_planned_geometry_is_bound_before_cache_reuse(tmp_path, kwargs):
    bundle, _, _ = _bundle(tmp_path, np.ones((3, 5), bool))
    with bundle.reader() as reader:
        reader.raw_crop_boundary_contacts('R', 1)
        with pytest.raises(ValueError, match='original planned geometry'):
            reader.raw_crop_boundary_contacts('R', 1, **kwargs)
    with pytest.raises(ValueError, match='original planned geometry'):
        bundle.raw_crop_boundary_contacts('R', 1, **kwargs)


def test_cached_boundary_query_retains_final_payload_mutation_check(tmp_path):
    bundle, _, _ = _bundle(tmp_path, np.ones((3, 5), bool))
    with pytest.raises(ValueError, match='payload changed'):
        with bundle.reader() as reader:
            reader.raw_crop_boundary_contacts('R', 1)
            path = bundle.directory/'masks.bin'
            raw = bytearray(path.read_bytes())
            raw[-1] ^= 1
            path.write_bytes(raw)
            reader.raw_crop_boundary_contacts('R', 1)  # Cache cannot waive transaction-final verification.


def test_shared_packed_record_cache_keeps_each_groups_boundary_geometry(tmp_path):
    mask = np.ones((3, 5), bool)
    original, box, canvas = _bundle(tmp_path, mask)
    index_path = original.directory/'index.json'
    index = json.loads(index_path.read_text('utf-8'))
    # Portable indexes can share an encoded mask record. The two descriptors
    # attach that same binary observation to differently positioned contexts.
    index['groups']['H'] = copy.deepcopy(index['groups']['G'])
    index['groups']['H'].update(group_id='H', context_bbox_yx=[0, 0, 3, 5])
    index['runs']['S'] = copy.deepcopy(index['runs']['R'])
    index['runs']['S'].update(run_id='S', group_id='H')
    raw = json.dumps(index).encode('utf-8')
    index_path.write_bytes(raw)
    manifest_path = original.directory/'manifest.json'
    manifest = json.loads(manifest_path.read_text('utf-8'))
    manifest.update(group_count=2, run_count=2)
    manifest['files']['index.json'].update(bytes=len(raw), sha256=hashlib.sha256(raw).hexdigest())
    manifest['evidence_fingerprint'] = fingerprint({key: value for key, value in manifest.items() if key != 'evidence_fingerprint'})
    manifest_path.write_text(json.dumps(manifest), encoding='utf-8')
    shared = SamEvidenceBundle.open(original.directory)
    with shared.reader() as reader:
        assert reader.raw_crop_boundary_contacts('R', 1) == raw_crop_boundary_contacts(mask, box, canvas)
        assert reader.raw_crop_boundary_contacts('S', 1) == raw_crop_boundary_contacts(mask, (0, 0, 3, 5), canvas)
        assert reader.stats['packed_boundary_contact_scans'] == 2


@pytest.mark.parametrize('shape', [(1, 4_000_001), (4_000_001, 1), (2001, 1999)])
def test_edge_scan_memory_stays_packed_plus_fixed_blocks(shape, monkeypatch):
    mask = np.zeros(shape, bool)
    mask[-1, -1] = True
    encoded, record = _record(mask)
    expected = raw_crop_boundary_contacts(mask, (5, 7, 5+shape[0], 7+shape[1]), (shape[0]+20, shape[1]+20))
    monkeypatch.setattr(np, 'unpackbits', lambda *a, **k: pytest.fail('No full mask expansion'))
    tracemalloc.start()
    try:
        result = _contacts(encoded, record)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert result == expected
    assert peak < max(2*1024**2, 4*record['packed_bytes']+512*1024)


def test_invalid_scalar_contacts_cannot_bypass_canonical_geometry():
    with pytest.raises(ValueError, match='side length'):
        crop_boundary_contacts_from_counts(dict(top=6, left=0, bottom=0, right=0), (0, 0, 3, 5), (3, 5))

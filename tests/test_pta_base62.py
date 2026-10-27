"""Augmentation tag encoding preserves the existing alphabet and collision IDs."""
import random
from types import SimpleNamespace
from unittest import mock

import pytest

from XTA import pta_dataset


def reference(digest, length):
    value = int.from_bytes(digest, 'big')
    chars = []
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for _ in range(int(length)):
        value, remainder = divmod(value, len(alphabet))
        chars.append(alphabet[remainder])
    return ''.join(reversed(chars))


@pytest.mark.parametrize('length', [-7, -1, 0, 1, 2, 3, 15, 16, 17, 43, 100])
def test_tags_keep_exact_padding_truncation_and_odd_widths(length):
    values = [0, 1, 61, 62, 3843, 3844, 62**16 - 1, 62**16,
              62**16 + 1, 2**256 - 1]
    generator = random.Random(9031)
    values.extend(generator.getrandbits(256) for _ in range(100))
    for value in values:
        digest = value.to_bytes(32, 'big')
        assert pta_dataset.base62_tag_from_digest(digest, length) == reference(digest, length)
    assert pta_dataset.base62_tag_from_digest(b'', length) == reference(b'', length)


def test_every_two_character_tag_uses_the_original_alphabet_order():
    alphabet = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'
    for value in range(62**2):
        assert pta_dataset.base62_tag_from_digest(value.to_bytes(2, 'big'), 2) == (
            alphabet[value // 62] + alphabet[value % 62]
        )


def test_truncated_collisions_keep_nonce_retry_and_global_tag_reservation():
    # Distinct full digests can share their least-significant sixteen base62 digits.
    digests = [(1).to_bytes(32, 'big'), (62**16 + 1).to_bytes(32, 'big'),
               (2).to_bytes(32, 'big')]
    collision = '0000000000000001'
    used = {collision}
    with mock.patch.object(pta_dataset, 'augmentation_digest', side_effect=digests) as digest:
        selected = pta_dataset.deterministic_augmentation_tag(
            SimpleNamespace(), 'source', 3, used_tags=used,
        )
    assert selected == '0000000000000002'
    assert used == {collision, selected}
    assert [call.kwargs['nonce'] for call in digest.call_args_list] == [0, 1, 2]
    assert all(call.kwargs['domain'] == 'tag' for call in digest.call_args_list)

import builtins
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import sys
import time
from types import SimpleNamespace
import zlib

import pytest

from XTA import gray_cache


@pytest.fixture(autouse=True)
def codecs(monkeypatch):
    gray_cache._resolve_gray_encoder.cache_clear()
    for name in ('imagecodecs','deflate','isal'):
        monkeypatch.setitem(sys.modules,name,None)
    gauges = []
    monkeypatch.setattr(gray_cache,'_log_gray_encoder',lambda codec,info:gauges.append((codec,info)))
    yield gauges
    gray_cache._resolve_gray_encoder.cache_clear()


def roundtrip(requested='auto'):
    raw = b'gray pixels'*4096
    actual, encoded = gray_cache._encode(raw,requested)
    assert gray_cache._decode(encoded,actual,len(raw),hashlib.sha256(raw).hexdigest()) == raw
    return actual, encoded


def test_missing_optional_imports_are_resolved_once(monkeypatch,codecs):
    original = builtins.__import__
    attempts = []
    def import_(name,*args,**kwargs):
        if name in ('imagecodecs','deflate','isal'):
            attempts.append(name)
        return original(name,*args,**kwargs)
    monkeypatch.setattr(builtins,'__import__',import_)
    for _ in range(8):
        assert roundtrip()[0] == 'zlib'
    info = gray_cache.gray_encoder_info()
    assert info['backend'] == 'stdlib.zlib'
    assert attempts == ['imagecodecs','deflate','isal']
    assert len(codecs) == 1 and codecs[0] == ('auto',info)
    assert len(info['fallback_reasons']) == 3
    json.dumps(info)
    info['fallback_reasons'].clear()
    assert len(gray_cache.gray_encoder_info()['fallback_reasons']) == 3


def test_zstd_remains_preferred(monkeypatch,codecs):
    fake = SimpleNamespace(__version__='fixture-zstd',
        zstd_encode=lambda raw,level:zlib.compress(raw,level),
        zstd_decode=lambda encoded,out:zlib.decompress(encoded))
    monkeypatch.setitem(sys.modules,'imagecodecs',fake)
    monkeypatch.setitem(sys.modules,'deflate',SimpleNamespace(zlib_compress=lambda *args:pytest.fail('native called')))
    assert roundtrip()[0] == 'zstd'
    assert gray_cache.gray_encoder_info() == dict(requested='auto',codec='zstd',backend='imagecodecs.zstd',
        version='fixture-zstd',level=1,fallback_reasons=[])
    assert len(codecs) == 1


def test_concurrent_first_use_proves_and_announces_once(monkeypatch,codecs):
    def encode(raw,level):
        time.sleep(.01)
        return zlib.compress(raw,level)
    monkeypatch.setitem(sys.modules,'imagecodecs',SimpleNamespace(zstd_encode=encode,
        zstd_decode=lambda encoded,out:zlib.decompress(encoded)))
    with ThreadPoolExecutor(max_workers=4) as pool:
        assert all(item[0] == 'zstd' for item in pool.map(lambda _:roundtrip(),range(8)))
    assert len(codecs) == 1


def test_native_zlib_uses_public_positional_api_and_existing_decoder(monkeypatch):
    calls = []
    def compress(raw,level):
        calls.append(level)
        return zlib.compress(raw,level)
    monkeypatch.setitem(sys.modules,'deflate',SimpleNamespace(__version__='0.9.0',zlib_compress=compress))
    actual,_encoded = roundtrip()
    info = gray_cache.gray_encoder_info()
    assert actual == 'zlib' and info['backend'] == 'libdeflate' and info['version'] == '0.9.0'
    assert calls == [1,1]
    assert gray_cache.SCHEMA == 'xta.gray_view_cache/1'


@pytest.mark.parametrize('version',['0.8.9','unknown',None,'0.9.0rc1'])
def test_unproven_or_old_libdeflate_skips_to_isal(monkeypatch,version):
    fake = SimpleNamespace(zlib_compress=lambda *args:pytest.fail('unproven binding called'))
    if version is not None: fake.__version__ = version
    monkeypatch.setitem(sys.modules,'deflate',fake)
    calls = []
    def compressobj(level,method,wbits):
        calls.append((level,method,wbits))
        return zlib.compressobj(level,method,wbits)
    monkeypatch.setitem(sys.modules,'isal',SimpleNamespace(__version__='fixture-isal',
        isal_zlib=SimpleNamespace(DEFLATED=zlib.DEFLATED,compressobj=compressobj)))
    assert roundtrip()[0] == 'zlib'
    info = gray_cache.gray_encoder_info()
    assert info['backend'] == 'isal'
    assert len(info['fallback_reasons']) == 2
    assert calls == [(1,zlib.DEFLATED,15)]*2


def test_explicit_zlib_keeps_stdlib_baseline(monkeypatch):
    monkeypatch.setitem(sys.modules,'imagecodecs',SimpleNamespace(zstd_encode=lambda *args:pytest.fail('Zstd called')))
    monkeypatch.setitem(sys.modules,'deflate',SimpleNamespace(__version__='0.9.0',
        zlib_compress=lambda *args:pytest.fail('native called')))
    raw = b'gray pixels'*4096
    assert gray_cache._encode(raw,'zlib') == ('zlib',zlib.compress(raw,1))
    assert gray_cache.gray_encoder_info('zlib')['backend'] == 'stdlib.zlib'


@pytest.mark.parametrize('requested',['auto','zstd'])
def test_corrupt_zstd_known_answer_fails_without_silent_fallback(monkeypatch,requested):
    monkeypatch.setitem(sys.modules,'imagecodecs',SimpleNamespace(
        zstd_encode=lambda raw,level:b'bad',zstd_decode=lambda encoded,out:b'wrong pixels'))
    with pytest.raises(gray_cache.GrayCacheError,match='known-answer'):
        gray_cache.gray_encoder_info(requested)


def test_gzip_native_stream_fails_instead_of_changing_cache_format(monkeypatch):
    def gzip_compress(raw,level):
        compressor = zlib.compressobj(level,zlib.DEFLATED,31)
        return compressor.compress(raw)+compressor.flush()
    monkeypatch.setitem(sys.modules,'deflate',SimpleNamespace(__version__='0.9.0',zlib_compress=gzip_compress))
    with pytest.raises(gray_cache.GrayCacheError,match='known-answer'):
        gray_cache.gray_encoder_info()


@pytest.mark.parametrize('missing',['import','attribute','not_implemented'])
def test_explicit_zstd_unavailable_is_clear_and_auto_falls_back(monkeypatch,missing):
    if missing == 'attribute':
        monkeypatch.setitem(sys.modules,'imagecodecs',SimpleNamespace())
    elif missing == 'not_implemented':
        def fail(*args,**kwargs): raise NotImplementedError('codec unavailable')
        monkeypatch.setitem(sys.modules,'imagecodecs',SimpleNamespace(zstd_encode=fail))
    with pytest.raises(gray_cache.GrayCacheError,match='Zstd.*unavailable'):
        gray_cache.gray_encoder_info('zstd')
    assert roundtrip()[0] == 'zlib'


def test_raw_and_invalid_requests():
    raw = b'x'*100
    assert gray_cache._encode(raw,'raw') == ('raw',raw)
    assert gray_cache.gray_encoder_info('raw')['level'] is None
    for invalid in ('lz4',[],None):
        with pytest.raises(gray_cache.GrayCacheError,match='Unknown requested'):
            gray_cache.gray_encoder_info(invalid)
        with pytest.raises(gray_cache.GrayCacheError,match='Unknown requested'):
            gray_cache._encode(raw,invalid)

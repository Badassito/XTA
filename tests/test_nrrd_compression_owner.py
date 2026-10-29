"""The stream owner is also the policy and lifecycle owner used by producers."""
import gzip
import io

import pytest

from XTA import nrrd_compression as compression, outputs


def test_raster_producer_uses_owner_admission_before_opening_output(monkeypatch, tmp_path):
    target = tmp_path / 'result.seg.nrrd'
    target.write_bytes(b'previous successful output')
    # Minimal metadata suffices because admission happens before payload publication.
    from types import SimpleNamespace
    ref = SimpleNamespace(source='global', shape=(1, 1, 1))
    plan = SimpleNamespace(stored_shape_tyx=(1, 1, 1), segment_extent_xyt=(0, 0, 0, 0, 0, 0))
    monkeypatch.setattr(outputs, '_resolve_live_ref_extent', lambda value: value)
    monkeypatch.setattr(outputs, '_nrrd_raster_plan', lambda *args, **kwargs: plan)
    monkeypatch.setattr(outputs, 'nrrd_slicer_header', lambda *args: {})
    monkeypatch.setattr(outputs, 'slicer_segmentation_header_fields', lambda *args, **kwargs: {})
    def reject(**kwargs):
        assert kwargs['expected_input_bytes'] == 1
        raise RuntimeError('codec admission sentinel')
    monkeypatch.setattr(compression, '_require_nrrd_member_codec', reject)
    with pytest.raises(RuntimeError, match='codec admission sentinel'):
        outputs.write_single_layer_nrrd_from_ref(ref, (1, 1, 1), target, z_shards=1)
    assert target.read_bytes() == b'previous successful output'
    assert sorted(p.name for p in tmp_path.iterdir()) == [target.name]


def test_failed_sink_settles_and_unregisters_owner_writer():
    class BrokenSink(io.BytesIO):
        def write(self, _data):
            raise OSError('sink failure sentinel')
    spec = compression._nrrd_member_codec_spec('zlib')
    writer = compression._MemberParallelGzipPayloadWriter(BrokenSink(), chunk_bytes=64, codec_spec=spec)
    with pytest.raises(OSError, match='sink failure sentinel'):
        with writer:
            writer.write(bytes(range(128)))
            writer.write_canonical_zeros(1025)
    assert writer.closed
    assert not writer._pending
    assert not writer._completed
    assert writer not in compression._NRRD_LIVE_WRITERS
    # Cleanup leaves the same owner's pool usable for a later embedded run.
    compression.shutdown_nrrd_gzip_executors()
    sink = io.BytesIO()
    with compression._MemberParallelGzipPayloadWriter(sink, codec_spec=spec) as stream:
        stream.write(b'recovery')
    assert gzip.decompress(sink.getvalue()) == b'recovery'
    compression.shutdown_nrrd_gzip_executors()

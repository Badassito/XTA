"""GPU retirement precedes CPU commit; unrelated frames do not cause repeated scans."""
import math
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from XTA import geometry, sam_integration
from XTA.lta_rendering import LtaPhysicalViewCacheRef
from tests.test_sam_gpu_rendering import FakeRenderer
from tests.test_sam_view_image_cache import SOURCE_SHAPE, context_for, crop_from, demand


@pytest.mark.parametrize('boundary', ['flush', 'revalidate'])
@pytest.mark.parametrize('fail', [False, True])
def test_gpu_renderer_retires_before_private_cache_finalization(tmp_path, monkeypatch, boundary, fail):
    context = context_for(tmp_path, np.zeros(SOURCE_SHAPE, np.uint8))
    view = geometry.get_view_infos(*SOURCE_SHAPE, cartesian_views=('sagittal',))[0]
    shape = (view.num_slices, 8,8)
    frames = {1:(1,2,7,6), 3:(2,1,6,7)}
    renderer = FakeRenderer(context)
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *_a, **_k:renderer)
    failure = OSError('controlled private image finalization failure')
    observed = []
    owner, name = (np.memmap, 'flush') if boundary == 'flush' else (LtaPhysicalViewCacheRef, 'revalidate')
    original = getattr(owner, name)
    def finalize(value, *args, **kwargs):
        if not observed:
            observed.append(boundary)
            assert renderer.calls == len(frames) and renderer.closed
            assert not context._caches and context._image_build_credit_bytes > 0
            if fail:
                raise failure
        return original(value, *args, **kwargs)
    monkeypatch.setattr(owner, name, finalize)
    try:
        if fail:
            with pytest.raises(OSError) as caught:
                context.image_provider(view, shape, demand(shape, frames))
            assert caught.value is failure and not context._caches
        else:
            reference = context.image_provider(view, shape, demand(shape, frames))
            for frame, box in frames.items():
                np.testing.assert_array_equal(crop_from(reference, frame, box), 201+frame%7)
        assert observed == [boundary] and renderer.closed
        assert not context._image_builds and context._image_build_credit_bytes == 0
    finally:
        context.close()


def test_unique_frame_cache_build_work_grows_linearly(tmp_path, monkeypatch):
    counts = []
    for frames in (128, 256):
        source = np.zeros((frames+1,3,4), np.uint8)
        context = context_for(tmp_path/str(frames), source)
        view = geometry.get_view_infos(*source.shape, cartesian_views=('transverse',))[0]
        renderer = FakeRenderer(context)
        monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer', lambda *_a, **_k:renderer)
        count = [0]
        def trace(frame, event, _argument):
            if frame.f_code.co_name == 'build_reference' and frame.f_code.co_filename == sam_integration.__file__:
                if event == 'line':
                    count[0] += 1
                return trace
            return None
        previous = sys.gettrace()
        try:
            sys.settrace(trace)
            reference = context.image_provider(view, source.shape,
                demand(source.shape, {frame:(0,0,2,3) for frame in range(frames)}))
        finally:
            sys.settrace(previous)
            context.close()
        assert renderer.calls == frames and len(reference.frame_crops) == frames
        counts.append(count[0])
    # Count work, not wall time: unrelated prior frames must not create a quadratic scan.
    assert 0 < counts[0] and counts[1] < 2.5*counts[0], counts


@pytest.mark.parametrize('native_count', [128, 1024])
@pytest.mark.parametrize('period', [180, 360])
def test_donor_lookup_is_logarithmic_and_preserves_cyclic_overlap(tmp_path, native_count, period):
    from XTA.sam_cyclic import mirror_bbox_yx
    height, width, evidence_count = 4,6,3*native_count-2
    records, addresses, payload = [], {}, bytearray()
    physical_boxes = ((0,0,3,3), (1,2,4,5), (0,0,4,6))
    def physical_image(frame):
        return np.arange(height*width, dtype=np.uint8).reshape(height,width)+frame%97
    for frame in range(evidence_count):
        cycle, native = divmod(frame, native_count)
        mirrored = period == 180 and bool(cycle%2)
        box = physical_boxes[cycle]
        if mirrored:
            box = mirror_bbox_yx(box, width)
        y0,x0,y1,x1 = box
        image = physical_image(native)
        if mirrored:
            image = image[:, ::-1]
        records.append((frame,*box,len(payload)))
        payload.extend(image[y0:y1,x0:x1].tobytes())
        addresses[frame] = dict(native_index=native, mirror_u=mirrored)
    path = tmp_path/'donor.dat'
    path.write_bytes(payload)
    reference = LtaPhysicalViewCacheRef(path, (evidence_count,height,width), 'uint8',
        'azimuthal_coronal', 'validated-test-geometry', len(payload), path.stat().st_mtime_ns, tuple(records))
    class CountedRecords:
        reads = 0
        def __len__(self):
            return len(reference.frame_crops)
        def __getitem__(self, index):
            self.reads += 1
            return reference.frame_crops[index]
        def __iter__(self):
            for index in range(len(self)):
                yield self[index]
    counted = CountedRecords()
    proxy = SimpleNamespace(frame_crops=counted, shape=reference.shape, open=reference.open)
    entry = dict(reference=proxy, geometry_identity='same', addresses=addresses,
                 native_shape=(native_count,height,width))
    context = context_for(tmp_path, np.zeros((native_count,height,width), np.uint8))
    try:
        for native in (0,native_count//2,native_count-1):
            target = np.full((height,width), 255, np.uint8)
            before = counted.reads
            missing = context._copy_cached_pixels(target, (0,0,height,width), native, 'same',
                                                   cache_entries=(entry,))
            assert counted.reads-before <= 3*math.ceil(math.log2(len(counted)))+9
            expected = physical_image(native)
            if native < native_count-2:
                assert missing == ()
                np.testing.assert_array_equal(target, expected)
            else:
                covered = np.zeros((height,width), bool)
                for y0,x0,y1,x1 in physical_boxes[:2]:
                    covered[y0:y1,x0:x1] = True
                np.testing.assert_array_equal(target[covered], expected[covered])
                assert (target[~covered] == 255).all()
                missing_mask = np.zeros_like(covered)
                for y0,x0,y1,x1 in missing:
                    missing_mask[y0:y1,x0:x1] = True
                np.testing.assert_array_equal(missing_mask, ~covered)
    finally:
        context.close()

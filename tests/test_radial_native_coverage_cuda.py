"""Opt-in actual CUDA coverage of parent pull and native Radial owner bitsets.

Run only under the root-controlled GPU_LOCK window:
YOLO_TTA_TEST_RADIAL_COVERAGE_CUDA=1. Default CPU collection skips every case.
"""
import os

import numpy as np
import pytest

from XTA.cylindrical_cuda_projection import RadialCudaProjector
from XTA.cylindrical_owner import RadialOwner
from tests.test_cylindrical_cuda_projection import contract
from tests.test_radial_native_coverage import analytic_domain, views


pytestmark = pytest.mark.skipif(os.environ.get('YOLO_TTA_TEST_RADIAL_COVERAGE_CUDA') != '1',
                               reason='actual CUDA Radial coverage requires an explicit GPU window')


def decode_packet(packet, shape):
    result = np.zeros((len(packet.records), *shape[1:]), np.uint8)
    for index, record in enumerate(packet.records):
        if not record.foreground:
            continue
        height, width = record.y1-record.y0, record.x1-record.x0
        raw = packet.payload[record.offset:record.offset+record.size]
        if packet.packed:
            crop = np.unpackbits(raw.reshape(height, (width+7)//8), axis=1,
                                 count=width, bitorder='little')
        else:
            crop = raw.reshape(height, width)
        result[index, record.y0:record.y1, record.x0:record.x1] = crop
    return result


def project_parent(source, view, shape):
    import torch
    assert torch.cuda.is_available(), 'requested actual CUDA qualification has no visible GPU'
    plan, metadata, boxes = contract(source, view, shape)
    result = np.zeros(shape, np.uint8)
    with RadialCudaProjector(source, plan, metadata, view, shape, boxes, True, 0,
                             reserve_bytes=0, block_bytes=2*shape[1]*shape[2]) as projector:
        for first in range(0, shape[0], 2):
            count = min(2, shape[0]-first)
            actual = projector.project(first, count)
            result[first:first+count] = actual
            for packed in (False, True):
                np.testing.assert_array_equal(decode_packet(projector.project_encoded(first, count, packed), shape), actual)
    return result


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_cuda_reduced_model_covers_anisotropic_native_caps_and_preserves_annulus(base):
    shape = (11, 15, 19)
    built = views((7, 9, 11), base)
    combined = np.zeros(shape, np.uint8)
    for view in built:
        combined |= project_parent(np.ones((view.num_slices, 3, 4), np.uint8), view, shape)
    np.testing.assert_array_equal(combined.astype(bool), analytic_domain(built[0], shape))


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
@pytest.mark.parametrize('tilt', (-23., 23.))
@pytest.mark.parametrize('direction', ('vertical', 'horizontal'))
def test_cuda_tilted_radial_native_height_footprint_matches_independent_roi(base, tilt, direction):
    shape = (11, 15, 19)
    built = [view for view in views((7, 9, 11), base, tilt=tilt) if view.tilt_direction == direction]
    combined = np.zeros(shape, np.uint8)
    for view in built:
        combined |= project_parent(np.ones((view.num_slices, 4, 4), np.uint8), view, shape)
    np.testing.assert_array_equal(combined.astype(bool), analytic_domain(built[0], shape))


@pytest.mark.parametrize('base', ('transverse', 'sagittal', 'coronal'))
def test_cuda_rational_restoration_preserves_closed_radius_and_black_middle_shell(base):
    shape = (21, 21, 21)
    built = views((7, 7, 7), base, minimum=1.)
    full, gap = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
    for view in built:
        data = np.ones((view.num_slices, 4, 4), np.uint8)
        full |= project_parent(data, view, shape)
        for index, radius in enumerate(view.radial_radii):
            if radius == 2.:
                data[index] = 0
        gap |= project_parent(data, view, shape)
    offsets = np.arange(21)-10
    a, b = np.meshgrid(offsets, offsets, indexing='ij')
    q = a*a+b*b
    plane = (q >= 9) & (q <= 81)
    chosen = np.argmin(np.abs(np.sqrt(q)[..., None]/3. - np.array((1., 2., 3.))), axis=-1)
    gap_plane = plane & (chosen != 1)
    transform = {'transverse': lambda p: np.broadcast_to(p, shape),
                 'sagittal': lambda p: np.broadcast_to(p[:, None, :], shape),
                 'coronal': lambda p: np.broadcast_to(p[:, :, None], shape)}[base]
    np.testing.assert_array_equal(full.astype(bool), transform(plane))
    np.testing.assert_array_equal(gap.astype(bool), transform(gap_plane))


@pytest.mark.parametrize('base,tilt,direction', (('transverse', None, ''), ('sagittal', None, ''),
    ('coronal', None, ''), ('transverse', -23., 'vertical'), ('coronal', 23., 'horizontal')))
def test_actual_radial_owner_bitset_covers_the_same_native_roi_across_shell_chunks(base, tilt, direction):
    import cupy as cupy
    assert cupy.cuda.runtime.getDeviceCount() > 0, 'requested owner CUDA qualification has no visible GPU'
    shape = (11, 15, 19)
    built = [view for view in views((7, 9, 11), base, tilt=tilt)
             if not tilt or view.tilt_direction == direction]
    combined = np.zeros(shape, np.uint8)
    for view in built:
        owner = RadialOwner(view, (3, 4), shape, reserve_bytes=0)
        try:
            for first in range(0, view.num_slices, 2):
                count = min(2, view.num_slices-first)
                device_source = cupy.ones((count, 3, 4), dtype=cupy.uint8)
                owner.consume(first, device_source)
                del device_source
            words = owner.host_words()
            bitmap = np.unpackbits(words.view(np.uint8), bitorder='little')[:np.prod(shape)].reshape(shape)
            combined |= bitmap
            packed = owner.host_packed_blocks()
            decoded = np.zeros(shape, np.uint8)
            for packet in packed.blocks:
                decoded[packet.first_z:packet.first_z+len(packet.records)] = decode_packet(packet, shape)
            np.testing.assert_array_equal(decoded, bitmap)
        finally:
            owner.close()
    np.testing.assert_array_equal(combined.astype(bool), analytic_domain(built[0], shape))

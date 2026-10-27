"""Resizing into caller-owned output storage avoids an extra full-cube copy."""
from unittest import mock

import numpy as np
import pytest

from XTA import media


@pytest.mark.parametrize('shape',[(9,6,7),(8,9,11),(5,6,7)])
@pytest.mark.parametrize('categorical',[False,True])
def test_supplied_cube_destination_matches_existing_pixels_without_allocating(tmp_path,shape,categorical):
    source=np.random.default_rng(61).integers(0,256,size=(5,6,7),dtype=np.uint8)
    if categorical:source=(source>128).astype(np.uint8)
    resize=(media.resize_categorical_volume_to_processing_cube_uint8 if categorical
            else media.resize_volume_to_processing_cube_gray8)
    expected=resize(source,shape,tmp_path/'reference.u8',workers=2,reserve_bytes=0)
    destination=np.empty(shape,np.uint8)
    with mock.patch.object(media,'allocate_workspace_array',side_effect=AssertionError('extra cube allocation')):
        actual=resize(source,shape,tmp_path/'unused.u8',workers=2,destination=destination)
    assert actual is destination
    np.testing.assert_array_equal(actual,expected)
    assert not (tmp_path/'unused.u8').exists()


@pytest.mark.parametrize('case',['shape','dtype','readonly','strided','overlap'])
def test_invalid_destination_fails_before_resize_or_allocation(tmp_path,case):
    source=np.zeros((5,6,7),np.uint8)
    target=(9,6,7)
    destination=np.empty(target,np.uint8)
    if case=='shape':destination=np.empty((8,6,7),np.uint8)
    elif case=='dtype':destination=np.empty(target,np.float32)
    elif case=='readonly':destination.flags.writeable=False
    elif case=='strided':destination=np.empty((9,6,14),np.uint8)[:,:,::2]
    else:
        target=source.shape
        destination=source.view()
    with mock.patch.object(media,'allocate_workspace_array',side_effect=AssertionError('invalid allocation')):
        for resize in (media.resize_volume_to_processing_cube_gray8,
                       media.resize_categorical_volume_to_processing_cube_uint8):
            with pytest.raises(ValueError,match='destination'):
                resize(source,target,tmp_path/'unused.u8',destination=destination)


def test_identity_source_can_be_its_own_destination(tmp_path):
    source=np.ones((3,4,5),np.uint8)
    assert media.resize_volume_to_processing_cube_gray8(source,source.shape,tmp_path/'unused',destination=source) is source
    assert media.resize_categorical_volume_to_processing_cube_uint8(source,source.shape,tmp_path/'unused',destination=source) is source

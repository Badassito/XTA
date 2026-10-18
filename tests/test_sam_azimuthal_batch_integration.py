"""The private cache consumes a native prefix once, with a genuine scalar control."""
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from XTA import geometry
from XTA.sam_gpu_rendering import SAM_GPU_IMAGE_CONTRACT
from tests.test_sam_view_image_cache import context_for, crop_from, demand


class PrefixRenderer:
    def __init__(self,context):
        self.context=context
        self.prefixes=[]
        self.scalar=[]
        self.closed=False
    def sampling_identity(self):
        return dict(contract=SAM_GPU_IMAGE_CONTRACT,backend='cuda',implementation_sha256='fixed-test')
    def render_compatible_prefix(self,view,frames,inverse,*,consume,**crop):
        assert self.context._image_build_credit_bytes>0
        assert not self.context._caches
        assert list((self.context.temp_dir/'sam_image_cache').glob('*.dat'))
        selected=tuple(frames[:2])
        self.prefixes.append(selected)
        for position,frame in enumerate(selected):
            consume(position,np.full((crop['output_height'],crop['output_width']),100+frame,np.uint8))
        return dict(batch_path_used=True,host_result_mode='sink',completed_images=2,effective_batch_size=2)
    def render(self,view,frame,inverse,**crop):
        self.scalar.append(frame)
        return np.full((crop['output_height'],crop['output_width']),100+frame,np.uint8)
    def close(self):self.closed=True


@pytest.mark.parametrize('enabled',[False,True])
def test_real_private_cache_grouping_and_disabled_baseline(tmp_path,monkeypatch,enabled):
    source=np.zeros((7,11,13),np.uint8)
    context=context_for(tmp_path,source)
    view=geometry.get_view_infos(*source.shape,cartesian_views=(),azimuthal_views=('coronal',),
        azimuthal_azimuth_angles=(30.,))[0]
    shape=(view.num_slices,8,8)
    frames={frame:(1,2,6,7) for frame in (0,1,2,3,4)}
    renderer=PrefixRenderer(context)
    monkeypatch.setenv('YOLO_TTA_SAM_AZIMUTHAL_BATCH',str(int(enabled)))
    monkeypatch.setattr('XTA.sam_gpu_rendering.try_gpu_crop_renderer',lambda *_a,**_k:renderer)
    try:
        reference=context.image_provider(view,shape,demand(shape,frames))
        for frame,bbox in frames.items():
            np.testing.assert_array_equal(crop_from(reference,frame,bbox),100+frame)
        assert context.rendered_frames==5 and context.rendered_pixels==125
        assert renderer.closed and len(context._cache_entries)==1
        assert not context._image_builds and context._image_build_credit_bytes==0
        assert renderer.prefixes==([(0,1),(2,3)] if enabled else [])
        assert renderer.scalar==([4] if enabled else [0,1,2,3,4])
    finally:
        context.close()

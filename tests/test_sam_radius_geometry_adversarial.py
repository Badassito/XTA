"""Independent finite-domain and strided checks of component ROI radius proof."""
import numpy as np
from scipy import ndimage

from XTA import sam_filtering


def full_plane_reference(raw,threshold):
    labels,count=ndimage.label(raw,structure=np.ones((3,3),bool))
    if count:
        padded=ndimage.distance_transform_edt(np.pad(raw,1))[1:-1,1:-1]
        radii=np.asarray(ndimage.maximum(padded,labels,np.arange(1,count+1)),np.float64).reshape(-1)
    else:
        radii=np.empty(0,np.float64)
    sizes=np.bincount(labels.reshape(-1),minlength=count+1)[1:]
    expected=np.r_[False,radii>threshold][labels]
    return expected,radii,sizes


def assert_exact(raw,threshold):
    original=raw.copy()
    expected,radii,sizes=full_plane_reference(raw,threshold)
    actual,diagnostic=sam_filtering.filter_sam_components(raw,threshold)
    np.testing.assert_array_equal(actual,expected)
    np.testing.assert_array_equal(raw,original)
    assert not actual.flags.writeable
    count=len(radii)
    assert diagnostic['raw_component_count']==count
    assert diagnostic['removed_component_count']==int((radii<=threshold).sum())
    assert diagnostic['retained_component_count']==int((radii>threshold).sum())
    assert diagnostic['omitted_component_records']==max(0,count-128)
    assert diagnostic['components']==[
        dict(component_id=i+1,foreground=int(sizes[i]),maximum_inscribed_radius=float(radii[i]),
             removed=bool(radii[i]<=threshold)) for i in range(min(count,128))]


def test_every_three_by_three_binary_pattern_preserves_ids_radii_and_equal_threshold():
    # Exhaustive finite domain covers isolated/diagonal/edge-touching components,
    # a central hole, full rectangles and nonrectangular interiors. Four exact
    # thresholds exercise whole-component removal, including sqrt(2) equality.
    bit_positions=np.arange(9,dtype=np.uint16)
    for bits in range(1<<9):
        raw=((np.uint16(bits)>>bit_positions)&1).astype(bool).reshape(3,3)
        for threshold in (.5,1.,np.sqrt(2.),2.):
            assert_exact(raw,threshold)


def test_strided_readonly_many_components_and_border_spur_match_global_reference():
    raw=np.zeros((79,103),bool)
    raw[:19,:17]=True
    raw[9,17:69]=True  # Attached one-pixel spur must survive with its body.
    raw[27::3,31::3]=True  # Exceeds the diagnostic-record limit.
    raw[36:49,57:74]=True
    raw[39:46,61:69]=False
    variants=(raw,raw[::-1,::-1],raw[:,::2],np.asfortranarray(raw))
    for variant in variants:
        variant.setflags(write=False)
        for threshold in (1.,2.0052219321148828,4.):
            assert_exact(variant,threshold)

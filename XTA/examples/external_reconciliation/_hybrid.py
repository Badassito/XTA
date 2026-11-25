"""Component agreement, selective quorum, bounded rescue and candidate-only fill."""
from __future__ import annotations
from fractions import Fraction
import numpy as np
from scipy import ndimage as ndi

from ._morphology import candidate_window, disk

CROSS = ndi.generate_binary_structure(2,1)


def fraction_pass(positive,total,fraction):
    ratio=Fraction(str(fraction))
    return ratio.denominator*positive >= ratio.numerator*total


def local_additions(candidate,accepted,radius,fraction):
    width=2*radius+1
    total=np.rint(ndi.uniform_filter(candidate.astype(np.float64),size=width,mode='constant')*width*width).astype(np.int64)
    positive=np.rint(ndi.uniform_filter(accepted.astype(np.float64),size=width,mode='constant')*width*width).astype(np.int64)
    return candidate & (total>0) & fraction_pass(positive,total,fraction)


def decide_slice(candidate,score,support,direct,*,satellites=False,rescue=False,fill=False,
                 agreement=.60,satellite_fraction=.15,rescue_radius=1,fill_radius=3):
    b2=candidate & (score>=1.5) & (support>=2) & (direct>=1)
    b3=candidate & (score>=3.) & (support>=3) & (direct>=1)
    labels,count=ndi.label(candidate,structure=CROSS)
    sizes=np.bincount(labels.ravel(),minlength=count+1)
    n2=np.bincount(labels[b2],minlength=count+1)
    n3=np.bincount(labels[b3],minlength=count+1)
    base_ids=fraction_pass(n2,sizes,agreement)
    base_ids[0]=False
    base=base_ids[labels]
    strict_ids=fraction_pass(n3,sizes,agreement)
    strict_ids[0]=False
    satellite_ids=np.zeros(count+1,bool)
    if satellites and np.any(base_ids):
        largest=int(sizes[base_ids].max())
        ratio=Fraction(str(satellite_fraction))
        # Classify every candidate component, including base-failing ones.
        # Rescue cannot bypass the stronger vote merely by failing the base gate.
        satellite_ids=(sizes>0) & (ratio.denominator*sizes <= ratio.numerator*largest)
        satellite_ids[0]=False
    satellite=satellite_ids[labels]
    keep=b2 & base
    if rescue:
        domain=b2 & ~base
        cores=ndi.binary_opening(b3 & ~base,structure=disk(rescue_radius),border_value=0)
        # Bounded geodesic growth stays within two-vote support; it cannot jump
        # across an absent-candidate gap into another component at larger scales.
        recovered=ndi.binary_dilation(cores,structure=CROSS,iterations=rescue_radius,mask=domain)
        keep |= recovered
    if satellites:
        keep &= ~satellite | (b3 & strict_ids[labels])
    if fill:
        eligible=candidate & base & ~satellite
        # Additions only: this stage cannot undo a preserved or rescued region.
        keep |= local_additions(eligible,keep & eligible,fill_radius,agreement)
    return keep & candidate


def build_hybrid(name,*,satellites=False,rescue=False,fill=False,agreement=.60,
                 satellite_fraction=.15,rescue_radius_fraction=.002,fill_radius_fraction=.005):
    def decide(block):
        candidate=block['candidate']
        shorter=min(candidate.shape[1:])
        rescue_radius=max(1,int(round(shorter*rescue_radius_fraction)))
        fill_radius=max(1,int(round(shorter*fill_radius_fraction)))
        result=np.zeros_like(candidate,dtype=bool)
        for z in range(candidate.shape[0]):
            window = candidate_window(candidate[z])
            if window is None:
                continue
            # The rectangle contains every component. Fill and rescue only
            # consider candidates, so exterior pixels are all constant zeros.
            plane_window = (z, *window)
            result[plane_window] = decide_slice(
                candidate[plane_window], block['score'][plane_window], block['support'][plane_window],
                block['prediction_support'][plane_window], satellites=satellites, rescue=rescue, fill=fill,
                agreement=agreement,satellite_fraction=satellite_fraction,
                rescue_radius=rescue_radius,fill_radius=fill_radius)
        return result
    return dict(name=name,mode='weighted',grouping='sections',threshold=1.5,min_sources=2,
                min_prediction_sources=1,island_weighting=False,angular_tolerance_deg=1.,
                provenance_weights=dict(prediction=1.,bridge=.35,mixed=.5),decide=decide)

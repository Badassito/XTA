"""Exact categorical native coverage for axis-aligned D1 processing masks.

Cartesian axes are permuted before the established terminal temporal/XY
restore. Tables are small, global-grid descriptors; no source volume is built.
Spatial area contraction retains the existing canonical projector instead.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import physical_view_name
from .outputs import _nrrd_sparse_resize_axis_map, _restore_source_indices_for_output_z

ORTHOGONAL_COVERAGE_CONTRACT = 'xta.d1_orthogonal_categorical_coverage/2'


@dataclass(frozen=True)
class OrthogonalCoveragePlan:
    processing_shape_tyx: tuple[int, int, int]
    canonical_shape_tyx: tuple[int, int, int]
    source_shape_tyx: tuple[int, int, int]
    input_axes_tyx: tuple[int, int, int]
    source_ranges_tyx: tuple[tuple[np.ndarray, np.ndarray], ...]

    def output_bbox_for_input_crop(self, frame_index, bbox_yx):
        y0,x0,y1,x1=map(int,bbox_yx)
        ranges=((int(frame_index),int(frame_index)+1),(y0,y1),(x0,x1))
        if any(not 0 <= first < stop <= size for (first,stop),size in zip(ranges,self.processing_shape_tyx)):
            raise ValueError('D1 input crop lies outside its declared processing grid')
        bounds=[]
        for axis,(starts,stops) in zip(self.input_axes_tyx,self.source_ranges_tyx):
            first,stop=ranges[axis]
            lo=int(np.searchsorted(stops,first,side='right'))
            hi=int(np.searchsorted(starts,stop,side='left'))
            if lo >= hi:
                return None
            bounds.extend((lo,hi))
        return tuple(bounds)


def orthogonal_canonical_shape(view, processing_shape_tyx):
    shape=tuple(map(int,processing_shape_tyx))
    if len(shape)!=3 or min(shape)<1 or shape[0]!=int(view.num_slices):
        raise ValueError('D1 processing shape must match the positive view frame count')
    if str(view.family)!='orthogonal':
        raise ValueError('D1 categorical coverage supports axis-aligned Cartesian views only')
    axes={'transverse':(0,1,2),'sagittal':(1,0,2),'coronal':(1,2,0)}.get(physical_view_name(view))
    if axes is None:
        raise ValueError('D1 categorical coverage requires a known Cartesian axis mapping')
    return tuple(shape[axis] for axis in axes),axes


def orthogonal_coverage_supported(view, processing_shape_tyx, output_shape_tyx):
    canonical,_=orthogonal_canonical_shape(view,processing_shape_tyx)
    output=tuple(map(int,output_shape_tyx))
    if len(output)!=3 or min(output)<1:
        raise ValueError('D1 source shape must contain three positive dimensions')
    area=canonical[1]>=output[1] and canonical[2]>=output[2]
    return not area or canonical[1:]==output[1:]


def build_orthogonal_coverage_plan(view, processing_shape_tyx, output_shape_tyx):
    canonical,axes=orthogonal_canonical_shape(view,processing_shape_tyx)
    output=tuple(map(int,output_shape_tyx))
    if not orthogonal_coverage_supported(view,processing_shape_tyx,output):
        raise ValueError('Spatial area contraction requires the canonical categorical projector')
    time_ranges=[_restore_source_indices_for_output_z(canonical[0],output[0],z) for z in range(output[0])]
    ranges=[(np.asarray([indices[0] for indices in time_ranges],np.int32),
             np.asarray([indices[-1]+1 for indices in time_ranges],np.int32))]
    for inside,outside in zip(canonical[1:],output[1:]):
        starts,stops=_nrrd_sparse_resize_axis_map(inside,outside,False)
        ranges.append((np.asarray(starts,np.int32),np.asarray(stops,np.int32)))
    ranges=[(np.frombuffer(starts.tobytes(),np.int32),np.frombuffer(stops.tobytes(),np.int32))
            for starts,stops in ranges]
    return OrthogonalCoveragePlan(tuple(map(int,processing_shape_tyx)),canonical,output,axes,tuple(ranges))


def coverage_launch_groups(plan, bbox_specs, slice_start):
    """Bucket source-cell bounding rectangles, keeping each launch under 2x work."""
    groups={}
    for local,y0,x0,height,width in np.asarray(bbox_specs):
        frame=int(slice_start)+int(local)
        bounds=plan.output_bbox_for_input_crop(frame,(int(y0),int(x0),int(y0+height),int(x0+width)))
        if bounds is None:continue
        size=(bounds[1]-bounds[0])*(bounds[3]-bounds[2])*(bounds[5]-bounds[4])
        bucket=1<<(int(size)-1).bit_length()
        groups.setdefault(bucket,[]).append((int(local),frame,*bounds))
    return tuple((size,np.ascontiguousarray(specs,dtype=np.int32)) for size,specs in sorted(groups.items()))


__all__=['build_orthogonal_coverage_plan','orthogonal_coverage_supported',
    'coverage_launch_groups','ORTHOGONAL_COVERAGE_CONTRACT']

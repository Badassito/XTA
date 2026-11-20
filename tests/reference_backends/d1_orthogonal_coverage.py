"""Bounded CPU oracle for D1 orthogonal categorical coverage, outside the production package."""
from __future__ import annotations

import math

import numpy as np

from XTA.d1_orthogonal_coverage import build_orthogonal_coverage_plan


def execute_orthogonal_coverage_reference(volume,view,output_shape_tyx,*,slice_start=0,memory_mib=64):
    """Bounded CPU reference for independent/chunked packed-kernel qualification."""
    source=np.asarray(volume)
    if source.ndim!=3 or source.dtype.kind not in 'bu' or np.any(source>1):
        raise ValueError('D1 coverage reference requires a binary TYX mask')
    full_shape=(int(view.num_slices),*source.shape[1:])
    plan=build_orthogonal_coverage_plan(view,full_shape,output_shape_tyx)
    if math.prod(plan.source_shape_tyx)>int(float(memory_mib)*1024**2):
        raise ValueError('D1 CPU coverage reference exceeds its bounded output budget')
    if not 0 <= int(slice_start) < full_shape[0] or int(slice_start)+source.shape[0]>full_shape[0]:
        raise ValueError('D1 coverage reference chunk lies outside its view')
    result=np.zeros(plan.source_shape_tyx,np.uint8)
    projected=source.transpose(plan.input_axes_tyx)
    stack_axis=plan.input_axes_tyx.index(0)
    ys=plan.source_ranges_tyx[1][0].astype(np.int64)
    xs=plan.source_ranges_tyx[2][0].astype(np.int64)
    out_y=np.arange(len(ys));out_x=np.arange(len(xs))
    if stack_axis==1:
        valid=(ys>=slice_start)&(ys<slice_start+source.shape[0])
        out_y=out_y[valid];ys=ys[valid]-int(slice_start)
    elif stack_axis==2:
        valid=(xs>=slice_start)&(xs<slice_start+source.shape[0])
        out_x=out_x[valid];xs=xs[valid]-int(slice_start)
    starts,stops=plan.source_ranges_tyx[0]
    for z,(first,stop) in enumerate(zip(starts,stops)):
        for source_z in range(int(first),int(stop)):
            if stack_axis==0:
                source_z-=int(slice_start)
                if not 0<=source_z<source.shape[0]:continue
            result[z][np.ix_(out_y,out_x)]|=projected[source_z][np.ix_(ys,xs)]
    return result

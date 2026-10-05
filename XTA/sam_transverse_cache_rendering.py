"""Bounded slab-major reuse of the unchanged OpenCV processing-t resize.

Only a lazy Transverse source with unchanged XY geometry opts in. Sampling
uses the existing canonical global affine remap; no processing cube is built.
"""
from __future__ import annotations

import numpy as np

from .runtime import runtime_telemetry, runtime_telemetry_phase


def native_crop_bbox(bbox, inverse, native_shape_yx):
    """The existing demand renderer's exact two-pixel native preparation guard."""
    y0,x0,y1,x1=map(int,bbox)
    corners=np.array([[x0,y0],[x1-1.,y0],[x0,y1-1.],[x1-1.,y1-1.]])
    mapped=corners@np.asarray(inverse)[:,:2].T+np.asarray(inverse)[:,2]
    height,width=map(int,native_shape_yx)
    result=(max(0,int(np.floor(mapped[:,1].min()))-2),
            max(0,int(np.floor(mapped[:,0].min()))-2),
            min(height,int(np.ceil(mapped[:,1].max()))+3),
            min(width,int(np.ceil(mapped[:,0].max()))+3))
    return result if result[0]<result[2] and result[1]<result[3] else None


def _pixels(box):
    return (box[2]-box[0])*(box[3]-box[1])


def _cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError('SAM transverse batch rendering cancelled')


@runtime_telemetry_phase('sam.transverse_cache.batch_render')
def _render_batch(decoded, out_shape, requests, roi, affine, canvas_width, budget, cancel_event):
    from ._deps import cv2
    from .sam_canvas_rendering import _remap_global_affine_crop
    from . import geometry
    geometry.require_forward_sampling('cpu',geometry.DataRole.INTENSITY)
    in_t=int(decoded.shape[0]);out_t=int(out_shape[0])
    y0,x0,y1,x1=roi
    frames=tuple(dict.fromkeys(frame for frame,_bbox in requests))
    native_bytes=len(frames)*_pixels(roi)
    planes=np.empty((len(frames),y1-y0,x1-x0),np.uint8)
    # Keep the same CV resize height and identity pixel-column axis. Spatial
    # chunk width changes allocation only; pixel arithmetic remains identical.
    pixel_cap=min(32760,(budget-native_bytes)//max(1,in_t+out_t))
    if pixel_cap<1:
        raise RuntimeError('SAM transverse batch exceeds admitted rendering workspace')
    resize_calls=slab_bytes=0;peak=native_bytes
    for column in range(x0,x1,pixel_cap):
        column_stop=min(x1,column+pixel_cap)
        row_count=max(1,pixel_cap//(column_stop-column))
        for row in range(y0,y1,row_count):
            _cancel(cancel_event)
            row_stop=min(y1,row+row_count)
            slab=np.ascontiguousarray(decoded[:,row:row_stop,column:column_stop],dtype=np.uint8)
            resized=cv2.resize(slab.reshape(in_t,-1),(slab.shape[1]*slab.shape[2],out_t),
                               interpolation=cv2.INTER_LINEAR)
            peak=max(peak,native_bytes+slab.nbytes+resized.nbytes)
            slab_bytes+=int(slab.nbytes);resize_calls+=1
            for position,frame in enumerate(frames):
                planes[position,row-y0:row_stop-y0,column-x0:column_stop-x0]=resized[frame].reshape(
                    row_stop-row,column_stop-column)
            del slab,resized
    outputs=[];held_output_bytes=0;sampling_pixels=0
    matrix=cv2.invertAffineTransform(np.asarray(affine,dtype=np.float32).astype(np.float64))
    positions={frame:position for position,frame in enumerate(frames)}
    for frame,bbox in requests:
        _cancel(cancel_event)
        stats={}
        image=_remap_global_affine_crop(planes[positions[frame]],matrix,
            output_origin_yx=bbox[:2],output_height=bbox[2]-bbox[0],output_width=bbox[3]-bbox[1],
            native_origin_xy=(x0,y0),output_canvas_width=canvas_width,
            max_workspace_bytes=budget-native_bytes-held_output_bytes,sampling_stats=stats)
        # The remapper's cap includes its own native view and current output;
        # reserving the entire batch again is conservative, never uncharged.
        held_output_bytes+=int(image.nbytes)
        sampling_pixels+=stats.get('sampled_output_pixels',0)
        outputs.append((frame,bbox,image))
    del planes
    counters=dict(batches=1,frames=len(requests),resize_calls=resize_calls,
        decoded_slab_bytes=slab_bytes,native_prepared_pixels=native_bytes,
        canonical_sampled_pixels=sampling_pixels,output_bytes=held_output_bytes,
        admitted_workspace_bytes=budget,peak_resize_workspace_bytes=peak)
    telemetry=runtime_telemetry()
    for key,value in counters.items():
        if key not in ('admitted_workspace_bytes','peak_resize_workspace_bytes'):
            telemetry.add('sam.transverse_cache.'+key,value)
    telemetry.gauge('sam.transverse_cache.last_workspace_bytes',budget)
    telemetry.gauge('sam.transverse_cache.last_resize_peak_bytes',peak)
    return outputs,counters


def iter_transverse_crop_batches(decoded, out_shape, requests, *, affine, inverse,
                                  canvas_width, max_workspace_bytes, cancel_event=None):
    """Yield ordered crop batches or ``None`` for unchanged independent rendering.

    Native batch planes and all retained output crops each use at most a quarter
    of the existing admitted cap. Slabs and canonical remap share the remaining
    workspace. Disjoint ROI unions that increase resize work stay independent.
    """
    requests=tuple((int(frame),tuple(map(int,bbox)))for frame,bbox in requests)
    budget=int(max_workspace_bytes);shape=tuple(map(int,out_shape))
    if budget<1:
        raise ValueError('SAM transverse batch workspace must be positive')
    if len(shape)!=3 or min(shape)<1 or int(decoded.shape[0])<1 or tuple(decoded.shape[1:])!=shape[1:]:
        raise ValueError('Batched Transverse resize requires unchanged positive XY geometry')
    if any(not 0<=frame<shape[0] or len(bbox)!=4 or min(bbox[:2])<0
           or bbox[0]>=bbox[2] or bbox[1]>=bbox[3] or bbox[3]>int(canvas_width)
           for frame,bbox in requests):
        raise ValueError('Malformed SAM transverse batch frame/crop demand')
    rois=[native_crop_bbox(bbox,inverse,shape[1:])for _frame,bbox in requests]
    index=0
    while index<len(requests):
        _cancel(cancel_event)
        selected=[];roi=None;individual_area=0;output_bytes=0
        for cursor in range(index,min(len(requests),index+256)):
            if rois[cursor] is None:break
            next_roi=rois[cursor] if roi is None else (min(roi[0],rois[cursor][0]),min(roi[1],rois[cursor][1]),
                max(roi[2],rois[cursor][2]),max(roi[3],rois[cursor][3]))
            candidate=selected+[requests[cursor]]
            frames=len({frame for frame,_bbox in candidate})
            area=individual_area+_pixels(rois[cursor]);pixels=output_bytes+_pixels(requests[cursor][1])
            remap_row=max(min(int(canvas_width),bbox[3]-bbox[1]+32)*64 for _frame,bbox in candidate)
            if (frames*_pixels(next_roi)>budget//4 or pixels>budget//4
                    or len(candidate)>1 and _pixels(next_roi)>area
                    or frames*_pixels(next_roi)+pixels+_pixels(next_roi)+remap_row>budget):
                break
            selected=candidate;roi=next_roi;individual_area=area;output_bytes=pixels
        if len(selected)<2:
            runtime_telemetry().add('sam.transverse_cache.independent_frames',1)
            yield [(requests[index][0],requests[index][1],None)],dict(independent_frames=1)
            index+=1
        else:
            outputs,counters=_render_batch(decoded,shape,selected,roi,affine,canvas_width,budget,cancel_event)
            yield outputs,counters
            del outputs
            index+=len(selected)

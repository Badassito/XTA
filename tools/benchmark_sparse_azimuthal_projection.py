"""CPU-only sparse Tilted-Azimuthal projection sanity benchmark with exact parity.

Synthetic masks are the sole input; production bridge quality is not a target.
The reference is the existing bounded NumPy destination-owned coverage path.
A small analytic physical row-band oracle also checks each selected source.
Use --source-root to compare an extracted predecessor with the current source.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
from pathlib import Path
import sys
import time

_source_parser=argparse.ArgumentParser(add_help=False)
_source_parser.add_argument('--source-root',type=Path,default=Path(__file__).resolve().parents[1])
_source_args,_=_source_parser.parse_known_args()
SOURCE_ROOT=_source_args.source_root.resolve()
sys.path.insert(0,str(SOURCE_ROOT))
import numpy as np

from XTA.config import TiltedViewGroup
from XTA.geometry import get_view_infos
from XTA.interpolation import INTERNAL_PACKED_CVOL_FORMAT,RawBBoxMaskStore,write_raw_bbox_mask_store
from XTA.sparse_projection import project_azimuthal_sparse_store


def fingerprint_store(path):
    digest=hashlib.sha256()
    store=RawBBoxMaskStore.open(path,mmap_payload=True)
    try:
        for frame in range(store.shape[0]):
            digest.update(np.packbits(store.decode_slice(frame).reshape(-1),bitorder='little').tobytes())
    finally:
        store.close()
    return digest.hexdigest()


def independent_row_band_oracle(data,view,target):
    """Scalar inverse physical cells; angle and column values deliberately agree.

    This uses no production coverage iterators, plans, or projection kernels.
    Native rows downsample by categorical max and upsample by nearest center.
    """
    work=(view.full_t,view.full_h,view.full_w)
    stack,vertical,horizontal={'transverse':(0,1,2),'sagittal':(1,0,2),
                               'coronal':(2,0,1)}[view.azimuthal_base_view]
    shear=vertical if view.tilt_direction=='vertical' else horizontal
    tangent=math.tan(math.radians(view.tilt_angle_deg))
    canvas=max(view.src_h,view.src_w)
    scale=float(np.float32(data.shape[1]/canvas))
    offset=float(np.float32((data.shape[1]-1)/2-(data.shape[1]/canvas)*(view.src_h-1)/2))
    expected=np.zeros(target,np.uint8)
    for position in np.ndindex(target):
        point=[(index+.5)*inside/outside-.5 for index,inside,outside in zip(position,work,target)]
        delta_y=point[vertical]-(work[vertical]-1)/2
        delta_x=point[horizontal]-(work[horizontal]-1)/2
        if math.hypot(delta_y,delta_x)>view.roi_radius+.5:continue
        shift=-tangent*(point[shear]-(work[shear]-1)/2)-view.tilt_frame_start
        local=point[stack]+shift
        if view.src_h>target[stack]:
            first=max(0,math.floor(position[stack]*view.src_h/target[stack]+shift*view.src_h/work[stack]))
            stop=min(view.src_h,math.ceil((position[stack]+1)*view.src_h/target[stack]+shift*view.src_h/work[stack]))
            rows=range(first,stop)
        else:
            if not -.5<=local<work[stack]-.5:continue
            rows=(min(view.src_h-1,max(0,round((local+.5)*view.src_h/work[stack]-.5))),)
        for native_row in rows:
            model_row=min(data.shape[1]-1,max(0,round(scale*native_row+offset)))
            expected[position]|=data[0,model_row,0]!=0
    return expected


def validate_independent_oracle(output,backends,workers):
    view=next(view for view in get_view_infos(7,9,11,cartesian_views=(),
        tilt_groups=[TiltedViewGroup(('sagittal',),(30.,),('horizontal',))],
        azimuthal_views=('tilted_sagittal',),azimuthal_azimuth_angles=(30.,),
        azimuthal_native_raster=4) if view.family=='azimuthal' and view.tilt_angle_deg<0)
    target=(11,15,19)
    data=np.ones((view.num_slices,3,3),np.uint8);data[:,1]=0
    expected=independent_row_band_oracle(data,view,target)
    source=output/'oracle_input.cvol'
    with contextlib.redirect_stdout(io.StringIO()):
        write_raw_bbox_mask_store(data,source,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    checks=[]
    for backend in backends:
        os.environ['YOLO_TTA_NATIVE_PULL_BACKEND']=backend
        path=output/('oracle_'+backend)
        with contextlib.redirect_stdout(io.StringIO()):
            result=project_azimuthal_sparse_store(source,view,path,out_shape_tyx=target,workers=workers)
        store=RawBBoxMaskStore.open(path,mmap_payload=True)
        try:actual=np.stack([store.decode_slice(z) for z in range(target[0])])
        finally:store.close()
        np.testing.assert_array_equal(actual,expected,err_msg='Independent scalar physical row-band oracle')
        checks.append(dict(backend=backend,actual_backend=result['backend'],mismatched_voxels=0,
                           foreground_voxels=int(np.count_nonzero(expected))))
    return dict(synthetic_only=True,oracle='Independent scalar physical row-band cells',
                target_shape_tyx=target,checks=checks)


def run(output,shape,target,workers,heatsoak_seconds,repetitions,backends=('numpy','compiled')):
    output=Path(output).resolve()
    output.mkdir(parents=True,exist_ok=False)
    view=next(view for view in get_view_infos(*shape,cartesian_views=(),
        tilt_groups=[TiltedViewGroup(('sagittal',),(30.,),('horizontal',))],
        azimuthal_views=('tilted_sagittal',),azimuthal_azimuth_angles=(15.,))
        if view.family=='azimuthal' and view.tilt_angle_deg<0)
    data=np.zeros((view.num_slices,128,128),np.uint8)
    rng=np.random.default_rng(150993)
    # Small positive bodies at separated angles/rows give a large conservative
    # destination bbox while retaining sparse encoded input and realistic work.
    for frame in range(view.num_slices):
        y0=18+(frame*7)%65; x0=12+(frame*11)%75
        data[frame,y0:y0+16,x0:x0+22]=(rng.random((16,22))<.75).astype(np.uint8)
    source=output/'synthetic.cvol'
    with contextlib.redirect_stdout(io.StringIO()):
        write_raw_bbox_mask_store(data,source,format_name=INTERNAL_PACKED_CVOL_FORMAT,workers=1)
    prior=os.environ.get('YOLO_TTA_NATIVE_PULL_BACKEND')
    records=[]
    def project(tag,backend,count):
        os.environ['YOLO_TTA_NATIVE_PULL_BACKEND']=backend
        started=time.perf_counter()
        with contextlib.redirect_stdout(io.StringIO()):
            result=project_azimuthal_sparse_store(source,view,output/tag,out_shape_tyx=target,workers=count)
        result['wall_seconds']=time.perf_counter()-started
        result['mask_sha256']=fingerprint_store(output/tag)
        result['case']=tag
        # Older sparse projectors were serial and did not publish this field.
        result.setdefault('projection_workers',1)
        return result
    try:
        oracle=validate_independent_oracle(output,backends,workers)
        reference=project('reference_numpy','numpy',1)
        warmed=project('compile_warm','compiled',1) if 'compiled' in backends else None
        if warmed is not None:assert warmed['mask_sha256']==reference['mask_sha256']
        heat_backend='compiled' if 'compiled' in backends else 'numpy'
        heat_workers=workers if heat_backend=='compiled' else 1
        start=time.monotonic(); iteration=0
        while time.monotonic()-start<heatsoak_seconds:
            result=project(f'heatsoak_{iteration:03d}',heat_backend,heat_workers)
            assert result['mask_sha256']==reference['mask_sha256']
            iteration+=1
        heatsoak_elapsed=time.monotonic()-start
        cases=([('numpy',1)] if 'numpy' in backends else [])+([('compiled',1),('compiled',workers)] if 'compiled' in backends else [])
        for repetition in range(repetitions):
            # Reverse each round to expose order/thermal drift in paired runs.
            for backend,count in cases if repetition%2==0 else reversed(cases):
                result=project(f'measured_{repetition:02d}_{backend}_{count}','compiled' if backend=='compiled' else 'numpy',count)
                assert result['mask_sha256']==reference['mask_sha256']
                records.append(result)
                print(json.dumps(dict(case=result['case'],seconds=result['wall_seconds'],
                    candidate_voxels=result['destination_candidate_voxels'],workers=result['projection_workers'])),flush=True)
    finally:
        if prior is None:os.environ.pop('YOLO_TTA_NATIVE_PULL_BACKEND',None)
        else:os.environ['YOLO_TTA_NATIVE_PULL_BACKEND']=prior
    report=dict(schema='xta.sparse_projection_cpu_benchmark/2',synthetic_only=True,gpu_used=False,
        source_root=str(SOURCE_ROOT),source_tool_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        invocation=sys.argv,independent_oracle=oracle,fixture_seed=150993,
        fixture_sha256=hashlib.sha256(data.tobytes()).hexdigest(),requested_backends=backends,
        heatsoak_backend=heat_backend,heatsoak_workers=heat_workers,
        timing_scope='Local warmed CPU sanity check; projection wall excludes correctness decode/hash and is not a cluster speed claim',
        shape_tyx=shape,target_shape_tyx=target,input_shape_tyx=list(data.shape),
        warm_reference=reference,warm_compiled=warmed,heatsoak_seconds=heatsoak_elapsed,
        heatsoak_iterations=iteration,records=records,
        source_sha256={name:hashlib.sha256((SOURCE_ROOT/'XTA'/name).read_bytes()).hexdigest()
                       for name in ('sparse_projection.py','projection_coverage_cpu.py')})
    (output/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--source-root',type=Path,default=SOURCE_ROOT)
    parser.add_argument('--backends',choices=('numpy','compiled'),nargs='+',default=('numpy','compiled'))
    parser.add_argument('--shape',type=int,nargs=3,default=(192,256,320))
    parser.add_argument('--target',type=int,nargs=3,default=(128,240,304))
    parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--heatsoak-seconds',type=float,default=20.)
    parser.add_argument('--repetitions',type=int,default=3)
    args=parser.parse_args()
    # GPU exclusion is explicit even when the surrounding shell enables it.
    os.environ['CUDA_VISIBLE_DEVICES']='-1'
    os.environ['YOLO_TTA_GPU_BACKPROJECT']='0'
    os.environ['YOLO_TTA_TELEMETRY']='0'
    run(args.output,tuple(args.shape),tuple(args.target),args.workers,args.heatsoak_seconds,args.repetitions,args.backends)


if __name__=='__main__':main()

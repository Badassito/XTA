"""Portable full-grid sampling of the frozen radius-weighted cubic policy.

Only mixed four-coefficient footprints require interpolation for binary output.
Uniform foreground/background shortcuts are exact sign proofs, not approximations.
Compact caches include all original foreground and three geometric-zero cells;
therefore in-source points beyond their four-tap domain have proven zero support.
Out-of-source queries remain invalid and are not conflated with that background.
"""
from __future__ import annotations
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import hashlib
import json
import math
import threading
import numpy as np
from scipy import ndimage as ndi

from .policy import POLICY, ALGORITHM_VERSION, package_source_identity

MIB = 1024 ** 2
NATIVE_BYTES_PER_QUERY = 512
BASE_BYTES_PER_NATIVE_VOXEL = 96

def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for data in iter(lambda: stream.read(4*MIB), b''):
            digest.update(data)
    return digest.hexdigest()

def _shape(value):
    if len(value) != 3 or any(isinstance(n,(bool,np.bool_)) or not isinstance(n,(int,np.integer)) or n < 1 for n in value):
        raise ValueError('Expected three positive integer TYX dimensions')
    return tuple(int(n) for n in value)

class FieldReference:
    def __init__(self, cache_dir, native_shape_tyx=None):
        self.cache_dir = Path(cache_dir).resolve()
        path = self.cache_dir/'COMPLETE.json'
        manifest = json.loads(path.read_text(encoding='utf-8'))
        if (manifest.get('schema') != 'pta-radius-reference-v1' or manifest.get('status') != 'complete'
                or manifest.get('policy') != POLICY or manifest.get('algorithm_version') != ALGORITHM_VERSION):
            raise ValueError('Field cache recipe differs')
        self.native_shape_tyx = _shape(manifest['native_shape_tyx'])
        self.low_shape_tyx = _shape(manifest['low_shape_tyx'])
        if native_shape_tyx is not None and _shape(native_shape_tyx) != self.native_shape_tyx:
            raise ValueError('Native dimensions differ from the source-bound cache')
        self.origin_tyx = tuple(int(n) for n in manifest['origin_tyx'])
        self.origin = np.array(self.origin_tyx, np.int64)
        self.native = np.array(self.native_shape_tyx, np.int64)
        self.low = np.array(self.low_shape_tyx, np.int64)
        self.empty = bool(manifest['empty_foreground'])
        self.phi = self.cap = self.source_mask = None
        self._planes = OrderedDict()
        self._plane_lock = threading.RLock()
        self.stats = {'native_voxels':0, 'mixed_interpolated_voxels':0,
                      'uniform_foreground_voxels':0, 'proven_background_voxels':0,
                      'outside_compact_cache_voxels':0, 'blocks':0}
        if manifest.get('padding_cells') != 3:
            raise ValueError('Three-cell complete-foreground padding is required')
        expected = set() if self.empty else {'mask','phi','cap_extended'}
        if set(manifest.get('fields',{})) != expected:
            raise ValueError('Coefficient inventory differs')
        loaded = []
        try:
            for name, dtype in [('mask',np.bool_),('phi',np.float32),('cap_extended',np.float32)] if not self.empty else []:
                item = manifest['fields'][name]
                if item.get('file') != name+'.npy':
                    raise ValueError('Unexpected coefficient filename')
                field = self.cache_dir/item['file']
                if field.is_symlink() or not field.is_file() or field.stat().st_size != item['bytes'] or _sha(field) != item['sha256']:
                    raise ValueError('Coefficient identity differs: '+name)
                value = np.load(field, mmap_mode='r', allow_pickle=False)
                loaded.append(value)
                if value.dtype != np.dtype(dtype) or list(value.shape) != manifest['shape_tyx']:
                    raise ValueError('Coefficient dtype/shape differs: '+name)
                if name == 'mask': self.source_mask = value
                elif name == 'phi': self.phi = value
                else: self.cap = value
        except BaseException:
            for value in loaded:
                value._mmap.close()
            raise
        self.provenance = {
            'policy':POLICY, 'algorithm_version':ALGORITHM_VERSION,
            'field_manifest':manifest, 'field_manifest_sha256':_sha(path),
            'cache_dir':str(self.cache_dir),
            'full_native_shape_tyx':list(self.native_shape_tyx),
            'outside_compact_reference': 'Exact zero-support sign: complete original foreground plus3zero cells contains every4tap stencil that can touch foreground. This is not invalid-query fallback or negative annotation authority.',
            'scalar_background': 'Outside compact coefficient support, only sign is defined: scalar=-1 sentinel and evaluated_scalar=False. No global distance-field magnitude claim.',
            'sampling': '(global native center index+.5)*low/native-.5-global coefficient origin',
            'formula': 'L=linear(phi)-linear(cap);C=B3(phi)-B3(cap);W=B3(cap)/.5;Q=(1-W)L+WC>=0;prefilterFalse',
            'guarantees': 'Full source-grid layout and declared fixed formula; no topology, coarse-center, coarse-background-lock or anatomical guarantees.'}

    def _plane_state(self, index):
        with self._plane_lock:
            if index in self._planes:
                self._planes.move_to_end(index)
                return self._planes[index]
            block = self.source_mask[index-1:index+3]
            any_fg = ndi.maximum_filter(np.any(block,axis=0),size=4,origin=-1,mode='constant',cval=0)
            all_fg = ndi.minimum_filter(np.all(block,axis=0),size=4,origin=-1,mode='constant',cval=0)
            caps = self.cap[index-1:index+3]
            cap_min = ndi.minimum_filter(np.min(caps,axis=0),size=4,origin=-1,mode='constant',cval=0)
            cap_max = ndi.maximum_filter(np.max(caps,axis=0),size=4,origin=-1,mode='constant',cval=0)
            if not np.isfinite(cap_min).all() or not np.isfinite(cap_max).all() or cap_min.min()<0 or cap_max.max()>.5:
                raise ValueError('Invalid cap coefficients')
            state = (any_fg, all_fg, cap_min, cap_max)
            self._planes[index] = state
            while len(self._planes)>2:
                self._planes.popitem(last=False)
            return state

    def close(self):
        self._planes.clear()
        for value in (self.phi,self.cap,self.source_mask):
            mmap = getattr(value,'_mmap',None)
            if mmap is not None and not mmap.closed:
                mmap.close()

def resolve_reference(reference_or_cache, native_shape_tyx=None):
    if isinstance(reference_or_cache,FieldReference):
        if native_shape_tyx is not None and _shape(native_shape_tyx)!=reference_or_cache.native_shape_tyx:
            raise ValueError('Native dimensions differ from the source-bound cache')
        return reference_or_cache
    if isinstance(reference_or_cache,dict):
        if 'cache_dir' not in reference_or_cache:
            raise ValueError('Returned reference manifest requires cache_dir')
        reference_or_cache=reference_or_cache['cache_dir']
    return FieldReference(reference_or_cache,native_shape_tyx)

def _sample(ref, native_coords, evaluate_scalar):
    coords=np.asarray(native_coords,np.float64)
    if coords.ndim<2 or coords.shape[0]!=3:
        raise ValueError('Expected native coordinates with shape(3,...)')
    shape=coords.shape[1:];flat=coords.reshape(3,-1);size=flat.shape[1]
    valid=np.isfinite(flat).all(axis=0)&np.all((flat>=0)&(flat<=ref.native[:,None]-1),axis=0)
    mask=np.zeros(size,bool);baseline=mask.copy();nearest=mask.copy();evaluated=mask.copy()
    scalar=np.full(size,np.nan,np.float64);linear=scalar.copy();cubic=scalar.copy();weight=np.full(size,np.nan,np.float32)
    scalar[valid]=-1;linear[valid]=-1;cubic[valid]=-1
    any_fg=np.zeros(size,bool);all_fg=np.zeros(size,bool);inside=np.zeros(size,bool)
    if not ref.empty and valid.any():
        q=(flat+.5)*(ref.low/ref.native)[:,None]-.5-ref.origin[:,None]
        inside=valid&np.all((q>=1)&(q<np.array(ref.phi.shape)[:,None]-2),axis=0)
        loc=np.flatnonzero(inside)
        cells=np.floor(q[:,loc]).astype(np.int64)
        cap_lo=np.zeros(size,np.float32);cap_hi=np.zeros(size,np.float32)
        for t in np.unique(cells[0]):
            take=cells[0]==t; positions=loc[take];y,x=cells[1,take],cells[2,take]
            a,f,cl,ch=ref._plane_state(int(t))
            any_fg[positions]=a[y,x];all_fg[positions]=f[y,x];cap_lo[positions]=cl[y,x];cap_hi[positions]=ch[y,x]
        ids=np.floor(q[:,loc]+.5).astype(np.int64)
        nearest[loc]=ref.source_mask[tuple(ids)]
        mask[all_fg]=True;baseline[all_fg]=True;scalar[all_fg]=linear[all_fg]=cubic[all_fg]=1
        need=inside if evaluate_scalar else (any_fg&~all_fg)
        ix=np.flatnonzero(need)
        if ix.size:
            points=q[:,ix]
            lo=np.floor(points.min(axis=1)).astype(int)-1
            hi=np.floor(points.max(axis=1)).astype(int)+3
            crop=tuple(slice(int(a),int(b)) for a,b in zip(lo,hi));local=points-lo[:,None]
            phi=np.asarray(ref.phi[crop]);cap=np.asarray(ref.cap[crop])
            if not np.isfinite(phi).all():raise ValueError('Nonfinite phi coefficient')
            # Dense diagnostic path is deliberately identical to the frozen sampler.
            # Binary path may use exact constant-cap shortcuts on each4tap footprint.
            if evaluate_scalar:
                groups=[(np.ones(len(ix),bool),None)]
            else:
                same=cap_lo[ix]==cap_hi[ix]
                groups=[(~same,None)]+[(same&(cap_lo[ix]==v),float(v)) for v in np.unique(cap_lo[ix][same])]
            for selector,constant in groups:
                if not selector.any():continue
                positions=ix[selector];query=local[:,selector]
                if constant==.5:
                    p1=ndi.map_coordinates(phi,query,order=1,prefilter=False,output=np.float32)
                    p3=ndi.map_coordinates(phi,query,order=3,prefilter=False,output=np.float32)
                    value=p3.astype(np.float64)-.5
                    lvalue=p1.astype(np.float64)-.5
                    cvalue=value;w=np.ones(value.shape)
                elif constant==0.:
                    p1=ndi.map_coordinates(phi,query,order=1,prefilter=False,output=np.float32)
                    value=p1.astype(np.float64);lvalue=value;cvalue=np.full(value.shape,np.nan);w=np.zeros(value.shape)
                else:
                    p1=ndi.map_coordinates(phi,query,order=1,prefilter=False,output=np.float32)
                    p3=ndi.map_coordinates(phi,query,order=3,prefilter=False,output=np.float32)
                    if constant is None:
                        c1=ndi.map_coordinates(cap,query,order=1,prefilter=False,output=np.float32)
                        c3=ndi.map_coordinates(cap,query,order=3,prefilter=False,output=np.float32)
                    else:c1=c3=np.full(p1.shape,constant,np.float32)
                    lvalue=p1.astype(np.float64)-c1.astype(np.float64);cvalue=p3.astype(np.float64)-c3.astype(np.float64)
                    w=np.clip(c3.astype(np.float64)/.5,0,1);value=(1-w)*lvalue+w*cvalue
                scalar[positions]=value;linear[positions]=lvalue;cubic[positions]=cvalue;weight[positions]=w
                mask[positions]=value>=0;baseline[positions]=lvalue>=0;evaluated[positions]=True
    background=valid&~any_fg
    return {'mask':mask.reshape(shape),'baseline':baseline.reshape(shape),
            'scalar':scalar.reshape(shape),'baseline_scalar':linear.reshape(shape),
            'cubic_scalar':cubic.reshape(shape),'weight':weight.reshape(shape),
            'nearest_cell':nearest.reshape(shape),'valid':valid.reshape(shape),
            'evaluated_scalar':evaluated.reshape(shape),'proven_background':background.reshape(shape),
            'uniform_foreground':all_fg.reshape(shape),'mixed':(any_fg&~all_fg).reshape(shape),
            'outside_compact_cache':(valid&~inside).reshape(shape)}

def sample_reference(reference_or_cache,native_coords,*,evaluate_scalar=True):
    ref=resolve_reference(reference_or_cache);own=ref is not reference_or_cache
    try:return _sample(ref,native_coords,bool(evaluate_scalar))
    finally:
        if own:ref.close()

def block_plan(native_shape_tyx,block_mib=128,workers=1):
    shape=_shape(native_shape_tyx)
    if isinstance(workers,bool) or not isinstance(workers,int) or workers<1:
        raise ValueError('workers must be a positive integer')
    if not math.isfinite(block_mib) or block_mib<=0:
        raise ValueError('block_mib must be positive and finite')
    per_worker=int(block_mib*MIB/workers)
    minimum=max(2*shape[2]*BASE_BYTES_PER_NATIVE_VOXEL,2*NATIVE_BYTES_PER_QUERY)
    if per_worker<minimum:
        raise ValueError(f'Aggregate native buffer budget requires at least {minimum*workers/MIB:.3f}MiB for one full-X row per worker')
    rows=min(shape[1],max(1,(per_worker//2)//(shape[2]*BASE_BYTES_PER_NATIVE_VOXEL)))
    mixed_queries=max(1,(per_worker//2)//NATIVE_BYTES_PER_QUERY)
    return {'aggregate_native_buffer_budget_mib':float(block_mib),'workers':workers,
            'rows_per_block':int(rows),'bytes_per_mixed_query_bound':NATIVE_BYTES_PER_QUERY,
            'base_bytes_per_native_voxel_bound':BASE_BYTES_PER_NATIVE_VOXEL,
            'mixed_queries_per_batch':int(mixed_queries),
            'scheduled_blocks':shape[0]*math.ceil(shape[1]/rows),
            'budget_scope':'Aggregate native coordinates/interpolation/results across workers; excludes immutable coarse coefficient mappings and bounded per-coarse-T classification maps. Field build has its own budget.'}

def iter_native_blocks(reference_or_cache,*,block_mib=128,workers=1):
    ref=resolve_reference(reference_or_cache);own=ref is not reference_or_cache
    plan=block_plan(ref.native_shape_tyx,block_mib,workers);ref.provenance['native_execution']=plan
    tsize,ysize,xsize=ref.native_shape_tyx;rows=plan['rows_per_block']
    x=np.arange(xsize,dtype=np.float64)
    qx=(x+.5)*(ref.low[2]/ref.native[2])-.5-ref.origin[2]
    inside_x=np.zeros(xsize,bool) if ref.empty else ((qx>=1)&(qx<ref.phi.shape[2]-2))
    valid_x=np.flatnonzero(inside_x);cell_x=np.floor(qx[inside_x]).astype(np.int64)
    def work(task):
        t,y,n=task
        output=np.zeros((n,xsize),np.uint8);total=output.size
        counts={'mixed_interpolated_voxels':0,'uniform_foreground_voxels':0,
                'proven_background_voxels':total,'outside_compact_cache_voxels':total}
        if ref.empty:return t,y,output,counts
        qt=(t+.5)*(ref.low[0]/ref.native[0])-.5-ref.origin[0]
        if qt<1 or qt>=ref.phi.shape[0]-2 or not valid_x.size:return t,y,output,counts
        ys=np.arange(y,y+n,dtype=np.float64)
        qy=(ys+.5)*(ref.low[1]/ref.native[1])-.5-ref.origin[1]
        valid_y=np.flatnonzero((qy>=1)&(qy<ref.phi.shape[1]-2))
        if not valid_y.size:return t,y,output,counts
        cell_y=np.floor(qy[valid_y]).astype(np.int64)
        any_fg,all_fg,_,_=ref._plane_state(int(math.floor(qt)))
        selection=np.ix_(cell_y,cell_x)
        any_grid=any_fg[selection];all_grid=all_fg[selection]
        mixed=any_grid&~all_grid
        output[np.ix_(valid_y,valid_x)]=all_grid
        row,col=np.nonzero(mixed)
        counts['mixed_interpolated_voxels']=int(len(row))
        counts['uniform_foreground_voxels']=int(all_grid.sum())
        counts['proven_background_voxels']=total-int(any_grid.sum())
        counts['outside_compact_cache_voxels']=total-len(valid_y)*len(valid_x)
        # Only mixed samples allocate floating coordinates/interpolation scratch.
        # Every global coordinate is still mapped ratio-first in _sample.
        batch=plan['mixed_queries_per_batch']
        for begin in range(0,len(row),batch):
            rr=valid_y[row[begin:begin+batch]];cc=valid_x[col[begin:begin+batch]]
            coords=np.empty((3,len(rr)),np.float64);coords[0]=t;coords[1]=y+rr;coords[2]=cc
            sampled=_sample(ref,coords,False)
            assert sampled['valid'].all() and sampled['mixed'].all()
            output[rr,cc]=sampled['mask']
        return t,y,output,counts
    tasks=((t,y,min(rows,ysize-y)) for t in range(tsize) for y in range(0,ysize,rows))
    def publish(result):
        t,y,block,counts=result;ref.stats['blocks']+=1;ref.stats['native_voxels']+=block.size
        for key,value in counts.items():ref.stats[key]+=value
        return t,y,block
    try:
        if workers==1:
            for task in tasks:yield publish(work(task))
        else:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                pending=[]
                for _ in range(workers):
                    task=next(tasks,None)
                    if task is not None:pending.append(executor.submit(work,task))
                while pending:
                    result=pending.pop(0).result()
                    task=next(tasks,None)
                    if task is not None:pending.append(executor.submit(work,task))
                    yield publish(result)
    finally:
        if own:ref.close()

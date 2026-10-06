"""Binary-authoritative SAM tail prefixes; no paired endpoint or tail cleanup."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from contextlib import nullcontext
from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from .sam_evidence import SamEvidenceBundle, fingerprint, _plain
from .sam_mask_reader import SamMaskReader

SCHEMA = 'xta.sam_extrapolation_selection/1'
_SOURCE = Path(__file__).resolve()
IMPLEMENTATION_SHA256 = hashlib.sha256(_SOURCE.read_bytes()).hexdigest()


def _purpose(bundle):
    if bundle.scope.get('evidence_purpose') != 'sam_extrapolation':
        raise ValueError('Extrapolation requires explicitly tagged one-seed tail evidence')
    if bundle.scope.get('source_stage') != 'post_interpolation':
        raise ValueError('SAM extrapolation requires its actual frozen post_interpolation baseline')


def _validate_run(reader, run):
    expected = tuple(map(int,run['expected_frames']))
    direction = 1 if run['direction']=='forward' else -1
    if not expected or expected != tuple(range(expected[0],expected[-1]+direction,direction)):
        raise ValueError('SAM extrapolation has malformed ordered frame coverage')
    if set(map(int,run['raw_mask_keys'])) != set(expected) or not run.get('complete'):
        raise RuntimeError('Missing SAM extrapolation observations are infrastructure failure')
    if len(run.get('seed_ids',()))!=1 or run.get('held_out_ids') or run.get('edge_ids'):
        raise ValueError('SAM extrapolation must have one frozen seed and no held-out endpoint')
    if tuple(run.get('injected_frames',())) != (expected[0],):
        raise ValueError('SAM extrapolation attempted an additional mask injection')
    group=reader.groups[run['group_id']]
    endpoints={endpoint['observation_id']:endpoint for endpoint in group['endpoints']}
    seed=endpoints.get(run['seed_ids'][0])
    terminal=endpoints.get(run['terminal_id'])
    if (seed is None or int(seed['frame_index'])!=expected[0] or terminal is None
            or int(terminal['frame_index'])!=int(run['terminal_frame'])):
        raise ValueError('SAM extrapolation seed/terminal identity differs from its actual native frame')
    if run.get('status') in {'failed','cancelled','infrastructure_invalid'}:
        raise RuntimeError('SAM extrapolation runtime infrastructure failure')
    for tile in run.get('tile_evidence',()):
        if tile.get('attempted') and (not tile.get('complete') or set(map(int,tile['raw_mask_keys']))!=set(expected)):
            raise RuntimeError('Missing SAM extrapolation tile observations are infrastructure failure')
    output = tuple(map(int,run['output_frames']))
    terminal = int(run['terminal_frame'])
    if (not output or output != tuple(range(terminal+direction,output[-1]+direction,direction))
            or not set(output).issubset(expected)):
        raise ValueError('SAM extrapolation distance must begin beyond the actual terminal')
    return expected, output


def select_sam_extrapolation(bundle, *, reader_cache_bytes=32*1024**2):
    """Stop only on the first raw SDK-empty mask or the declared horizon.

    A tiled hypothesis tests its full retained halos for emptiness, before
    ownership or observation subtraction. Unknown/unseeded owners cannot
    manufacture an empty prediction. Output still uses only owned-core pixels.
    Object scores, removal bookkeeping, radius, area, contact, crop borders and
    frame-to-frame overlap do not change this binary-mask stopping rule.
    """
    if not isinstance(bundle,SamEvidenceBundle):
        bundle=SamEvidenceBundle.open(bundle)
    _purpose(bundle)
    if not bundle.manifest.get('complete') or bundle.unfinalized_tile_runs:
        raise RuntimeError('Incomplete SAM extrapolation evidence is infrastructure failure')
    selected, frames_by_run, receipts = [], {}, {}
    with bundle.reader(max_cache_bytes=reader_cache_bytes) as reader:
        for rid,run in bundle.runs.items():
            expected, output = _validate_run(reader,run)
            group = bundle.groups[run['group_id']]
            if not group.get('complete',True):
                raise RuntimeError('Unresolved extrapolation geometry cannot own tracker predictions')
            tail = []
            stop_frame = None
            published = 0
            for frame in expected:
                raw = (reader.halo_union_mask(rid,frame) if run.get('generation_mode')=='tiled'
                       else reader.raw_mask(rid,frame))
                if not raw.any():
                    stop_frame = frame
                    break
                if frame not in output:
                    continue
                tail.append(frame)
                # This accounting never controls whether tracking continues.
                candidate = reader.raw_mask(rid,frame) & reader.group_mask(group['group_id'],f'write:{frame}')
                published += int(candidate.sum())
            frames_by_run[rid] = tail
            if published:
                selected.append(rid)
            receipts[rid] = dict(run_id=rid,terminal_id=run['terminal_id'],
                terminal_frame=int(run['terminal_frame']),seed_ids=list(run['seed_ids']),
                planned_frames=list(expected),planned_output_frames=list(output),
                effective_output_frames=tail,stop_reason='raw_empty' if stop_frame is not None else 'distance_limit',
                stop_frame=stop_frame,selected=bool(published),output_foreground=published,
                stopping_mask_domain='full_raw_tile_halo_union' if run.get('generation_mode')=='tiled' else 'raw_model_crop',
                mask_filters_applied=False,other_masks_authoritative=False)
    policy = dict(schema=SCHEMA,version=1,stop_on_raw_empty=True,stop_on_distance=True,
        source_stage='post_interpolation',terminal_radius_role='seed_admission_only',
        predicted_mask_filters='none',overlap_policy='continue_and_subtract_baseline',
        implementation_sha256=IMPLEMENTATION_SHA256)
    receipt = dict(schema=SCHEMA,evidence_purpose='sam_extrapolation',
        evidence_fingerprint=bundle.evidence_fingerprint,selected_run_ids=selected,
        selected_frames_by_run=frames_by_run,run_receipts=receipts,resolved_policy=policy,
        policy_hash=fingerprint(policy),source_stage='post_interpolation',
        generation_early_stop=False,selection_effective_prefix=True)
    receipt['selection_identity']=fingerprint(receipt)
    return receipt


def _validate_receipt(bundle,receipt):
    _purpose(bundle)
    if receipt.get('schema')!=SCHEMA or receipt.get('evidence_fingerprint')!=bundle.evidence_fingerprint:
        raise ValueError('SAM extrapolation receipt does not match its immutable evidence')
    unsigned={k:v for k,v in receipt.items() if k!='selection_identity'}
    if fingerprint(unsigned)!=receipt.get('selection_identity'):
        raise ValueError('SAM extrapolation selection receipt was modified')
    if not set(receipt['selected_run_ids']).issubset(bundle.runs):
        raise ValueError('SAM extrapolation selection has unknown run owners')


class _PublicationReuseLimit(Exception):
    """Use the existing streaming reconstruction when metadata cannot be reused."""


@dataclass(frozen=True,slots=True)
class _PublicationOwner:
    run_id: str
    group_id: str
    stored_frame: int
    native_bbox_yx: tuple
    mirror_u: bool


@dataclass(frozen=True,slots=True)
class _PublicationIndex:
    reader: object
    owners_by_frame: Mapping
    run_ids_by_direction: tuple
    roots_by_direction: tuple
    selection_identity: str
    policy_hash: str
    charged_metadata_bytes: int
    uses_cyclic: bool


def _publication_index(reader,receipt,*,max_metadata_bytes):
    """Authenticate one detached snapshot, then retain only native owner metadata.

    Credit bounds both the temporary receipt copy and the immutable index. This
    is a reuse limit: callers retain the validated streaming path when it is too
    small, rather than omitting any selected owner or rejecting valid evidence.
    """
    reader._require_active()
    charge=0
    def reserve(amount):
        nonlocal charge
        charge+=amount
        if charge>max_metadata_bytes:
            raise _PublicationReuseLimit
    def snapshot(value):
        if isinstance(value,Mapping):
            # Include dictionary slots and the temporary fingerprint expansion.
            reserve(256+256*len(value))
            return MappingProxyType({str(key):snapshot(item) for key,item in value.items()})
        if isinstance(value,(list,tuple)):
            reserve(128+32*len(value))
            return tuple(snapshot(item) for item in value)
        plain=_plain(value)
        reserve(128+4*len(plain) if isinstance(plain,str) else 128)
        return plain
    frozen=snapshot(receipt)
    _validate_receipt(reader,frozen)
    shape=tuple(reader.scope['shape_tyx'])
    owners={}
    run_ids=[[],[]]
    roots=[set(),set()]
    crops={}
    target_addresses={}
    uses_cyclic=False
    for rid in frozen['selected_run_ids']:
        run=reader.runs[rid]
        if run['direction'] not in ('forward','backward'):
            raise ValueError('SAM extrapolation selected owner has an invalid direction')
        direction=0 if run['direction']=='forward' else 1
        group=reader.groups[run['group_id']]
        reserve(1024)
        run_ids[direction].append(rid)
        roots[direction].add(group['terminal_id'])
        bbox=tuple(group['context_bbox_yx'])
        if not (0<=bbox[0]<bbox[2]<=shape[1] and 0<=bbox[1]<bbox[3]<=shape[2]):
            raise ValueError('SAM extrapolation selected crop is outside its native canvas')
        reserve(128+64*len(run['output_frames']))
        output=set(run['output_frames'])
        for stored in frozen['selected_frames_by_run'][rid]:
            reserve(512)
            if (isinstance(stored,(bool,np.bool_)) or not isinstance(stored,(int,np.integer))
                    or stored not in output or str(stored) not in run['raw_mask_keys']
                    or f'write:{stored}' not in group['mask_keys']):
                raise ValueError('SAM extrapolation selected frame differs from its declared output ownership')
            native=int(stored)
            mirror=False
            if group.get('frame_addressing'):
                from .sam_cyclic import address_for_unfolded_index,mirror_bbox_yx
                uses_cyclic=True
                address=group['frame_addresses'][str(stored)]
                native=int(address['native_index'])
                if native not in target_addresses:
                    target_addresses[native]=address_for_unfolded_index(native,shape[0])
                mirror=bool(address['mirror_u'])^bool(target_addresses[native]['mirror_u'])
            if not 0<=native<shape[0]:
                raise ValueError('SAM extrapolation selected frame is outside its native canvas')
            crop_key=(run['group_id'],mirror)
            if crop_key not in crops:
                crops[crop_key]=mirror_bbox_yx(bbox,shape[2]) if mirror else bbox
            if native not in owners:
                reserve(1024)
                owners[native]=[[],[]]
            owners[native][direction].append(_PublicationOwner(rid,run['group_id'],
                int(stored),crops[crop_key],mirror))
    return _PublicationIndex(reader,MappingProxyType({frame:(tuple(values[0]),tuple(values[1]))
        for frame,values in sorted(owners.items())}),tuple(tuple(values) for values in run_ids),
        tuple(tuple(sorted(values)) for values in roots),frozen['selection_identity'],
        frozen['policy_hash'],charge,uses_cyclic)


def _indexed_publication_crop(reader,owner):
    mask=reader.raw_mask(owner.run_id,owner.stored_frame)&reader.group_mask(
        owner.group_id,f'write:{owner.stored_frame}')
    if owner.mirror_u:
        mask=mask[:,::-1]
    y0,x0,y1,x1=owner.native_bbox_yx
    if mask.shape!=(y1-y0,x1-x0):
        raise ValueError('SAM extrapolation raw crop shape differs from its native owner')
    return owner.native_bbox_yx,mask


def _indexed_publication_plane(reader,index,frame,*,direction,shape_yx):
    """Reconstruct only this frame's owners inside their authenticated transaction."""
    reader._require_active()
    if index.reader is not reader:
        raise ValueError('SAM extrapolation owner index belongs to another reader transaction')
    plane=np.zeros(shape_yx,bool)
    for owner in index.owners_by_frame.get(frame,((),()))[direction]:
        (y0,x0,y1,x1),mask=_indexed_publication_crop(reader,owner)
        plane[y0:y1,x0:x1]|=mask
    return plane


def iter_selected_extrapolation_crops(bundle,receipt,*,frame=None,direction=None):
    """Yield (run_id, native_frame, native_bbox_yx, mask) from sparse raw owners."""
    if not isinstance(bundle,(SamEvidenceBundle,SamMaskReader)):
        bundle=SamEvidenceBundle.open(bundle)
    _validate_receipt(bundle,receipt)
    transaction=nullcontext(bundle) if isinstance(bundle,SamMaskReader) else bundle.reader()
    with transaction as reader:
        for rid in receipt['selected_run_ids']:
            run=bundle.runs[rid]
            sign=1 if run['direction']=='forward' else -1
            if direction is not None and sign!=direction:
                continue
            group=bundle.groups[run['group_id']]
            for original_frame in receipt['selected_frames_by_run'][rid]:
                native_frame=original_frame
                bbox=tuple(group['context_bbox_yx'])
                if group.get('frame_addressing'):
                    from .sam_cyclic import address_for_unfolded_index,transform_crop_between_frame_addresses
                    address=group['frame_addresses'][str(original_frame)]
                    native_frame=int(address['native_index'])
                if frame is not None and native_frame!=frame:
                    continue
                mask=reader.raw_mask(rid,original_frame) & reader.group_mask(group['group_id'],f'write:{original_frame}')
                if group.get('frame_addressing'):
                    target=address_for_unfolded_index(native_frame,bundle.scope['shape_tyx'][0])
                    mask,bbox=transform_crop_between_frame_addresses(mask,bbox,address,target,bundle.scope['shape_tyx'][2])
                yield rid,native_frame,bbox,mask


def selected_extrapolation_plane(bundle,receipt,frame,shape_yx=None,*,direction=None):
    if not isinstance(bundle,(SamEvidenceBundle,SamMaskReader)):
        bundle=SamEvidenceBundle.open(bundle)
    shape=tuple(shape_yx or bundle.scope['shape_tyx'][1:])
    plane=np.zeros(shape,bool)
    for _,_,bbox,mask in iter_selected_extrapolation_crops(bundle,receipt,frame=frame,direction=direction):
        y0,x0,y1,x1=bbox
        plane[y0:y1,x0:x1] |= mask
    return plane

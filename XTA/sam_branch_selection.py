"""Receipt-defined publication of qualified, endpoint-connected SAM branches.

Radius filtering remains the raw-mask interpreter. This independent extension
only removes candidate ownership: selected edges retain the connected component
that joins their original endpoints under their fixed local attachment contract.
Raw proposals, sibling evidence and the original write contracts remain unchanged.
"""
from __future__ import annotations

from collections.abc import Mapping
import base64
import hashlib
import json
from pathlib import Path
from types import MappingProxyType
import zlib

import numpy as np
from scipy import ndimage

from .sam_evidence import _freeze, _plain, fingerprint


SCHEMA = 'xta.sam_branch_selection/1'
_SOURCE_PATH = Path(__file__).resolve()
IMPLEMENTATION_SHA256 = hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest()
_MAX_EDGE_RECORDS = 100_000
_MAX_RUN_RECORDS = 200_000
_MAX_RECIPE_BYTES = 64*1024**2
BRANCH_WORKSPACE_BYTES_PER_VOXEL = 32
BRANCH_WORKSPACE_FIXED_BYTES = 2*1024**2
_QUALIFIED_PACKED_OWNER_IMPLEMENTATIONS = {
    # Phase-two prototype: same authenticated owner-plane encoding, with only
    # the original narrow observed-attachment contract at consumers.
    '660f2e4eb8bab05b7315419fd6b64a3af8011bb9dac9b7dab5b614f4e3b8d6ea': 'narrow_only',
    '37a9eeee35a90d6bca89de115714a9d08d1390086bce7d92b4fc01e24cbd4859': 'tracked_known',
    'fbc427e0ce3476fd6019f375325b6ad09a5b698d8581885094f6ae32638754d5': 'tracked_known',
    # Qualified v25.1.0 job151147 source archive; immutable-prefix reuse only
    # changes metadata ownership, not this packed owner/attachment contract.
    '254fbb676b0585baf9f0dfc9a680884a70270c3987177cafd4aa2fc511dabe42': 'tracked_known',
}


def branch_workspace_bytes(shape):
    """Conservative fresh numeric workspace; caches/spools require extra headroom."""
    dims = tuple(map(int, shape))
    if len(dims) != 3 or min(dims) <= 0:
        raise ValueError('SAM branch workspace requires a positive three-dimensional shape')
    return int(np.prod(dims))*BRANCH_WORKSPACE_BYTES_PER_VOXEL+BRANCH_WORKSPACE_FIXED_BYTES


def _positive_label_membership(labels, selected):
    """Bounded bool lookup avoids np.isin's masked int64 offset temporaries."""
    if not labels.size or not selected:
        return np.zeros(labels.shape, bool)
    maximum = int(labels.max())
    lookup = np.zeros(maximum+1, bool)
    for identifier in selected:
        if 0 < int(identifier) <= maximum:
            lookup[int(identifier)] = True
    return lookup[labels]


def _readonly(mask):
    return np.frombuffer(np.ascontiguousarray(mask, dtype=bool).tobytes(), dtype=bool).reshape(mask.shape)


def _assert_unchanged():
    if hashlib.sha256(_SOURCE_PATH.read_bytes()).hexdigest() != IMPLEMENTATION_SHA256:
        raise RuntimeError('SAM branch-selection implementation changed during selection/replay')


def _source_filter(value):
    """Remove publication restrictions when reconstructing their source owners."""
    if isinstance(value, Mapping) and 'mask_filter' in value:
        value = value['mask_filter']
    # Reader snapshots expose their radius spec without branch ownership.
    return getattr(value, 'spec', value)


def _owner_roles(value):
    names = ('direct_run_ids', 'source_partial_run_ids', 'target_partial_run_ids')
    roles = ({name: sorted(set(map(str, value.get(name, ())))) for name in names}
             if isinstance(value, Mapping) else {names[0]: sorted(set(map(str, value))), names[1]: [], names[2]: []})
    owners = [identifier for name in names for identifier in roles[name]]
    if not owners or len(owners) != len(set(owners)):
        raise ValueError('SAM edge owner roles must be nonempty and disjoint')
    return roles, tuple(sorted(owners))


def _pack_owner_support(support, frames):
    result = {}
    for frame, plane in zip(frames, support):
        foreground = int(np.count_nonzero(plane))
        if not foreground:
            continue
        packed = np.packbits(plane.reshape(-1), bitorder='little').tobytes()
        compressed = zlib.compress(packed, level=6)
        result[str(frame)] = dict(shape=list(plane.shape), foreground=foreground,
            packed_bytes=len(packed), sha256=hashlib.sha256(packed).hexdigest(),
            compressed_sha256=hashlib.sha256(compressed).hexdigest(),
            data=base64.b64encode(compressed).decode('ascii'))
    return result


def decode_owner_support_plane(record, *, max_plane_bytes=64*1024**2):
    """Decode one authenticated packed plane; receipt resources grant no allocation."""
    shape = tuple(map(int, record['shape']))
    if len(shape) != 2 or min(shape) <= 0 or int(np.prod(shape)) > int(max_plane_bytes):
        raise MemoryError('SAM branch support plane exceeds the current decoder budget')
    packed_bytes = (int(np.prod(shape))+7)//8
    if record['packed_bytes'] != packed_bytes:
        raise ValueError('SAM branch support packed shape differs')
    if len(record['data']) > (packed_bytes+packed_bytes//1000+1024)*4//3+8:
        raise ValueError('SAM branch compressed support exceeds its bounded plane size')
    compressed = base64.b64decode(record['data'], validate=True)
    if hashlib.sha256(compressed).hexdigest() != record['compressed_sha256']:
        raise ValueError('SAM branch support compressed checksum differs')
    decoder = zlib.decompressobj()
    packed = decoder.decompress(compressed, packed_bytes+1)
    if len(packed) != packed_bytes or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError('Malformed SAM branch support packed payload')
    if hashlib.sha256(packed).hexdigest() != record['sha256']:
        raise ValueError('SAM branch support packed checksum differs')
    plane = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=int(np.prod(shape)), bitorder='little').reshape(shape)
    if int(np.count_nonzero(plane)) != int(record['foreground']):
        raise ValueError('SAM branch support foreground count differs')
    return _readonly(plane)


def packet_identity(record):
    """Hash actual packed content, not merely its declared digest."""
    return fingerprint(_plain(record))


def candidate_branch_identity(recipe, run_id, frame):
    owners = [(edge_id, recipe['edges'][edge_id]['owner_support'][str(run_id)].get(str(int(frame))))
              for edge_id in recipe['selected_edge_ids_by_run'].get(str(run_id), ())]
    return fingerprint(dict(write_domain=recipe.get('write_domain', 'edge_write'), owners=_plain(owners)))


def _seed_rooted_raw(bundle, group, run_id, mask_filter, connectivity, max_group_bytes):
    from .sam_mask_reader import SamMaskReader
    run = bundle.runs[run_id]
    frames = tuple(sorted(map(int, run['expected_frames'])))
    if not frames or frames != tuple(range(frames[0], frames[-1]+1)):
        raise ValueError('SAM seed-rooted raw coverage must use contiguous native frame indices')
    y0, x0, y1, x1 = group['context_bbox_yx']
    if branch_workspace_bytes((len(run['expected_frames']), int(y1-y0), int(x1-x0))) > int(max_group_bytes):
        raise MemoryError('SAM seed-rooted owner exceeds its configured topology workspace')
    if isinstance(bundle, SamMaskReader):
        spec = _source_filter(mask_filter)
        identity = None if spec is None else spec.get('sha256')
        return bundle._product(('seed_rooted_raw', bundle.evidence_fingerprint, identity, run_id, int(connectivity)),
            lambda: _uncached_seed_rooted_raw(bundle, group, run_id, mask_filter, connectivity, max_group_bytes))
    return _uncached_seed_rooted_raw(bundle, group, run_id, mask_filter, connectivity, max_group_bytes)


def _label_foreground_crop(volume, structure):
    occupied = [np.flatnonzero(np.any(volume, axis=tuple(other for other in range(3) if other != axis)))
                for axis in range(3)]
    if not all(len(axis) for axis in occupied):
        return np.empty((0,0,0), np.int32), None
    lower = tuple(int(axis[0]) for axis in occupied)
    upper = tuple(int(axis[-1])+1 for axis in occupied)
    labels,_ = ndimage.label(volume[tuple(slice(a,b) for a,b in zip(lower,upper))], structure=structure)
    return labels, (lower,upper)


def _uncached_seed_rooted_raw(bundle, group, run_id, mask_filter, connectivity, max_group_bytes):
    """Actual tracker support connected to its injected original seed, never a target reference."""
    from .sam_mask_reader import effective_raw_mask
    run = bundle.runs[run_id]
    frames = tuple(sorted(map(int, run['expected_frames'])))
    y0, x0, y1, x1 = group['context_bbox_yx']
    shape = (len(frames), int(y1-y0), int(x1-x0))
    if branch_workspace_bytes(shape) > int(max_group_bytes):
        raise MemoryError('SAM seed-rooted owner exceeds its configured topology workspace')
    raw = np.zeros(shape, bool)
    for offset, frame in enumerate(frames):
        if str(frame) in run['raw_mask_keys']:
            raw[offset] = effective_raw_mask(bundle, run_id, frame, _source_filter(mask_filter))
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    labels, bounds = _label_foreground_crop(raw, structure)
    endpoints = {str(item['observation_id']): item for item in group['endpoints']}
    seed_labels = set()
    for seed_id in run.get('seed_ids', ()):
        seed = endpoints[str(seed_id)]
        frame = int(seed['frame_index'])
        if frame in frames and bounds is not None:
            reference = bundle.group_mask(group['group_id'], f'endpoint:{seed_id}')
            lower,upper = bounds
            offset = frames.index(frame)
            if lower[0] <= offset < upper[0]:
                seed_labels.update(set(map(int, np.unique(labels[offset-lower[0]][reference[lower[1]:upper[1],lower[2]:upper[2]]]))) - {0})
    rooted = np.zeros(shape, bool)
    if bounds is not None:
        lower,upper = bounds
        rooted[tuple(slice(a,b) for a,b in zip(lower,upper))] = _positive_label_membership(labels, seed_labels)
    return frames, _readonly(rooted)


def _original_plane(bundle, group, frame):
    y0, x0, y1, x1 = group['context_bbox_yx']
    original = np.zeros((y1-y0, x1-x0), bool)
    for name in ('known_foreground', 'unrelated'):
        key = f'{name}:{frame}'
        if key not in group['mask_keys']:
            raise ValueError('Expanded SAM branch requires complete original-detector contracts')
        original |= bundle.group_mask(group['group_id'], key)
    for endpoint in group['endpoints']:
        if int(endpoint['frame_index']) == int(frame):
            original |= bundle.group_mask(group['group_id'], f"endpoint:{endpoint['observation_id']}")
    return original


def _remember_owner_spool(spool, run_id, owned, frames):
    """Keep compact owner products only inside unused live topology credit."""
    if spool is None or spool['limit']-spool['bytes'] < 1024:
        return
    planes, charge = {}, 1024
    for offset, frame in enumerate(frames):
        if not owned[offset].any():
            continue
        packed = _pack_owner_support(owned[offset:offset+1], (frame,))[str(frame)]
        charge += len(json.dumps(packed, separators=(',', ':')).encode('utf-8'))*4+1024
        if spool['bytes']+charge > spool['limit']:
            return  # Overflow changes reuse, never qualification or pixels.
        planes[str(frame)] = packed
    spool['owners'][run_id] = planes
    spool['bytes'] += charge


def _prune_spooled_owner(planes, support, frames):
    result = {}
    for offset, frame in enumerate(frames):
        packed = planes.get(str(frame))
        if packed is None:
            continue
        owned = decode_owner_support_plane(packed)
        if not np.any(owned & ~support[offset]):
            result[str(frame)] = packed
        else:
            result.update(_pack_owner_support((owned & support[offset])[None], (frame,)))
    return result


def _owner_additions(bundle, group, edge_id, run_id, frames, mask_filter, connectivity, max_group_bytes, write_domain, *, rooted_source=None):
    owner_frames, rooted = (rooted_source if rooted_source is not None else
        _seed_rooted_raw(bundle, group, run_id, mask_filter, connectivity, max_group_bytes))
    index = {frame: offset for offset, frame in enumerate(owner_frames)}
    result = np.zeros((len(frames), *rooted.shape[1:]), bool)
    for offset, frame in enumerate(frames):
        if frame not in index:
            continue
        raw = rooted[index[frame]]
        if write_domain == 'edge_write':
            if str(frame) in bundle.runs[run_id]['candidate_mask_keys']:
                result[offset] = raw & bundle.candidate_mask(run_id, frame) & bundle.group_mask(group['group_id'], f'edge_write:{edge_id}:{frame}')
        elif frames[0] < frame < frames[-1]:
            result[offset] = raw & ~_original_plane(bundle, group, frame)
            if bundle.runs[run_id].get('generation_mode') == 'tiled':
                result[offset] &= bundle.availability_mask(run_id, frame)
    return result


def _edge_geometry(bundle, group_id, edge_id):
    group = bundle.groups[str(group_id)]
    edge = next((edge for edge in group.get('edges', ()) if str(edge['edge_id']) == str(edge_id)), None)
    if edge is None:
        raise ValueError('SAM branch selection references an unknown edge')
    endpoints = {str(item['observation_id']): item for item in group['endpoints']}
    source, target = endpoints[str(edge['source_id'])], endpoints[str(edge['target_id'])]
    lo, hi = sorted((int(source['frame_index']), int(target['frame_index'])))
    frames = tuple(range(lo, hi+1))
    if not set(frames).issubset(group['frame_indices']):
        raise ValueError('SAM branch interval is outside its fixed group')
    y0, x0, y1, x1 = map(int, group['context_bbox_yx'])
    return group, edge, source, target, frames, (len(frames), y1-y0, x1-x0)


def _connected_edge_path(bundle, group_id, edge_id, owner_roles, mask_filter, connectivity, max_group_bytes, write_domain='edge_write', crop_boundary_policy='reject', *, owner_spool=None):
    """Rebuild exact local path and diagnostics without consulting predictions as seeds."""
    group, edge, source, target, frames, shape = _edge_geometry(bundle, group_id, edge_id)
    if connectivity not in (6, 18, 26):
        raise ValueError('SAM branch connectivity must be 6, 18 or 26')
    # Boolean unions, an int32 label volume, EDT-independent work and returned
    # immutable support share this same bounded admission estimate as topology.
    if isinstance(max_group_bytes, bool) or int(max_group_bytes) <= 0 or branch_workspace_bytes(shape) > int(max_group_bytes):
        raise MemoryError('SAM connected branch exceeds its configured topology workspace')
    if write_domain not in ('edge_write', 'fixed_context'):
        raise ValueError('SAM branch write domain must be edge_write or fixed_context')
    if crop_boundary_policy not in ('reject', 'retain_censored'):
        raise ValueError('SAM branch crop boundary policy must be reject or retain_censored')
    owner_roles, run_ids = _owner_roles(owner_roles)
    for run_id in run_ids:
        run = bundle.runs[run_id]
        if str(run['group_id']) != str(group_id) or str(edge_id) not in run.get('edge_ids', ()):
            raise ValueError('SAM branch owner does not own the declared group/edge')
    for name, origin, opposite in (('source_partial_run_ids', str(edge['source_id']), str(edge['target_id'])),
                                   ('target_partial_run_ids', str(edge['target_id']), str(edge['source_id']))):
        for run_id in owner_roles[name]:
            run = bundle.runs[run_id]
            seeds, held = set(map(str, run.get('seed_ids', ()))), set(map(str, run.get('held_out_ids', ())))
            # Walk-back seeds are original observations outside the edge ends;
            # their uniquely held-out endpoint establishes the same ancestry.
            if opposite in seeds or (origin not in seeds and not (opposite in held and origin not in held)):
                raise ValueError('SAM paired prefix role differs from its actual original seed/held-out identities')
    additions = np.zeros(shape, bool)
    role_additions = {name: np.zeros(shape, bool) for name, owners in owner_roles.items() if owners}
    tracked_known = np.zeros(shape, bool) if write_domain == 'fixed_context' else None
    for name, identifiers in owner_roles.items():
        for run_id in identifiers:
            owner_frames, rooted = _seed_rooted_raw(bundle, group, run_id, mask_filter, connectivity, max_group_bytes)
            owned = _owner_additions(bundle, group, edge_id, run_id, frames, mask_filter,
                connectivity, max_group_bytes, write_domain, rooted_source=(owner_frames, rooted))
            role_additions[name] |= owned
            _remember_owner_spool(owner_spool, run_id, owned, frames)
            if write_domain == 'fixed_context':
                owner_index = {frame: offset for offset, frame in enumerate(owner_frames)}
                for offset, frame in enumerate(frames):
                    if frame in owner_index:
                        tracked_known[offset] |= rooted[owner_index[frame]] & bundle.group_mask(group_id, f'known_foreground:{frame}')
            del owned, rooted
        if name in role_additions:
            additions |= role_additions[name]
    local = np.zeros(shape, bool)
    for offset, frame in enumerate(frames):
        write_key = f'edge_write:{edge_id}:{frame}'
        contract_key = f'edge_contract:{edge_id}:{frame}'
        if write_key not in group['mask_keys'] or contract_key not in group['mask_keys']:
            raise ValueError('SAM connected branch requires its exact per-edge write/attachment contracts')
        write = bundle.group_mask(group_id, write_key)
        contract = bundle.group_mask(group_id, contract_key)
        local[offset] = additions[offset] & contract if write_domain == 'edge_write' else additions[offset]
        known_key = f'known_foreground:{frame}'
        if known_key in group['mask_keys']:
            local[offset] |= bundle.group_mask(group_id, known_key) & contract
        if write_domain == 'fixed_context':
            # An observed continuation outside the old silhouette corridor can
            # attach a branch only where its actual seed-rooted tracker saw it.
            local[offset] |= tracked_known[offset]
    source_mask = bundle.group_mask(group_id, f"endpoint:{source['observation_id']}")
    target_mask = bundle.group_mask(group_id, f"endpoint:{target['observation_id']}")
    source_offset, target_offset = int(source['frame_index'])-frames[0], int(target['frame_index'])-frames[0]
    local[source_offset] |= source_mask
    local[target_offset] |= target_mask
    structure = ndimage.generate_binary_structure(3, {6: 1, 18: 2, 26: 3}[connectivity])
    labels, foreground_bounds = _label_foreground_crop(local, structure)
    if foreground_bounds is None:
        lower = upper = (0,0,0)
        source_labels = target_labels = set()
    else:
        lower,upper = foreground_bounds
        def endpoint_labels(offset, mask):
            if not lower[0] <= offset < upper[0]:
                return set()
            return set(map(int, np.unique(labels[offset-lower[0]][mask[lower[1]:upper[1],lower[2]:upper[2]]]))) - {0}
        source_labels = endpoint_labels(source_offset, source_mask)
        target_labels = endpoint_labels(target_offset, target_mask)
    crop = tuple(slice(a,b) for a,b in zip(lower,upper))
    common = source_labels & target_labels
    contributors = {name: set(map(int, np.unique(labels[mask[crop]]))) - {0} for name, mask in role_additions.items()}
    meeting_labels, meeting_voxels = set(), 0
    if ('source_partial_run_ids' in role_additions and 'target_partial_run_ids' in role_additions
            and role_additions['source_partial_run_ids'][1:-1].any() and role_additions['target_partial_run_ids'][1:-1].any()):
        source_partial, target_partial = role_additions['source_partial_run_ids'].copy(), role_additions['target_partial_run_ids'].copy()
        source_partial[[0,-1]], target_partial[[0,-1]] = False, False
        meeting = ndimage.binary_dilation(source_partial, structure=structure) & target_partial
        meeting[[0,-1]] = False
        meeting_voxels = int(np.count_nonzero(meeting))
        meeting_labels = set(map(int, np.unique(labels[meeting[crop]]))) - {0}
    admitted = contributors.get('direct_run_ids', set()) | meeting_labels
    common = sorted(common & admitted)
    support = np.zeros(shape, bool)
    if foreground_bounds is not None:
        support[crop] = additions[crop] & _positive_label_membership(labels, common)
    bounds = [list(lower),list(upper)] if foreground_bounds is not None else None
    count = int(np.count_nonzero(support))
    crop_contacts = 0
    if write_domain == 'fixed_context':
        canvas = tuple(bundle.scope.get('shape_tyx', group.get('native_shape_tyx', ())))
        gy0, gx0, gy1, gx1 = map(int, group['context_bbox_yx'])
        internal = (gy0 > 0, gx0 > 0, len(canvas) != 3 or gy1 < canvas[1], len(canvas) != 3 or gx1 < canvas[2])
        for plane in support:
            crop_contacts += sum(int(np.count_nonzero(border)) for flag, border in zip(internal,
                (plane[0], plane[:, 0], plane[-1], plane[:, -1])) if flag)
    diagnostic = dict(edge_id=str(edge_id), group_id=str(group_id), source_id=str(edge['source_id']),
        target_id=str(edge['target_id']), eligible_run_ids=list(run_ids), owner_roles=owner_roles,
        write_domain=write_domain, local_native_interval=[frames[0], frames[-1]],
        supporting_component_labels=common, bounds_zyx=bounds,
        connected=bool(common) and count > 0 and (not crop_contacts or crop_boundary_policy=='retain_censored'),
        internal_crop_edge_pixels=crop_contacts,
        extent_censored=bool(crop_contacts),
        spatial_extent_status='context_censored' if crop_contacts else 'complete_within_fixed_context',
        partial_meeting_voxels=meeting_voxels,
        reasons=(['expanded_branch_extent_censored_by_context'] if crop_boundary_policy=='retain_censored'
                 else ['expanded_branch_requires_larger_crop']) if crop_contacts else [],
        local_addition_voxels=count, candidate_addition_voxels=int(np.count_nonzero(additions)),
        removed_disconnected_voxels=int(np.count_nonzero(additions))-count)
    if tracked_known is not None:
        tracked = np.zeros(shape, bool)
        if foreground_bounds is not None:
            tracked[crop] = tracked_known[crop] & _positive_label_membership(labels, common)
        diagnostic['_tracked_known_support'] = _readonly(tracked)
    else:
        diagnostic['_tracked_known_support'] = None
    return _readonly(support), diagnostic


def build_connected_edge_selection(bundle, mask_filter, group_id, eligible_by_edge, *, connectivity=6,
                                   max_group_bytes=256*1024**2, write_domain='edge_write', crop_boundary_policy='reject',
                                   owner_spool_bytes=32*1024**2):
    """Return a portable restriction recipe and diagnostics for eligible owners.

    Only connected edges enter the recipe. Directional prefixes may meet in its
    interior: neither owner is required here to reach the opposite endpoint.
    The policy supplies eligible owners after its own seed/quality predicates.
    """
    _assert_unchanged()
    from .sam_filtering import _spec
    spec = _spec(_source_filter(mask_filter))
    filter_identity = None if spec is None else spec['sha256']
    edges, by_run, diagnostics = {}, {}, {}
    if isinstance(owner_spool_bytes, bool) or not isinstance(owner_spool_bytes, int) or owner_spool_bytes < 0:
        raise ValueError('SAM owner spool capacity must be a nonnegative byte count')
    group = bundle.groups[str(group_id)]
    y0,x0,y1,x1 = group['context_bbox_yx']
    fresh_headroom = max(0, int(max_group_bytes)-branch_workspace_bytes((len(group['frame_indices']), int(y1-y0), int(x1-x0))))
    for edge_id, run_ids in sorted(eligible_by_edge.items()):
        owner_spool = dict(limit=min(int(owner_spool_bytes), fresh_headroom), bytes=0, owners={})
        support, diagnostic = _connected_edge_path(bundle, group_id, edge_id, run_ids, mask_filter,
            connectivity, max_group_bytes, write_domain, crop_boundary_policy, owner_spool=owner_spool)
        tracked_known_support = diagnostic.pop('_tracked_known_support')
        diagnostics[str(edge_id)] = diagnostic
        if not diagnostic['connected']:
            del support, tracked_known_support, owner_spool
            continue
        edges[str(edge_id)] = {key: value for key, value in diagnostic.items()
                              if key not in ('connected', 'candidate_addition_voxels', 'removed_disconnected_voxels')}
        group, _, _, _, frames, _ = _edge_geometry(bundle, group_id, edge_id)
        owners = {}
        for run_id in diagnostic['eligible_run_ids']:
            spooled = owner_spool['owners'].get(run_id)
            if spooled is not None:
                packed = _prune_spooled_owner(spooled, support, frames)
            else:
                owned = support & _owner_additions(bundle, group, edge_id, run_id, frames, mask_filter,
                    connectivity, max_group_bytes, write_domain)
                packed = _pack_owner_support(owned, frames)
            if packed:
                owners[run_id] = packed
        discarded = sorted(set(diagnostic['eligible_run_ids']) - set(owners))
        diagnostic['discarded_noncontributing_run_ids'] = discarded
        diagnostic['contributing_run_ids'] = sorted(owners)
        if not owners:
            edges.pop(str(edge_id))
            del support, tracked_known_support, owner_spool
            continue
        edges[str(edge_id)]['eligible_run_ids'] = sorted(owners)
        edges[str(edge_id)]['owner_roles'] = {name: [identifier for identifier in identifiers if identifier in owners]
                                              for name, identifiers in diagnostic['owner_roles'].items()}
        edges[str(edge_id)]['owner_support'] = owners
        edges[str(edge_id)]['tracked_known_support'] = (_pack_owner_support(tracked_known_support, frames)
                                                      if tracked_known_support is not None else {})
        for run_id in sorted(owners):
            by_run.setdefault(run_id, []).append(str(edge_id))
        # Release this edge's dense products before rooting the next edge.
        del support, tracked_known_support, owner_spool
    recipe = dict(schema=SCHEMA, mode='qualified_connected_edges', evidence_fingerprint=bundle.evidence_fingerprint,
        implementation_sha256=IMPLEMENTATION_SHA256, mask_filter_sha256=filter_identity,
        connectivity=int(connectivity), max_group_bytes=int(max_group_bytes), edges=edges,
        write_domain=write_domain,
        crop_boundary_policy=crop_boundary_policy,
        selected_edge_ids_by_run={key: sorted(value) for key, value in sorted(by_run.items())})
    recipe['sha256'] = fingerprint(recipe)
    if len(json.dumps(recipe, separators=(',', ':')).encode('utf-8')) > _MAX_RECIPE_BYTES:
        raise MemoryError('SAM branch selection exceeds its bounded packed metadata budget')
    return recipe, diagnostics


def merge_connected_edge_selections(recipes):
    """Merge already qualified groups without changing their owner/path decisions."""
    recipes = list(recipes)
    if not recipes:
        raise ValueError('SAM branch merge requires a base recipe')
    base = {key: _plain(value) for key, value in recipes[0].items()
            if key not in ('edges', 'selected_edge_ids_by_run', 'sha256')}
    edges, by_run = {}, {}
    for recipe in recipes:
        if {key: _plain(value) for key, value in recipe.items()
                if key not in ('edges', 'selected_edge_ids_by_run', 'sha256')} != base:
            raise ValueError('SAM branch groups have inconsistent evidence/filter/geometry identities')
        for edge_id, record in recipe['edges'].items():
            if edge_id in edges:
                raise ValueError('SAM branch merge duplicates an edge')
            edges[str(edge_id)] = _plain(record)
        for run_id, selected in recipe['selected_edge_ids_by_run'].items():
            by_run.setdefault(str(run_id), set()).update(map(str, selected))
    result = dict(base, edges=edges, selected_edge_ids_by_run={key: sorted(value) for key, value in sorted(by_run.items())})
    result['sha256'] = fingerprint(result)
    if len(json.dumps(result, separators=(',', ':')).encode('utf-8')) > _MAX_RECIPE_BYTES:
        raise MemoryError('SAM branch selection exceeds its bounded packed metadata budget')
    return result


def merge_branch_selections(bundle, mask_filter, recipes, *, connectivity=6,
                            max_group_bytes=256*1024**2, write_domain='edge_write', crop_boundary_policy='reject'):
    """Seal a full-scope extension, including an explicitly empty selection."""
    recipes = list(recipes)
    if recipes:
        return merge_connected_edge_selections(recipes)
    from .sam_filtering import _spec
    radius = _spec(_source_filter(mask_filter))
    result = dict(schema=SCHEMA, mode='qualified_connected_edges', evidence_fingerprint=bundle.evidence_fingerprint,
        implementation_sha256=IMPLEMENTATION_SHA256, mask_filter_sha256=None if radius is None else radius['sha256'],
        connectivity=int(connectivity), max_group_bytes=int(max_group_bytes), write_domain=write_domain,
        crop_boundary_policy=crop_boundary_policy,
        edges={}, selected_edge_ids_by_run={})
    result['sha256'] = fingerprint(result)
    return result


def _json_size(value):
    return len(json.dumps(_plain(value), separators=(',', ':')).encode('utf-8'))


def _branch_index_bytes(edge_count, owner_count, owner_edges):
    """Conservative shallow-index charge; packed records are shared, not cached."""
    return 1024 + 256*(int(edge_count)+int(owner_count)) + 16*int(owner_edges)


class _BranchPrefix(Mapping):
    """Reader-only immutable indexes over fully validated packed records.

    This is deliberately not a portable receipt: it has no whole-prefix SHA.
    Public receipt validation therefore cannot mistake it for sealed evidence.
    Finalization uses the ordinary merge, fingerprint and complete validator.
    """

    __slots__ = ('_header', '_edges', '_by_run', '_edge_bytes', '_owner_bytes',
                 '_owner_edges', 'serialized_bytes', 'index_bytes')

    def __init__(self, header, edges, by_run, edge_bytes, owner_bytes, owner_edges):
        object.__setattr__(self, '_header', header)
        object.__setattr__(self, '_edges', MappingProxyType(edges))
        object.__setattr__(self, '_by_run', MappingProxyType(by_run))
        object.__setattr__(self, '_edge_bytes', int(edge_bytes))
        object.__setattr__(self, '_owner_bytes', int(owner_bytes))
        object.__setattr__(self, '_owner_edges', int(owner_edges))
        # JSON object order cannot affect its byte count. Include the exact
        # eventual 64-character SHA field and the commas between map entries.
        empty = dict(header, edges={}, selected_edge_ids_by_run={}, sha256='0'*64)
        size = (_json_size(empty) + edge_bytes + owner_bytes
                + max(0, len(edges)-1) + max(0, len(by_run)-1))
        if size > _MAX_RECIPE_BYTES:
            raise MemoryError('SAM branch selection exceeds its bounded packed metadata budget')
        if len(edges) > _MAX_EDGE_RECORDS or len(by_run) > _MAX_RUN_RECORDS:
            raise ValueError('SAM branch selection exceeds its bounded edge/owner inventory')
        object.__setattr__(self, 'serialized_bytes', size)
        object.__setattr__(self, 'index_bytes', _branch_index_bytes(len(edges), len(by_run), owner_edges))

    def __setattr__(self, name, value):
        raise TypeError('SAM branch prefixes are immutable')

    def __delattr__(self, name):
        raise TypeError('SAM branch prefixes are immutable')

    def __getitem__(self, key):
        if key == 'edges':
            return self._edges
        if key == 'selected_edge_ids_by_run':
            return self._by_run
        return self._header[key]

    def __iter__(self):
        yield from self._header
        yield 'edges'
        yield 'selected_edge_ids_by_run'

    def __len__(self):
        return len(self._header)+2


def _validated_branch_overlay(prefix, incoming, *, max_index_bytes):
    """Join immutable validated chunks, or request the original bounded path.

    Only the reader calls this after validating the incoming public recipe and
    authenticating the prefix snapshot's transaction. Never use a receipt flag
    to bypass validation. The allowance covers simultaneously live old/new and
    combined indexes; the ordinary full merge remains the low-credit fallback.
    """
    recipes = [recipe for recipe in (prefix, incoming) if recipe is not None]
    edge_count = sum(len(recipe['edges']) for recipe in recipes)
    owner_count = sum(len(recipe['selected_edge_ids_by_run']) for recipe in recipes)
    owner_edges = sum(recipe._owner_edges if isinstance(recipe, _BranchPrefix) else
        sum(len(ids) for ids in recipe['selected_edge_ids_by_run'].values()) for recipe in recipes)
    if 3*_branch_index_bytes(edge_count, owner_count, owner_edges) > int(max_index_bytes):
        return None
    base = {key: value for key, value in incoming.items()
            if key not in ('edges', 'selected_edge_ids_by_run', 'sha256')}
    if prefix is not None and {key: value for key, value in prefix.items()
            if key not in ('edges', 'selected_edge_ids_by_run', 'sha256')} != base:
        raise ValueError('SAM branch groups have inconsistent evidence/filter/geometry identities')
    if prefix is None:
        edges, by_run, edge_bytes, owner_bytes = {}, {}, 0, 0
    else:
        edges, by_run = dict(prefix['edges']), dict(prefix['selected_edge_ids_by_run'])
        edge_bytes = (prefix._edge_bytes if isinstance(prefix, _BranchPrefix) else
            _json_size(edges)-2-max(0, len(edges)-1))
        owner_bytes = (prefix._owner_bytes if isinstance(prefix, _BranchPrefix) else
            _json_size(by_run)-2-max(0, len(by_run)-1))
    for edge_id, record in incoming['edges'].items():
        if edge_id in edges:
            raise ValueError('SAM branch merge duplicates an edge')
        edges[edge_id] = record
    incoming_edges = incoming['edges']
    edge_bytes += _json_size(incoming_edges)-2-max(0, len(incoming_edges)-1)
    for run_id, ids in incoming['selected_edge_ids_by_run'].items():
        if run_id in by_run:
            owner_bytes -= _json_size({run_id: by_run[run_id]})-2
            merged = tuple(sorted(set(by_run[run_id]).union(ids)))
            owner_edges -= len(by_run[run_id])+len(ids)-len(merged)
        else:
            merged = ids
        by_run[run_id] = merged
        owner_bytes += _json_size({run_id: merged})-2
    return _BranchPrefix(MappingProxyType(base), edges, by_run,
        edge_bytes, owner_bytes, owner_edges)


def branch_selection_from_value(value):
    """Retrieve the extension from a receipt or an immutable reader snapshot."""
    if isinstance(value, Mapping) and 'branch_selection' in value:
        return value['branch_selection']
    if isinstance(value, Mapping) and 'mask_filter' in value:
        snapshot = value['mask_filter']
        return getattr(snapshot, 'branch_selection', None)
    return getattr(value, 'branch_selection', None)


def validate_branch_selection(value, bundle, *, mask_filter_sha256=None):
    """Validate bounded, complete owner recipes without decoding a mask."""
    _assert_unchanged()
    recipe = branch_selection_from_value(value)
    required = isinstance(value, Mapping) and value.get('resolved_policy', {}).get('version') in (6, 7)
    if recipe is None:
        if required:
            raise ValueError('SAM branch-aware support requires its retained branch selection recipe')
        return None
    if not isinstance(recipe, Mapping) or recipe.get('schema') != SCHEMA or recipe.get('mode') != 'qualified_connected_edges':
        raise ValueError('Unsupported SAM branch selection schema/mode')
    if recipe.get('sha256') != fingerprint({key: _plain(item) for key, item in recipe.items() if key != 'sha256'}):
        raise ValueError('SAM branch selection changed or its fingerprint is invalid')
    if len(json.dumps(_plain(recipe), separators=(',', ':')).encode('utf-8')) > _MAX_RECIPE_BYTES:
        raise MemoryError('SAM branch selection exceeds its bounded packed metadata budget')
    historical = recipe.get('implementation_sha256') in _QUALIFIED_PACKED_OWNER_IMPLEMENTATIONS
    if (recipe.get('implementation_sha256') != IMPLEMENTATION_SHA256 and not historical or recipe.get('evidence_fingerprint') != bundle.evidence_fingerprint
            or recipe.get('mask_filter_sha256') != mask_filter_sha256):
        raise ValueError('SAM branch selection implementation, evidence or radius filter differs')
    if recipe.get('connectivity') not in (6, 18, 26) or isinstance(recipe.get('max_group_bytes'), bool) or not isinstance(recipe.get('max_group_bytes'), int) or recipe['max_group_bytes'] <= 0:
        raise ValueError('SAM branch selection geometry/resource bounds are malformed')
    if recipe.get('crop_boundary_policy', 'reject') not in ('reject','retain_censored'):
        raise ValueError('SAM branch crop boundary policy is malformed')
    edges, by_run = recipe.get('edges'), recipe.get('selected_edge_ids_by_run')
    if not isinstance(edges, Mapping) or not isinstance(by_run, Mapping) or len(edges) > _MAX_EDGE_RECORDS or len(by_run) > _MAX_RUN_RECORDS:
        raise ValueError('SAM branch selection exceeds its bounded edge/owner inventory')
    expected = {}
    for edge_id, record in edges.items():
        group_id = str(record['group_id'])
        _, edge, source, target, frames, shape = _edge_geometry(bundle, group_id, str(edge_id))
        if (record.get('source_id') != edge['source_id'] or record.get('target_id') != edge['target_id']
                or list(record.get('local_native_interval', ())) != [frames[0], frames[-1]]):
            raise ValueError('SAM branch selection endpoint/frame addresses differ')
        _, owners = _owner_roles(record.get('owner_roles', record.get('eligible_run_ids', ())))
        if list(owners) != list(record.get('eligible_run_ids', ())):
            raise ValueError('SAM branch selection owner role inventory differs')
        if not owners or len(owners) != len(set(owners)) or len(owners) > _MAX_RUN_RECORDS:
            raise ValueError('SAM branch selection needs unique eligible owners')
        for run_id in owners:
            if str(run_id) not in bundle.runs or bundle.runs[str(run_id)]['group_id'] != group_id or str(edge_id) not in bundle.runs[str(run_id)].get('edge_ids', ()):
                raise ValueError('SAM branch selection references an ineligible owner')
            expected.setdefault(str(run_id), set()).add(str(edge_id))
        if set(record.get('owner_support', {})) != set(owners):
            raise ValueError('SAM packed branch paths differ from their eligible owners')
        for run_id, planes in record['owner_support'].items():
            if not isinstance(planes, Mapping) or len(planes) > len(frames):
                raise ValueError('SAM packed branch frame inventory exceeds its interval')
            for frame, plane in planes.items():
                if int(frame) not in frames or tuple(plane.get('shape', ())) != shape[1:]:
                    raise ValueError('SAM packed branch frame/shape differs from its fixed crop')
        if historical and _QUALIFIED_PACKED_OWNER_IMPLEMENTATIONS[recipe['implementation_sha256']]=='narrow_only' and 'tracked_known_support' in record:
            raise ValueError('Historical SAM branch receipt cannot invent a newer observed-attachment proof')
        if (not historical or _QUALIFIED_PACKED_OWNER_IMPLEMENTATIONS[recipe['implementation_sha256']]=='tracked_known') and 'tracked_known_support' not in record:
            raise ValueError('SAM branch receipt lacks its tracked observed-attachment proof')
        known_planes = record.get('tracked_known_support', {})
        if not isinstance(known_planes, Mapping) or len(known_planes) > len(frames):
            raise ValueError('SAM tracked attachment frame inventory exceeds its interval')
        for frame, plane in known_planes.items():
            if int(frame) not in frames or tuple(plane.get('shape', ())) != shape[1:]:
                raise ValueError('SAM tracked attachment frame/shape differs from its fixed crop')
        if not record.get('supporting_component_labels') or any(isinstance(label, bool) or not isinstance(label, int) or label <= 0 for label in record['supporting_component_labels']):
            raise ValueError('SAM branch selection needs connected positive component labels')
    actual = {str(key): set(map(str, identifiers)) for key, identifiers in by_run.items()}
    if expected != actual or any(len(identifiers) != len(set(identifiers)) for identifiers in by_run.values()):
        raise ValueError('SAM branch selection owner-to-edge mapping differs from connected paths')
    if isinstance(value, Mapping) and 'selected_run_ids' in value and set(map(str, value['selected_run_ids'])) - set(actual):
        raise ValueError('SAM branch selection differs from its selected contributors')
    return _freeze(_plain(recipe))


def connected_edge_path_from_recipe(bundle, recipe, edge_id, mask_filter, *, max_group_bytes=256*1024**2):
    """Reconstruct and verify one saved path recipe for publication consumers."""
    record = recipe['edges'][str(edge_id)]
    support, observed = _connected_edge_path(bundle, record['group_id'], str(edge_id), record.get('owner_roles', record['eligible_run_ids']),
        mask_filter, int(recipe['connectivity']), int(max_group_bytes), recipe.get('write_domain', 'edge_write'),
        recipe.get('crop_boundary_policy', 'reject'))
    for key in ('supporting_component_labels', 'bounds_zyx', 'local_addition_voxels', 'local_native_interval'):
        if _plain(record.get(key)) != _plain(observed[key]):
            raise ValueError('SAM connected branch support differs from its retained component recipe')
    if not observed['connected']:
        raise ValueError('SAM selected branch no longer joins its original endpoints')
    return support


def branch_attachment_mask(bundle, group_id, edge_id, frame, value):
    """Fixed local observations plus the selected model-supported known proof.

    This is attachment evidence only. It never becomes an output addition.
    A final-survival consumer must additionally intersect it with actual final
    foreground, so removed observed pixels cannot be restored by this accessor.
    """
    group = bundle.groups[str(group_id)]
    known = bundle.group_mask(group_id, f'known_foreground:{int(frame)}')
    contract = bundle.group_mask(group_id, f'edge_contract:{edge_id}:{int(frame)}')
    result = known & contract
    from .sam_filtering import _spec
    radius = _spec(_source_filter(value))
    if hasattr(bundle, '_branch_selection'):
        recipe = bundle._branch_selection(value, radius)
    else:
        recipe = validate_branch_selection(value, bundle, mask_filter_sha256=None if radius is None else radius['sha256'])
    if recipe is not None and recipe.get('write_domain') == 'fixed_context' and str(edge_id) in recipe['edges']:
        packed = recipe['edges'][str(edge_id)].get('tracked_known_support', {}).get(str(int(frame)))
        if packed is not None:
            def decode():
                return decode_owner_support_plane(packed, max_plane_bytes=bundle.max_mask_bytes)
            proof = (bundle._product(('branch_known', bundle.evidence_fingerprint, str(group_id), str(edge_id), int(frame), packet_identity(packed)), decode)
                     if hasattr(bundle, '_product') else decode())
            if np.any(proof & ~known):
                raise ValueError('SAM tracked attachment proof invents original detector pixels')
            result |= proof
    return _readonly(result)


def apply_edge_selection_to_candidate(bundle, run_id, frame, candidate, value, *, recipe=None, path_reader=None):
    """Retain this owner's pixels only on its selected connected edge paths."""
    if recipe is None:
        from .sam_filtering import _spec
        radius = _spec(_source_filter(value))
        recipe = validate_branch_selection(value, bundle, mask_filter_sha256=None if radius is None else radius['sha256'])
    if recipe is None:
        return candidate
    permitted = np.zeros(candidate.shape, bool)
    for edge_id in recipe['selected_edge_ids_by_run'].get(str(run_id), ()):
        record = recipe['edges'][edge_id]
        lo, hi = record['local_native_interval']
        if not lo <= int(frame) <= hi:
            continue
        packed = record['owner_support'][str(run_id)].get(str(int(frame)))
        if packed is None:
            continue
        support = decode_owner_support_plane(packed) if path_reader is None else path_reader(edge_id, str(run_id), int(frame))
        if recipe.get('write_domain', 'edge_write') == 'edge_write':
            permitted |= support & bundle.group_mask(record['group_id'], f'edge_write:{edge_id}:{int(frame)}')
        elif lo < int(frame) < hi:
            permitted |= support & ~_original_plane(bundle, bundle.groups[record['group_id']], int(frame))
    return _readonly(candidate & permitted)

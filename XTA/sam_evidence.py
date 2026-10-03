"""Portable, indexed SAM proposal evidence with exact overlapping mask ownership.

The three-file bundle is deliberately separate from binary NRRD layer readers.
Each group mask and each run/frame mask is bit-packed and independently deflated;
selection always reconstructs accepted contributors, never subtracts from a union.
No SAM or detector runtime is imported here.
"""
from __future__ import annotations

from collections.abc import Mapping
import hashlib
import json
import os
import operator
from pathlib import Path
import shutil
from types import MappingProxyType
import uuid
import zlib

import numpy as np

SCHEMA = "xta.sam_proposals/1"
TILE_EVIDENCE_SCHEMA = "xta.sam_run_tiles/1"


def _plain(value):
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_plain(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        raise TypeError("Dense SAM masks belong in indexed payloads, not proposal metadata")
    if isinstance(value, Path):
        return str(value)
    return value


def _freeze(value):
    if isinstance(value, dict):
        return MappingProxyType({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    return value


def _json_bytes(value):
    return json.dumps(_plain(value), sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")


def fingerprint(value):
    """Stable fingerprint of a JSON-compatible generation or policy snapshot."""
    return hashlib.sha256(_json_bytes(value)).hexdigest()


def _file_hash(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def _mask(value, shape=None):
    result = np.asarray(value)
    if result.ndim != 2 or any(v <= 0 for v in result.shape):
        raise ValueError("SAM evidence masks require a nonempty two-dimensional canvas")
    if shape is not None and result.shape != tuple(shape):
        raise ValueError("SAM evidence mask shape disagrees with its fixed context crop")
    if not np.isin(result, (0, 1)).all():
        raise ValueError("SAM evidence masks must be binary")
    return np.asarray(result, dtype=np.bool_)


def _group_shape(group):
    y0, x0, y1, x1 = map(int, group["context_bbox_yx"])
    if y0 < 0 or x0 < 0 or y1 <= y0 or x1 <= x0:
        raise ValueError("Invalid SAM context crop")
    return y1 - y0, x1 - x0


def _frame_integer(value, name):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be an integer')
    try:
        return operator.index(value)
    except TypeError as error:
        raise ValueError(f'{name} must be an integer') from error


def _shape_tyx(value, name):
    shape = tuple(_frame_integer(item, name) for item in value)
    if len(shape) != 3 or any(item < 1 for item in shape):
        raise ValueError(f'{name} must contain three positive dimensions')
    return shape


def _scope_frame_geometry(scope):
    addressing = scope.get('frame_addressing')
    shape = (_shape_tyx(scope['shape_tyx'], 'SAM native shape') if 'shape_tyx' in scope else None)
    if not addressing:
        if 'evidence_shape_tyx' in scope and tuple(scope['evidence_shape_tyx']) != shape:
            raise ValueError('Extended SAM frames require an explicit cyclic closure')
        return dict(cyclic=False, native_shape_tyx=shape, evidence_shape_tyx=shape, addressing=None)
    from .sam_cyclic import validate_cyclic_frame_addressing, IMPLEMENTATION_SHA256
    validate_cyclic_frame_addressing(addressing)
    if scope.get('cyclic_implementation_sha256') not in (None, IMPLEMENTATION_SHA256):
        raise ValueError('Cyclic SAM helper implementation identity differs from saved evidence')
    native = _shape_tyx(addressing['native_shape_tyx'], 'SAM cyclic native shape')
    evidence = _shape_tyx(addressing['evidence_shape_tyx'], 'SAM cyclic evidence shape')
    if shape != native or ('evidence_shape_tyx' in scope and tuple(scope['evidence_shape_tyx']) != evidence):
        raise ValueError('Cyclic SAM scope shape differs from its native/evidence closure')
    return dict(cyclic=True, native_shape_tyx=native, evidence_shape_tyx=evidence,
                addressing=addressing, cyclic_implementation_sha256=IMPLEMENTATION_SHA256)


def _group_frame_geometry(group, scope_geometry):
    addressing = group.get('frame_addressing')
    if not scope_geometry['cyclic']:
        if addressing or group.get('frame_addresses'):
            raise ValueError('Cyclic SAM group requires its scope native frame closure')
        return dict(addresses=None, bbox_yx=tuple(group['context_bbox_yx']))
    if not addressing:
        raise ValueError('Cyclic SAM group is missing its declared frame addresses')
    from .sam_cyclic import validate_cyclic_frame_addressing
    frames = tuple(_frame_integer(frame, 'SAM unfolded frame') for frame in group['frame_indices'])
    if not frames or frames != tuple(range(frames[0], frames[-1] + 1)):
        raise ValueError('Cyclic SAM group frames must be distinct increasing contiguous indices')
    addresses = validate_cyclic_frame_addressing(addressing, expected_frames=frames)
    recipe = scope_geometry['addressing']
    for key in ('schema', 'native_shape_tyx', 'evidence_shape_tyx', 'alias_frames', 'period_degrees'):
        left, right = addressing.get(key), recipe.get(key)
        if isinstance(left, (list, tuple)):
            left, right = tuple(left), tuple(right)
        if left != right:
            raise ValueError('Cyclic SAM group and scope frame closure disagree')
    duplicate = group.get('frame_addresses')
    if duplicate is None:
        raise ValueError('Cyclic SAM group is missing its explicit native frame-address map')
    duplicated = validate_cyclic_frame_addressing({**dict(recipe), 'addresses': duplicate}, expected_frames=frames)
    if any(dict(addresses[frame]) != dict(duplicated[frame]) for frame in frames):
        raise ValueError('Cyclic SAM duplicated native addresses disagree')
    native = scope_geometry['native_shape_tyx']
    if group.get('native_shape_tyx') is not None and tuple(group['native_shape_tyx']) != native:
        raise ValueError('Cyclic SAM group native shape differs from its scope')
    bbox = tuple(_frame_integer(value, 'SAM cyclic crop bound') for value in group['context_bbox_yx'])
    if len(bbox) != 4 or not (0 <= bbox[0] < bbox[2] <= native[1] and 0 <= bbox[1] < bbox[3] <= native[2]):
        raise ValueError('Cyclic SAM crop is outside its native working canvas')
    for endpoint in group.get('endpoints', ()):
        frame = _frame_integer(endpoint['frame_index'], 'SAM endpoint unfolded frame')
        if frame not in addresses:
            raise ValueError('Cyclic SAM endpoint lacks its declared frame address')
        address = addresses[frame]
        if (not isinstance(endpoint.get('original_observation_id'), str) or not endpoint['original_observation_id']
                or _frame_integer(endpoint.get('native_frame_index'), 'SAM endpoint native frame') != address['native_index']
                or type(endpoint.get('mirror_u')) is not bool or endpoint['mirror_u'] != address['mirror_u']):
            raise ValueError('Cyclic SAM original endpoint identity or mirror address is corrupted')
        lineage = endpoint.get('lineage', {})
        for key, expected in (('native_frame_index', address['native_index']),
                              ('unfolded_frame_index', frame), ('cycle_index', address['cycle_index']),
                              ('mirror_u', address['mirror_u']),
                              ('original_observation_id', endpoint['original_observation_id'])):
            if key in lineage and (type(lineage[key]) is not type(expected) or lineage[key] != expected):
                raise ValueError('Cyclic SAM endpoint lineage disagrees with its native closure')
    return dict(addresses=addresses, bbox_yx=bbox)


def _evidence_frame_geometry(scope, groups, runs):
    geometry = _scope_frame_geometry(scope)
    geometry['groups'] = {str(key): _group_frame_geometry(group, geometry) for key, group in groups.items()}
    if geometry['cyclic']:
        original_frames = {}
        for group in groups.values():
            for endpoint in group.get('endpoints', ()):
                original = endpoint['original_observation_id']
                frame = endpoint['native_frame_index']
                if original in original_frames and original_frames[original] != frame:
                    raise ValueError('One original SAM observation is assigned to different native frames')
                original_frames[original] = frame
        for run in runs.values():
            group_id = str(run['group_id'])
            if group_id not in geometry['groups']:
                raise ValueError('Cyclic SAM run references an unknown group')
            allowed = geometry['groups'][group_id]['addresses']
            for key in ('expected_frames', 'observed_frames', 'injected_frames'):
                if any(_frame_integer(frame, 'SAM run unfolded frame') not in allowed for frame in run.get(key, ())):
                    raise ValueError('Cyclic SAM run coverage lies outside its declared address map')
            expected = set(run.get('expected_frames', ()))
            observed = set(run.get('observed_frames', ()))
            if not observed.issubset(expected):
                raise ValueError('Cyclic SAM observed frames lie outside expected coverage')
            for owner in (run, *run.get('tile_evidence', ())):
                owner_frames = {_frame_integer(frame, 'SAM owner unfolded frame') for frame in owner.get('observed_frames', ())}
                if not owner_frames.issubset(expected):
                    raise ValueError('Cyclic SAM tile coverage lies outside its original run')
                keys = ('raw_mask_keys', 'candidate_mask_keys', 'availability_mask_keys') if owner is run else ('raw_mask_keys',)
                for key in keys:
                    if key not in owner:
                        continue
                    saved = owner[key]
                    if (not isinstance(saved, Mapping) or set(saved) != {str(frame) for frame in owner_frames}):
                        raise ValueError('Cyclic SAM mask addresses differ from their observed frame coverage')
    return geometry


def evidence_frame_geometry(bundle):
    """Return validated immutable native closure for a bundle or active reader."""
    owner = getattr(bundle, 'bundle', bundle)
    if owner.scope.get('frame_addressing'):
        from .sam_cyclic import assert_cyclic_implementation_unchanged
        assert_cyclic_implementation_unchanged()
    geometry = getattr(owner, '_frame_geometry', None)
    if geometry is None:
        geometry = _freeze(_evidence_frame_geometry(owner.scope, owner.groups, owner.runs))
    return geometry


def native_output_shape_tyx(bundle):
    shape = evidence_frame_geometry(bundle)['native_shape_tyx']
    if shape is None:
        raise ValueError('SAM native publication requires its recorded shape_tyx')
    return tuple(shape)


class SamEvidenceWriter:
    """Stream bounded masks into an atomically published, immutable bundle.

    Group metadata uses planner names. Named masks are ``endpoint:ID``,
    ``evaluation:ID``, ``acceptance:FRAME`` and ``write:FRAME``. Optional
    ``permitted:ID`` masks mark known allowable sibling support in an endpoint
    evaluation domain; ``unrelated:FRAME`` masks permit contact diagnostics.
    """

    def __init__(self, directory, scope_metadata, *, max_mask_bytes=64 * 1024 * 1024,
                 max_payload_bytes=32 * 1024**3):
        self.directory = Path(directory).resolve()
        if self.directory.exists():
            raise FileExistsError(f"SAM evidence destination must be fresh: {self.directory}")
        self.directory.parent.mkdir(parents=True, exist_ok=True)
        self.staging = self.directory.parent / ("." + self.directory.name + ".stage-" + uuid.uuid4().hex)
        self.staging.mkdir()
        self._stream = (self.staging / "masks.bin").open("wb")
        self.max_mask_bytes, self.max_payload_bytes = int(max_mask_bytes), int(max_payload_bytes)
        if self.max_mask_bytes <= 0 or self.max_payload_bytes <= 0:
            self.abort()
            raise ValueError("SAM evidence budgets must be positive")
        self.scope = _plain(scope_metadata)
        if self.scope.get('frame_addressing'):
            from .sam_cyclic import IMPLEMENTATION_SHA256
            self.scope.setdefault('cyclic_implementation_sha256', IMPLEMENTATION_SHA256)
        try:
            self._scope_frame_geometry = _scope_frame_geometry(self.scope)
        except BaseException:
            self.abort()
            raise
        self.groups, self.runs, self.records = {}, {}, {}
        self._pending_tiles = {}
        self._closed = False

    def _put(self, key, value, shape):
        if key in self.records:
            raise ValueError(f"Duplicate SAM mask identity: {key}")
        value = _mask(value, shape)
        if value.size > self.max_mask_bytes:
            raise MemoryError("SAM evidence mask exceeds configured memory limit")
        packed = np.packbits(value.reshape(-1), bitorder="little").tobytes()
        encoded = zlib.compress(packed, level=6)
        offset = self._stream.tell()
        if offset + len(encoded) > self.max_payload_bytes:
            raise OSError("SAM evidence payload exceeds configured staging disk limit")
        self._stream.write(encoded)
        self.records[key] = dict(offset=offset, bytes=len(encoded), shape=list(shape),
                                 packed_bytes=len(packed), sha256=hashlib.sha256(packed).hexdigest(),
                                 compressed_sha256=hashlib.sha256(encoded).hexdigest(),
                                 foreground=int(np.count_nonzero(value)))
        return key

    def add_group(self, group, masks):
        group = _plain(group)
        identity = str(group["group_id"])
        if identity in self.groups:
            raise ValueError(f"Duplicate SAM group: {identity}")
        shape = _group_shape(group)
        frames = list(map(int, group["frame_indices"]))
        if not frames or frames != sorted(set(frames)):
            raise ValueError("SAM group frames must be distinct increasing native frame indices")
        if frames != list(range(frames[0], frames[-1] + 1)):
            raise ValueError("SAM group topology requires contiguous declared native frames")
        endpoints = group.get("endpoints", [])
        ids = [str(v["observation_id"]) for v in endpoints]
        if len(set(ids)) != len(ids):
            raise ValueError("Duplicate endpoint identity within SAM group")
        if any(int(v["frame_index"]) not in frames for v in endpoints):
            raise ValueError("SAM endpoint is outside its declared group frames")
        _group_frame_geometry(group, self._scope_frame_geometry)
        if group.get("status") in {"incomplete", "unresolved", "invalid"} or not group.get("complete", True):
            # Resource-limited inventories are metadata, never successful crops.
            # Optional endpoint_local masks retain native observation silhouettes
            # without allocating the oversized unresolved group canvas.
            keys={}
            for name in sorted(masks):
                value=masks[name]
                keys[name]=self._put(f"g/{identity}/{name}",value,np.asarray(value).shape)
            group.update(complete=False, geometry_contract_status="unresolved",mask_keys=keys)
            self.groups[identity] = group
            return
        required = {f"{kind}:{frame}" for kind in ("acceptance", "write") for frame in frames}
        required.update(f"endpoint:{identity}" for identity in ids)
        required.update(f"evaluation:{identity}" for identity in ids)
        if required - set(masks):
            raise ValueError(f"Missing SAM geometry/reference masks: {sorted(required - set(masks))}")
        for endpoint in endpoints:
            observed = _mask(masks[f"endpoint:{endpoint['observation_id']}"], shape)
            if not observed.any():
                raise ValueError("Original SAM endpoint reference must contain foreground")
            write = _mask(masks[f"write:{int(endpoint['frame_index'])}"], shape)
            if np.any(write & observed):
                raise ValueError("SAM write domain may not repaint an original observation")
        for frame in frames:
            if np.any(_mask(masks[f"write:{frame}"], shape) & ~_mask(masks[f"acceptance:{frame}"], shape)):
                raise ValueError("SAM write domain must be a subset of predeclared acceptance")
            for name in masks:
                if name.startswith("edge_write:") and name.endswith(f":{frame}"):
                    if np.any(_mask(masks[name], shape) & ~_mask(masks[f"write:{frame}"], shape)):
                        raise ValueError("Branch-specific SAM write domain must be a subset of group write")
        for edge in group.get("edges", []):
            if str(edge["source_id"]) not in ids or str(edge["target_id"]) not in ids:
                raise ValueError("SAM edge references an unknown original endpoint")
        # Sort identities, not materialized values: a streaming Mapping can
        # produce one endpoint crop plane at a time without an O(refs * crop)
        # transient list of dense arrays before compression.
        group["mask_keys"] = {name: self._put(f"g/{identity}/{name}", masks[name], shape)
                              for name in sorted(masks)}
        group.setdefault("complete", group.get("status", "complete") not in {"incomplete", "unresolved"})
        self.groups[identity] = group

    def add_run_tile(self, parent_run_id, tile_descriptor, raw_masks):
        """Stream full raw halos before completing their original hypothesis.

        A skipped empty-original-seed footprint is metadata with no raw masks;
        it is unavailable spatial coverage, never a successful empty prediction.
        """
        parent_run_id = str(parent_run_id)
        tile = _plain(tile_descriptor)
        group_id, tile_id = str(tile["group_id"]), str(tile["tile_id"])
        if group_id not in self.groups or parent_run_id in self.runs:
            raise ValueError("SAM tile must precede its original run and reference a known group")
        pending = self._pending_tiles.setdefault(parent_run_id, {})
        if tile_id in pending:
            raise ValueError("Duplicate independent SAM tile identity")
        group = self.groups[group_id]
        gy0,gx0,gy1,gx1 = map(int,group["context_bbox_yx"])
        y0,x0,y1,x1 = map(int,tile["crop_bbox_yx"])
        a0,b0,a1,b1 = map(int,tile["ownership_bbox_yx"])
        if not (gy0<=y0<=a0<a1<=y1<=gy1 and gx0<=x0<=b0<b1<=x1<=gx1):
            raise ValueError("Independent SAM tile/owner lies outside the fixed original crop")
        expected = list(map(int,tile["expected_frames"]))
        if not expected or len(expected)!=len(set(expected)) or set(expected)-set(group["frame_indices"]):
            raise ValueError("Independent SAM tile coverage lies outside its original group")
        raw_masks = {int(frame):mask for frame,mask in raw_masks.items()}
        if set(raw_masks)-set(expected):
            raise ValueError("SAM tile observation lies outside expected coverage")
        attempted = bool(tile.get("attempted", True))
        if not attempted and raw_masks:
            raise ValueError("Unattempted SAM tile cannot own successful raw predictions")
        endpoint_ids={str(v["observation_id"]) for v in group["endpoints"]}
        if not tile.get("seed_ids") or set(map(str,tile["seed_ids"]))-endpoint_ids:
            raise ValueError("SAM tile seed must retain original observation identities")
        endpoints={str(v["observation_id"]):v for v in group["endpoints"]}
        if any(int(endpoints[key]["frame_index"])!=expected[0] for key in tile["seed_ids"]):
            raise ValueError("Independent SAM tile seed must match the injected original frame")
        source_seed=np.zeros((y1-y0,x1-x0),bool)
        for key in tile["seed_ids"]:
            source=self._read_staged(group["mask_keys"][f"endpoint:{key}"])
            source_seed |= source[y0-gy0:y1-gy0,x0-gx0:x1-gx0]
        count=int(source_seed.sum())
        if "seed_foreground" in tile and int(tile["seed_foreground"])!=count:
            raise ValueError("Independent SAM tile seed geometry differs from its original observation")
        if attempted!=bool(count):
            raise ValueError("Only empty original seed intersections may be unavailable SAM tiles")
        tile["seed_foreground"]=count
        tile.update(schema=TILE_EVIDENCE_SCHEMA,parent_run_id=parent_run_id,group_id=group_id,
            tile_id=tile_id,expected_frames=expected,observed_frames=sorted(raw_masks),attempted=attempted,
            raw_mask_keys={str(frame):self._put(f"t/{parent_run_id}/{tile_id}/raw/{frame}",mask,(y1-y0,x1-x0))
                           for frame,mask in sorted(raw_masks.items())})
        tile["complete"] = attempted and bool(tile.get("complete", True)) and set(raw_masks)==set(expected)
        tile.setdefault("status", "generated_complete" if tile["complete"] else "generated_incomplete" if attempted else "unavailable_empty_original_seed")
        pending[tile_id] = tile

    def add_run(self, run, raw_masks, candidate_masks=None, *, availability_masks=None):
        run = _plain(run)
        identity, group_id = str(run["run_id"]), str(run["group_id"])
        if identity in self.runs or group_id not in self.groups:
            raise ValueError("Duplicate SAM run or unknown group identity")
        group = self.groups[group_id]
        if not group.get("complete", True):
            raise ValueError("Cannot generate a SAM run from an unresolved family/geometry contract")
        shape = _group_shape(group)
        raw_masks = {int(k): v for k, v in raw_masks.items()}
        expected = list(map(int, run["expected_frames"]))
        if not expected or len(expected) != len(set(expected)) or set(expected) - set(group["frame_indices"]):
            raise ValueError("SAM expected coverage is malformed or outside its group")
        direction = run["direction"]
        direction = "forward" if direction in (1, "forward") else "backward" if direction in (-1, "backward") else None
        if direction is None or expected != sorted(expected, reverse=direction == "backward"):
            raise ValueError("SAM direction must match ordered expected native frames")
        if set(raw_masks) - set(expected):
            raise ValueError("SAM observed frame lies outside expected coverage")
        endpoint_ids = {str(v["observation_id"]) for v in group["endpoints"]}
        if set(map(str, run.get("seed_ids", ()))) - endpoint_ids or set(map(str, run.get("held_out_ids", ()))) - endpoint_ids:
            raise ValueError("SAM run references an unknown endpoint")
        run.update(direction=direction, expected_frames=expected, observed_frames=sorted(raw_masks),
                   raw_mask_keys={}, candidate_mask_keys={})
        tiles = self._pending_tiles.get(identity, {})
        if tiles:
            if any(tile["group_id"]!=group_id or tile["expected_frames"]!=expected for tile in tiles.values()):
                raise ValueError("Independent SAM tile lineage/coverage differs from its original run")
            if any(set(tile["seed_ids"])-set(run.get("seed_ids",())) for tile in tiles.values()):
                raise ValueError("SAM tile seed lineage differs from its original hypothesis")
            # Core rectangles partition the shared canvas. A sweep over the few
            # declared tile rows validates ownership without allocating a mask.
            gy0,gx0,gy1,gx1=map(int,group["context_bbox_yx"])
            boundaries=sorted({gy0,gy1,*[int(v) for tile in tiles.values() for v in (tile["ownership_bbox_yx"][0],tile["ownership_bbox_yx"][2])]})
            for lo,hi in zip(boundaries,boundaries[1:]):
                spans=sorted((int(tile["ownership_bbox_yx"][1]),int(tile["ownership_bbox_yx"][3])) for tile in tiles.values()
                             if int(tile["ownership_bbox_yx"][0])<=lo and int(tile["ownership_bbox_yx"][2])>=hi)
                cursor=gx0
                for left,right in spans:
                    if left!=cursor:
                        raise ValueError("SAM tile ownership must partition the whole crop without gaps/overlap")
                    cursor=right
                if cursor!=gx1:
                    raise ValueError("SAM tile ownership does not cover the original crop")
            run.update(generation_mode="tiled",tile_evidence_schema=TILE_EVIDENCE_SCHEMA,
                       tile_evidence=[tiles[key] for key in sorted(tiles)],raw_domain="fixed_owned_core_assembly")
            run["tracker_scores"] = None
            run["aggregate_tracker_probability_status"] = "undefined_for_independent_tile_assembly"
            bad=[tile for tile in tiles.values() if tile["attempted"] and
                 (tile.get("status") in {"failed","cancelled","infrastructure_invalid"}
                  or tile.get("structurally_valid") is False or tile.get("runtime_receipt",{}).get("prediction_valid") is False)]
            if bad:
                run.update(structurally_valid=False,status="infrastructure_invalid")
            if any(tile["attempted"] and not tile["complete"] for tile in tiles.values()):
                run["complete"]=False
        elif run.get("generation_mode") == "tiled" or self.scope.get("sam_crop_mode")=="tiled":
            raise ValueError("Tiled SAM original run requires retained tile evidence")
        available = None if availability_masks is None else {int(frame):mask for frame,mask in availability_masks.items()}
        if available is not None and set(available)!=set(raw_masks):
            raise ValueError("SAM spatial coverage must match original raw frame addresses")
        if tiles and available is None:
            raise ValueError("Tiled SAM evidence requires explicit owned-core spatial coverage")
        if available is not None:
            run["availability_mask_keys"] = {}
        supplied = None if candidate_masks is None else {int(k): v for k, v in candidate_masks.items()}
        if supplied is not None and set(supplied) != set(raw_masks):
            raise ValueError("SAM candidate mask coverage must match raw evidence coverage")
        # Geometry was already serialized; decoding one write plane keeps reads bounded.
        self._stream.flush()
        for frame, raw in sorted(raw_masks.items()):
            raw = _mask(raw, shape)
            if available is not None:
                coverage = _mask(available[frame],shape)
                if np.any(raw & ~coverage):
                    raise ValueError("SAM raw support cannot come from an unavailable owner core")
                if tiles:
                    expected_raw,expected_available=np.zeros(shape,bool),np.zeros(shape,bool)
                    for tile in tiles.values():
                        if str(frame) not in tile["raw_mask_keys"]:
                            continue
                        ty0,tx0,ty1,tx1=map(int,tile["crop_bbox_yx"])
                        cy0,cx0,cy1,cx1=map(int,tile["ownership_bbox_yx"])
                        tile_raw=self._read_staged(tile["raw_mask_keys"][str(frame)])
                        expected_raw[cy0-gy0:cy1-gy0,cx0-gx0:cx1-gx0] = tile_raw[cy0-ty0:cy1-ty0,cx0-tx0:cx1-tx0]
                        expected_available[cy0-gy0:cy1-gy0,cx0-gx0:cx1-gx0] = True
                    if not np.array_equal(raw,expected_raw) or not np.array_equal(coverage,expected_available):
                        raise ValueError("SAM original raw/availability must reproduce fixed native tile ownership")
                run["availability_mask_keys"][str(frame)] = self._put(f"r/{identity}/available/{frame}",coverage,shape)
            branch_keys = [f"edge_write:{edge_id}:{frame}" for edge_id in run.get("edge_ids", ())]
            if branch_keys:
                if any(key not in group["mask_keys"] for key in branch_keys):
                    raise ValueError("SAM run branch identity has no declared write domain")
                write = np.zeros(shape, bool)
                for key in branch_keys:
                    write |= self._read_staged(group["mask_keys"][key])
            else:
                write = self._read_staged(group["mask_keys"][f"write:{frame}"])
            candidate = raw & write if supplied is None else _mask(supplied[frame], shape)
            if np.any(candidate & ~(raw & write)):
                raise ValueError("SAM candidate additions must be a subset of raw support and fixed write domain")
            run["raw_mask_keys"][str(frame)] = self._put(f"r/{identity}/raw/{frame}", raw, shape)
            run["candidate_mask_keys"][str(frame)] = self._put(f"r/{identity}/candidate/{frame}", candidate, shape)
        run["complete"] = bool(run.get("complete", True)) and set(raw_masks) == set(expected)
        run.setdefault("status", "generated_complete" if run["complete"] else "generated_incomplete")
        run.setdefault("pass_index", 1)
        run.setdefault("injected_frames", [expected[0]])
        self.runs[identity] = run
        self._pending_tiles.pop(identity,None)

    def _read_staged(self, key):
        self._stream.flush()
        with (self.staging / "masks.bin").open("rb") as stream:
            return _decode_mask(stream, self.records[key], self.max_mask_bytes)

    def commit(self, *, complete=True):
        _evidence_frame_geometry(self.scope, self.groups, self.runs)
        if self._closed:
            raise RuntimeError("SAM evidence writer is already closed")
        index = dict(groups=self.groups, runs=self.runs, masks=self.records)
        if self._pending_tiles:
            if complete:
                raise ValueError("Cannot publish complete SAM evidence with unfinalized tiled original runs")
            index["unfinalized_tile_runs"] = self._pending_tiles
        encoded_index = _json_bytes(index)
        if len(encoded_index) > 64 * 1024**2:
            raise MemoryError("SAM evidence index exceeds bounded metadata budget")
        self._stream.flush()
        os.fsync(self._stream.fileno())
        self._stream.close()
        (self.staging / "index.json").write_bytes(encoded_index)
        manifest = dict(schema=SCHEMA, complete=bool(complete), scope=self.scope,
                        group_count=len(self.groups), run_count=len(self.runs), mask_count=len(self.records),
                        files={name: dict(bytes=(self.staging/name).stat().st_size,
                                          sha256=_file_hash(self.staging/name)) for name in ("index.json", "masks.bin")})
        if self._pending_tiles or any(run.get("tile_evidence") for run in self.runs.values()):
            manifest["tile_evidence_schema"] = TILE_EVIDENCE_SCHEMA
        manifest["evidence_fingerprint"] = fingerprint(manifest)
        (self.staging / "manifest.json").write_bytes(_json_bytes(manifest))
        os.replace(self.staging, self.directory)
        self._closed = True
        return SamEvidenceBundle.open(self.directory, max_mask_bytes=self.max_mask_bytes)

    close = commit

    def abort(self):
        stream = getattr(self, "_stream", None)
        if stream is not None and not stream.closed:
            stream.close()
        staging = getattr(self, "staging", None)
        if staging is not None and staging.exists():
            # This UUID-owned staging directory is the only tree removed here.
            shutil.rmtree(staging)
        self._closed = True

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        if not self._closed:
            self.abort()


def _decode_mask(stream, record, max_mask_bytes):
    shape = tuple(map(int, record["shape"]))
    if len(shape) != 2 or any(v <= 0 for v in shape) or int(np.prod(shape)) > max_mask_bytes:
        raise ValueError("Invalid or oversized SAM evidence mask shape")
    packed_size = (shape[0] * shape[1] + 7) // 8
    if int(record["packed_bytes"]) != packed_size:
        raise ValueError("SAM evidence packed shape mismatch")
    if not 0 <= int(record["bytes"]) <= packed_size + packed_size // 1000 + 1024:
        raise ValueError("SAM evidence compressed mask exceeds bounded record size")
    stream.seek(int(record["offset"]))
    encoded = stream.read(int(record["bytes"]))
    if hashlib.sha256(encoded).hexdigest() != record["compressed_sha256"]:
        raise ValueError("SAM evidence compressed-mask checksum mismatch")
    decoder = zlib.decompressobj()
    packed = decoder.decompress(encoded, packed_size + 1)
    if len(packed) != packed_size or not decoder.eof or decoder.unused_data or decoder.unconsumed_tail:
        raise ValueError("Malformed SAM evidence mask payload")
    if hashlib.sha256(packed).hexdigest() != record["sha256"]:
        raise ValueError("SAM evidence mask checksum mismatch")
    unpacked = np.unpackbits(np.frombuffer(packed, dtype=np.uint8), count=shape[0]*shape[1], bitorder="little")
    result = np.frombuffer(unpacked.tobytes(), dtype=np.bool_).reshape(shape)
    if int(np.count_nonzero(result)) != int(record["foreground"]):
        raise ValueError("SAM evidence foreground count disagrees with indexed payload")
    return result


class SamEvidenceBundle:
    """Read-only descriptors and lazy bounded mask access; no model imports."""

    @classmethod
    def open(cls, directory, *, verify=True, max_mask_bytes=64 * 1024 * 1024):
        self = cls()
        self.directory = Path(directory).resolve()
        self.max_mask_bytes = int(max_mask_bytes)
        manifest = json.loads((self.directory / "manifest.json").read_text("utf-8"))
        if manifest.get("schema") != SCHEMA:
            raise ValueError("Unsupported SAM proposal evidence schema")
        saved = manifest.get("evidence_fingerprint")
        if saved != fingerprint({k: v for k, v in manifest.items() if k != "evidence_fingerprint"}):
            raise ValueError("SAM evidence manifest fingerprint mismatch")
        if set(manifest.get("files", {})) != {"index.json", "masks.bin"}:
            raise ValueError("SAM evidence files must use the portable fixed-file contract")
        for name, entry in manifest["files"].items():
            path = self.directory / name
            if not path.resolve().is_relative_to(self.directory) or path.stat().st_size != entry["bytes"]:
                raise ValueError("Missing or escaped SAM evidence payload")
            if verify and _file_hash(path) != entry["sha256"]:
                raise ValueError("SAM evidence file checksum mismatch")
        if manifest["files"]["index.json"]["bytes"] > 64 * 1024**2:
            raise MemoryError("SAM evidence index exceeds bounded metadata budget")
        index = json.loads((self.directory / "index.json").read_text("utf-8"))
        self.groups, self.runs, self.records = _freeze(index["groups"]), _freeze(index["runs"]), _freeze(index["masks"])
        self.unfinalized_tile_runs = _freeze(index.get("unfinalized_tile_runs",{}))
        self.manifest, self.scope = _freeze(manifest), _freeze(manifest["scope"])
        total = manifest["files"]["masks.bin"]["bytes"]
        for key, record in self.records.items():
            if int(record["offset"]) < 0 or int(record["bytes"]) < 0 or int(record["offset"]) + int(record["bytes"]) > total:
                raise ValueError(f"Invalid SAM evidence payload bounds: {key}")
        if (len(self.groups), len(self.runs), len(self.records)) != (manifest["group_count"], manifest["run_count"], manifest["mask_count"]):
            raise ValueError("SAM evidence record counts disagree with manifest")
        self._frame_geometry = _freeze(_evidence_frame_geometry(self.scope, self.groups, self.runs))
        return self

    def mask(self, key):
        with (self.directory / "masks.bin").open("rb") as stream:
            return _decode_mask(stream, self.records[str(key)], self.max_mask_bytes)

    def group_mask(self, group_id, name):
        return self.mask(self.groups[str(group_id)]["mask_keys"][str(name)])

    def raw_mask(self, run_id, frame):
        return self.mask(self.runs[str(run_id)]["raw_mask_keys"][str(int(frame))])

    def candidate_mask(self, run_id, frame):
        return self.mask(self.runs[str(run_id)]["candidate_mask_keys"][str(int(frame))])

    def availability_mask(self, run_id, frame):
        run=self.runs[str(run_id)]
        keys=run.get("availability_mask_keys",{})
        if str(int(frame)) in keys:
            return self.mask(keys[str(int(frame))])
        if run.get("generation_mode")=="tiled":
            raise ValueError("Independent tiled SAM run is missing owned-core availability evidence")
        shape=_group_shape(self.groups[run["group_id"]])
        return np.frombuffer(bytes([1])*int(np.prod(shape)),dtype=np.bool_).reshape(shape)

    def tile_raw_mask(self, run_id, tile_id, frame):
        run=self.runs[str(run_id)]
        tile=next((tile for tile in run.get("tile_evidence",()) if tile["tile_id"]==str(tile_id)),None)
        if tile is None:
            raise ValueError("Unknown independent SAM tile identity")
        return self.mask(tile["raw_mask_keys"][str(int(frame))])

    def halo_union_mask(self, run_id, frame):
        run=self.runs[str(run_id)]
        tiles=run.get("tile_evidence",())
        if not tiles:
            if run.get("generation_mode")=="tiled":
                raise ValueError("Tiled SAM evidence has no retained raw halos")
            return self.raw_mask(run_id,frame)
        group=self.groups[run["group_id"]]
        y0,x0,_,_=group["context_bbox_yx"]
        union=np.zeros(_group_shape(group),bool)
        for tile in tiles:
            if str(int(frame)) not in tile["raw_mask_keys"]:
                continue
            a0,b0,a1,b1=tile["crop_bbox_yx"]
            union[a0-y0:a1-y0,b0-x0:b1-x0] |= self.tile_raw_mask(run_id,tile["tile_id"],frame)
        return np.frombuffer(union.tobytes(),dtype=np.bool_).reshape(union.shape)

    def measure_effective_halo_union(self, run_id, frame, mask_filter=None):
        from .sam_filtering import _spec,filter_sam_components
        spec=_spec(mask_filter)
        raw=self.halo_union_mask(run_id,frame)
        group_id=self.runs[str(run_id)]["group_id"]
        threshold=0. if spec is None else float(spec["thresholds_by_group"][group_id])
        return filter_sam_components(raw,threshold,enabled=False if spec is None else bool(spec["enabled"]))

    def reader(self, *, max_cache_bytes=32 * 1024**2):
        """Create a bounded mask/filter reader with transaction integrity checks."""
        from .sam_mask_reader import SamMaskReader
        return SamMaskReader(self, max_cache_bytes=max_cache_bytes)

    @property
    def evidence_fingerprint(self):
        return self.manifest["evidence_fingerprint"]

    def assert_unchanged(self):
        """Verify publication identity after lazy reads and before publishing replay."""
        current = json.loads((self.directory / "manifest.json").read_text("utf-8"))
        if current != _plain(self.manifest):
            raise ValueError("SAM evidence manifest changed during selection/replay")
        for name, entry in self.manifest["files"].items():
            path = self.directory / name
            if path.stat().st_size != int(entry["bytes"]) or _file_hash(path) != entry["sha256"]:
                raise ValueError("SAM evidence payload changed during selection/replay")


def iter_selected_planes(bundle, selection, *, direction=None, pass_index=None):
    """Yield exact (group_id, stored/unfolded_frame, cropped union) evidence.

    Cyclic aliases remain unfolded here for compatibility with measurement
    consumers. Native publication must use iter_selected_native_crops instead.
    """
    from .sam_mask_reader import SamMaskReader
    if isinstance(bundle, SamMaskReader):
        yield from _iter_selected_planes(bundle, selection, direction=direction, pass_index=pass_index)
    else:
        with bundle.reader() as reader:
            yield from _iter_selected_planes(reader, selection, direction=direction, pass_index=pass_index)


def _iter_selected_planes(bundle, selection, *, direction=None, pass_index=None):
    from .sam_mask_reader import effective_candidate_mask
    evidence_frame_geometry(bundle)
    if selection.get('resolved_policy', {}).get('version') in (4,5) and 'mask_filter' not in selection:
        raise ValueError('Guarded SAM support requires its retained mask filter specification')
    snapshot = bundle.filter_snapshot(selection)
    selected = set(selection["selected_run_ids"])
    unknown = selected - set(bundle.runs)
    if unknown:
        raise ValueError(f"Selection references unknown SAM runs: {sorted(unknown)}")
    for group_id in sorted(bundle.groups):
        group = bundle.groups[group_id]
        runs = [bundle.runs[key] for key in sorted(selected) if bundle.runs[key]["group_id"] == group_id
                and (direction is None or bundle.runs[key]["direction"] == direction)
                and (pass_index is None or int(bundle.runs[key]["pass_index"]) == int(pass_index))]
        for frame in group["frame_indices"]:
            plane = np.zeros(_group_shape(group), dtype=bool)
            for run in runs:
                if str(frame) in run["candidate_mask_keys"]:
                    plane |= effective_candidate_mask(bundle, run["run_id"], frame, snapshot)
            plane.setflags(write=False)
            yield group_id, int(frame), plane


def iter_selected_native_crops(bundle, selection, *, direction=None, pass_index=None):
    """Yield group, stored frame, native frame, native bbox and exact mask.

    Stored ownership stays intact. Each selected contributor union is folded
    by its saved address; spatial/temporal overlaps collapse only when the
    native publication consumer explicitly ORs these crops.
    """
    from .sam_cyclic import address_for_unfolded_index, transform_crop_between_frame_addresses
    geometry = evidence_frame_geometry(bundle)
    for group_id, stored_frame, mask in iter_selected_planes(bundle, selection,
            direction=direction, pass_index=pass_index):
        group_geometry = geometry['groups'][str(group_id)]
        bbox = tuple(group_geometry['bbox_yx'])
        addresses = group_geometry['addresses']
        native_frame = stored_frame
        if addresses is not None:
            address = addresses[stored_frame]
            native_frame = int(address['native_index'])
            target = address_for_unfolded_index(native_frame, geometry['native_shape_tyx'][0],
                period_degrees=geometry['addressing']['period_degrees'])
            mask, bbox = transform_crop_between_frame_addresses(mask, bbox, address, target,
                geometry['native_shape_tyx'][2])
        mask.setflags(write=False)
        yield group_id, stored_frame, native_frame, tuple(bbox), mask


def selected_native_plane(bundle, selection, frame, *, direction=None, pass_index=None, shape_yx=None):
    """Rebuild one folded native binary plane from retained effective owners."""
    from .sam_cyclic import address_for_unfolded_index, transform_crop_between_frame_addresses
    from .sam_mask_reader import SamMaskReader, effective_candidate_mask
    if not isinstance(bundle, SamMaskReader):
        with bundle.reader() as reader:
            return selected_native_plane(reader, selection, frame, direction=direction,
                pass_index=pass_index, shape_yx=shape_yx)
    geometry = evidence_frame_geometry(bundle)
    native_shape = geometry['native_shape_tyx']
    frame = _frame_integer(frame, 'SAM native output frame')
    if native_shape is None:
        if shape_yx is None:
            raise ValueError('SAM native plane requires its recorded shape_tyx')
    elif not 0 <= frame < native_shape[0]:
        raise ValueError('SAM native output frame is outside its saved canvas')
    shape = tuple(native_shape[1:] if shape_yx is None else shape_yx)
    if native_shape is not None and shape != tuple(native_shape[1:]):
        raise ValueError('SAM native plane shape differs from its saved closure')
    selected = set(selection['selected_run_ids'])
    if selected - set(bundle.runs):
        raise ValueError('Selection references unknown SAM runs')
    if selection.get('resolved_policy', {}).get('version') in (4,5) and 'mask_filter' not in selection:
        raise ValueError('Guarded SAM support requires its retained mask filter specification')
    snapshot = bundle.filter_snapshot(selection)
    plane = np.zeros(shape, np.uint8)
    for run_id in sorted(selected):
        run = bundle.runs[run_id]
        if ((direction is not None and run['direction'] != direction) or
                (pass_index is not None and int(run['pass_index']) != int(pass_index))):
            continue
        group_geometry = geometry['groups'][str(run['group_id'])]
        addresses = group_geometry['addresses']
        for key in run['candidate_mask_keys']:
            stored_frame = int(key)
            address = addresses[stored_frame] if addresses is not None else None
            if int(address['native_index'] if address is not None else stored_frame) != frame:
                continue
            mask = effective_candidate_mask(bundle, run_id, stored_frame, snapshot)
            bbox = tuple(group_geometry['bbox_yx'])
            if address is not None:
                target = address_for_unfolded_index(frame, native_shape[0],
                    period_degrees=geometry['addressing']['period_degrees'])
                mask, bbox = transform_crop_between_frame_addresses(mask, bbox, address, target, shape[1])
            y0,x0,y1,x1 = bbox
            if mask.shape != (y1-y0,x1-x0) or not (0<=y0<y1<=shape[0] and 0<=x0<x1<=shape[1]):
                raise ValueError('SAM native candidate crop lies outside its published plane')
            plane[y0:y1,x0:x1] |= mask
    return plane


def load_sam_online_selection(bundle, *, policy_hash=None, selected_run_ids=None,
                              max_receipt_bytes=64 * 1024 * 1024):
    """Read the immutable online selection used by published contributors.

    A legacy bundle with no retained receipt has explicitly unfiltered semantics.
    New generator scopes declare that their receipt is required; losing it cannot
    silently turn filtered published support back into raw candidate support.
    """
    if not isinstance(bundle, SamEvidenceBundle):
        bundle = SamEvidenceBundle.open(bundle)
    path = bundle.directory / "online_selection.json"
    exported = bundle.directory / "export.json"
    if path.exists() != exported.exists():
        raise ValueError("Incomplete SAM online-selection export sidecars")
    if not path.exists():
        path = bundle.directory.parent / "selection.json"
    if not path.exists():
        if bundle.scope.get("selection_receipt_required", False):
            raise ValueError("Published SAM component filtering requires its retained online selection receipt")
        return None
    if path.stat().st_size > int(max_receipt_bytes):
        raise ValueError("SAM online selection receipt exceeds its bounded metadata budget")
    encoded = path.read_bytes()
    receipt = json.loads(encoded)
    if (receipt.get("schema") != "xta.sam_selection/1"
            or receipt.get("evidence_fingerprint") != bundle.evidence_fingerprint
            or set(receipt.get("selected_run_ids", ())) - set(bundle.runs)):
        raise ValueError("SAM online selection receipt does not belong to its proposal bundle")
    if (bundle.scope.get("selection_receipt_required", False) or receipt.get('resolved_policy', {}).get('version') in (4,5)) and "mask_filter" not in receipt:
        raise ValueError("Published SAM component filtering requires its retained mask filter specification")
    tiled=bundle.scope.get("sam_crop_mode")=="tiled" or any(run.get("generation_mode")=="tiled" for run in bundle.runs.values())
    if tiled and (receipt.get("resolved_policy",{}).get("version") not in (3,5) or "mask_filter" not in receipt
                  or receipt.get("tiled_quality_contract",{}).get("schema")!="xta.sam_tiled_quality/1"):
        raise ValueError("Published tiled SAM support requires its retained quality-v3 filter/halo contract")
    # Validation belongs to the same central interpreter used for effective
    # support. A portable sidecar cannot downgrade new filtered outputs to old
    # unfiltered semantics merely by changing its stated policy version.
    from .sam_filtering import _spec as validate_mask_filter
    validate_mask_filter(receipt)
    if policy_hash and receipt.get("policy_hash") != str(policy_hash):
        raise ValueError("SAM published policy identity differs from its retained selection")
    if selected_run_ids is not None and set(map(str, selected_run_ids)) - set(receipt["selected_run_ids"]):
        raise ValueError("SAM published contributors differ from its retained selection")
    if exported.exists():
        metadata = json.loads(exported.read_text(encoding="utf-8"))
        expected = metadata.get("online_selection", {})
        if (metadata.get("schema") != "xta.sam_export/1"
                or metadata.get("evidence_fingerprint") != bundle.evidence_fingerprint
                or expected.get("path") != "online_selection.json"
                or int(expected.get("bytes", -1)) != len(encoded)
                or expected.get("sha256") != hashlib.sha256(encoded).hexdigest()):
            raise ValueError("SAM exported online selection checksum or identity differs")
    return _freeze(receipt)


def export_sam_evidence(source, destination):
    """Copy raw ownership plus available online measurements without inference.

    The raw three-file bundle remains byte-identical. Two optional fixed-name
    sidecars preserve the original online selection and its checksum, so export
    does not replace measured history with a new policy's recomputed results.
    """
    bundle = source if isinstance(source, SamEvidenceBundle) else SamEvidenceBundle.open(source)
    load_sam_online_selection(bundle)
    receipt_path = bundle.directory / "online_selection.json"
    export_path = bundle.directory / "export.json"
    if receipt_path.exists() != export_path.exists():
        raise ValueError("Incomplete SAM online-selection export sidecars")
    exported_receipt = receipt_path.exists()
    if not exported_receipt:
        receipt_path = bundle.directory.parent / "selection.json"
    receipt_bytes = None
    if receipt_path.is_file():
        if receipt_path.stat().st_size > 64 * 1024**2:
            raise ValueError("SAM online selection receipt exceeds its bounded metadata budget")
        receipt_bytes = receipt_path.read_bytes()
        receipt = json.loads(receipt_bytes)
        if (receipt.get("schema") != "xta.sam_selection/1"
                or receipt.get("evidence_fingerprint") != bundle.evidence_fingerprint
                or set(receipt.get("selected_run_ids", ())) - set(bundle.runs)):
            raise ValueError("SAM online selection receipt does not belong to its proposal bundle")
        if exported_receipt:
            exported = json.loads(export_path.read_text(encoding="utf-8"))
            expected = exported.get("online_selection", {})
            if (exported.get("schema") != "xta.sam_export/1"
                    or exported.get("evidence_fingerprint") != bundle.evidence_fingerprint
                    or expected.get("path") != "online_selection.json"
                    or expected.get("bytes") != len(receipt_bytes)
                    or expected.get("sha256") != hashlib.sha256(receipt_bytes).hexdigest()):
                raise ValueError("SAM exported online selection checksum or identity differs")
    destination = Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("SAM evidence export destination must be fresh")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = destination.parent / ("." + destination.name + ".export-" + uuid.uuid4().hex)
    staging.mkdir()
    try:
        for name in ("manifest.json", "index.json", "masks.bin"):
            shutil.copyfile(bundle.directory / name, staging / name)
        if receipt_bytes is not None:
            (staging / "online_selection.json").write_bytes(receipt_bytes)
            (staging / "export.json").write_bytes(_json_bytes(dict(
                schema="xta.sam_export/1", evidence_fingerprint=bundle.evidence_fingerprint,
                online_selection=dict(path="online_selection.json", bytes=len(receipt_bytes),
                    sha256=hashlib.sha256(receipt_bytes).hexdigest()),
                semantics="Retained online measurements and selection; raw bundle unchanged.")))
        SamEvidenceBundle.open(staging)
        bundle.assert_unchanged()
        if receipt_bytes is not None and receipt_path.read_bytes() != receipt_bytes:
            raise RuntimeError("SAM online selection changed during export")
        os.replace(staging, destination)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return dict(directory=str(destination), evidence_fingerprint=bundle.evidence_fingerprint,
                complete=bool(bundle.manifest["complete"]),
                online_selection_retained=receipt_bytes is not None,
                file_count=5 if receipt_bytes is not None else 3)

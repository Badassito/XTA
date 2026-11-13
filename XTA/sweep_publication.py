"""Package completed scalar TTA branches as stock Slicer segments and count overlays.

A review grid is either ``grouped`` (stock grouped .seg.nrrd files plus
per-cutoff direct coverage labelmaps) or ``reference_only``. When a
low-quality grid exists, the native grid is reference-only: the review keeps
its branch origins and per-segment identity rows for edit transfer, but never
reads, counts or re-encodes native mask payloads. Those masks stay in their
scalar branch NRRDs.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
import threading
from urllib.parse import quote
import uuid

from .grouped_seg_nrrd import GroupedSegment, write_grouped_seg_nrrd
from .json_publication import write_json_atomic
from .reconciliation_io import FileLayer, _budget, _same_geometry, read_layer_manifest
from .sweep_coverage import DirectCoverage, write_coverage_colors


SCHEMA = "xta.sweep_review/1"
GROUPED = "grouped"
REFERENCE_ONLY = "reference_only"
_MAX_CHANNELS = 64
# Bound the padded volume created by bundling multiple otherwise sparse masks.
_MAX_GROUP_BYTES = 2 * 1024**3
_MAX_WORKERS = 8
_EMPTY = (0, -1, 0, -1, 0, -1)


def _relative(path, base):
    return Path(os.path.relpath(path, base)).as_posix()


def _identifier(*parts):
    # Hash only names/identities, never the volume payload.
    encoded = json.dumps(parts, ensure_ascii=True, separators=(",", ":"))
    return uuid.uuid5(uuid.NAMESPACE_URL, "xta.sweep_review/1/" + encoded).hex


def _read_identity(path):
    content = path.read_bytes()
    return content, hashlib.sha256(content).hexdigest()


def _manifest_paths(branch):
    candidates = list((branch / "nrrd").glob("*_nrrd_manifest.json"))
    candidates += list((branch / "low_quality").glob("*/nrrd/*_nrrd_manifest.json"))
    grids = {}
    for path in sorted(candidates):
        rel = path.relative_to(branch)
        grid_id = "native" if rel.parts[0] == "nrrd" else f"low_quality/{rel.parts[1]}"
        if grid_id in grids:
            raise ValueError(f"Multiple source NRRD manifests for grid {grid_id} in {branch}")
        if not path.resolve().is_relative_to(branch):
            raise ValueError(f"Source NRRD manifest escapes its scalar output directory: {path}")
        grids[grid_id] = path
    return grids


def _role(metadata):
    kind = metadata.get("mask_kind")
    stage = str(metadata.get("stage", ""))
    if (kind == "yolo" and metadata.get("layer_role") == "additive_component"
            and metadata.get("recomposition_op") == "union" and
            (stage == "pre_interpolation" or stage == "pre_tile_interpolation"
             or stage.endswith("_pre_tile_interpolation"))):
        return "direct"
    if kind in ("bridge", "extrapolation"):
        return "generated"
    if metadata.get("source") == "global" and stage == "final_output_after_all_postprocessing":
        return "global"
    return "derived"


def _extent(layer):
    if not layer.header:  # explicitly empty layers can have no payload
        return _EMPTY
    values = tuple(int(v) for v in layer.header["Segment0_Extent"].split())
    if values[1] < values[0]:
        return _EMPTY
    return tuple(v + origin for i, origin in enumerate(reversed(layer.offset_tyx))
                 for v in values[2*i:2*i+2])


def _union(a, b):
    if a == _EMPTY:
        return b
    if b == _EMPTY:
        return a
    return tuple(v for i in range(3) for v in (min(a[2*i], b[2*i]), max(a[2*i+1], b[2*i+1])))


def _volume(extent):
    return math.prod(max(0, extent[i+1]-extent[i]+1) for i in (0,2,4))


@dataclass
class _Member:
    cutoff: int
    source: FileLayer
    info: dict
    extent: tuple

    def descriptor(self, sweep_id, grid_id, coverage):
        def read(z0, z1):
            slab = self.source.read_slab(z0, z1)
            if self.info["role"] == "direct":
                coverage.observe(z0, slab, self.extent)
            return slab

        tags = {}
        for entry in self.source.header.get("Segment0_Tags", "").split("|"):
            key, separator, value = entry.partition(":")
            if separator and key:
                tags[key] = value
        tags.update({"XTA.SweepID": sweep_id, "XTA.GridID": grid_id,
                "XTA.SegmentID": self.info["segment_id"], "XTA.LogicalID": self.info["logical_id"],
                "XTA.CutoffCent": str(self.cutoff), "XTA.Role": self.info["role"],
                "XTA.Model": quote(self.info["model_name"], safe="._-/"),
                "XTA.LayerKey": quote(self.info["layer_key"], safe="._-/")})
        return GroupedSegment(self.info["segment_id"], self.info["display_name"],
                              tuple(self.info["color_rgb"]), read, tags, self.extent)


def _member(cutoff, layer, sweep_id, grid_id):
    meta = layer.metadata
    filename = meta["filename"]
    name = meta.get("segment_name") or layer.header.get("Segment0_Name") or filename
    color = meta.get("segment_color_rgb")
    if color is None:
        color = [float(c) for c in layer.header.get("Segment0_Color", "0.5 0.5 0.5").split()]
    model = str(meta.get("model_name", ""))
    key = str(meta.get("layer_key", ""))
    role = _role(meta)
    logical_id = _identifier(model, key or meta.get("suffix") or filename, meta.get("suffix", ""), role)
    sid = "xta_" + _identifier(sweep_id, grid_id, cutoff, logical_id)
    info = {k: meta.get(k, "") for k in
            ("view_name", "physical_view_name", "view_family", "source", "mask_kind", "stage",
             "layer_role", "recomposition_op", "tile_config_id", "tta_aug_id")}
    info.update(segment_id=sid, logical_id=logical_id, cutoff_cent=cutoff, model_name=model,
                layer_key=key, role=role, segment_name=name, display_name=f"{name} [conf {cutoff/100:.2f}]",
                color_rgb=color, source_filename=filename)
    return _Member(cutoff, layer, info, _extent(layer))


def _packs(members):
    """Group one cutoff's layers at a time, without combining distant tiny ROIs.

    A group never mixes cutoffs, so a review can load a subset of cutoffs as
    whole files. Within a cutoff, like layers of one model/view stay adjacent.
    """
    ordered = sorted(members, key=lambda m: (m.cutoff, m.info["model_name"], m.info["view_name"],
                     m.info["layer_key"], m.info["logical_id"]))
    group, extent, active_volume = [], _EMPTY, 0
    for member in ordered:
        combined = _union(extent, member.extent)
        covered = active_volume + _volume(member.extent)
        expanded_size = len(group) + 1
        padded = _volume(combined) * expanded_size
        if group and (member.cutoff != group[0].cutoff or len(group) >= _MAX_CHANNELS
                      or padded > _MAX_GROUP_BYTES or (padded > 8*1024**2 and padded > max(1, covered)*4)):
            yield group
            group, extent, active_volume = [], _EMPTY, 0
        group.append(member)
        extent = _union(extent, member.extent)
        active_volume += _volume(member.extent)
    if group:
        yield group


def _grid_sources(grid_id, branches, review, scratch, sweep_id, memory_mib, source_mode, resources):
    """Read each cutoff's manifest and NRRD headers; payloads stay unopened."""
    origins, members = [], []
    geometry = None
    for cutoff, branch, run_digest, paths, run_id in branches:
        path = paths[grid_id]
        data, manifest_digest = _read_identity(path)
        collection = resources.enter_context(read_layer_manifest(path, workspace=scratch,
            memory_mib=memory_mib, max_open=_MAX_CHANNELS, include_all_layers=True,
            manifest_bytes=data, prefer_fast_gzip=True))
        layer_run_id = collection.manifest.get("run_id")
        if ("run_id" in collection.manifest and not layer_run_id) or layer_run_id != run_id:
            raise ValueError(f"Scalar run and NRRD manifest run_id differ: {path}")
        if geometry is None:
            geometry = collection.geometry
        elif not _same_geometry(geometry, collection.geometry):
            raise ValueError(f"Physical geometry changed at cutoff {cutoff} on {grid_id}")
        evidence_manifest = branch / "reconciliation_evidence" / "manifest.json"
        origins.append({"cutoff_cent":cutoff, "source_mode":source_mode, "run_id":run_id,
                        "branch_dir":_relative(branch, review),
                        "run_manifest":_relative(branch / "manifest.json", review),
                        "run_manifest_sha256":run_digest, "source_manifest":_relative(path, review),
                        "source_manifest_sha256":manifest_digest,
                        "observed_confidence_manifest":(_relative(evidence_manifest, review)
                            if evidence_manifest.is_file() else None)})
        members.extend(_member(cutoff, layer, sweep_id, grid_id) for layer in collection)
    ids = [m.info["segment_id"] for m in members]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate source layer identity on {grid_id}")
    return geometry, origins, members


def _grid_record(grid_id, geometry, payload, origins, groups, overlays):
    return {"grid_id":grid_id, "payload":payload, "shape_tyx":list(geometry.shape_tyx),
            "space":geometry.space, "directions_xyz":geometry.directions_xyz,
            "origin_xyz":geometry.origin_xyz, "branches":origins, "groups":groups, "coverage":overlays}


def _reference_grid(grid_id, branches, review, scratch, sweep_id, memory_mib, source_mode):
    """Publish identities only; the edit export binds these rows to the preview grid."""
    with ExitStack() as resources:
        geometry, origins, members = _grid_sources(grid_id, branches, review, scratch, sweep_id,
                                                   memory_mib, source_mode, resources)
    rows = [dict(m.info) for m in sorted(members, key=lambda m: (m.cutoff, m.info["model_name"],
            m.info["view_name"], m.info["layer_key"], m.info["logical_id"]))]
    record = _grid_record(grid_id, geometry, REFERENCE_ONLY, origins, [], [])
    record["segments"] = rows
    record["reference_reason"] = ("A low-quality review grid exists. Native masks remain in their scalar "
                                  "branch NRRDs; these rows bind exported preview edits to them.")
    return record


class _Cancelled(Exception):
    """Another cutoff failed; stop before starting further group files."""


def _publish_grid(grid_id, branches, stage, review, scratch, sweep_id, memory_mib, source_mode, workers=1):
    destination = stage / grid_id
    destination.mkdir(parents=True)
    with ExitStack() as resources:
        geometry, origins, members = _grid_sources(grid_id, branches, review, scratch, sweep_id,
                                                   memory_mib, source_mode, resources)
        by_cut = defaultdict(list)
        for member in members:
            by_cut[member.cutoff].append(member)
        direct_by_cut = {cutoff: [m.info["segment_id"] for m in by_cut[cutoff] if m.info["role"] == "direct"]
                         for cutoff, *_ in branches}
        table = destination / "direct_layer_count.ctbl"
        write_coverage_colors(table, max(map(len, direct_by_cut.values()), default=0))
        # File numbers are fixed before any worker starts, so output names are deterministic.
        plan, index = [], 0
        for cutoff, *_ in branches:
            packs = []
            for pack in _packs(by_cut[cutoff]):
                index += 1
                packs.append((index, pack))
            plan.append((cutoff, packs))
        failed = threading.Event()

        def publish_cutoff(cutoff, packs):
            # Every member of a cutoff shares that cutoff's reader collection and
            # coverage spool; one worker owns both, so neither needs a lock.
            counter = DirectCoverage(scratch, geometry, len(direct_by_cut[cutoff]))
            try:
                groups, votes = [], 0
                for index, pack in packs:
                    if failed.is_set():
                        raise _Cancelled()
                    path = destination / f"group_{index:04d}.seg.nrrd"
                    descriptors = [m.descriptor(sweep_id, grid_id, counter) for m in pack]
                    try:
                        stats = write_grouped_seg_nrrd(path, shape_tyx=geometry.shape_tyx, segments=descriptors,
                                                        geometry=geometry, memory_mib=memory_mib, temp_dir=scratch)
                    finally:
                        for m in pack:
                            m.source.close()
                    records = []
                    for member, written in zip(pack, stats["segments"], strict=True):
                        records.append({**member.info, "empty_segment":written["empty_segment"],
                                        "extent_xyz":written["extent_xyz"], "layer":written["layer"],
                                        "foreground_voxels":written["foreground_voxels"]})
                        if member.info["role"] == "direct":
                            votes += written["foreground_voxels"]
                    groups.append({"path":_relative(path, stage), "cutoff_cent":cutoff, "segments":records,
                                   "bytes":stats["bytes"], "stored_shape_tyx":stats["stored_shape_tyx"],
                                   "offset_xyz":stats["offset_xyz"]})
                path = destination / f"conf_{cutoff:03d}_coverage.label.nrrd"
                info = counter.write(path, sweep_id=sweep_id, grid_id=grid_id, cutoff_cent=cutoff,
                                     memory_mib=memory_mib)
                if info["voxel_votes"] != votes:
                    raise RuntimeError(f"Direct coverage count conservation failed for {grid_id}, cutoff {cutoff}")
                overlay = {"cutoff_cent":cutoff, "path":_relative(path, stage),
                           "color_table":_relative(table, stage), "role":"direct_layer_count",
                           "source_segment_ids":direct_by_cut[cutoff], **info}
                return groups, overlay
            except BaseException:
                failed.set()
                raise
            finally:
                counter.close()

        workers = max(1, min(int(workers), len(plan)))
        if workers == 1:
            results = [publish_cutoff(cutoff, packs) for cutoff, packs in plan]
        else:
            with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="xta-review") as pool:
                try:
                    futures = [pool.submit(publish_cutoff, cutoff, packs) for cutoff, packs in plan]
                    for future in futures:
                        future.exception()  # wait; an interrupt here stops further group files
                except BaseException:
                    failed.set()
                    raise
            # The pool has drained. Report the first real failure, not a cancellation.
            errors = [f.exception() for f in futures if f.exception() is not None]
            real = [e for e in errors if not isinstance(e, _Cancelled)]
            if errors:
                raise (real or errors)[0]
            results = [f.result() for f in futures]
    groups = [group for cut_groups, _ in results for group in cut_groups]
    overlays = [overlay for _, overlay in results]
    return _grid_record(grid_id, geometry, GROUPED, origins, groups, overlays)


def _independent_identity(run, cutoff, branch):
    from .config import _confidence_hundredths

    if run.get("launcher", {}).get("mode") != "tta":
        raise ValueError(f"Existing scalar run must be TTA: {branch}")
    config = run.get("resolved_configuration", {})
    recorded = config.get("conf")
    if recorded is None or _confidence_hundredths(str(recorded), "recorded cutoff") != cutoff:
        raise ValueError(f"Recorded scalar cutoff differs from the supplied cutoff: {branch}")
    inputs = run.get("inputs", {})
    source = inputs.get("source") or {}
    digest = source.get("sha256")
    if digest:
        source_id = ("sha256", str(digest).lower())
    else:
        fields = ("path", "size_bytes", "modified_time_ns", "change_or_creation_time_ns", "device", "file_id")
        if any(source.get(k) is None for k in fields):
            raise ValueError(f"Scalar run lacks a recorded source content digest or complete file identity: {branch}")
        source_id = tuple(source[k] for k in fields)
    def valid_digest(value):
        return isinstance(value, str) and len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)
    if digest and not valid_digest(digest):
        raise ValueError(f"Invalid recorded source SHA256: {branch}")
    model_ids = []
    for backend, model in sorted((inputs.get("models") or {}).items()):
        if model is None:
            continue
        artifacts = model.get("artifacts") or []
        if not artifacts or any(not valid_digest(a.get("sha256")) for a in artifacts):
            raise ValueError(f"Scalar run lacks recorded model artifact SHA256s: {branch}")
        model_ids.append((backend, tuple(a["sha256"].lower() for a in artifacts)))
    if not model_ids:
        raise ValueError(f"No model artifact identities in scalar run: {branch}")
    shape = inputs.get("source_shape_t_y_x")
    if not isinstance(shape, list) or len(shape) != 3 or any(type(v) is not int or v <= 0 for v in shape):
        raise ValueError(f"Scalar run lacks the source volume dimensions: {branch}")
    return source_id, tuple(shape), tuple(model_ids)


def publish_sweep_review(output_dir, branches, *, sweep_id, memory_mib=128,
                         source_mode="confidence_sweep", review_name="review", temp_dir=None,
                         workers=None):
    """Atomically publish a new review directory for one multi-cutoff scalar-input family.

    Each (hundredth, branch output directory) must designate a completed scalar
    run below output_dir. With no requested scalar NRRDs there is nothing to package.
    Paths in review/manifest.json are relative to that manifest.
    Seek-heavy coverage spools and intermediate compression use temp_dir or the
    system temporary filesystem. Complete review assets stage beside their destination.

    When a low-quality grid exists, the native grid is published reference-only
    (identities and branch origins, no payload reads); otherwise native is the
    reviewable grouped grid. A grouped grid publishes its cutoffs concurrently
    on up to ``workers`` threads (default: one per cutoff, at most 8, bounded by
    the CPU count). Each worker uses its own ``memory_mib`` slab budget.
    """
    _budget(memory_mib)
    if workers is None:
        from .workspace import _cpu_count
        workers = min(_MAX_WORKERS, max(1, int(_cpu_count())))
    if isinstance(workers, bool) or not isinstance(workers, int) or workers < 1:
        raise ValueError("workers must be a positive integer")
    output_dir = Path(output_dir).resolve()
    if source_mode not in ("confidence_sweep", "independent_scalar"):
        raise ValueError("Unknown sweep publication source mode")
    if (not isinstance(review_name, str) or not review_name.strip() or review_name in (".", "..")
            or "/" in review_name or "\\" in review_name or ":" in review_name):
        raise ValueError("Review name must be a single directory name")
    review = output_dir / review_name
    if not isinstance(sweep_id, str) or not sweep_id or not sweep_id.isascii() or not all(
        c.isalnum() or c in "-_" for c in sweep_id
    ):
        raise ValueError("Invalid sweep ID")
    if review.exists():
        raise FileExistsError(f"Review output already exists: {review}")
    resolved = []
    seen = set()
    common_identity = first_config = None
    different_options = set()
    for cutoff, value in branches:
        if isinstance(cutoff, bool) or not isinstance(cutoff, int) or not 0 <= cutoff <= 100 or cutoff in seen:
            raise ValueError("Sweep cutoffs must be distinct integer hundredths in [0,100]")
        seen.add(cutoff)
        branch = Path(value).resolve()
        if (not branch.is_relative_to(output_dir) or branch == output_dir
                or review.is_relative_to(branch) or branch.is_relative_to(review)):
            raise ValueError(f"Scalar branch lies outside the sweep output: {branch}")
        run_path = branch / "manifest.json"
        if not run_path.resolve().is_relative_to(branch):
            raise ValueError(f"Scalar run manifest escapes its output directory: {run_path}")
        run_data, digest = _read_identity(run_path)
        run = json.loads(run_data)
        identity = run.get("confidence_sweep") or {}
        if run.get("status") != "complete":
            raise ValueError(f"Scalar branch is incomplete: {branch}")
        if source_mode == "confidence_sweep":
            if identity.get("sweep_id") != sweep_id or identity.get("cutoff_hundredths") != cutoff:
                raise ValueError(f"Scalar branch has the wrong sweep/cutoff identity: {branch}")
        else:
            current_identity = _independent_identity(run, cutoff, branch)
            configuration = run["resolved_configuration"]
            if common_identity is None:
                common_identity, first_config = current_identity, configuration
            elif current_identity != common_identity:
                raise ValueError(f"Independent scalar inputs, source shape or model identities differ: {branch}")
            different_options.update(k for k in first_config.keys() | configuration.keys()
                                     if k != "conf" and first_config.get(k) != configuration.get(k))
        run_id = run.get("run_id")
        if "run_id" in run and (not isinstance(run_id, str) or not run_id.strip()):
            raise ValueError(f"Invalid scalar run_id: {branch}")
        resolved.append((cutoff, branch, digest, _manifest_paths(branch), run_id))
    if not resolved:
        raise ValueError("Sweep publication requires at least one cutoff")
    resolved.sort(key=lambda b: b[0])
    grids = set(resolved[0][3])
    if any(set(row[3]) != grids for row in resolved):
        raise ValueError("Scalar branches exported different review grids")
    if not grids:
        if source_mode == "independent_scalar":
            raise ValueError("Independent runs contain no decomposed NRRD manifests to package")
        return None
    if temp_dir is not None:
        temp_dir = Path(temp_dir).expanduser().resolve()
        if temp_dir.is_relative_to(review):
            raise ValueError("The temporary directory must be outside the review being published")
        temp_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xta-review-", dir=output_dir) as staging, \
            tempfile.TemporaryDirectory(prefix="xta-review-work-", dir=temp_dir) as workspace:
        scratch = Path(workspace)
        stage = Path(staging) / "review"
        stage.mkdir()
        reference_native = any(g != "native" for g in grids)
        products = []
        for grid_id in sorted(grids, key=lambda g: (g != "native", g)):
            if grid_id == "native" and reference_native:
                products.append(_reference_grid(grid_id, resolved, review, scratch, sweep_id,
                                                memory_mib, source_mode))
            else:
                products.append(_publish_grid(grid_id, resolved, stage, review, scratch, sweep_id,
                                              memory_mib, source_mode, workers))
        write_json_atomic(stage / "manifest.json", {
            "schema":SCHEMA, "status":"complete", "sweep_id":sweep_id, "source_mode":source_mode,
            "independent_scalar_configuration_differences":sorted(different_options),
            "cutoffs_cent":sorted(seen), "default_cutoff_cent":min(seen), "grids":products,
            "coverage_semantics":"Count of contributing published direct YOLO prediction layers at the selected cutoff; zero denotes no direct layer. Correlated views can contribute.",
            "generated_only_display":"Accent generated/final foreground where the selected cutoff's current direct layer count is zero. Generated and Global layers contribute no votes.",
            "logical_identity":"A logical ID identifies a model/layer export slot across grids and cutoffs. Independently generated geometry in that slot may differ.",
            "grid_payloads":"A grouped grid has stock .seg.nrrd groups, each holding one cutoff, and per-cutoff direct coverage labelmaps. A reference_only grid lists its segment identities and branch origins but no review payload; open its scalar branch NRRDs directly.",
            "source_retention":"Scalar branch NRRDs remain in their original directories. Review groups carry the same binary masks, original colors and per-cutoff identities."
        })
        stage.rename(review)
    return review / "manifest.json"

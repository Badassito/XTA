"""Adapter metadata and edit snapshots for the full-field cutoff editor."""
from __future__ import annotations

import os
from pathlib import Path
import tempfile

from .native_review_store import NativeReviewStore, _file_stamp
from .review_archive import ReviewLayer, write_review_archive


def _edit_source(store):
    source = store.source_record
    return source.get("preview_edit") if isinstance(source, dict) else None


def preview_manifest(store):
    """Build display descriptors without requiring grouped NRRD files."""
    if store.grid_role != "preview":
        raise ValueError("Choose a preview review archive for full-field review")
    if _edit_source(store) is not None:
        raise ValueError("Open the source study to edit layers. This archive is an exported PTA snapshot.")
    if store.global_layer_id is None:
        raise ValueError("A preview review needs a Global layer")
    rows = []
    for lid, layer in store.layers.items():
        for cutoff in layer["cutoffs"]:
            row = dict(layer["segments_by_cutoff"][str(cutoff)])
            row.update(logical_id=lid, cutoff_cent=cutoff, role=layer["role"],
                       segment_name=layer["name"], color_rgb=layer["color"],
                       extent_xyz=layer["extent_by_cutoff"][cutoff],
                       empty_segment=row["foreground_voxels"] == 0)
            if layer["role"] == "global":
                row["suffix"] = "Global_final_output"
            if row.get("layer_role") is None:
                row["layer_role"] = "additive_component" if layer["additive"] else "checkpoint"
            if row.get("recomposition_op") is None:
                row["recomposition_op"] = "union" if layer["additive"] else "select"
            rows.append(row)
    grid = dict(grid_id=store.grid_id, payload="archive", shape_tyx=store.shape,
                native_shape_tyx=store.native_shape_tyx, axial_crop=store.axial_crop,
                groups=[dict(path=store.path.name, segments=rows, bytes=store.file_stamp[2])],
                **store.geometry)
    manifest = dict(schema="xta.sweep_review/1", status="complete", source_mode="confidence_sweep",
                    sweep_id=store.sweep_id, archive_identity=store.identity,
                    cutoffs_cent=store.cutoffs, default_cutoff_cent=store.default_cutoff, grids=[grid])
    exact = (store.source_record or {}).get("confidence_threshold_exact")
    if exact is not None:
        manifest["confidence_threshold_exact"] = exact
    return manifest


def read_preview_manifest(path):
    with NativeReviewStore(path) as store:
        return preview_manifest(store)


def write_preview_edits(directory, source_path, expected_identity, layers, *, edit_revision,
                        stale_cutoffs=(), stale_policy="refuse"):
    """Seal selected Global snapshots without copying unchanged study layers."""
    from .json_publication import _fsync_parent_directory, _publish_sweep_directory

    directory = Path(directory).expanduser().absolute()
    if os.path.lexists(directory):
        raise FileExistsError(f"Preview edit export requires a new directory: {directory}")
    directory.parent.mkdir(parents=True, exist_ok=True)
    with NativeReviewStore(source_path, expected_identity=expected_identity) as store:
        if store.grid_role != "preview" or _edit_source(store) is not None:
            raise ValueError("Export preview edits from a complete editable preview study")
        global_id = store.global_layer_id
        if global_id is None:
            raise ValueError("Preview study has no Global")
        source = dict(store.source_record or {})
        choices = tuple(layers)
        cuts = tuple(layer.metadata["cutoff_cent"] for layer in choices)
        if not cuts or len(set(cuts)) != len(cuts) or not set(cuts).issubset(store.cutoffs):
            raise ValueError("Preview export needs distinct cutoffs from the source study")
        with tempfile.TemporaryDirectory(prefix=".xta-preview-edits-", dir=directory.parent) as work:
            stage = Path(work) / "edited"
            stage.mkdir()
            for cutoff, layer in zip(cuts, choices):
                row = dict(layer.metadata)
                if row["logical_id"] != global_id or row["role"] != "global":
                    raise ValueError("Only the selected edited Global is a PTA snapshot")
                part = stage if len(cuts) == 1 else stage / f"cut_{cutoff:03d}"
                record = dict(source, preview_edit=dict(
                    archive_identity=store.identity, cutoff_cent=cutoff, revision=edit_revision,
                    stale=cutoff in stale_cutoffs, stale_policy=stale_policy,
                ))
                write_review_archive(
                    part / "preview.xtareview", review_id=store.sweep_id, grid_id=store.grid_id,
                    shape_tyx=store.shape, native_shape_tyx=store.native_shape_tyx,
                    geometry=store.geometry, cutoffs_cent=(cutoff,),
                    layers=(ReviewLayer(row, layer.read_slab, layer.bounds_tyx),),
                    source_record=record, axial_crop=store.axial_crop,
                )
            if _file_stamp(store.path) != store.file_stamp:
                raise ValueError("Preview source archive changed while exporting edits")
            _publish_sweep_directory(stage, directory)
            _fsync_parent_directory(directory.parent)
    return directory

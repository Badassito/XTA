"""Cut a Slicer edits bundle or an XTA NRRD and its MKV at a preview marker.

Use the standard axial view with L on the right and the original XTA preview
coordinate frame. The plane passes through the marker, across every row and
frame. Python 3.10+, NumPy, ffmpeg and ffprobe are required.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from zipfile import BadZipFile

if __package__:
    from .prepare_pta_input import preview_cut_columns as cut_columns
    from .xta_cut_edits import inspect_edits, write_edits
    from .xta_cut_nrrd import probe_nrrd, write_nrrd
    from .xta_cut_video import probe_video, write_video
else:
    from prepare_pta_input import preview_cut_columns as cut_columns
    from xta_cut_edits import inspect_edits, write_edits
    from xta_cut_nrrd import probe_nrrd, write_nrrd
    from xta_cut_video import probe_video, write_video


def preview_shape(native_xyz, manifest_path, scale):
    if manifest_path is None:
        if not 0 < scale < 1:
            raise ValueError("Preview scale must be between 0 and 1.")
        # XTA rounds each scaled dimension to the nearest positive multiple of four.
        return tuple(4 * max(1, math.floor(n * scale / 4 + Fraction(1, 2))) for n in native_xyz)

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or manifest.get("quality") != "low_quality":
        raise ValueError("Select the original XTA low-quality NRRD manifest.")
    if manifest.get("exported_axes", "(X, Y, t)") != "(X, Y, t)":
        raise ValueError("Unsupported manifest axis order. Expected (X, Y, t).")
    for key in ("full_quality_output_shape_tyx", "output_shape_tyx"):
        shape = manifest.get(key)
        if not isinstance(shape, list) or len(shape) != 3 or any(type(n) is not int or n < 1 for n in shape):
            raise ValueError(f"The manifest needs three positive integer dimensions in {key}.")
    if tuple(reversed(manifest["full_quality_output_shape_tyx"])) != native_xyz:
        raise ValueError("The MKV dimensions or frame count differ from the manifest's full grid.")
    return tuple(reversed(manifest["output_shape_tyx"]))


def _file_state(path):
    stat = path.stat()
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns


def nrrd_cut(info, requested_grid, native, low, first_native, first_preview):
    shape = (info.width, info.height, info.frames)
    # A cropped segmentation names its full grid; its stored box sits at its origin.
    reference = getattr(info, "reference_xyz", None)
    grids = {"native": (native, first_native), "preview": (low, first_preview)}
    grid = requested_grid
    if grid is None:
        grid = next((name for name, (size, _) in grids.items()
                     if (reference == size if reference is not None
                         else shape == size and info.origin_xyz == (0, 0, 0))), None)
        if grid is None:
            described = f"reference XYZ {reference}" if reference is not None else f"XYZ {shape}"
            raise ValueError(f"NRRD {described} at origin {info.origin_xyz} is not a full native "
                             f"{native} or preview {low} grid. For a known crop, specify --nrrd-grid.")
    size, first_removed = grids[grid]
    if reference is not None and reference != size:
        raise ValueError(f"NRRD reference XYZ {reference} differs from the {grid} grid {size}.")
    if any(start < 0 or start + length > bound for start, length, bound
           in zip(info.origin_xyz, shape, size)):
        raise ValueError(f"NRRD at {info.origin_xyz} with XYZ {shape} extends outside the {grid} grid {size}.")
    return grid, max(0, min(info.width, first_removed - info.origin_xyz[0]))


def run(args):
    if args.edits:
        if args.preview_manifest or args.preview_scale is not None or args.nrrd_grid:
            raise ValueError("--edits reads both grids from the Slicer export. Omit preview and NRRD grid options.")
        if args.mode == "zero":
            raise ValueError("--edits produces a physically narrower bundle and requires crop mode.")
    elif args.preview_manifest is None and args.preview_scale is None:
        raise ValueError("Supply --edits, --preview-manifest, or --preview-scale for the original preview grid.")
    mode = args.mode or ("crop" if args.edits else "zero")
    if not args.dry_run and (not (args.nrrd or args.edits) or args.output_dir is None):
        raise ValueError("Writing requires --edits or --nrrd, and a new --output-dir.")
    target = args.output_dir
    if not args.dry_run and os.path.lexists(target):
        raise FileExistsError(f"Choose a new output directory: {target}")
    if args.edits and target:
        bundle_path = args.edits.resolve()
        output_path = target.resolve()
        if output_path.is_relative_to(bundle_path) or bundle_path.is_relative_to(output_path):
            raise ValueError("The output and source edit bundle directories must be disjoint.")
    inputs = [p.resolve() for p in (args.mkv, args.nrrd, args.preview_manifest) if p]
    if args.edits:
        inputs.extend(args.edits / name for name in ("edits.json", "source_manifest.json"))
    before = {p: _file_state(p) for p in inputs}
    video = probe_video(args.mkv)
    native = (video.width, video.height, video.frames)
    if args.edits:
        bundle = inspect_edits(args.edits)
        if bundle.native_xyz != native:
            raise ValueError(f"The original MKV XYZ {native} differs from the bundle's native grid {bundle.native_xyz}.")
        low = bundle.preview_xyz
        print(f"Edited volume {bundle.stem}; source MKV {args.mkv.name}.", flush=True)
    else:
        low = preview_shape(native, args.preview_manifest, args.preview_scale)
    first_preview, first_native = cut_columns(args.ras, low, native)
    if not 0 < first_native < native[0]:
        raise ValueError("The marker must retain and remove at least one native column.")
    print(f"Original preview XYZ {low}; MKV XYZ {native}.", flush=True)
    print(f"Screen-right (L): preview i >= {first_preview}; native i >= {first_native} "
          f"(zero-based, all rows and frames).", flush=True)
    plan = {
        "mode": mode, "marker_preview_ras": [float(v) for v in args.ras],
        "preview_shape_xyz": low, "native_shape_xyz": native,
        "first_removed_preview_i": first_preview, "first_removed_native_i": first_native,
        "preview_manifest": str(args.preview_manifest.resolve()) if args.preview_manifest else None,
        "preview_scale": float(args.preview_scale) if args.preview_scale else None,
        "mkv_source": str(args.mkv.resolve()),
    }
    if args.nrrd is not None:
        label = probe_nrrd(args.nrrd)
        grid, first_nrrd = nrrd_cut(label, args.nrrd_grid, native, low, first_native, first_preview)
        cropped_segmentation = label.reference_xyz is not None
        # A cropped segmentation beyond the plane is written as Slicer's empty one-voxel image.
        if mode == "crop" and first_nrrd == 0 and not cropped_segmentation:
            raise ValueError("This NRRD lies entirely beyond the plane. Use zero mode for an empty mask.")
        plan.update(nrrd_source=str(args.nrrd.resolve()), nrrd_grid=grid,
                    nrrd_origin_xyz=label.origin_xyz, first_removed_nrrd_i=first_nrrd)
        if cropped_segmentation:
            plan["nrrd_reference_xyz"] = label.reference_xyz
        print(f"NRRD uses the {grid} grid at {label.origin_xyz}; "
              f"retain its first {first_nrrd} of {label.width} columns.", flush=True)
    if args.edits:
        if not 0 < first_preview <= low[0]:
            raise ValueError("The cropped edit bundle needs at least one retained preview column.")
        print(f"Cropped preview XYZ {(first_preview, *low[1:])}; "
              f"cropped MKV XYZ {(first_native, *native[1:])}.", flush=True)
    if args.dry_run:
        return

    reserved = {"cut.json", "edits", "preview"} if args.edits else {"cut.json"}
    if args.mkv.name.casefold() in reserved or (args.nrrd and args.nrrd.name.casefold() in reserved | {args.mkv.name.casefold()}):
        raise ValueError("Input filenames conflict with an output filename.")
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".xta-cut-", dir=target.parent) as tmp:
        staged = Path(tmp) / "pair"
        staged.mkdir()
        if args.edits:
            print("Cropping Slicer snapshots and writing the low-quality Global NRRD...", flush=True)
            preview_dir = staged / "preview"
            preview_dir.mkdir()
            preview_nrrd = preview_dir / (args.mkv.stem + ".nrrd")
            plan["source_edits_directory"] = str(args.edits.resolve())
            plan["preview_nrrd"] = preview_nrrd.relative_to(staged).as_posix()
            plan["edits_output"] = write_edits(args.edits, staged / "edits", preview_nrrd,
                                               bundle, first_preview, first_native, args.ras)
        else:
            print(f"Writing NRRD ({mode})...", flush=True)
            reference_cut = {} if label.reference_xyz is None else {
                "reference_first_removed": first_native if grid == "native" else first_preview}
            plan["nrrd_output"] = write_nrrd(args.nrrd, staged / args.nrrd.name, label, first_nrrd, mode,
                                             **reference_cut)
        print("Encoding lossless MKV...", flush=True)
        plan["mkv_output"] = write_video(args.mkv, staged / args.mkv.name, video, first_native, mode)
        for path, state in before.items():
            if _file_state(path) != state:
                raise ValueError(f"Input changed during processing: {path}")
        (staged / "cut.json").write_text(json.dumps(plan, indent=2) + "\n", encoding="utf-8")
        if os.path.lexists(target):
            raise FileExistsError(f"Output directory already exists: {target}")
        staged.rename(target)
    print(f"Wrote {target.resolve()}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--mkv", type=Path, required=True, help="Original full-resolution FFV1 gray8 MKV")
    label = parser.add_mutually_exclusive_group()
    label.add_argument("--edits", type=Path, help="Slicer export directory containing edits.json and NPZ snapshots")
    label.add_argument("--nrrd", type=Path, help="Scalar XTA NRRD on the native or preview grid")
    parser.add_argument("--nrrd-grid", choices=("preview", "native"),
                        help="Parent grid of a cropped NRRD, e.g. preview for Global_union.seg.nrrd. "
                             "A full raster with zero origin, or a cropped layer that records its "
                             "Segmentation_ReferenceImageExtent, is recognized automatically.")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--preview-manifest", type=Path, help="Original low-quality *_nrrd_manifest.json")
    source.add_argument("--preview-scale", type=Fraction, help="XTA fractional downbin used, e.g. 0.2")
    parser.add_argument("--ras", type=Fraction, nargs=3, required=True, metavar=("R", "A", "S"),
                        help="Marker in the original, untransformed XTA preview coordinate frame")
    parser.add_argument("--mode", choices=("zero", "crop"),
                        help="crop physically shortens X (default for --edits); zero clears samples (default for --nrrd)")
    parser.add_argument("--output-dir", type=Path, help="New directory to receive both files")
    parser.add_argument("--dry-run", action="store_true", help="Print cut columns without writing files")
    args = parser.parse_args(argv)
    try:
        run(args)
    except (ValueError, OSError, subprocess.CalledProcessError, EOFError) as exc:
        parser.exit(1, f"Cut failed: {exc}\n")
    except (BadZipFile, KeyError) as exc:
        parser.exit(1, f"Invalid export bundle: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())

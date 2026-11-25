"""Command-line entry points for paired-volume registration and transfer."""

from __future__ import annotations

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def _build_parser():
    parser = argparse.ArgumentParser(prog="xta register", description=__doc__, allow_abbrev=False)
    cmds = parser.add_subparsers(dest="command", required=True)
    fit = cmds.add_parser(
        "fit", help="Estimate a fixed-to-moving mapping from two grayscale slice stacks"
    )
    for role in ("fixed", "moving"):
        fit.add_argument(
            f"--{role}", required=True, type=Path, help=f"{role.title()} native grayscale movie"
        )
        fit.add_argument(
            f"--{role}-header",
            type=Path,
            help="Reconstruction text header containing Pixel Size and grid dimensions",
        )
        fit.add_argument(
            f"--{role}-spacing",
            type=float,
            nargs=3,
            metavar=("X", "Y", "Z"),
            help="Native voxel sizes in the same length unit for both scans",
        )
    fit.add_argument("--output", required=True, type=Path, help="Destination mapping ZIP")
    fit.add_argument("--pair", help="Human-readable acquisition pair identifier")
    fit.add_argument(
        "--step",
        type=int,
        help="Power-of-two sampling step for the initial fit (default based on volume dimensions)",
    )
    fit.add_argument(
        "--exclude-first",
        type=int,
        nargs="+",
        default=[0],
        metavar="N",
        help="Exclude the first N native frames from field detection and fitting, or give two counts",
    )
    fit.add_argument(
        "--black-exterior",
        action="store_true",
        help="Infer each scan's acquisition field from a black exterior",
    )
    fit.add_argument(
        "--contrast",
        choices=("bright", "both"),
        default="bright",
        help="Fit bright features above the local background, or use signed contrast for bright and dark features",
    )
    fit.add_argument(
        "--affine",
        action="store_true",
        help="Evaluate a full affine refinement with shear and anisotropic scale on held-out patches",
    )
    fit.add_argument(
        "--broad-search",
        action="store_true",
        help="Search additional arbitrary 3D rotations and translations on CPU",
    )
    fit.add_argument(
        "--pyramid-only",
        action="store_true",
        help="Use the validated pyramid fit without additional native patch refinement",
    )
    fit.add_argument("--threads", type=int, default=4)
    fit.add_argument(
        "--work-dir", type=Path, help="Scratch filesystem for temporary decoded native stacks"
    )
    inspect = cmds.add_parser(
        "inspect", help="Print the mapping, acquisition geometry and validation record"
    )
    inspect.add_argument("packet", type=Path)
    video = cmds.add_parser(
        "video", help="Reslice a full grayscale movie onto the other scan's grid"
    )
    review = cmds.add_parser("review", help="Compare every fixed and registered slice in an MP4")
    review.add_argument("--packet", type=Path, required=True)
    review.add_argument("--fixed", type=Path, required=True, help="Original fixed movie")
    review.add_argument(
        "--registered-video", type=Path, required=True, help="Moving movie on the fixed grid"
    )
    review.add_argument(
        "--registered-metadata", type=Path, help="Verification and geometry sidecar"
    )
    review.add_argument("--output", type=Path, required=True)
    review.add_argument("--fixed-window", type=float, nargs=2, metavar=("LOW", "HIGH"))
    review.add_argument("--moving-window", type=float, nargs=2, metavar=("LOW", "HIGH"))
    review.add_argument("--panel-width", type=int, default=480)
    review.add_argument("--fps", type=float, default=30)
    yolo = cmds.add_parser(
        "yolo",
        help="Reslice annotated voxel slabs into partial native-target YOLO labels and exact masks",
    )
    nrrd = cmds.add_parser(
        "nrrd", help="Resample an integer labelmap or Slicer .seg.nrrd using nearest neighbors"
    )
    pairs = cmds.add_parser(
        "pairs",
        help="Write flat, matching PNG and YOLO TXT slices for PTA",
    )
    for cmd in (video, yolo, nrrd, pairs):
        cmd.add_argument("--packet", type=Path, required=True)
        cmd.add_argument("--output", type=Path, required=True)
        cmd.add_argument(
            "--direction",
            choices=("fixed-to-moving", "moving-to-fixed"),
            default="moving-to-fixed" if cmd is video else "fixed-to-moving",
            help="Direction of movie transfer for video, or of label transfer for label commands",
        )
    for cmd in (video, nrrd):
        cmd.add_argument("--input", type=Path, required=True)
        cmd.add_argument("--work-dir", type=Path, default=Path(tempfile.gettempdir()))
    video.add_argument("--workers", type=int, default=4, help="Concurrent output planes")
    video.add_argument("--threads", type=int, default=3, help="FFmpeg threads")
    video.add_argument("--fps", type=float, help="Override output playback rate")
    for cmd in (yolo, pairs):
        cmd.add_argument(
            "--labels",
            type=Path,
            required=True,
            help="Directory of original numerically indexed YOLO slice labels",
        )
        cmd.add_argument(
            "--index-base",
            type=int,
            choices=(0, 1),
            default=1,
            help="Index of the first source movie frame in annotation filenames",
        )
        cmd.add_argument("--workers", type=int, default=2)
    nrrd.add_argument(
        "--source-reference",
        type=Path,
        help="NRRD of the native source scan, stored in the movie's column, row and frame order",
    )
    nrrd.add_argument(
        "--source-grid-aligned",
        action="store_true",
        help="Assert native source index axes, using a Slicer crop offset when present",
    )
    nrrd.add_argument(
        "--target-reference",
        type=Path,
        help="NRRD of the native target scan, in the same storage order, for world origin and directions",
    )
    nrrd.add_argument(
        "--coverage-output",
        type=Path,
        help="Override the destination of the companion observation coverage NRRD",
    )
    nrrd.add_argument(
        "--encoding", choices=("raw", "gzip"), default="gzip", help="Encoding for output NRRDs"
    )
    nrrd.add_argument("--block-rows", type=int, default=128, help="Target rows to resample at once")
    pairs.add_argument(
        "--registered-video",
        type=Path,
        required=True,
        help="Other scan's intensities already resliced onto the original label grid",
    )
    pairs.add_argument(
        "--registered-metadata",
        type=Path,
        help="Override the paired movie verification/geometry sidecar",
    )
    pairs.add_argument(
        "--assume-registered-grid",
        action="store_true",
        help="Assert registered-grid identity for a separately produced movie without a sidecar",
    )
    pairs.add_argument(
        "--flag-first",
        type=int,
        default=0,
        metavar="N",
        help="Flag the first N reference planes in the pair provenance for artifact review",
    )
    return parser


def _execute(args):
    if args.command == "fit":
        from .workflow import fit_movies

        options = vars(args).copy()
        del options["command"]
        return fit_movies(**options)
    from .packet import load_packet

    packet = load_packet(args.packet)
    if args.command == "inspect":
        return packet.metadata
    if args.command == "review":
        from .review import render_review

        return render_review(
            packet,
            args.fixed,
            args.registered_video,
            args.output,
            fixed_window=args.fixed_window,
            moving_window=args.moving_window,
            panel_width=args.panel_width,
            fps=args.fps,
            registered_metadata=args.registered_metadata,
        )
    if args.command == "video":
        from .video import export_registered_video

        return export_registered_video(
            args.input,
            packet,
            args.output,
            args.work_dir,
            direction=args.direction,
            workers=args.workers,
            threads=args.threads,
            fps=args.fps,
        )
    if args.command == "nrrd":
        from .nrrd import transfer_nrrd_labelmap

        return transfer_nrrd_labelmap(
            packet,
            args.input,
            args.output,
            direction=args.direction,
            source_reference=args.source_reference,
            target_reference=args.target_reference,
            source_grid_aligned=args.source_grid_aligned,
            coverage_output=args.coverage_output,
            work_dir=args.work_dir,
            encoding=args.encoding,
            block_rows=args.block_rows,
        )
    if args.command == "yolo":
        from .labels import transfer_yolo_labels

        info = transfer_yolo_labels(
            packet,
            args.labels,
            args.output,
            direction=args.direction,
            filename_index_base=args.index_base,
            workers=args.workers,
        )
        return {
            "output": str(args.output),
            **{
                key: info[key]
                for key in ("positive_target_planes", "positive_class_pixels", "yolo_total_rows")
            },
        }
    from .pairs import export_image_label_pairs

    info = export_image_label_pairs(
        packet,
        args.registered_video,
        args.labels,
        args.output,
        direction=args.direction,
        filename_index_base=args.index_base,
        workers=args.workers,
        registered_metadata=args.registered_metadata,
        assume_registered_grid=args.assume_registered_grid,
        artifact_start_z=args.flag_first,
    )
    return {
        "output": str(args.output),
        **{key: val for key, val in info.items() if isinstance(val, (int, float, bool))},
    }


def main(argv=None):
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        with redirect_stdout(sys.stderr):
            result = _execute(args)
    except (OSError, ValueError, RuntimeError, ImportError, subprocess.SubprocessError) as exc:
        parser.exit(1, f"xta register: {exc}\n")
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.command == "fit" and result["status"] != "validated":
        sys.exit(3)

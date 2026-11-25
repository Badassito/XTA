"""Build the native-resolution backing file for XTA's Slicer review."""
from __future__ import annotations

import argparse
import os
from pathlib import Path
import sqlite3
import sys
import tempfile
import time

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from XTA.native_review_store import prepare_native_review
from XTA.run_transport import TransportError


def prepare_update_both(archive, *, existing_journal=None, progress=None):
    """Publish an initial Update Both sidecar, retaining the original archive."""
    from XTA.native_review_initial import initial_update_path, prepare_initial_update

    archive = Path(archive).expanduser().resolve(strict=True)
    destination = initial_update_path(archive)
    if os.path.lexists(destination):
        raise FileExistsError(destination)
    if existing_journal is not None:
        return prepare_initial_update(archive, existing_journal, destination, trusted_legacy=True)

    from XTA.native_review_edits import NativeReviewEdits
    from XTA.native_review_session import NativeReviewSession
    from XTA.native_review_store import NativeReviewStore

    with (tempfile.TemporaryDirectory(prefix=".xta-initial-update-", dir=archive.parent) as temporary,
          NativeReviewStore(archive) as store):
        journal = Path(temporary) / "update.edits.sqlite"
        with NativeReviewSession(store, journal, initial_update_path=False) as session:
            result = NativeReviewEdits(session).update_both(progress=progress)
        return prepare_initial_update(store, journal, destination, global_voxels=result["global_voxels"])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path,
                        help="Published review/manifest.json or an existing .xtareview for --prepare-update-both")
    parser.add_argument("output", type=Path, nargs="?", help="New .xtareview file when converting a manifest")
    parser.add_argument("--chunk-shape", type=int, nargs=3, metavar=("Z", "Y", "X"),
                        help="Native chunk dimensions. X must be a multiple of 8")
    parser.add_argument("--intensity", type=Path, help="Optional matching source intensity")
    parser.add_argument("--prepare-update-both", action="store_true",
                        help="Prepare the first Update Both offline for a fast, undoable action in Slicer")
    parser.add_argument("--from-existing-journal", type=Path,
                        help="Reuse a trusted journal containing only the initial Update Both of this archive")
    parser.add_argument("--quiet", action="store_true", help="Suppress preparation progress")
    args = parser.parse_args(argv)
    if args.from_existing_journal is not None and (args.output is not None or not args.prepare_update_both):
        parser.error("--from-existing-journal requires an existing .xtareview and --prepare-update-both")
    if args.output is None:
        if args.source.suffix.lower() != ".xtareview" or not args.prepare_update_both:
            parser.error("Supply a new .xtareview output, or pass --prepare-update-both with an existing archive")
        if args.intensity is not None or args.chunk_shape is not None:
            parser.error("--intensity and --chunk-shape apply when creating an archive from a manifest")
    last_message = -float("inf")

    def report(phase, completed, count):
        nonlocal last_message
        now = time.monotonic()
        if phase == "intensity":
            print("Preparing native intensity", file=sys.stderr, flush=True)
        elif phase == "index":
            print("Indexing native mask blocks", file=sys.stderr, flush=True)
        elif phase == "start" or completed == count or now - last_message >= 2:
            print(f"Preparing native masks ({completed}/{count} groups)", file=sys.stderr, flush=True)
            last_message = now

    last_update = (None, None, -float("inf"))
    update_phases = {
        "label": "Finding connected components",
        "write_global": "Writing Global",
        "write_global_and_clip": "Writing Global and clipping layers",
    }

    def report_update(phase, cutoff, blocks):
        nonlocal last_update
        now = time.monotonic()
        if (phase, cutoff) != last_update[:2] or now - last_update[2] >= 2:
            description = update_phases.get(phase, phase.replace("_", " ").capitalize())
            print(f"{description} at cutoff {cutoff / 100:.2f} ({blocks:,} blocks)",
                  file=sys.stderr, flush=True)
            last_update = (phase, cutoff, now)

    try:
        if args.output is None:
            archive = args.source
        else:
            if args.prepare_update_both:
                from XTA.native_review_initial import initial_update_path

                sidecar = initial_update_path(args.output.expanduser().resolve())
                if os.path.lexists(sidecar):
                    raise FileExistsError(sidecar)
            archive = prepare_native_review(args.source, args.output, chunk_shape=args.chunk_shape or (32, 64, 64),
                                            intensity_path=args.intensity, progress=None if args.quiet else report)
            print(archive, flush=True)
        if args.prepare_update_both:
            if not args.quiet:
                notice = ("Checking the supplied initial Update Both journal"
                          if args.from_existing_journal else "Computing initial Update Both")
                print(notice, file=sys.stderr, flush=True)
            prepared = prepare_update_both(
                archive, existing_journal=args.from_existing_journal, progress=None if args.quiet else report_update,
            )
            print(prepared, flush=True)
    except (OSError, KeyError, TypeError, ValueError, EOFError, sqlite3.Error, TransportError) as error:
        parser.exit(2, f"prepare_native_review: {error}\n")


if __name__ == "__main__":
    main()

"""Pack/unpack a lossless single-file run transfer without changing producers."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from XTA.run_transport import pack_run, unpack_run, verify_bundle, verify_external_outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="operation", required=True)
    pack = subcommands.add_parser("pack", help="Keep originals and publish one checked ZIP64 envelope")
    pack.add_argument("source", type=Path, help="Exact run folder; parent folders are never traversed")
    pack.add_argument("destination", type=Path, help="Fresh .zip path outside the run folder")
    pack.add_argument("--scope", choices=("diagnostics", "all"), default="diagnostics",
                      help="Default: diagnostics plus scientific evidence, with public NRRDs separate; all includes NRRDs")
    pack.add_argument("--run-status", choices=("unknown", "in_progress", "failed", "complete"), default="unknown",
                      help="Optional explicit source run state; archive validity is not scientific completion")
    unpack = subcommands.add_parser("unpack", help="Verify then publish a fresh directory; use a short Windows root")
    unpack.add_argument("source", type=Path)
    unpack.add_argument("destination", type=Path, help="Fresh short directory, e.g. C:\\XTA\\r151147")
    verify = subcommands.add_parser("verify", help="Check all member CRCs, SHA-256 and inventory")
    verify.add_argument("source", type=Path)
    verify.add_argument("--outputs-root", type=Path,
                        help="Also check separately transferred NRRDs against the archive's external path/SHA inventory")
    args = parser.parse_args(argv)
    try:
        if args.operation == "pack":
            result = pack_run(args.source, args.destination, scope=args.scope, run_status=args.run_status)
        elif args.operation == "unpack":
            result = unpack_run(args.source, args.destination)
        else:
            result = verify_bundle(args.source)
            external = verify_external_outputs(result, args.outputs_root) if args.outputs_root is not None else None
    except (OSError, ValueError, zipfile.BadZipFile) as error:
        print(json.dumps({"operation": args.operation, "status": "failed",
                          "error_type": type(error).__name__, "error": str(error)}), file=sys.stderr)
        return 2
    summary = {"operation": args.operation, "status": "verified", "scope": result["scope"],
                      "source_run": result["source_run"], "files": len(result["files"]),
                      "external_nrrds": len(result.get("external_outputs", [])),
                      "omitted": len(result["omitted"])}
    if args.operation == "verify" and external is not None:
        summary["external_verification"] = external
        if external["status"] != "verified":
            summary["status"] = "archive_verified_external_outputs_incomplete"
    print(json.dumps(summary, indent=2))
    if args.operation == "verify" and external is not None and external["status"] != "verified":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

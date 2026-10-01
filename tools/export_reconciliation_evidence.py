"""Export matched compact masks and confidence for portable saved-evidence replay."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from XTA.confidence_export import export_compact_evidence


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--run_manifest", type=Path, help="Completed source run's manifest.json")
    source.add_argument("--sam_bundle", type=Path, help="Indexed SAM proposal bundle directory to copy without inference")
    parser.add_argument("--compact_manifest", type=Path, help="Matching low-quality NRRD layer manifest (required with --run_manifest)")
    parser.add_argument("--output", type=Path, required=True, help="Fresh portable-package directory")
    parser.add_argument("--memory_mib", type=int, default=256, help="Crop/plane workspace budget")
    parser.add_argument("--allow_native_projection", action="store_true",
                        help="Explicitly convert one deferred native view at a time on CPU")
    parser.add_argument("--max_staging_mib", type=int, default=32768,
                        help="Maximum disk staging permitted for explicit native-view conversion")
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.sam_bundle is not None:
        if args.compact_manifest is not None:
            parser.error('--compact_manifest does not apply to --sam_bundle')
        if args.allow_native_projection:
            parser.error('--allow_native_projection does not apply to --sam_bundle; SAM evidence retains its view-native coordinates')
        from XTA.sam_evidence import export_sam_evidence
        result = export_sam_evidence(args.sam_bundle, args.output)
        print(json.dumps(result, indent=2))
        return 0
    if args.compact_manifest is None:
        parser.error('--run_manifest requires --compact_manifest')
    last = [0.]
    def progress(stage, current, total):
        now = time.monotonic()
        if stage == "layers" or current == total or now-last[0] >= 15:
            print(f"Compact evidence {stage}: {current}/{total}", flush=True)
            last[0] = now
    result = export_compact_evidence(args.run_manifest, args.compact_manifest, args.output,
        memory_mib=args.memory_mib, allow_native_projection=args.allow_native_projection,
        max_staging_mib=args.max_staging_mib, progress=progress)
    print(json.dumps({"output": str(args.output.resolve()), "layers": result["layer_count"],
                      "confidence_layers": result["confidence_layer_count"], "paths": result["paths"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

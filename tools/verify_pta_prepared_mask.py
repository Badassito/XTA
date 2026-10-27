#!/usr/bin/env python3
"""Stream a prepared gzip NRRD, checking its receipt and sampled cubic planes."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrrd", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    receipt = json.loads(args.receipt.read_text(encoding="utf-8"))
    samples = json.loads(args.samples.read_text(encoding="utf-8"))
    full = tuple(samples["native_shape_tyx"])
    low = tuple(samples["preview_shape_tyx"])
    expected = {r["native_t"]: r for r in samples["planes"]}
    compressed_hash = hashlib.sha256()
    with args.nrrd.open("rb") as source:
        for chunk in iter(lambda: source.read(4 * 1024**2), b""):
            compressed_hash.update(chunk)
    payload_hash = hashlib.sha256()
    count = 0
    plane_results = []
    with args.nrrd.open("rb") as source:
        header = []
        while True:
            line = source.readline()
            if line in (b"\n", b"\r\n"):
                break
            if not line or source.tell() > 65536:
                raise ValueError("Invalid header")
            header.append(line)
        with gzip.GzipFile(fileobj=source, mode="rb") as payload:
            plane_bytes = full[1] * full[2]
            for t in range(full[0]):
                data = payload.read(plane_bytes)
                if len(data) != plane_bytes:
                    raise ValueError("Truncated native plane")
                plane = np.frombuffer(data, np.uint8)
                if np.any(plane > 1):
                    raise ValueError("Nonbinary mask")
                count += int(np.count_nonzero(plane))
                payload_hash.update(data)
                if t in expected:
                    actual_hash = hashlib.sha256(data).hexdigest()
                    plane_results.append({"native_t": t,
                                          "expected_cubic_sha256": expected[t]["cubic_binary_sha256"],
                                          "actual_sha256": actual_hash,
                                          "matches": actual_hash == expected[t]["cubic_binary_sha256"]})
                if t % 200 == 0:
                    print(f"Streamed native plane {t}/{full[0]}", flush=True)
            if payload.read(1):
                raise ValueError("Excess payload")
    # Calculate the full nearest-neighbor count without generating native voxels.
    spec = importlib.util.spec_from_file_location("prepper", samples["prepper"])
    helper = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(helper)
    bundle = Path(samples["bundle"])
    _, _, selected, provenance = helper.load_bundle(bundle, samples["stem"])
    preview = np.zeros(low, np.uint8)
    for entry in selected:
        helper.insert_snapshot(preview, bundle / entry["file"], entry, 1024**3)
    weights = [np.bincount(np.floor((np.arange(f) + .5) * l / f).astype(int), minlength=l)
               for l, f in zip(low, full)]
    nearest_count = sum(int(np.sum(preview[t].astype(np.int64) * weights[1][:, None] * weights[2][None, :])) * int(weights[0][t])
                        for t in range(low[0]))
    scaled_preview_count = int(np.count_nonzero(preview)) * math.prod(full) / math.prod(low)
    receipt_mask = receipt["mask"]
    result = {"method": "One gzip streaming pass with one native plane allocated; full payload hash/count, compressed hash and 24 cubic-plane hashes checked; exact full native nearest count computed using per-preview-cell repetition counts",
              "nrrd": str(args.nrrd.resolve()), "header": b"".join(header).decode("ascii"),
              "receipt": str(args.receipt.resolve()), "receipt_mask": receipt_mask,
              "actual_compressed_sha256": compressed_hash.hexdigest(),
              "actual_payload_sha256": payload_hash.hexdigest(), "actual_foreground_voxels": count,
              "compressed_sha256_matches": compressed_hash.hexdigest() == receipt_mask["sha256"],
              "payload_sha256_matches": payload_hash.hexdigest() == receipt_mask["binary_payload_sha256"],
              "foreground_count_matches": count == receipt_mask["foreground_voxels"],
              "bundle_identity_matches": provenance == receipt["identity"]["bundle"],
              "native_shape_tyx": list(full), "preview_shape_tyx": list(low),
              "full_exact_nearest_foreground_voxels": nearest_count,
              "actual_vs_exact_nearest_net_voxels": count - nearest_count,
              "actual_vs_exact_nearest_percent": 100 * (count / nearest_count - 1),
              "preview_foreground_scaled_by_voxel_volume_ratio": scaled_preview_count,
              "actual_vs_scaled_preview_percent": 100 * (count / scaled_preview_count - 1),
              "sampled_plane_checks": plane_results,
              "sampled_plane_matches": sum(r["matches"] for r in plane_results)}
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in result.items() if k not in ("header", "sampled_plane_checks", "receipt_mask")}), flush=True)
    if not (result["compressed_sha256_matches"] and result["payload_sha256_matches"] and result["foreground_count_matches"]
            and result["bundle_identity_matches"] and all(r["matches"] for r in plane_results)):
        raise AssertionError("Prepared output verification mismatch; inspect JSON")


if __name__ == "__main__":
    main()

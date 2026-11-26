"""Export one observed prediction score as a scalar uint8 NRRD on the XTA review grid."""
from __future__ import annotations

import argparse
from contextlib import nullcontext
import gzip
import json
import math
import os
from pathlib import Path
import sys
import tempfile

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from XTA.confidence_evidence import ConfidenceEvidenceRef
from XTA.confidence_export import EXPORT_SCHEMA, SUPPORTED_EVIDENCE_SCHEMAS, _inside, _shape, retained_evidence_manifest


def export_score(run_manifest, model, layer, output, *, memory_mib=256, max_staging_mib=32768,
                 allow_native_projection=False, temp_dir=None):
    run_manifest, output = Path(run_manifest).resolve(), Path(output).resolve()
    if output.exists():
        raise FileExistsError(f"Score output already exists: {output}")
    for name, value in (("memory_mib", memory_mib), ("max_staging_mib", max_staging_mib)):
        if isinstance(value, bool) or not math.isfinite(float(value)) or value <= 0:
            raise ValueError(f"{name} must be positive and finite")
    run = json.loads(run_manifest.read_bytes())
    if run.get("status") != "complete":
        raise ValueError("Score export requires a completed TTA run or portable compact package")
    catalog_path = retained_evidence_manifest(run_manifest, run=run)
    folder = catalog_path.parent
    catalog = json.loads(catalog_path.read_bytes())
    if catalog.get("schema") not in SUPPORTED_EVIDENCE_SCHEMAS:
        raise ValueError("Unsupported confidence evidence catalog")
    if catalog.get("run_id") != run.get("run_id"):
        raise ValueError("Score catalog and run manifest refer to different runs")
    matches = [entry for entry in catalog["layers"]
               if (entry["model_name"],entry["layer_key"]) == (model,layer)]
    if len(matches) != 1:
        raise ValueError(f"Expected one recorded confidence layer for model {model!r}, layer {layer!r}; found {len(matches)}")
    entry = matches[0]
    ref = ConfidenceEvidenceRef.open(_inside(folder, entry["directory"], "Score directory"))
    compact = run.get("compact_evidence_export")
    if compact is None:
        shape = _shape(run["inputs"]["source_shape_t_y_x"], "run source grid")
        grid = "source"
    else:
        if compact.get("schema") != EXPORT_SCHEMA:
            raise ValueError("Unsupported compact evidence grid")
        shape = _shape(compact["output_shape_tyx"], "portable compact grid")
        grid = "compact_source"
    if ((ref.model_name, ref.layer_key) != (model, layer) or ref.source_shape != shape
            or tuple(entry["output_shape_tyx"]) != shape):
        raise ValueError("Score metadata, selected identity and requested source grid disagree")
    native = ref.coordinate_space != "source"
    if native and not allow_native_projection:
        raise ValueError("Deferred native evidence requires --allow-native-projection for source-grid output")
    # The existing native reader splits its workspace between projection and the output plane.
    # Source-aligned block readers may decode batches of up to 8 MiB in addition to a plane.
    budget = int(float(memory_mib) * 1024**2)
    plane_bytes = math.prod(shape[1:])
    reserve = 512*1024
    block_workspace = 18*1024**2 if ref.metadata.get("schema") == "xta.confidence_evidence/2" else 0
    if budget < reserve + 4*plane_bytes + block_workspace:
        raise ValueError("memory_mib is too small for this score plane and its decoding workspace")
    if temp_dir is not None and native:
        Path(temp_dir).mkdir(parents=True, exist_ok=True)
    provenance = ref.metadata.get("provenance", {})
    record = {"schema":"xta.observed_score_nrrd/1", "model_name":model, "layer_key":layer,
              "run_id":run.get("run_id"), "grid":grid, "stored_coordinate_space":ref.coordinate_space,
              "native_projection_applied":native, "output_shape_tyx":shape,
              "score_semantics":ref.metadata["score_semantics"], "unknown":"score_zero",
              "quantization":ref.metadata.get("quantization", "uint8 score code on a 0..255 scale"),
              "observed_confidence":provenance.get("observed_confidence") or entry.get("observed_confidence")}
    # These are the same canonical unit-LPS geometries used by XTA's source and compact mask writers.
    header = ["NRRD0005","type: uint8","dimension: 3","space: left-posterior-superior",
              "sizes: " + " ".join(str(n) for n in reversed(shape)),
              "space directions: (1,0,0) (0,1,0) (0,0,1)","kinds: domain domain domain",
              "space origin: (0,0,0)","encoding: gzip",
              "xta_observed_score:=" + json.dumps(record, ensure_ascii=True, separators=(",",":"))]
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent, prefix=".xta-score-") as stage:
        staged = Path(stage) / "score.nrrd"
        workspace = tempfile.TemporaryDirectory(dir=temp_dir, prefix="xta-score-projection-") if native else nullcontext(None)
        with workspace as work, ref.source_reader(work, memory_mib=(budget-reserve)/1024**2,
                                                   max_staging_mib=max_staging_mib) as reader:
            records = getattr(reader, "records", None)
            if records and any(int(row[5]) > budget-reserve-3*plane_bytes for row in records):
                raise ValueError("memory_mib is too small for a stored score crop")
            with staged.open("xb") as raw:
                raw.write(("\n".join(header)+"\n\n").encode("ascii"))
                with gzip.GzipFile(filename="",fileobj=raw,mode="wb",compresslevel=3,mtime=0) as compressed:
                    for z in range(shape[0]):
                        scores, known = reader(z,z+1)
                        data = memoryview(scores).cast("B")
                        for first in range(0,len(data),256*1024):
                            compressed.write(data[first:first+256*1024])
                        del data, scores, known
        if os.name == "nt":
            staged.rename(output)   # Windows rename fails if the destination exists.
        else:
            os.link(staged, output)  # Publish a fresh file atomically.
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-manifest", type=Path, required=True,
                        help="Completed run, sweep cutoff or portable compact manifest.json")
    parser.add_argument("--model", required=True, help="Exact model_name in the confidence catalog")
    parser.add_argument("--layer", required=True, help="Exact layer_key in the confidence catalog")
    parser.add_argument("--output", type=Path, required=True, help="Fresh scalar .nrrd output")
    parser.add_argument("--memory-mib", type=int, default=256)
    parser.add_argument("--allow-native-projection", action="store_true", help="Project a deferred native view onto the source grid")
    parser.add_argument("--max-staging-mib", type=int, default=32768)
    parser.add_argument("--temp", type=Path, help="Fast temporary filesystem for projection (default: system temp)")
    a=parser.parse_args(argv)
    record=export_score(a.run_manifest, a.model, a.layer, a.output, memory_mib=a.memory_mib,
                        max_staging_mib=a.max_staging_mib, allow_native_projection=a.allow_native_projection,
                        temp_dir=a.temp)
    print(json.dumps({"output":str(a.output.resolve()), **record},indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())

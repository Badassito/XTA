Transfer one ZIP64 diagnostics/evidence archive while keeping public NRRDs separate:

```powershell
python tools/run_transport.py pack C:/XTA/run C:/XTA/run-diagnostics.zip --run-status failed
python tools/run_transport.py verify C:/XTA/run-diagnostics.zip
python tools/run_transport.py verify C:/XTA/run-diagnostics.zip --outputs-root C:/XTA/run
python tools/run_transport.py unpack C:/XTA/run-diagnostics.zip C:/XTA/restored
```

Choose a fresh archive outside the source folder and a fresh extraction directory.
Use a short extraction root on Windows. The utility uses extended Windows API
paths, but external viewers and copy programs may still require shorter paths.
Original loose files remain available; there is no automatic pipeline archive or
deletion. Stored ZIP members avoid recompressing already compressed payloads.

`diagnostics` is the default scope. It includes telemetry, configuration/logs,
sealed SAM evidence files (`manifest.json`, `index.json`, `masks.bin`),
selection/generation/retry receipts, CVOL component stores and reconciliation
evidence. Scientific files retain their schemas, exact bytes and separate typing;
raw masks are never classified as telemetry. Public NRRD payloads stay outside
the archive and appear in an explicit external path/size/SHA-256 inventory. Copy
those NRRDs separately when needed. Their manifests remain in the archive.
`--scope all` optionally includes public NRRDs too. Unpack scientific evidence
before using its checked loaders. The archive is a transport envelope, never a
scientific qualification.

`verify --outputs-root DIR` additionally checks the separately transferred
NRRD files under their original relative paths. Missing or corrupt NRRDs return
exit code 3 with a separate transfer availability report; source run status is
unchanged. Archive structure/checksum failures return exit code 2.

Future NRRD layer basenames stay identical through 120 characters, including
`.seg.nrrd`. Longer names use a readable prefix and full SHA-256 suffix. Full
logical segment labels, colors and provenance remain in Slicer headers and
manifests, whose `filename` field resolves the actual physical name. Full and
low-quality mirrors share the same basename. Historical filenames remain
readable; this does not rename existing source files.

Source run status defaults to `unknown` unless the source manifest supplies an
explicit state. `--run-status` records a caller declaration. Missing local files
remain unknown; this utility cannot establish cluster-to-local transfer
completeness. Packing a partial run is supported. Source files or inventory that
change during packing abort the operation without publishing the archive.
Snapshot a stopped/quiescent run, or retry a live run after its writers settle.

The packer does not follow links/reparse points or visit source-parent folders.
It excludes only identified launcher-owned `temp` and known private producer
staging directories. Arbitrary `.tmp` files and unrecognized directories remain
in `all`. Safe extraction refuses existing destinations, unsafe paths, links,
duplicate/colliding names and inventory mismatches, and verifies every CRC and
SHA-256 before publishing the restored root. A failed extraction leaves no
completed destination.

Telemetry can be analyzed directly without extraction:

```powershell
python tools/analyze_pipeline_trace.py C:/XTA/run-diagnostics.zip
python tools/lta_trace_summary.py C:/XTA/lta-run.zip
```

Loose JSONL/directory inputs continue to work. Per-process streams, complete
records, measurement frequency, task identities and torn-tail warnings are
preserved. Scientific payloads are not parsed as telemetry.

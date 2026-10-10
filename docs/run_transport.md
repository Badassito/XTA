TTA retains SAM evidence during execution in `OUTPUT/sam-artifacts.tar`.
`sam-artifacts.tar.lock` coordinates writers and reader snapshots; the companion
lock must be writable. Long interpolation and extrapolation
scope paths are logical TAR/PAX members, addressed as
`OUTPUT/sam-artifacts.tar#/sam_interpolation/.../evidence` or
`OUTPUT/sam-artifacts.tar#/sam_extrapolation/.../evidence`.
Evidence bundles, CVOL stores and JSON receipts commit as they become available.
Active builders use short temporary stages; successful stages retire after
publication. Public NRRDs, telemetry and ordinary run manifests remain files
outside this container.

A stopped or interrupted run's SAM container can be transferred as one file.
SAM scientific readers, replay and export tools read its committed members
directly, without extraction. They also accept legacy directory bundles. Member
streams are bounded to their payloads and retain scientific checksum and identity
checks. A torn final append leaves earlier committed transactions readable;
the next writer overwrites from the last committed boundary. Container integrity does not
establish a complete scope or run, and required selection receipts still apply.
After transfer, references use the local container path with the same logical
member path; saved provenance retains its recorded identities.

`run_transport` optionally creates a separate ZIP64 envelope for a stopped run's
diagnostics and evidence, while keeping public NRRDs separate:

```powershell
python tools/run_transport.py pack C:/XTA/run C:/XTA/run-diagnostics.zip --run-status failed
python tools/run_transport.py verify C:/XTA/run-diagnostics.zip
python tools/run_transport.py verify C:/XTA/run-diagnostics.zip --outputs-root C:/XTA/run
python tools/run_transport.py unpack C:/XTA/run-diagnostics.zip C:/XTA/restored
```

Choose a fresh archive outside the source folder and a fresh extraction directory.
Use a short extraction root on Windows. The utility uses extended Windows API
paths, but external viewers and copy programs may still require shorter paths.
Packing preserves the source run files, including its live-format SAM container.
Stored ZIP members avoid recompressing already compressed payloads. This optional
transport envelope is separate from SAM's during-run transactional publication.

`diagnostics` is the default scope. It includes telemetry, configuration/logs,
the SAM container and any legacy loose sealed SAM evidence, selection/generation/
retry receipts, CVOL component stores and reconciliation evidence. Scientific
files retain their schemas, exact bytes and separate typing;
raw masks are never classified as telemetry. Public NRRD payloads stay outside
the archive and appear in an explicit external path/size/SHA-256 inventory. Copy
those NRRDs separately when needed. Their manifests remain in the archive.
`--scope all` optionally includes public NRRDs too. Restoring an outer ZIP writes
the SAM container as one file; scientific APIs then read its members directly.
Legacy loose scientific payloads are restored to their original directory layout.
The transport envelope is never a scientific qualification.

`verify --outputs-root DIR` additionally checks the separately transferred
NRRD files under their original relative paths. Missing or corrupt NRRDs return
exit code 3 with a separate transfer availability report; source run status is
unchanged. Archive structure/checksum failures return exit code 2.

NRRD layer basenames stay identical through 120 characters, including
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

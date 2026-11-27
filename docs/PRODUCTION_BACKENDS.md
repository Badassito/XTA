# Production backend policy

Production bulk CPU operations use the [native CPU extension](native_cpu.md),
built ahead of time. The scalar references in the test suite provide independent
oracles for those kernels.

## Deployment scope

- H100 is the primary production GPU; A100 is the fallback target.
- Volta/V100 is outside the supported deployment scope.
- Development hardware provides correctness checks and local performance sanity
  measurements. Production throughput, model engines and their dependencies
  require target-specific validation.

## Admission principles

Optimized GPU implementations and viable compiled CPU implementations belong in
production. A compiled CPU stage can be useful while GPUs are occupied with
inference. OpenCV, SciPy, NumPy, and PyTorch CPU operators already execute native
code; a function called a reference is not necessarily interpreted or inefficient.

Reference implementations also provide independent test oracles. Their existence
does not require automatic admission for every production workload. Standard
platform wheels include the CPU extension. Source launches require an existing
matching build. A missing or invalid artifact raises an actionable error.
Configuration, CLI help and version discovery remain dependency-light. Compiled
kernel failures propagate; bulk union-find and packed publication do not replay
a partially completed operation through a slow alternate implementation.

Admission must distinguish missing dependencies, unsupported input contracts,
temporary device occupancy, memory limits, and failures after publication. Check
availability before expensive allocations/publication where possible. Retrying
after an unsafe partial publication remains forbidden. Native-array operations
with distinct layout or conversion contracts remain supported.

## Production dispatch

| Operation | Production route | Reference handling |
| --- | --- | --- |
| Union-find batches | Native graph loop | Independent scalar oracle in `tests/reference_backends/topology.py` |
| Topology adjacency | Compiled row runs and bounded hash; native NumPy for distinct legacy numeric contracts | Row-run refusal keeps a compiled hash alternative; no replay after a failed compiled batch |
| Spherical CPU pull | Prepared native FP64 kernel, including confidence and preflight reads | Independent vectorized oracle in `tests/reference_backends/spherical.py` |
| Radial CPU pull | Factored native projection; bounded compiled direct pull when the ownership plan is too large | Independent vectorized oracle in `tests/reference_backends/radial.py` |
| Interpolation candidates | Native scan; workspace grows only after overflow, up to the scanned-window bound | Historical Python planner in `tests/reference_backends/interpolation.py` |
| D1 orthogonal categorical coverage | Bbox-bounded CUDA pull kernel or compiled crop restore over `d1_orthogonal_coverage` address tables | Bounded CPU oracle in `tests/reference_backends/d1_orthogonal_coverage.py` |
| LTA rendering | Shared native-frame cache and tile-window rendering | Full-frame CPU render/restore reference in `tests/reference_backends/lta_rendering.py` |
| Compact relabel, keep-object LUTs, sparse scatter and packed publication | Required compiled kernels | Differential tests preserve independent dense/array references |
| CPU slice labeling and layout-specific operations | OpenCV CCL or vectorized native-array work | These remain supported native CPU algorithms, not interpreted voxel-loop fallbacks |

Compiled interpolation workspace retries preserve valid fragmented input without
truncating candidates or switching backends. The initial budget does not allocate
for an arbitrarily large requested candidate count. Radial plan refusal keeps a
bounded compiled route instead of materializing a full coordinate map or running
the old single-worker NumPy reference.

Slice-local topology stores prefer `uint16` while each slice fits its 65,535
foreground-component capacity. If a slice exceeds that capacity, the complete
unpublished label workspace is discarded after its workers settle and the pass
automatically restarts with `uint32`; memory/disk admission is recomputed.
Counts, areas, root lookup tables and adjacent-slice connectivity are rebuilt
from the original mask. This promotion never reuses narrowed partial labels or
changes a shared environment setting. Valid slice-local overflow needs no manual
rerun. Compact global labels already use `uint32`. Only a verified capacity
signal triggers promotion; unrelated failures retain their established backend
error contracts.

The `YOLO_TTA_TOPOLOGY_COMPILED_KERNELS`, `YOLO_TTA_INTERPOLATION_COMPILED_KERNELS`,
`YOLO_TTA_CPU_SPHERICAL_COMPILED`, `YOLO_TTA_NRRD_PARALLEL_MIRROR_ENCODE`,
`YOLO_TTA_NRRD_PARALLEL_EXTENT_SCAN`, `YOLO_TTA_TOPOLOGY_SPARSE_CPU_LABELING`,
`YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16`, and
`YOLO_TTA_INTERPOLATION_CACHE_BRIDGE_SECTIONS` opt-outs are retired. The
compiled Spherical CPU path is independent of approximate GPU geometry
preferences. Compact-relabel thread-pool tuning
(`YOLO_TTA_INTERPOLATION_COMPACT_WORKERS` and
`YOLO_TTA_INTERPOLATION_COMPACT_RELABEL_ROWS`) is also retired with that duplicate
implementation. Native NumPy/OpenCV paths required for aliasing, dtype conversion,
explicit CPU inference, or different algorithms retain their separate contracts.

`YOLO_TTA_V1613_D1_OWNER_PIPELINE` is retired; `YOLO_TTA_V1613_FAST_BUNDLE=0` is
the single D1 backout. Unlike the retired switch, which turned off only the D1
pipeline, it turns off the whole fast bundle. A TTA run with
`--enable_2D_hole_fill yolo` that should match the retired switch's D1-off masks
needs `YOLO_TTA_PROTO_HOLE_TREATMENT=close` and `YOLO_TTA_PROTO_HOLE_RADIUS=2`
to restore the bundle's default resident-proto closing. That recipe is necessary
but not sufficient for exact parity. The retired switch's D1-off runs also let
a completed dense view borrow an idle worker GPU for backprojection while other
devices still inferred; no current mode does that. `YOLO_TTA_V1613_FAST_BUNDLE=0`
also lowers the default GPU union retirement lanes from 3 to 2
(`YOLO_TTA_GPU_UNION_RETIREMENT_LANES=3` restores them).

## Reproducible CPU comparisons

`tools/benchmark_cpu_backends.py` has two explicit modes:

```powershell
python -B tools/benchmark_cpu_backends.py --mode check --workload smoke `
  --output-dir ../Scratch/Experiments/REVIEW_NAME/backend-check

python -B tools/benchmark_cpu_backends.py --mode benchmark --workload scaled `
  --threads 1 --heatsoak-seconds 60 --repeats 5 `
  --output-dir ../Scratch/Experiments/REVIEW_NAME/backend-benchmark
```

Run these development tools from the complete source checkout or source
distribution, which include `tests/reference_backends`. The installed production
wheel excludes the test oracles. Use a fresh output directory. Default mode is
`check`; it makes no performance claim. Timing mode runs each implementation before sampling, heatsoaks the CPU, and
alternates
trial order, and records individual wall/process times and numerical comparisons.
Compiled cases must execute a compiled implementation or report failure; they do
not silently time the reference under a compiled label. All outputs and caches
belong under Scratch. GPU work is excluded from this CPU comparison.

Fixtures include a bounded native-coordinate Spherical plane, coherent and
fragmented adjacency, constrained hash spills, graph unions, and interpolation
candidate searches. They do not cover full pipeline wall time, production model
quality, build/setup costs, or process-wide peak memory. Benchmark receipts
identify source hashes, work sizes, and shared setup/canonicalization costs.

## GIL-free CPU hot spots

The TTA main process runs many threads at once. A loop of short NumPy, zlib,
file or hashing calls releases and re-takes the GIL on every call, and each
re-take can wait a full switch interval (5 ms) behind busy threads. The paths
below release the GIL around batches of numerical work or file reads.
Each one matches the previous result byte for byte.

- NRRD layer producer (`XTA/nrrd_sparse_kernels.py`): fused batch kernels.
  `YOLO_TTA_NRRD_PRODUCER_BATCH_MIB` (default 64) sets the per-thread batch
  buffer; 16 lowers peak memory at some speed cost.
- Azimuthal angular key plane (`XTA/projection_coverage_cpu.py`).
- SAM parent score restore (`XTA/confidence_native_inflate.py`): native inflate
  and scatter. `YOLO_TTA_CONFIDENCE_NATIVE_INFLATE=off` restores the Python path.
  On Windows it stays off unless the qualified native zlib is configured, as for
  the encoder.
- Encoded block publication (`consume_encoded_block` in `XTA/interpolation.py`).
- Terminal score masking (`XTA/confidence_capture_cpu.py`).
- RAM headroom (`XTA/workspace.py`, `native/cpu_memory.cpp`). On Linux each
  sample reads meminfo and the applicable cgroup limits, usage and reclaimable
  pages in order, with the GIL released during I/O. Process membership and
  canonical mount proofs are cached per PID. Every admission reads current
  pressure. An unreadable usage counter under a finite limit grants zero
  headroom. Windows retains its physical-memory query through psutil.
- SAM cyclic, filter, reader, branch and policy modules hash their own source
  once at import; no address, mask or selection call re-reads a source file.

Numerical calls load the same ahead-of-time extension within each process.
No CPU compilation or specialization runs during the pipeline.

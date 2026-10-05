# Production backend policy

Production numerical modules require Numba; slow scalar
reference implementations live in the test suite instead of serving as automatic
recovery from a missing or failed compiler.

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
does not require automatic admission for every production workload. Numba is a
base dependency (`numba>=0.61.2`); llvmlite is supplied through Numba.
Missing or broken imports raise an actionable error with the original cause.
`NUMBA_DISABLE_JIT=1` is rejected so compiled kernels cannot quietly become
interpreted loops.
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
| Union-find batches | Required Numba graph loop | Independent scalar oracle in `tests/reference_backends/topology.py` |
| Topology adjacency | Compiled row runs and bounded hash; native NumPy for distinct legacy numeric contracts | Row-run refusal keeps a compiled hash alternative; no replay after a failed compiled batch |
| Spherical CPU pull | Required prepared Numba FP64 kernel, including confidence and preflight reads | Independent vectorized oracle in `tests/reference_backends/spherical.py` |
| Radial CPU pull | Factored Numba projection; bounded compiled direct pull when the ownership plan is too large | Independent vectorized oracle in `tests/reference_backends/radial.py` |
| Interpolation candidates | Required Numba scan; workspace grows only after overflow, up to the scanned-window bound | Historical Python planner in `tests/reference_backends/interpolation.py` |
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
changes a shared environment setting. Set
`YOLO_TTA_INTERPOLATION_LOCAL_LABEL_UINT16=0` to select `uint32` immediately;
manual reruns are no longer required for valid slice-local overflow. Compact
global labels already use `uint32`. Only a verified capacity signal triggers
promotion; unrelated failures retain their established backend error contracts.

The `YOLO_TTA_TOPOLOGY_COMPILED_KERNELS`,
`YOLO_TTA_INTERPOLATION_COMPILED_KERNELS`, and
`YOLO_TTA_CPU_SPHERICAL_COMPILED` opt-outs are retired. The compiled Spherical CPU
path is independent of approximate GPU geometry preferences. Compact-relabel
thread-pool tuning (`YOLO_TTA_INTERPOLATION_COMPACT_WORKERS` and
`YOLO_TTA_INTERPOLATION_COMPACT_RELABEL_ROWS`) is also retired with that duplicate
implementation. Native NumPy/OpenCV paths required for aliasing, dtype conversion,
explicit CPU inference, or different algorithms retain their separate contracts.

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
`check`; it makes no performance claim. Timing mode warms the real implementations,
heatsoaks the CPU, alternates
trial order, and records individual wall/process times and numerical comparisons.
Compiled cases must execute a compiled implementation or report failure; they do
not silently time the reference under a compiled label. All outputs and caches
belong under Scratch. GPU work is excluded from this CPU comparison.

Fixtures include a bounded native-coordinate Spherical plane, coherent and
fragmented adjacency, constrained hash spills, graph unions, and interpolation
candidate searches. They do not cover full pipeline wall time, production model
quality, cold compiler/setup costs, or process-wide peak memory. Benchmark receipts
identify source hashes, work sizes, and shared setup/canonicalization costs.

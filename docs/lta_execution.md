# LTA execution ownership

Production LTA uses native transverse slices at `--angle 0`. Each positive
annotation tracks within the interval bounded by the nearest annotated slice
on each side. With two or more annotated slices, tracking starts at the
first annotation and ends at the last. A lone positive annotation can
seed tracking toward both ends of the scan. An explicit empty YOLO label
is a hard background boundary. Missing labels leave slices unannotated.

On unannotated slices between two positive annotations, LTA blends
matching structures with center-aligned signed distances in the region
tracked from both directions. Predictions outside their shared region are
retained. Every checkpoint and the final native mask restore entire annotated
planes, including background pixels. The manifest records the half-open
tracking intervals in `execution.temporal_propagation`.

The run manifest describes the next annotation under
`execution.annotation_suggestions`. Within the pixels examined by both
directions on an unannotated slice, it ranks disagreement by the fraction of
their predicted foreground union where they differ. It compares slices
with foreground in that shared region. Equal scores choose the lower native
slice index. `next_slice` gives the zero-based transverse `slice_index`,
the source `encoded_index` when available, and `exemplar_encoded_index`
using the configured `--exemplar_index_origin`. It is null if no eligible
plane has a nonzero disagreement.

The tiled production path in `XTA/lta_execution.py` runs the initial bounded
window graph, then continues through the canonical temporal frontier in
`XTA/lta_frontier_execution.py`. The dynamic crop backend retains its own
sealed predecessor scheduling in `XTA/lta_dynamic_execution.py`.

The tiled relay frontier is a coordinator-local SQLite database. It starts in
memory when the live host and cgroup budget allow it. The database is limited
to a 16 MiB default or one sixty-fourth of available allocation headroom,
whichever is smaller. A separate bounded retry buffer holds packed seed offers.
An oversized batch or a full database moves the committed state to the run's
scratch directory and continues there. Set `XTA_LTA_FRONTIER_MEMORY_MIB=0` to
start on the filesystem selected by `--temp`. Frontier diagnostics record the
active backing and database size.

`XTA/lta_window_execution.py` owns the shared window graph execution: worker
dispatch and leases, reduction of directional proposals, verified dogfood
continuations, coverage ledger and snapshot lifetimes, and ordered audit and
relay commits. Proposals within a direction can be accumulated in completion
order. Opposing directions are combined after the frontier settles. Audit and
relay acknowledgment follow plan order. A continuation receives the settled graph
state and either supplies the next generation or finishes the run. Hole filling
remains disabled unless the run explicitly enables it.

The older generation admission and monotone relay mask revision logic lives
in `tests/reference_backends/lta_generation.py`. Tests and developer comparison
tools select that backend explicitly. Production does not import it. It uses
the same window engine, worker adapter, and artifact validation as production,
so it is a reference for relay admission rather than an independent window
execution implementation. These comparisons require a complete source checkout;
the installed runtime package does not include the `tests` package.

`tools/lta_window_gpu_smoke.py` compares whole-chain and window graph reference
execution. `tools/lta_relay_frontier_smoke.py` can replay a selected source tree
with the reference backend or, with `--canonical-frontier`, production. Use
immutable cache and seed fixtures, retain their hashes and exact source identity,
and compare native unions, authoritative seed retention, and per-object evidence.
Package telemetry fingerprints cover package Python sources; reference backend
bytes require a separate identity in comparison evidence. These bounded
functional checks make no throughput or model-quality claim. Reserve
`Scratch/Temp/GPU_LOCK` before starting the GPU replay and keep it until the
workers have stopped.

# LTA execution ownership

The tiled production path in `XTA/lta_execution.py` runs the initial bounded
window graph, then continues through the canonical temporal frontier in
`XTA/lta_frontier_execution.py`. The dynamic crop backend retains its own
sealed predecessor scheduling in `XTA/lta_dynamic_execution.py`.

`XTA/lta_window_execution.py` owns the shared window graph execution: worker
dispatch and leases, immediate native union reduction, verified dogfood
continuations, coverage ledger and snapshot lifetimes, and ordered audit and
relay commits. Native union OR may follow completion order; audit and relay
acknowledgment follow plan order. A continuation receives the settled graph
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

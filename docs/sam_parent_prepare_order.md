# SAM parent preparation order

The bounded parent queue mixes long input preparation with older suppliers.
Among expensive parents, one successful admission selects the largest recorded
native dense-input footprint, followed by three successful admissions in
immutable arrival order. Equal sizes use arrival order. The footprint is a
restore/scan cost proxy, not a prediction of total SAM runtime or a memory grant.
It uses the required bytes already recorded when the completed detector input
enters the queue; it does not read or modify image/mask data to set priority.

Orthogonal and tilted parents keep their existing protected cheap lanes, dense
capacity, and transient protections. They do not advance the expensive turn.
The configured parent worker limit is unchanged: a task still owns its lane
through preparation, retries, selection, and publication. Existing dense,
startup, physical-memory, transient, and cancellation checks remain in force.

A failed fit or lock probe neither advances the turn nor blocks later fitting
work. Scan progress persists across bounded scheduler passes independently of
arrival age, with a frozen scan cohort so newly arriving parents cannot hide a
fitting parent across short budget slices. Cheap and expensive candidates share
denial progress: a failed cheap probe keeps its resource protection but yields
the next probe to an untried candidate. Every parent is probed at most once per
pass. Successful expensive
handoffs reset the scan and advance the turn; all finite, continuously fitting
older parents retain regular FIFO opportunities, including under new arrivals.
This changes ordering only, with no new executor, speculative scientific work,
or task inventory change.

Terminal cancellation clears queue ordinals, scan cohorts, denial progress and
policy cursors; the existing resource cancellation and ownership cleanup still
settle the actual work.

Each successful handoff emits one `sam_parent_prepare_admitted` event containing
the policy turn, cost, immutable queue ordinal/age, age and size ranks, worker
occupancy, and fixed-size denial counters. Aggregate denial counters also appear
in the staging gauge. The records do not serialize queued owners or arrays.
Abort clears queue-local policy counters as well as pending identifiers; use the
emitted admission events and earlier snapshots for an aborted run's history.
Evaluate throughput and tail latency on a real run with identical task inventory;
fixed-duration scheduling replay cannot predict the resulting contention or
walltime.

When all configured parent workers are occupied, candidate scanning pauses before
choosing a parent or changing its priority, scan progress, or cheap-lane turn.
The aggregate `worker_limit` counter counts these blocked checks; they are not
candidate fit denials. Cancellation cleanup and invalid bounded-request errors
remain independent of worker availability. Candidate-specific memory, protected
cheap-lane and condition denials still permit bounded backfill.
Earlier versions counted each candidate visited while full, so their raw
`worker_limit` totals are not directly comparable with the new blocked-check
totals.

Admission records flatten the five candidate denial counters as
`candidate_denials_<reason>` and the five aggregate counters as
`denial_totals_<reason>`. All 24 fields are scalar and fit the runtime trace's
32-field limit; nested counter dictionaries would be discarded by that writer.

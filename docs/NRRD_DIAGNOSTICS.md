# Live NRRD member-writer diagnostics

The parent telemetry gauge `nrrd.member_stream.live` samples active member writers on the existing `RuntimeSystemSampler` cadence. The default is five seconds; `YOLO_TTA_TELEMETRY_SAMPLE_SECONDS` controls it with a one-second minimum. No extra sampler thread is created. `sample_monotonic_ns` marks the actual diagnostic sample. Other telemetry flushes can repeat the same gauge, so group or filter records by this timestamp before plotting.

`active_writers` counts open member writers and `waiting_writers` counts those inside a blocking `Future` wait at sample time. `waiting_by_cause` splits that live count among `zero_descriptor`, `window`, and `close`. `oldest_wait_seconds` is the longest current wait. `completed_wait_seconds_process_lifetime` advances when waits end, before a writer closes; it is a separate process-lifetime gauge, not a replacement for the existing aggregate counter.

`oldest_head_age_seconds` and `oldest_head` describe the oldest submitted compression future at a blocked writer's ordered output head. The bounded descriptor includes the layer basename, member sequence, wait cause, age, pending-future count, ready-descriptor count, and charged in-flight bytes. Counts are captured on entry to the current wait. The `future_stage` values are:

- `queued`: the executor future has not started.
- `running`: the executor task has started. This does not prove it is using CPU at that instant; it may be waiting inside a codec or the runtime.
- `done`: the future finished before the sampler observed the writer processing it.
- `unknown`: no matching future is available for the ordered head, for example when a test injects a future or a ready prefix is being processed.

The existing `nrrd.member_stream.pending_wait_seconds`, `zero_descriptor_wait_seconds`, `close_wait_seconds`, and related counters retain their prior contract. Each writer publishes those accumulated values when it closes, so their increase in a telemetry window does **not** locate wait onset or measure concurrent blocked writers. Use the live gauge for concurrency and queue-head age; use the close-published counters for exact completed-writer totals. Short waits between sampler ticks may appear only in the latter totals.

Sampling failures are isolated from NRRD output. A final provider sample runs after the system sampler stops and before final telemetry flush. Disabling telemetry or `YOLO_TTA_TELEMETRY_SYSTEM_SAMPLER` disables the live sampler path. These diagnostics do not change sink-worker count, the 128 ready-descriptor cap, compression, or output ordering.

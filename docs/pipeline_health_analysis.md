# Offline pipeline capture analysis

Use `tools/analyze_pipeline_health.py` for reproducible summaries and comparisons.
It does not import a GPU runtime or change a running job. Put generated files in
the sibling `Scratch/Experiments/<task>` directory. Outputs inside the repository
or an input directory are rejected; extraction also refuses an existing output
directory. Keep original captures unchanged.

## Reuse a compact extract

```powershell
python -B tools/analyze_pipeline_health.py --compact-dir ../Scratch/Experiments/run157592-review-20261017/shared --output ../Scratch/Experiments/log-analysis-tools-20261017/run157592.json
python -B tools/analyze_pipeline_health.py --compact-dir ../Scratch/Experiments/run157592-review-20261017/shared --compare-compact-dir ../Scratch/Experiments/run157581-review-20261016/final/shared --output ../Scratch/Experiments/log-analysis-tools-20261017/comparison.json
```

`--until-seconds 600` limits the analyzed prefix. Comparisons always use the
shorter captured endpoint, optionally further limited by this argument. They
retain each full-capture summary separately. Future phase completions, grants,
and counters are excluded from prefix calculations. Full-capture outcome and
source metadata remain clearly labeled as full-capture context.

Supported compact inputs contain `stream-summary.json`, `time-origin.json`,
`gpu-sam-samples.jsonl`, and `events.jsonl`. Optional counter histories, final
telemetry record, transport inventory, stdout and execution provenance add
coverage. Existing shared extracts are labeled `legacy.shared-extract/1`;
missing fields remain unknown. An unfinished extraction is rejected.

## Create a compact extract once

Select the parent process explicitly; do not combine worker and parent clocks.

```powershell
python -B tools/analyze_pipeline_health.py ../Scratch/Experiments/Results/RUN.ready/telemetry-RUN-PARENT.jsonl --stdout ../Scratch/Experiments/Results/RUN.ready/RUN.txt --extract-dir ../Scratch/Experiments/log-analysis-tools-20261017/RUN-compact --output ../Scratch/Experiments/log-analysis-tools-20261017/RUN.json
```

Extraction reads one uncompressed JSONL's initial byte prefix and hashes those
bytes in that same pass. Appended bytes are excluded. It retains source size,
hash, parse warnings and before/after stability. Interrupted output has partial
files and no completed summary marker. The completion marker is atomically
published last. Reports then read the compact files, never rescan the raw input.
Archive validation and safe extraction remain the responsibility of the
[existing transport tool](run_transport.md).

## Interpretation

The summary method is `xta.pipeline-capture-summary/cap15-v1`. Each unique GPU
acquisition holds until the earliest of the next acquisition, its own capture
endpoint, and 15 seconds after acquisition. The interval is then clipped to the
requested window. There is no extrapolation before the first sample. The final
sample has the same cap. Reports include covered and unknown seconds; GPU
utilization and memory each have independent coverage. Fleet means require a
complete, consistent utilization vector. Device indices refer to the recorded
physical inventory, not an assumed CUDA ordinal mapping.

Queue classes require integer counts and a SAM sample aligned within 0.1 seconds
of GPU acquisition. Capacity comes from recorded slots or that same snapshot's
running plus free slots. `full_ready` means all recorded slots are running and
ready work exists. Empty, underfilled, unknown-capacity and unaligned intervals
remain separate. These conditional intervals do not match identical jobs across
runs and do not establish causality.

Transport stability, extraction completion, telemetry closure and pipeline
outcome are different fields. A `final=true` telemetry record is not proof of
success. Explicit incomplete phase/requested-work evidence overrides a printed
walltime. Printed pipeline walltime is retained separately from the telemetry
endpoint; neither substitutes for the other.

Phase durations may overlap across scopes. Thread CPU excludes helper threads
and waits. Admission summaries preserve event counts, reasons, funding field
observations, first grants and per-lease ownership. An admitted event's reported
wait remains usable without its request event. An unmatched request is unresolved;
time since request is right-censored and does not prove it is still blocked.
It is excluded from completed-wait medians. Owner expiry is not credit return.
When all fields exist, promised funding subtracts parent-funded image bytes to
avoid counting that reservation twice. Missing fields remain unavailable.

Retry attempt-bound rows are diagnostic scalar maxima across possibly different
scopes. They are not additive peak measurements or allocation certificates.
Unknown-owner flags and schema version are retained. An absent row does not
prove that no retry happened. Counter resets suppress affected deltas rather
than subtracting across a reset.

The original raw-stream invocation remains available for legacy health attention
windows and CUDA/NVML owner attribution. Its existing API and attention rules are
unchanged; it intentionally has different final-sample/window semantics from the
versioned capture summary. Do not compare their aggregate values as one method.

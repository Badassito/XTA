# TTA confidence sweeps

The `--conf` range form selects multiple operating points in one TTA invocation.
For example, run this on a GPU with an instance model:

```text
python -m XTA --mode tta --input sample.mkv --model gpu:best.pt --device 0 --enable_cartesian transverse sagittal coronal --conf 0.50:0.10:1.00 --interpolation_backend sdf --interpolation_distance 30 --save nrrd low_quality:0.20 summary --output sample_sweep
```

The form is `START:STEP:END`. Use values from 0 through 1 on the hundredth
grid and a positive step that lands on the end. Both endpoints are included.
The example computes 0.50, 0.60, 0.70, 0.80, 0.90 and 1.00. The single-value
form retains its original precision and defaults. In a range, the secondary
gate `--min_conf` defaults to zero. If supplied explicitly, the same secondary
gate applies to every branch. For comparisons with existing scalar runs,
specify the same gate explicitly in those runs.

For `--task segment`, the primary cutoff applies to detection confidence. For
`--task semantic`, it applies to foreground probability. Each backend uses its
existing scalar comparison and mask decoding. Each cutoff has its own cleaned
masks, SDF or SAM interpolation decisions, extrapolation, reconciliation and
finalization. Ordinary SAM model, device and interpolation options still apply.
An empty result at a high cutoff is a valid computed branch.

## Compute and output

For eligible ranges the lowest-cutoff child dispatches **view batches**. The
worker renders and preprocesses the batch once, evaluates the model once, and
uses the live outputs to produce each cutoff's native mask and any requested observed scores.
The scalar writer already owns the lowest-cutoff pixels, so its entry in the
shared journal carries frame counts only. Higher cutoffs retain their exact
mask crops and any requested score crops and can share a stored crop when the decoder proves the
accepted predictions are identical. For a
long grid, the tiled direct CUDA decoder can group candidates by score and
accumulate prototype maxima from higher to lower thresholds. Small grids and
the TensorRT ring use their native cutoff decoding on the shared forward.

With NRRD decomposition and interpolation/extrapolation off, the normal scalar
route handles the lowest cutoff. Compatible higher-cutoff masks from the same
view share a categorical geometry traversal. The publisher stages layers
on family scratch as views finish. The final native and preview NRRDs are
grouped by cutoff. Masks which cross between cutoffs can require more than one
geometry traversal. Radial sibling traversals evaluate only the output voxels
their shell patch and native boxes can reach; the result is identical to the
full traversal. Projected sibling stores use row-packed bits on scratch.

Publication runs beside inference, so it must keep pace with the GPUs.
Sibling rank projection uses the GPU when a device can lend a shared slice
beside inference (radial, tilted, azimuthal and spherical views), and the
CPU otherwise. Both routes produce identical outputs. Several publishers each
own one view at a time. The scheduler waits only when the native journal
nears its quota or the queue reaches its safety cap:

| Variable | Default | Meaning |
|---|---|---|
| `XTA_YOLO_SWEEP_PUBLISHERS` | visible CPUs / 8, 2 to 16 | concurrent view publications |
| `XTA_YOLO_SWEEP_PROJECTION_LANES` | CPUs / (2 x publishers), 1 to 8 | CPU projection or encoding lanes per publication |
| `XTA_YOLO_SWEEP_JOURNAL_HIGH_WATER` | 0.75 | journal fraction at which the scheduler waits for publication |
| `XTA_YOLO_SWEEP_PUBLISH_BACKLOG` | 16 x publishers, at least 64 | safety cap on queued views |
| `XTA_YOLO_SWEEP_RANK_GPU` | 1 | 0 restores the historical CUDA admission for sibling projection (no shared lease beside inference, no wait policy) |
| `XTA_YOLO_SWEEP_RANK_GPU_OVERLAP_GIB` | the ordinary slice | largest GPU slice a sibling projection may lease beside inference |
| `XTA_YOLO_SWEEP_RANK_GPU_HEADROOM_GIB` | 8 | extra free device memory required for an opted-in slice above the ordinary one |
| `XTA_YOLO_SWEEP_RANK_GPU_WAIT_SECONDS` | 60 | how long a busy GPU is retried for azimuthal views before the CPU route; after one full-length timeout later views wait at most 2 s until a slot is admitted again |
| `XTA_YOLO_SWEEP_RANK_GPU_RADIAL_WAIT_SECONDS` | 15 | the same for radial views, whose bounded CPU route is short |
| `XTA_YOLO_SWEEP_RANK_GPU_TILTED` | 0 | 1 lets tilted views wait for the GPU; by default they use the GPU only when a slot is free, since their row-bounded CPU route is cheap |
| `XTA_YOLO_SWEEP_EVIDENCE_PROCESSES` | visible CPUs / 8, at least 4, at most the evidence layers publications can queue | worker processes that write sibling confidence evidence; 0 keeps it on producer threads |
| `XTA_YOLO_SWEEP_NRRD_PROCESSES` | visible CPUs / 6, at least 4, at most the sibling layers publications can queue | worker processes that write sibling native and preview `.seg.nrrd` files; 0 keeps them on producer threads |
| `XTA_YOLO_SWEEP_NATIVE_DECODE` | native when qualified | `python` decodes sibling journal rows with Python's zlib instead of the nogil kernels |

The ordinary shared slice (`YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB`) is 4 GiB, or
6 GiB in a sweep producer so the largest sibling projections fit inside the
memory that inference sizing already leaves free. Floor projections keep
priority over sibling projections for those slices.

Each count accepts a positive integer or `auto`. A GPU-projected view packs its
planes on up to 4 host threads. The scheduler also keeps going when the bytes
above the journal mark could not be freed by the queued publications (workers
then take the exact scalar route if the quota fills). Sibling NRRD encoding uses one
shared pool sized to what concurrent publications can queue (at most a third
of the CPUs). Requested observed evidence is written beside those encodes.
The family `manifest.json` reports `upstream_fanout.early_publication` with
the backlog peak, the journal's peak fill and the seconds the scheduler waited.
A nonzero wait means publication, not inference, limited the run.
`fanout_progress.json` lists each projector backend with `cuda_declines` (why
a view took the CPU route). Telemetry adds `sweep.publication.view_seconds`,
per-family `sweep.publication.<family>_seconds`, per-stage
`sweep.publication.stage[.<family>].{rank_input,projection,confidence_thread,layers_barrier}_seconds`,
`sweep.publication.scheduler_wait_seconds` and `sweep.rank_gpu.*` admission counters.

Requested sibling confidence evidence and journal decoding run outside the producer's
Python interpreter lock. Each sibling's journal rows are decoded by nogil
kernels that call the native zlib (`Native view journal decode:` in the log
names the backend; when the library does not qualify the Python route is used
and the bytes are identical). Evidence layers are written by spawn-started
worker processes that reopen the sealed journal view; the schedule passes the
identity its readers were opened with (`native_journal_identity`), and a
caller that supplies other crops keeps evidence on producer threads. If a
worker dies, the pool joins every worker, the affected layers are rewritten on
producer threads (identical bytes) and later views use threads. Telemetry adds
`sweep.publication.evidence_process.{jobs,cpu_seconds,wall_seconds,queue_seconds,thread_fallback_layers}`,
the `broken` gauge and `stage.confidence_process_seconds`; the process CPU
gauge of the producer does not include its workers.

Sibling layer files are written the same way. The producer's sinks keep every
layer name, colour, manifest row and publication record; NRRD worker processes
write the native and preview files and return the plans they wrote, so the
files and rows equal the thread path byte for byte. Layers a worker cannot
reproduce (live arrays, z-sharded layers, a hardware or substituted codec) stay
on producer threads, as does everything after a worker dies (the pool joins its
workers and rewrites the affected layers on threads). `YOLO_TTA_NRRD_GZIP_WORKERS`
is the producer-wide member-compression budget: each NRRD worker uses
`max(2, budget / XTA_YOLO_SWEEP_NRRD_PROCESSES)` threads (budget = the setting,
else half the CPUs), and the producer keeps the full value for the layers it
still writes. Workers write their own per-process telemetry files, so sum
`nrrd.*` counters across those files; the producer adds
`sweep.publication.nrrd_process.{jobs,wall_seconds,cpu_seconds,queue_seconds,thread_fallback_layers}`.
The log states which path each pool took (`Sweep NRRD files: ...`,
`Sweep confidence evidence: ...`); a pool that cannot start (process or
descriptor limits) prints a warning and the run continues on threads.

Intermediate sweep files (native journal, projected stores, decoded source,
rank backing and each cutoff's scratch) are written once, shared between the
family's processes and deleted. On a node with ample memory, `--temp` on a
RAM-backed filesystem such as `/dev/shm` avoids their disk writeback; the family
workspace holds roughly the source volume plus the projected stores and the
native journal (tens of GB at 1929x3064x3023). tmpfs pages count against the
job's memory limit and cannot be reclaimed, so choose it only when that much
memory is spare; if a job is killed, the workspace stays until `/dev/shm` is
cleared. `manifest.json` records the workspace and whether it is memory-backed
in `work_storage`.

After the producer, the later cutoffs finish their independent final union,
reconciliation, Global, other requested outputs and receipts. When the producer
recorded no detector fallback and published its shared source (and resized
cube), they run concurrently; their output lines carry a `[conf 0.NN]` prefix.
`XTA_YOLO_SWEEP_CONSUMER_PARALLEL` takes a positive integer or `auto` (the
default), which admits one concurrent cutoff per three source-volume sizes of
available memory, capped by the job's cgroup headroom. When memory is known,
each concurrent child budgets about 40% of its share for dense workspaces and
40% for transient parent work. Inference counts toward the dense budget.
Smaller explicit limits still apply. Where physical-core topology is available,
active children use disjoint CPU masks that keep SMT siblings together.
Existing augmentation, interpolation and output worker overrides are divided
among the slots. SAM, CUDA SDF, CUDA `keep_objects` and retained native views
that can backproject on CUDA run in cutoff order when they share GPUs.
`manifest.json` records the decision in `consumer_execution`. The Slicer
bundle is assembled after those branches settle.

For SDF or SAM, the shared native masks let later children bypass
the detector on covered views. Tracking, bridge decisions, interpolation and
publication of their results remain independent per cutoff. The same applies
where a policy needs branch-specific processing. An uncaptured view runs the
existing scalar pipeline. Current fast paths target **batch 1, full-frame**
direct CUDA instance inference (including eligible TensorRT rings) and
OpenVINO CPU inference. Hybrid CPU/GPU selection, tiling, external augmentation,
additional detector cleanup gates, image capture and unsupported backends can
select the scalar route. The run records the reason. A singleton range has no
sibling work to share.

**Checking a partial run.** Read `fanout_progress.json` at the sweep root.
`rendered_frames`, `forward_calls` and `mask_decodes` count work reported by the
participating producer tasks. Each cutoff lists completed native views,
projected views, and publication decisions. An empty view may settle without
a scalar layer file; `published_layers` counts the logical layers actually
written. `geometry_passes` includes the floor's separate ordinary projections
and the executed sibling traversals. `sibling_projectors` breaks down the
shared work by backend and includes crop reads and dense reads. The file is
refreshed at task/view handoff
and can lag work in flight. `producer_complete` marks the end of this upstream
stage. The top-level `manifest.json` gives the final family status, scope of
reuse, fallback counts, and actual cutoff receipts.

Use fast local scratch (`--temp`) for large studies. The native mask/score
journal is bounded by `XTA_YOLO_SWEEP_NATIVE_GIB` (default 64 GiB), and can be
retired one view at a time after early projection/publication. Projected stores
are retained for the corresponding cutoff's finalization. Their admission uses
available scratch with headroom; `XTA_YOLO_SWEEP_PROJECTED_GIB` can set an
explicit cap. `fanout_storage.json` records their live, peak and retired bytes
when early publication is active. With no cap, the producer writes these
counters after the floor views settle or when an unfinished producer closes.
Set `XTA_YOLO_SWEEP_QUOTA_DIR=/dev/shm` on a single Linux node to place just
the shared SQLite reservation ledger and its journal on tmpfs. Leave `--temp`
on a disk with room for the mask, score and replay payloads. The sweep creates
a private directory when the quota root differs from `--temp`, records it in
`work_storage.native_quota`, and passes the same database path to every branch.
The coordinator removes the directory after the workers settle.
For comparison or diagnosis,
`XTA_YOLO_SWEEP_VIEW_MAJOR=0` selects the previous cutoff-major execution order.
Eligible scalar fallback uses its separate transient raw-forward cache
(`XTA_YOLO_SWEEP_CACHE_GIB`, default 64 GiB). Its reuse still depends on
matching device, precision, renderer, batch and available space. Source/model
fingerprints are reused across cutoffs.

The top-level `manifest.json` records the family status, cutoff receipts and
forward-cache statistics. Branch receipts and per-cutoff binary media live in
`confidence_cutoffs/conf_NNN/`, where `NNN` is the integer cutoff in
hundredths. With `--save nrrd`, the canonical segmentations live in
`review/`. Open `review/manifest.json` to select the native grid or a requested
preview grid. Each grouped `.seg.nrrd` holds layers from one cutoff. Groups
can be split to bound their size and the space occupied by distant crops.
The catalogue records segment identity, channel and reference geometry for
edit export and native recovery. Keep the review and branch receipts together.
A Python reader can pass a `review/<grid>/sources/conf_NNN/*_nrrd_manifest.json`
catalogue to `XTA.reconciliation_io.read_layer_manifest` and read the selected
layer by slabs. The group index identifies its gzip byte range. Readers that
stop after the first gzip member, including `pynrrd` 1.1.3's ordinary
`nrrd.read`, cannot decode every layer of a multi-member group.
A store-backed Slicer review writes its edit log beside
the sweep folder, in `<sweep folder>.cutoff_edits/`; keep that folder with the
sweep.

`--save summary` writes one family `summary.txt` with the existing scalar
summary text under a heading for each validated cutoff in requested order.
`--save high_quality` writes `heatmaps/native.mkv`; `--save low_quality`
writes `heatmaps/low_quality/<token>.mp4` for each preview tier. Each video
shows how many finalized cutoff masks contain each voxel on that grid.
Zero support preserves the gray source. The `heatmaps` records in the family
manifest give the exact cutoffs, colors, frame rate and opacity.
The separate binary exports preserve eligibility at each cutoff, including
non-nested results.

For requested NRRDs or heatmaps the coordinator tests directory publication
on the output filesystem before it launches a cutoff. On a POSIX filesystem
that rejects exclusive rename, the publisher reserves the final directory
name with `mkdir` and renames the completed directory over its own empty
reservation. If that promotion fails, the family manifest records the completed
output-side stage in `completed_publication_stages`, relative to the family
root. This stage survives cleanup of node-local `--temp`. Failures while
generating a product can still require the original scratch inputs.
The family manifest records elapsed seconds for attempted heatmap and review
publication in `publication_seconds`. The job log also reports each completed
review grid's elapsed time and compressed group bytes.

When a scalar cutoff exits normally and records omitted SAM work, XTA retains
its outputs and continues through the remaining cutoffs. The family finishes
with a warning and status `incomplete`. Its manifest lists the affected cutoffs
and their SAM omissions. Check family status and the published cutoff list
before using partial results.
Set `YOLO_TTA_SAM_REQUIRE_COMPLETE=1` to make a SAM omission fail its scalar
process. A nonzero scalar exit stops the sweep.

You can also package completed independent scalar runs from the same source
and model. From the source checkout, for instance:

```text
python tools/pack_confidence_sweep.py --root STUDY --branch 0.50 run_A --branch 0.80 run_B
```

Here `run_A` and `run_B` are existing scalar output directories beneath
`STUDY`, with their original run and NRRD manifests. The packer writes
`STUDY/review/manifest.json` and the grouped files. Use `--name` for a different
destination name. Cutoffs are packaged concurrently (`--workers`, default one
per cutoff, at most 8 and at most the CPU count, each with its own
`--memory-mib`). The source runs and the review form one movable study tree.
Slicer opens such a review as per-cutoff segments, because masks from
independent runs need not nest across cutoffs.

Confidence-mode reconciliation retains observed numeric evidence. Use
`--save confidence` to keep it independently of reconciliation in scalar runs
or confidence sweeps. The manifest
in `reconciliation_evidence/` identifies the model, layer, task-specific score
meaning and coordinate space. These bytes describe surviving observed
predictions before interpolation. Zero includes unknown support. Generated
bridges, hole fill and manual additions do not gain a measured YOLO score.
To inspect one model and layer as an ordinary uint8 scalar NRRD, use the
exact `model_name` and `layer_key` from that catalog, for example:

```text
python tools/export_observed_score.py --run-manifest sample_sweep/confidence_cutoffs/conf_050/manifest.json --model best --layer transverse__tta_a0__fullframe__yolo__pre_interpolation --allow-native-projection --output transverse_score.nrrd
```

The export streams planes. Deferred native-view scores use bounded source-grid
projection; `--temp` selects its staging location. A portable compact confidence
bundle can supply the input manifest for an export at its own resolution.

## Review and snip in 3D Slicer

1. Load the matching intensity volume. Install/load the XTA toolbar as described
   in the [Slicer cleanup guide](slicer_pta_cleanup.md). Use **Importer** with
   the sweep's `review/manifest.json`. Importer offers only grouped grids,
   with the smallest LQ tier first. Older reviews can describe the native grid
   by reference to scalar NRRDs. By default it opens the grid **store-backed**: every cutoff is read
   into one in-memory store and stays editable. Importer asks which cutoff to
   display, with the review default first, and, when the grid has an edit log,
   whether to resume it or start fresh. In the Slicer Python console the
   equivalent entry point is
   `ImportXTASegments(source_manifest=r"/path/to/sample_sweep/review/manifest.json", cutoff_cents=70)`.
   Importer loads per-cutoff segments instead when you choose
   **Per-cutoff segments (legacy)…** or pass a list of cutoffs or
   `cutoff_cents="all"`, and on its own for reviews packaged from independent
   scalar runs, native-resolution grids and stores that would not fit in
   memory. It estimates that memory from the manifest before reading any mask.
2. Scroll over the intensity in any slice view. The 2D coverage colors count
   visible direct prediction layers at the displayed cutoff. One, 2–3, 4–7,
   8–15 and 16+ contributions have distinct colors. Generated-only and
   Global-only support get separate accents where there are no direct votes.
   Correlated views can contribute. Use **2D Cutoff** to change the displayed
   operating point; a store-backed review rewrites its working segments at
   that cutoff. With per-cutoff segments, a cutoff that is not loaded shows
   its published direct counts read-only, without the generated-only and
   Global-only accents.
3. Place F points, L lines or C/CC curves and use **Remove Islands** or
   **Remove Slices**. On a store-backed review one click edits every cutoff
   with **Scope: All Cutoffs**, the default: the marked islands go at the
   displayed and stricter cutoffs, and at looser cutoffs only the looser
   growth that belongs to them (the union basin). **Scope:
   Displayed+Stricter** leaves looser cutoffs unchanged. Each click reports
   what it removed at each cutoff, flags layers that lost much, and asks first
   above 500,000 voxels. Segment Editor erase applies at every cutoff and
   paint at the displayed and looser cutoffs. **Undo** and **Redo** (Ctrl+Z /
   Ctrl+Y) revert any of these, and an edit log beside the sweep folder keeps
   them across a crash or a reopen. With per-cutoff segments each mask uses
   its own connectivity at its own cutoff. By default (**Scope: Displayed**)
   one click changes only the masks at the displayed cutoff, as when one
   cutoff was reviewed at a time; the scope button also offers *Displayed and
   stricter* and *All visible cutoffs*. Lower cutoffs can join a marked island
   to much larger regions, so a wider scope can remove far more at those
   cutoffs. The two kinds remember separate scopes, and a choice on one never
   carries over to the other; with no review open, the button sets the
   per-cutoff scope. Ordinary segments are not affected by the scope. The displayed counts follow the edits.
4. A store-backed review shows only the displayed cutoff's Global in 3D.
   Per-cutoff segments show the displayed cutoff's complete segments with
   their native colors and smoothed surfaces; **3D: Show All Cutoffs** shows
   every loaded cutoff. A coverage image is a display layer; the complete
   binary geometry supplies the 3D surface and the training mask. Global
   update actions run at every cutoff of a store-backed review, as one undo
   entry, and otherwise use the contributors belonging to each loaded cutoff.

Use **Export XTA Edits** to preserve branch identities and edited masks for
native processing. The toolbar button exports the displayed cutoff; in the
console `ExportXTAEdits(review_manifest)` exports every cutoff of a
store-backed review, or every loaded cutoff of per-cutoff segments, and
`cutoff_cents=[...]` a subset. When a store-backed review's Global is stale,
export first shows how much Update Global would change it at each exported
cutoff, because Update Global keeps only the largest 6-connected island;
**Export as is** is the default. Keep each LQ/native pair with the intensity
and run that produced it. A 2D coverage count alone cannot recover the
contributing masks.

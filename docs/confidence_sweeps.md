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
uses the live outputs to produce each cutoff's native mask and observed scores.
The scalar writer already owns the lowest-cutoff pixels, so its entry in the
shared journal carries frame counts only. Higher cutoffs retain their exact
mask and score crops and can share a stored crop when the decoder proves the
accepted predictions are identical. For a
long grid, the tiled direct CUDA decoder can group candidates by score and
accumulate prototype maxima from higher to lower thresholds. Small grids and
the TensorRT ring use their native cutoff decoding on the shared forward.

With NRRD decomposition and interpolation/extrapolation off, the normal scalar
route handles the lowest cutoff. Compatible higher-cutoff masks from the same
view share a categorical geometry traversal. The publisher writes their
complete native and requested preview NRRDs into their own cutoff directories
as views finish. Masks which cross between cutoffs can require more than one
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

The ordinary shared slice (`YOLO_TTA_GPU_PROJECTION_OVERLAP_GIB`) is 4 GiB, or
6 GiB in a sweep producer so the largest sibling projections fit inside the
memory that inference sizing already leaves free. Floor projections keep
priority over sibling projections for those slices.

Each count accepts a positive integer or `auto`. A GPU-projected view packs its
planes on up to 4 host threads. The scheduler also keeps going when the bytes
above the journal mark could not be freed by the queued publications (workers
then take the exact scalar route if the quota fills). Sibling NRRD encoding uses one
shared pool sized to what concurrent publications can queue (at most a third
of the CPUs); each cutoff's observed evidence is written beside those encodes.
The family `manifest.json` reports `upstream_fanout.early_publication` with
the backlog peak, the journal's peak fill and the seconds the scheduler waited.
A nonzero wait means publication, not inference, limited the run.
`fanout_progress.json` lists each projector backend with `cuda_declines` (why
a view took the CPU route). Telemetry adds `sweep.publication.view_seconds`,
per-family `sweep.publication.<family>_seconds`, per-stage
`sweep.publication.stage[.<family>].{rank_input,projection,confidence_thread,layers_barrier}_seconds`,
`sweep.publication.scheduler_wait_seconds` and `sweep.rank_gpu.*` admission counters.

Intermediate sweep files (native journal, projected stores, rank backing) are
written once and deleted. On a node with ample memory, `--temp` on a RAM-backed
filesystem such as `/dev/shm` avoids their disk writeback; the family workspace
holds roughly the source volume plus the projected stores (tens of GB at
1929x3064x3023).

After the producer, the later cutoffs finish their independent final union,
reconciliation, Global, other requested outputs and receipts. When the producer
recorded no detector fallback and published its shared source (and resized
cube), they run concurrently; their output lines carry a `[conf 0.NN]` prefix.
`XTA_YOLO_SWEEP_CONSUMER_PARALLEL` takes a positive integer or `auto` (the
default), which admits one concurrent cutoff per three source-volume sizes of
available memory, capped by the job's cgroup headroom. Otherwise they run in
cutoff order. `manifest.json` records the decision in `consumer_execution`. The
Slicer bundle is assembled after those branches settle.

For SDF or SAM, retained native masks and scores let later children bypass
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
when early publication is active. For comparison or diagnosis,
`XTA_YOLO_SWEEP_VIEW_MAJOR=0` selects the previous cutoff-major execution order.
Eligible scalar fallback uses its separate transient raw-forward cache
(`XTA_YOLO_SWEEP_CACHE_GIB`, default 64 GiB). Its reuse still depends on
matching device, precision, renderer, batch and available space. Source/model
fingerprints are reused across cutoffs.

Each scalar result is in `confidence_cutoffs/conf_NNN/`, where `NNN` is the
integer cutoff in hundredths. The top-level `manifest.json` gives the status,
branch paths and forward-cache statistics. A successful NRRD review has
`review/manifest.json`, stock grouped `.seg.nrrd` files and per-cutoff direct
coverage labelmaps. Each requested preview tier is a grouped review grid:
each `.seg.nrrd` group holds one cutoff, so Slicer can load a subset of
cutoffs. When a preview tier exists, the native grid is reference-only. The
review lists its per-cutoff source manifests and segment identities for edit
export, but never reads or re-encodes the native masks; they stay in the
branch NRRDs. Without a preview tier, native is the grouped review grid. Keep
the review and its corresponding branch directories together for edit export
and native handoff.

When a scalar cutoff exits normally and records omitted SAM work, XTA retains
its outputs and continues through the remaining cutoffs. The family finishes
with a warning and status `incomplete`. Its manifest lists the affected cutoffs
and their SAM omissions. Every branch remains available in its scalar directory.
The grouped Slicer review requires a complete family, so XTA skips it in this case.
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

Sweeps retain observed numeric evidence. For a single scalar run request
`--save confidence` to keep it independently of reconciliation. The manifest
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
   with the smallest LQ tier first; a reference-only native grid is never
   loaded. It then asks which cutoffs to load as editable masks, with the
   review default first. Run Importer again to add cutoffs. In the Slicer
   Python console the equivalent entry point is
   `ImportXTASegments(source_manifest=r"/path/to/sample_sweep/review/manifest.json", cutoff_cents=[70])`
   (`cutoff_cents="all"` loads every cutoff). Memory for the live count and
   display arrays scales with the chosen grid, in addition to Slicer's masks
   and surface meshes for the loaded cutoffs.
2. Scroll over the intensity in any slice view. The 2D coverage colors count
   visible direct prediction layers at the displayed cutoff. One, 2–3, 4–7,
   8–15 and 16+ contributions have distinct colors. Generated-only and
   Global-only support get separate accents where there are no direct votes.
   Correlated views can contribute. Use **2D Cutoff** to change the displayed
   operating point. A cutoff that is not loaded shows its published direct
   counts read-only, without the generated-only and Global-only accents.
3. Place F points, L lines or C/CC curves and use **Remove Islands** or
   **Remove Slices**. Each mask uses its own connectivity at its own cutoff.
   By default (**Scope: Displayed**) one click changes only the masks at the
   displayed cutoff, as when one cutoff was reviewed at a time. The scope
   button also offers *Displayed and stricter* and *All visible cutoffs*.
   Lower cutoffs can join a marked island to much larger regions, so a wider
   scope can remove far more at those cutoffs. Ordinary segments are not
   affected by the scope. The displayed counts follow the edits.
4. The 3D view shows the displayed cutoff's complete segments with their
   native colors and smoothed surfaces. **3D: Show All Cutoffs** shows every
   loaded cutoff. A coverage image is a display layer; the complete binary
   geometry supplies the 3D surface and the training mask. Global update
   actions use the contributors belonging to the corresponding loaded cutoff.

Use **Export XTA Edits** to preserve branch identities and edited masks for
native processing. The toolbar button exports the displayed cutoff; in the
console `ExportXTAEdits(review_manifest)` exports every loaded cutoff and
`cutoff_cents=[...]` a subset. Keep each LQ/native pair with the intensity and
run that produced it. A 2D coverage count alone cannot recover the contributing
masks.

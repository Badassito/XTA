# XTA architecture

XTA has three production modes: test-time augmentation (TTA), pretraining
augmentation (PTA), and label-time augmentation (LTA). This document describes
the current data flow and ownership boundaries. Detailed measurements, rejected
experiments, and release history live outside this repository in the workspace's
`Scratch/Data/XTA/History/Experiment_Log.md`. Those local experiment files are not
part of an installed package or source bundle. The complete pre-trim architecture
document is available in Git with `git show 597fc45:ARCHITECTURE.md`.

## Entry points and shared contracts

`GPT-6-Astra-Ultra_v24.0.5_SLURM.py`, the installed `xta` command, and
`python -m XTA` enter `XTA.cli.run()`. The CLI selects exactly one mode and
validates that mode's grammar before importing its heavy runtime. `tta_mode`
enters `pipeline.main`, `pta_mode` resolves `PtaConfig` before `pta_runtime`
enters `pta.main`, and `lta_mode` resolves `LtaConfig` before `lta_runtime`
plans and executes propagation. Dependency-heavy runtimes initialize only
after their mode has been selected.

| Shared boundary | Owner |
| --- | --- |
| Immutable geometry and sampling identity | [sampling](XTA/unification/sampling.py), [contracts](XTA/unification/contracts.py), [views](XTA/unification/views.py), [tiles](XTA/unification/tiles.py), [geometry](XTA/geometry.py), [render batches](XTA/render_batch.py) |
| Source and worker resources | [workspace](XTA/workspace.py), [media](XTA/media.py), [runtime](XTA/runtime.py), [TTA workers](XTA/workers.py) |
| Input/output identity and manifests | [launch context](XTA/unification/context.py), [manifest helpers](XTA/unification/manifest.py), [TTA manifest](XTA/unification/tta_manifest.py) |
| Image-space and source-space projection | [backprojection](XTA/backprojection.py), [sparse projection](XTA/sparse_projection.py), the `cylindrical_*`, `spherical_*`, and `tilted_azimuthal_projection*` modules |
| Common filtering and output | [Gaussian](XTA/gaussian.py), [finalization](XTA/finalization.py), [outputs](XTA/outputs.py), [packed publication](XTA/packed_publication.py), [memory grants](XTA/publication_memory.py) |

A `RasterPlan` fixes a concrete view, output raster, optional tile, and sampling
policy. Its digest identifies the request and is recorded with output provenance.
`ForwardSamplingPolicy` specifies the order of coordinate transforms, sampling
kernels, and boundary rules. A runtime binds a data role and implementation
through `require_forward_sampling()`; missing bindings fail rather than silently
selecting another kernel. Image intensity and categorical labels use distinct
sampling rules. Categorical rendering uses nearest spatial taps. Tilted
Cartesian and Azimuthal label planes blend adjacent stack planes and threshold
binary foreground and coverage at 0.5; shell views use nearest voxel sampling.

`RenderRequestBatch` describes logical frame addresses and plan identities;
`render_batch.RenderBatch` owns the actual rendered frames consumed by a model
or image sink. Full frames and tiles share geometry. Cartesian, tilted
Cartesian, Azimuthal, Radial cylindrical, and Spherical QSC views use their
own physical coordinates. Radial and Spherical patches retain their shell,
radius, and patch origin so backprojection can reconstruct source support.
A view is projected to source coordinates only after its own frame and variant
ownership has settled. Source-volume restoration to native shape is distinct
from the working cube.

External policies own transforms *after* shared rendering. TTA and PTA accept
CPU and GPU policy paths through `--augmentation`. Policy identity and selected
backend are recorded in manifests. A policy must preserve its paired image and
label contract; unsupported combinations fail during validation. PTA executes the
same source bytes whose SHA-256 it verifies, rather than reloading a policy or
accepting cached bytecode after the identity check. Ultralytics adapters check
the private APIs they replace and fail worker startup if a required patch cannot
be installed.

## Environment controls after v24.0.4 cleanup

The supported defaults now directly select ROI-only CPU mask resizing, GPU
proto union, retina flattening and eligible GPU warping, tiled proto composition, bilinear
Tilted intensity sampling, slice-local interpolation labels, bounded bridge
merges, and run-based topology adjacency. Their old comparison switches no
longer select alternate behavior. Categorical sampling, generic compact label
IDs, and automatic eligibility and failure fallbacks retain their separate
contracts.

Output publication uses cropped CVOL stores, eligible compact Spherical CPU
publication, crop-row NRRD streaming, extent skipping, and live immutable global
layers. Final fusion uses grouped restore and the eligible native sparse CPU
path. Grouping unions before resampling can preserve subpixel support that
separate integer AREA restores would round away; old-path voxel equality is
not the acceptance criterion.

Worker GPUs retain inference-first ownership, full-frame workers write bounded
direct unions, and result queues wake the scheduler through the result pump.
Split-view hole filling runs after view completion; single-lease eligibility
still permits device filling. Resource controls remain available where they
change concurrency or peak storage. Two redundant booleans were consolidated:
use `YOLO_TTA_GPU_INPUT_STAGING_BATCHES=0` to disable input staging and its eager
source warmup, and `YOLO_TTA_HYBRID_GPU_STEALBACK_MAX_FRACTION=0` to disable hybrid
GPU assistance. These replace `YOLO_TTA_GPU_INPUT_STAGING=0` and
`YOLO_TTA_HYBRID_GPU_STEALBACK=0`, respectively.

## Required compiled CPU backends

Numerical runtimes require Numba alongside NumPy, OpenCV and SciPy. Missing or
broken Numba imports raise a diagnostic instead of selecting interpreted bulk
work. Compiled topology, interpolation, Spherical/Radial CPU projection, sparse
scatter and packed publication propagate kernel failures. Configuration and CLI
help/version discovery keep their lightweight import boundary.

Independent slow implementations live in `tests/reference_backends` for
qualification. Interpolation grows its compiled workspace for valid fragmented
windows; Radial uses a bounded compiled direct pull when a factored ownership
plan cannot fit. Native-array operations that preserve aliasing/conversion
semantics and explicit OpenVINO CPU inference remain supported. GPU eligibility
and safe fallback to compiled CPU remain separate from compiler availability.

H100 is the primary deployment target and A100 is the fallback target; Volta/V100
is outside the supported deployment scope. See [production backend policy](docs/PRODUCTION_BACKENDS.md)
for the test-reference boundary and development benchmark commands.

## TTA: inference to source union

The TTA pipeline is coordinated by `pipeline`, `tta_scheduler`, `tta_prediction`,
`tta_lifecycle`, `tta_terminal`, and `tta_outputs`. `view_prepare` owns admitted
parent preparation, component-projection reservations, and inference-to-postprocess
lease handoffs through explicit state objects. The scheduler owns mutable
task identities, backend claims, worker liveness, parent memory credits, and
result transport. An execution target is a local CUDA GPU or a socket-local
OpenVINO process. Worker processes receive picklable contracts; bulk source and
result data use shared descriptors or artifact paths. CUDA, TensorRT, and
OpenVINO are initialized inside their respective runtime owners.
Immutable task descriptors keep the scheduler's shape and byte accounting
stable across refill passes; mutable owner, credit, and readiness decisions
are evaluated at selection time.

Each physical view renders frames and optional tiles/augmentations, performs
model inference, assembles the accepted masks, and settles its image-space
result. Completed physical views can project to source space while other views
continue inference. `assembly` and `tta_terminal` reduce these independent
layers into a single source union. `interpolation` can add bridges between
eligible frames; those bridges retain their own provenance. `finalization`
applies requested global postprocessing after the full union exists.

`--task segment` (default) composes YOLO instance masks from detections and
prototypes in `inference`. `--task semantic` decodes raw one- or two-channel
logits in `semantic_inference`: sigmoid for one foreground channel, or softmax
for background/foreground channels. `--conf` thresholds foreground probability
before binary cleanup, including thresholds below 0.5. `--min_conf` retains a
connected component only if its maximum encoded confidence reaches the
threshold. Models with unsupported class counts or baked argmax output fail
rather than inventing missing probabilities. `tools/export_semantic_logits.py`
exports a raw-logit ONNX/OpenVINO graph when the ordinary export path bakes an
argmax. `semantic_cuda` and `semantic_trt` provide eligible CUDA cleanup and a
private two-context TensorRT ring; the instance pipeline retains its separate
inference path. GPU failures after a partially consumed task abort it rather
than replaying writes.

TTA augmentation runs each policy pass independently and records the policy
snapshot. Each pass owns its result and inverse-validity support; passes
contribute to the terminal union while interpolation uses the base pass.
Supported CUDA paths map accepted masks directly to owned parent windows.
Parent admission charges all passes in a policy group against the dense
memory window; file-backed results and retained outputs remain separate
storage consumers. Inference, completed canvases, source projection, and
publication have separate credits so one backlog cannot grow without bound.

`--reconciliation POLICY.py` evaluates completed source-space component layers
before global postprocessing. The additive layers remain available; the policy
changes the derived final mask. `reconciliation_policy` resolves external
policy identity; `reconciliation_components`, `reconciliation_geometry`,
`reconciliation_io`, and `reconciliation_runtime` consume bounded source slabs
and write its manifest. A plain union can reuse the assembled union. Weighted
and confidence policies read independent evidence. `confidence_*` modules
capture native pieces or source-aligned uint8 maxima of detector scores
(segment) or foreground probabilities (semantic); zero means unknown.
Interpolation does not fabricate confidence. Native evidence conversion is
explicit and bounded. `confidence_export` can match numeric evidence to
selected compact masks without loading the NRRD masks.

TTA can save the final source-volume semantic mask as grayscale PNGs with
`0` background and `1` foreground. This differs from `--save images`, which
saves rendered model inputs. NRRD layers and their manifests preserve view,
model, geometry, and policy identities. `tta_outputs` publishes a complete
run manifest only after model workers, projection, finalization, and all output
futures settle.

### TTA resource and failure boundaries

A parent canvas holds its dense memory credit through assembly and projection
until its immutable component backing is published. File-mode results and
support sidecars have separate accounting. The scheduler can lend an idle
worker GPU to eligible Radial or Spherical projection after queued inference
drains; CPU projection can proceed while admission waits. A projection may
switch at the first unpublished source slice. A failed preflight leaves CPU
progress intact. A failure after CUDA publication starts aborts that layer
rather than replaying partial output.
The exact compiled Spherical CPU pull is required for CPU projection and is
independent of approximate GPU geometry controls. The Radial CUDA owner packs
eligible bitsets on the device before transferring their bounded slice payloads;
it skips the full-bitset download for a proven-empty output. Both Radial packing
flags, `YOLO_TTA_RADIAL_GPU_BITSET_COMPACTION` and
`YOLO_TTA_PACKED_OWNER_PUBLICATION`, allow the legacy host export path for
diagnosis. D1 confidence retirement can mask a uint8 score crop on the device
before one host transfer when the crop has at least 16,384 pixels. Set
`YOLO_TTA_D1_GPU_MASK_MIN_PIXELS=0` to request that path for every CUDA crop,
or `YOLO_TTA_D1_GPU_MASK_CONFIDENCE=0` to select the two-crop host path.

Completed view layers are reduced by a single source-union writer. A dense
handoff credit bounds waiting source-sized volumes. Final sink joins precede
the complete manifest. TTA first writes an `in_progress` manifest and replaces
it only after successful completion. Reconciliation cannot consume unfinished
confidence publication.

## PTA: deterministic dataset construction

`pta` owns candidate membership, augmentation versions, split assignment, and
output identities before asynchronous rendering begins. `pta_classification`
tests foreground occupancy for background sampling, using native categorical
planes and cached full/tile index maps where possible. If semantic output keeps
every candidate (`--background_percent 1`) and has no offline copies, it skips
the costly all-view classification while retaining missing-label eligibility
and a foreground-preservation check. The summary marks unavailable class totals
as unmeasured. `pta_scheduler` packs retained candidates into source-frame work
so one decoded or resident volume serves many neighboring outputs.

`--task segment|semantic` selects YOLO task semantics for PTA. `--save labels`
continues the polygon-backed path. `--save semantic` writes class-index PNGs
under split-local `masks/` directories, paired by stem with images:
`0` background, `1` foreground, `255` ignore. The generated dataset YAML uses
`nc: 1`, `names: ['0']`, and `masks_dir: masks`. A missing YOLO label file is
unknown coverage in forced partial-label mode; an existing empty file is known
background. Foreground and coverage travel as separate categorical volumes
through geometry and paired augmentation; only publication combines them into
the 0/1/255 plane. A custom GPU policy used for partial semantic labels must
declare `mask_independent_geometry = True` before the same sampled transform
can be replayed for coverage. The bundled GPU policies declare this contract.
`--save binary` writes retained exact binary masks and grouped lossless videos
through `pta_binary`.

CPU rendering uses persistent workers and shared volume memory. Offline GPU
augmentation uses one persistent process per selected GPU. `pta_rendering`
prepares view plans; `pta_workers` renders and applies policies. GPU owners keep
foreground and coverage volumes resident when VRAM admission succeeds.
`pta_cuda_masks` coordinates their lifetime; `pta_cuda_cartesian`,
`pta_cuda_azimuthal`, and `pta_cuda_shells` project categorical values directly
to the final full or tile raster. Intensity and categorical sampling remain
separate. GPU projection events fence policy reads and source retirement.
Unsupported geometry or insufficient prelaunch VRAM selects the CPU categorical
reference path; a launched CUDA failure propagates. The manifest records which
backend actually rendered categorical items.

`pta_batch_pipeline` and `pta_gpu_publication` overlap bounded GPU policy
batches with encoding and host publication. Output tensors are snapshotted
before a policy can reuse buffers. `pta_publication` writes and verifies
image and label files; the nvJPEG batch uses staged files and atomic renames,
while ordinary image, text-label, and semantic PNG writers write their final
paths directly. `nvjpeg` and `nvtiff` select explicit GPU image encoders;
CPU formats use their requested encoders. When semantic output is requested
without polygon or binary output, foreground and coverage combine on GPU
before one class-index download. With those other outputs, publication also
downloads the separate planes and combines semantic IDs on the host. PNG
compression and filesystem writes use bounded host workers. A batch holds its
queue reservations until all image, semantic-mask, and label writes settle;
failures join admitted work before the worker reports failure.

PTA validates output ownership before cleaning an existing generated dataset.
Requested and effective image formats are recorded independently. Its complete
manifest is written last, after every file and worker closes. Unforced partial
volumes and sequences with encoded-frame gaps retain native depth and restrict
labels to Transverse full frames or tiles. Forced partial volumes without
encoded-frame gaps can use 3-D views, including supported shells, with unknown
semantic coverage preserved through projection. Fully labeled contiguous data
can use all supported views.

## LTA: authoritative seeds and bounded propagation

LTA takes a target volume, aligned exemplar masks, and a verified local SAM
bundle. Production uses native Transverse, angle-zero overlapping tiles.
`lta_inputs` discovers and validates the target and exemplar identities.
`lta_runtime` plans views and tile grids. `lta_scheduler` assigns one physical
view owner for its render cache and backprojection; idle devices may help with
unopened SAM sessions. `lta_workers` owns persistent GPU processes and each
process's model/tracker state. A live session stays on one worker. Results
commit in plan order even when execution completes out of order.

`lta_propagation` and `lta_windows` advance authoritative anchors through
bounded temporal windows, with independent backward and forward branches.
Sessions admit at most 30 frames and 128 objects. Authoritative foreground
is never replaced by a tracker prediction. `lta_tiles` and
`lta_tile_tracking` plan overlaps and spatial relays; `lta_relay_episodes`,
`lta_frontier`, and `lta_frontier_execution` settle new support over bounded
temporal waves. Relay identity includes lineage, tile, prompt frame, and
direction. Complete same-event masks merge before coverage checks. A subset
already visited in the same lineage and direction can hand off; new support
remains eligible. The finite growth guard fails publication if relays have not
settled.

Workers return sparse cropped, row-packed foreground and coverage artifacts.
The coordinator verifies each packet, ORs its indexed crops into the private
view, and releases it before admitting more work. `lta_coverage` stores the
ephemeral visited-transition ledger; `lta_union_artifacts` owns the sparse
view union. Once a physical view is complete, it receives final two-dimensional
hole fill and one backprojection into the native union. `lta_postprocessing`
writes requested filter checkpoints; exact authoritative foreground is
restored in the final output after destructive filters. `lta_outputs` publishes
the mandatory final NRRD and complete manifest after checkpoint and identity
checks. A checkpoint can remain after a later failure; only a complete manifest
marks run success.

`lta_sam` resolves and audits the local SAM 3.1 assets; `lta_worker_adapter`
owns model construction and tracker calls. The production feature path in
`lta_tracker_features` skips unused grounding detection while retaining the
tracker's expected visual features. GPU workers divide the visible CPU
allocation and record their process/device identities. Increasing
`--lta_workers_per_gpu` creates independent model contexts, so model memory and
CUDA scheduling scale with that count. LTA does not use YOLO's `--task` flag.
Cross-anchor identity matching remains a diagnostic facility in
`lta_tracklets`; production merges independent anchor chains by recall union.

SAM import and construction run in the isolated model worker. XTA restores the
caller's CUDA-matmul and cuDNN TF32 settings after those operations, including
failure paths. SAM's model-process BF16 autocast behavior remains confined to
that worker; a predictor is not constructed inside a shared TTA/PTA process.

## Publication, diagnostics, and validation

Large source and mask arrays use explicit ownership: shared mappings, worker
descriptors, bounded RAM credits, or path-backed artifacts. Native shell
payloads can use parent-owned Linux memfds with spill to disk under pressure.
Immutable component stores may outlive a dense canvas, but each retained layer
still consumes filesystem capacity. Scratch placement can therefore affect
RAM usage when its filesystem is memory backed.

Retiring a shared NumPy mapping does not invalidate live views or independent
arrays backed by the same mmap. The mapping's last reference governs unmapping
and memfd-owner release. Named scratch deletion follows that lifetime and checks
file identity before unlinking and skips an observed replacement. Callers retain
ownership of their unique scratch names until retirement completes; borrowed input
paths and keep-temp artifacts are not deletion targets. Windows
deletion may briefly wait for the mapping destructor. Owners must release their
references explicitly; the retirement wait helper can verify cleanup and reports
consumers that still retain the mapping. Forked workers reset inherited
retirement queues and cannot delete their parent's pending scratch files.

`outputs` and `nrrd_spans` stream native NRRD rows and crops; publication
records durability and releases source references after completion. Optional
native codecs and hardware copy/compression helpers live in `native/`,
`intel_compression`, and `intel_dsa`. Encoder/backend admission is explicit:
a requested unavailable accelerator fails or uses the documented supported
fallback before an output starts. Partial artifacts and worker failures never
produce a complete run manifest.

Run, LTA, NRRD-sidecar and confidence JSON writers share `json_publication`.
They reject non-finite values, sync a unique sibling stage, replace the destination,
and sync its directory where supported. Failures are not acknowledged as successful
publication. Same-destination writes serialize; independent destinations can sync
concurrently. Forked workers reset inherited publication locks.

Runtime diagnostics separate queue waits, rendering, inference, projection,
encoding, publication, and worker retirement. These are overlapping stage
timings, so summing them does not recover wall time. Validation tools under
`tools/` check geometry, model backends, semantic decoding, PTA categorical
projection, and LTA propagation. Tests live under `tests/`. Benchmark receipts
and full workload evidence belong outside the repository in
`Scratch/Data/XTA/History` and task-specific Scratch directories.

Release qualification runs the full suite from the repository root before
inventory verification and source-bundle construction:

```powershell
python -B tools/qualify_release.py --output-dir ../Scratch/Releases/v24.0.5-validation
```

The default requires a clean Git checkout and bundles committed Git bytes.
`--snapshot` explicitly validates an uncommitted development tree and labels its
bundle accordingly. Each step must pass without changing the source tree;
the receipt and logs stay in the requested Scratch directory. Tests establish
Ultralytics settings outside the checkout and check for leaked dependency stubs
and TF32 settings. CUDA tests share the workspace's atomic `Scratch/Temp/GPU_LOCK`.
The gate requires an available CUDA device and successful execution of the
CUDA geometry/policy cases that exposed the TF32 leak. `--snapshot --cpu-only`
records explicitly reduced coverage and does not qualify a release.
